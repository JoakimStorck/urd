"""
Interaktivt läge för URD.

    $ urd
    urd> Vem är proprefekt vid IIT?
    ...
    urd> Berätta mer.
    ...
    urd> .quit

Motsvarar `python` utan argument: en tolk snarare än ett kommando.
Skillnaden mot `urd ask` är inte bekvämlighet utan att sessionen LEVER
— QUD, aktiva dokument och rework-tillstånd finns kvar mellan frågor,
vilket är hela poängen med URD:s samtalsmodell. Varje `urd ask` startar
en ny process och tappar allt det.

TVÅ DESIGNBESLUT

SAMMA BETEENDE MED OCH UTAN SERVER. Sessionshanteringen ligger i
RagService.converse, så samtalsminnet fungerar likadant lokalt som
över HTTP.

LAT UPPSTART. Modellerna laddas först när de behövs, inte när läget
startar. En användare som går in för att köra .hjälp eller .status ska
inte vänta en halvminut på embeddingmodell och cross-encoder. Kör en
server redan används den i stället, och då laddas ingenting lokalt.

PUNKTKOMMANDON. Allt som inte inleds med punkt är en fråga till
dokumentbeståndet. Det gör gränssnittet förutsägbart: ingen fråga kan
råka tolkas som ett kommando, oavsett hur den formuleras. Samma
konvention som psql och sqlite3, och den lämnar utrymme för ett
framtida urd-språk utan att kollidera med naturligt språk.

ENGELSKT GRÄNSSNITT, SVENSKA SVAR. Kommandonamn, flaggor och
utskrifter är engelska, av samma skäl som CLI:t och API:t är det: ett
gränssnitt går inte att byta språk på i efterhand utan att bryta
någons vana eller något skript. SVAREN följer källorna och är svenska
— verktyget talar engelska, dokumenten talar svenska, och det är två
olika saker.

Koden och kommentarerna förblir svenska. De bär mätningar och
motiveringar i domänens egna termer (proprefekt mot prefekt,
bedömningsgrunder mot arbetsuppgifter), och översatta tappar de sin
precision.
"""

from __future__ import annotations

import sys

import requests
import typer

from app.schemas import ChatResponse

PROMPT = "urd> "
CONTINUATION = "...  "

BANNER = """URD interactive mode. Type a question, or .help for commands.
The session lives until you type .new or exit."""

HELP = """Commands (anything else is treated as a question):

  .help             show this
  .status           backend, session, loaded models
  .new              start a new session (forgets conversation history)
  .sources on|off   show or hide sources in answers
  .debug on|off     show or hide technical detail
  .attest <term>    look up what the corpus attests about a role
                    (.attest "N.N." --person to look up a person)
  .whoami           who the server thinks you are
  .login            sign in with your user name and password
  .logout           end the session on the server
  .stop             shut down the SERVER (the session continues locally)
  .quit             leave interactive mode (Ctrl-D also works)

The session is kept between questions: follow-ups such as "berätta mer"
and "stämmer det?" work as they do in the web interface.

Questions are answered in the language of the sources — the interface
is English, the documents are Swedish."""

# Svenska namn behålls som ODOKUMENTERADE alias. De kostar en
# uppslagstabell och ingen dubblerad logik, och de skyddar
# muskelminne och befintliga heredoc-skript från att brytas av
# språkbytet. Att de inte står i HELP är avsiktligt: nya användare ska
# lära sig ett namn per kommando, inte två.
#
# Tas de bort någon gång ska det vara ett beslut, inte en städning —
# därför denna kommentar.
_ALIASES = {
    "hjälp": "help", "hjalp": "help", "h": "help", "?": "help",
    "ny": "new",
    "källor": "sources", "kallor": "sources",
    "stopp": "stop",
    "avsluta": "quit", "exit": "quit", "q": "quit",
    "vemärjag": "whoami", "vemarjag": "whoami",
    "loggain": "login", "loggaut": "logout",
}


class Repl:
    """Tillståndet för en interaktiv session."""

    def __init__(
        self,
        server_url: str,
        show_sources: bool,
        show_debug: bool,
        note=None,
        token: str | None = None,
        scripted: bool = False,
    ):
        # note() skriver statusrader dit de hör: stdout i terminal,
        # stderr i skriptläge. Default håller äldre anropare fungerande.
        self.note = note or typer.echo
        self.server_url = server_url.rstrip("/")
        # Token mot servern. Interaktivt läge saknade tidigare varje
        # sätt att ange den, vilket gjorde hela läget oanvändbart så
        # snart auth_enabled slogs på — bara 'urd connect' kunde
        # autentisera. None när servern inte kräver bevis.
        self.token = token
        # Skriptläge frågar ALDRIG efter lösenord. getpass läser från
        # stdin när det inte finns någon terminal, och skulle då äta
        # nästa fråga ur heredocen och skicka den som lösenord.
        self.scripted = scripted
        # Namnet servern gav oss, för prompten. None tills vi frågat.
        self.principal: str | None = None
        # Sant bara när sessionen skapades HÄR. En token som kom via
        # --token eller URD_TOKEN tillhör användaren och ska inte
        # loggas ut när läget avslutas — det vore att förstöra något
        # vi lånat.
        self._own_session = False
        self.show_sources = show_sources
        self.show_debug = show_debug
        self.session_id: str | None = None
        self.use_server: bool | None = None   # avgörs vid första frågan
        self._rag = None                      # laddas lat
        self.turns = 0

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # -- identitet ---------------------------------------------------

    def prompt(self) -> str:
        """Prompten, med namnet när servern gett oss ett."""
        return f"{self.principal}@urd> " if self.principal else PROMPT

    def whoami(self, quiet: bool = False) -> str | None:
        """Fråga servern vem vi är. Uppdaterar prompten."""
        if not self.token:
            if not quiet:
                typer.echo("Not signed in.")
            return None
        try:
            resp = requests.get(
                self.server_url + "/whoami", headers=self._headers(), timeout=10
            )
        except requests.RequestException as e:
            if not quiet:
                typer.echo(f"Could not reach the server: {e}")
            return None
        if resp.status_code in (401, 403):
            self.principal = None
            if not quiet:
                typer.echo("The server does not accept this token.")
            return None
        if not resp.ok:
            if not quiet:
                typer.echo(f"Server error {resp.status_code}.")
            return None
        data = resp.json()
        self.principal = data.get("name")
        if not quiet:
            grupper = ", ".join(data.get("groups") or []) or "no groups"
            typer.echo(f"  {self.principal}  [{grupper}]")
        return self.principal

    def login(self) -> bool:
        """
        Fråga efter namn och lösenord och byt dem mot en session.

        Anropas både av .login och automatiskt när ett anrop avvisas.
        Returnerar True när vi har ett bevis efteråt.
        """
        if self.scripted:
            typer.echo(
                "The server requires authentication. Set URD_TOKEN or pass "
                "--token; scripted mode cannot prompt for a password.",
                err=True,
            )
            return False

        import getpass
        try:
            namn = input("user: ").strip()
            if not namn:
                return False
            lösen = getpass.getpass("password: ")
        except (EOFError, KeyboardInterrupt):
            typer.echo("")
            return False
        if not lösen:
            return False

        try:
            resp = requests.post(
                self.server_url + "/login",
                json={"name": namn, "password": lösen},
                timeout=60,
            )
        except requests.RequestException as e:
            typer.echo(f"Could not reach the server: {e}")
            return False

        if resp.status_code == 429:
            typer.echo(resp.json().get("detail", "Too many attempts."))
            return False
        if resp.status_code == 421:
            # TLS-spärren. Beskedet från servern förklarar varför.
            typer.echo(resp.json().get("detail", "Password login requires TLS."))
            return False
        if not resp.ok:
            typer.echo("Wrong user name or password.")
            return False

        data = resp.json()
        self.token = data["token"]
        self._own_session = True
        self.principal = (data.get("principal") or {}).get("name")
        typer.echo(f"Welcome {self.principal}!")
        return True

    def logout(self, quiet: bool = False) -> None:
        """
        Avsluta sessionen på servern.

        Bara vår EGEN session avslutas. En token som kom med --token
        eller URD_TOKEN tillhör användaren och rörs inte — den kan
        dessutom vara ett maskinkonto, som inte går att logga ut.
        """
        if not self.token or not self._own_session:
            if not quiet:
                typer.echo("No session of my own to end.")
            return
        try:
            requests.post(
                self.server_url + "/logout", headers=self._headers(), timeout=10
            )
        except requests.RequestException:
            # Servern kan vara nere. Sessionen går ändå ut av sig
            # själv, och att gnälla vid avslut hjälper ingen.
            pass
        self.token = None
        self._own_session = False
        självnamn, self.principal = self.principal, None
        if not quiet:
            typer.echo(f"Signed out{f' ({självnamn})' if självnamn else ''}.")

    # -- backend -----------------------------------------------------

    def _resolve_backend(self) -> None:
        """Avgör en gång om servern ska användas, och säg det tydligt."""
        if self.use_server is not None:
            return
        try:
            resp = requests.get(self.server_url + "/health", timeout=2)
            self.use_server = resp.ok
        except requests.RequestException:
            self.use_server = False

        if self.use_server:
            self.note(f"Backend: server ({self.server_url})")
        else:
            self.note(
                "Backend: local. Loading models — this takes a moment and "
                "happens only once."
            )

    def _rag_service(self):
        if self._rag is None:
            from app.retrieval import RagService
            self._rag = RagService()
        return self._rag

    # -- frågor ------------------------------------------------------

    def ask(self, question: str) -> None:
        self._resolve_backend()

        if self.use_server:
            payload = {"question": question}
            if self.session_id:
                payload["session_id"] = self.session_id
            try:
                resp = requests.post(
                    self.server_url + "/chat", json=payload,
                    headers=self._headers(), timeout=300,
                )
            except requests.ConnectionError:
                typer.echo(
                    f"Lost contact with the server at {self.server_url}. "
                    "Run .status once it is back up."
                )
                self.use_server = None      # pröva om vid nästa fråga
                return
            if not resp.ok:
                if resp.status_code in (401, 403):
                    # LAT INLOGGNING. Användaren behöver inte känna
                    # till något inloggningskommando: frågan hålls,
                    # beviset hämtas, frågan skickas om. Ett enda
                    # försök — lyckas inte inloggningen är det inte
                    # rätt läge att fråga igen automatiskt.
                    typer.echo("The request requires authentication.")
                    if not self.login():
                        return
                    self.ask(question)
                    return
                typer.echo(
                    f"Server error {resp.status_code}: {resp.text[:200]}"
                )
                return
            response = ChatResponse.model_validate(resp.json())
            if response.session_id:
                self.session_id = response.session_id
        else:
            # Lokalt läge har samtalsminne sedan sessionshanteringen
            # flyttades till kärnan: converse äger QUD, drift och
            # rework-tillstånd, och fungerar likadant med och utan
            # server.
            rag = self._rag_service()
            response = rag.converse(question, session_id=self.session_id)
            if getattr(response, "session_id", None):
                self.session_id = response.session_id

        self.turns += 1
        from app.cli import _print_response
        _print_response(
            response,
            show_sources=self.show_sources,
            show_debug=self.show_debug,
        )

    def _attest(self, args: list[str]) -> None:
        """
        Slå upp en term i Attest utan att lämna sessionen.

        Läser bara .urd/attest.db och rör inte Qdrant, så det fungerar
        med servern igång.
        """
        if not args:
            typer.echo("Give a term:  .attest proprefekt")
            typer.echo("Look up a person: .attest \"F. Lastname\" --person")
            return

        by_subject = any(a in ("--person", "--subjekt") for a in args)
        term = " ".join(a for a in args if not a.startswith("--")).strip('"')

        try:
            from app import attest
            conn = attest.connect()
        except Exception as e:
            typer.echo(f"Attest unavailable: {e}")
            return

        fn = attest.lookup_subject if by_subject else attest.lookup_object
        cands = fn(conn, term)
        if not cands:
            typer.echo(f"No observations for {term!r}.")
            return

        for c in cands:
            flag = "  [AMBIGUOUS ONLY]" if c.ambiguous_only else ""
            avser = f" for {', '.join(c.scopes)}" if c.scopes else ""
            typer.echo(f"  {c.subject} — {c.object}{avser}{flag}")
            typer.echo(
                f"      relevance {c.relevance:.2f}, {c.documents} documents, "
                f"{c.first_date or '?'} – {c.last_date or '?'}"
            )
        typer.echo("  (attestation, not truth)")

    # -- kommandon ---------------------------------------------------

    def command(self, line: str) -> bool:
        """Kör ett punktkommando. Returnerar False om läget ska avslutas."""
        parts = line[1:].strip().split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]
        cmd = _ALIASES.get(cmd, cmd)

        if cmd == "quit":
            return False

        if cmd == "stop":
            # Avslutar SERVERN, inte sessionen. Namnet är avsiktligt
            # skilt från .avsluta för att skillnaden ska synas.
            import subprocess
            subprocess.run(["python", "-m", "app.cli", "stop"], check=False)
            self.use_server = None
            return True

        if cmd == "help":
            typer.echo(HELP)

        elif cmd == "new":
            self.session_id = None
            self.turns = 0
            typer.echo("New session. Conversation history forgotten.")

        elif cmd == "attest":
            self._attest(args)

        elif cmd == "whoami":
            self.whoami()

        elif cmd == "login":
            if self.token and self.principal:
                typer.echo(f"Already signed in as {self.principal}.")
            else:
                self.login()

        elif cmd == "logout":
            self.logout()

        elif cmd == "status":
            backend = {
                None: "undecided (resolved at first question)",
                True: f"server ({self.server_url})",
                False: "local",
            }[self.use_server]
            typer.echo(f"  backend:  {backend}")
            typer.echo(f"  identity: {self.principal or 'not signed in'}")
            typer.echo(f"  session:  {self.session_id or 'none yet'}")
            typer.echo(f"  turns:    {self.turns}")
            typer.echo(f"  sources:  {'on' if self.show_sources else 'off'}")
            typer.echo(f"  debug:    {'on' if self.show_debug else 'off'}")
            typer.echo(
                f"  models:   {'loaded' if self._rag else 'not loaded'}"
            )

        elif cmd == "sources":
            self.show_sources = _on_off(args, self.show_sources)
            typer.echo(f"Sources: {'on' if self.show_sources else 'off'}")

        elif cmd == "debug":
            self.show_debug = _on_off(args, self.show_debug)
            # Debug på återställer också URD:s egen INFO-logg, som är
            # dämpad i läget för att inte stå mellan fråga och svar.
            import logging as _logging
            level = _logging.INFO if self.show_debug else _logging.WARNING
            for name in ("app.retrieval", "app.api", "app.grammar",
                         "app.predication", "app.attest"):
                _logging.getLogger(name).setLevel(level)
            typer.echo(f"Debug: {'on' if self.show_debug else 'off'}")

        else:
            typer.echo(f"Unknown command: .{cmd}   (.help lists them all)")

        return True


def _on_off(args: list[str], current: bool) -> bool:
    """
    Tolka on/off; utan argument växlas värdet.

    Svenska former behålls av samma skäl som kommandoaliasen: de kostar
    ingenting och skyddar muskelminne. De dokumenteras inte.
    """
    if not args:
        return not current
    return args[0].lower() in ("on", "true", "yes", "1", "på", "pa", "ja")


def run(
    server_url: str,
    show_sources: bool,
    show_debug: bool,
    token: str | None = None,
) -> int:
    """
    Kör slingan tills användaren avslutar. Returnerar exitkod.

    SKRIPTLÄGE NÄR STDIN INTE ÄR EN TERMINAL. Läget är då fortfarande
    samma tolk, men uppträder som ett filter:

        urd <<'EOF'
        Vem är proprefekt vid IIT?
        Berätta mer.
        EOF

    Skillnaden mot tjugo `urd ask` är att modellerna laddas EN gång och
    att sessionen lever mellan frågorna — samma skäl som gör läget
    värt att ha alls.

    Fyra saker skiljer skriptläget från terminalläget:

      - prompt, banner och avsked tystas; input() skriver annars
        prompten till stdout mitt i utdata
      - allt som inte är svar går till stderr, så att stdout kan
        pipas vidare
      - frågan ekas före svaret, annars blir tio svar omöjliga att
        para ihop med sina frågor
      - ett fel sätter exitkod 1

    Maskinläsbar utdata hör INTE hit. Frestelsen är att lägga till ett
    JSON-läge och köra regressionsfrågor i en heredoc, men då byggs
    `urd test` om, sämre och utan assertions. Skriptlägets värde är
    utforskande batcharbete i en varm session.
    """
    scripted = not sys.stdin.isatty()

    def note(text: str) -> None:
        """Statusrader: stderr i skriptläge, stdout i terminal."""
        typer.echo(text, err=scripted)

    try:
        import readline  # noqa: F401  — ger radhistorik och redigering
    except ImportError:
        pass

    # URD:s egen INFO-logg hör hemma i serverloggen, inte mellan frågan
    # och svaret i en prompt. Diagnostiken finns kvar i debug-blocket
    # och i JSONL-spåren; .debug på visar den i läget.
    import logging as _logging
    for _name in ("app.retrieval", "app.api", "app.grammar",
                  "app.predication", "app.attest", "app.synonyms",
                  "app.concepts", "app.question_operations"):
        _logging.getLogger(_name).setLevel(_logging.WARNING)

    repl = Repl(server_url, show_sources, show_debug, note=note, token=token,
                scripted=scripted)
    # Har vi redan ett bevis, fråga en gång vem det tillhör — då kan
    # prompten bära namnet från första raden. Tyst: saknas eller duger
    # inte token märks det ändå vid första frågan, och ett felmeddelande
    # före bannern vore brus.
    if token:
        repl.whoami(quiet=True)
    if not scripted:
        typer.echo(BANNER)
        typer.echo("")

    failed = False

    while True:
        try:
            line = input("" if scripted else repl.prompt())
        except EOFError:              # Ctrl-D, eller slut på indata
            if not scripted:
                typer.echo("")
            break
        except KeyboardInterrupt:     # Ctrl-C avbryter raden, inte läget
            typer.echo("")
            continue

        line = line.strip()
        # Tomma rader och #-kommentarer hör till skript. I terminalen
        # är de ofarliga.
        if not line or line.startswith("#"):
            continue

        if line.startswith("."):
            if not repl.command(line):
                break
            continue

        if scripted:
            # Utan eko går tio svar inte att para ihop med sina frågor.
            typer.echo(f"> {line}")

        try:
            repl.ask(line)
        except KeyboardInterrupt:
            # Avbrott mitt i en fråga ska inte fälla sessionen.
            typer.echo("\n(interrupted)")
        except Exception as e:
            # I skriptläge är felet ett resultat: stderr och exitkod.
            typer.echo(f"Error: {e}", err=scripted)
            failed = True

    # AUTOMATISK UTLOGGNING. En session som skapades här ska inte
    # överleva läget — då blir en glömd utloggning omöjlig snarare än
    # osannolik. Bara vår egen session avslutas; en token som kom med
    # --token eller URD_TOKEN tillhör användaren och rörs inte.
    repl.logout(quiet=True)

    if not scripted:
        typer.echo("Bye.")
    return 1 if failed else 0
