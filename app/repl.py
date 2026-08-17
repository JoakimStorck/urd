"""
Interaktivt läge för URD.

    $ urd
    urd> Vem är proprefekt vid IIT?
    ...
    urd> Berätta mer.
    ...
    urd> .avsluta

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
"""

from __future__ import annotations

import sys

import requests
import typer

from app.schemas import ChatResponse

PROMPT = "urd> "
CONTINUATION = "...  "

BANNER = """URD interaktivt läge. Skriv en fråga, eller .hjälp för kommandon.
Sessionen lever tills du skriver .ny eller avslutar."""

HELP = """Kommandon (allt annat tolkas som en fråga):

  .hjälp            visa detta
  .status           backend, session, laddade modeller
  .ny               starta ny session (glömmer samtalshistorik)
  .källor på|av     visa eller dölj källor i svaren
  .debug på|av      visa eller dölj teknisk info
  .attest <term>    slå upp vad beståndet belägger om en roll
                    (.attest "N.N." --person för uppslag på person)
  .stopp            avsluta SERVERN (sessionen fortsätter lokalt)
  .avsluta          avsluta läget (även Ctrl-D)

Sessionen behålls mellan frågor: följdfrågor som "berätta mer" och
"stämmer det?" fungerar som i webbgränssnittet."""


class Repl:
    """Tillståndet för en interaktiv session."""

    def __init__(self, server_url: str, show_sources: bool, show_debug: bool):
        self.server_url = server_url.rstrip("/")
        self.show_sources = show_sources
        self.show_debug = show_debug
        self.session_id: str | None = None
        self.use_server: bool | None = None   # avgörs vid första frågan
        self._rag = None                      # laddas lat
        self.turns = 0

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
            typer.echo(f"Backend: server ({self.server_url})")
        else:
            typer.echo(
                "Backend: lokal. Laddar modeller — detta tar en stund och "
                "sker bara en gång."
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
                    self.server_url + "/chat", json=payload, timeout=300
                )
            except requests.ConnectionError:
                typer.echo(
                    f"Tappade kontakten med servern på {self.server_url}. "
                    "Kör .status när den är uppe igen."
                )
                self.use_server = None      # pröva om vid nästa fråga
                return
            if not resp.ok:
                typer.echo(f"Serverfel {resp.status_code}: {resp.text[:200]}")
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
            typer.echo("Ange en term:  .attest proprefekt")
            typer.echo("Uppslag på person: .attest \"Anna Andersson\" --person")
            return

        by_subject = any(a in ("--person", "--subjekt") for a in args)
        term = " ".join(a for a in args if not a.startswith("--")).strip('"')

        try:
            from app import attest
            conn = attest.connect()
        except Exception as e:
            typer.echo(f"Attest otillgängligt: {e}")
            return

        fn = attest.lookup_subject if by_subject else attest.lookup_object
        cands = fn(conn, term)
        if not cands:
            typer.echo(f"Inga observationer för {term!r}.")
            return

        for c in cands:
            flag = "  [ENDAST TVETYDIGA]" if c.ambiguous_only else ""
            avser = f" för {', '.join(c.scopes)}" if c.scopes else ""
            typer.echo(f"  {c.subject} — {c.object}{avser}{flag}")
            typer.echo(
                f"      relevans {c.relevance:.2f}, {c.documents} dokument, "
                f"{c.first_date or '?'} – {c.last_date or '?'}"
            )
        typer.echo("  (beläggning, inte sanning)")

    # -- kommandon ---------------------------------------------------

    def command(self, line: str) -> bool:
        """Kör ett punktkommando. Returnerar False om läget ska avslutas."""
        parts = line[1:].strip().split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("avsluta", "quit", "exit", "q"):
            return False

        if cmd in ("stopp", "stop"):
            # Avslutar SERVERN, inte sessionen. Namnet är avsiktligt
            # skilt från .avsluta för att skillnaden ska synas.
            import subprocess
            subprocess.run(["python", "-m", "app.cli", "stop"], check=False)
            self.use_server = None
            return True

        if cmd in ("hjälp", "hjalp", "help", "h", "?"):
            typer.echo(HELP)

        elif cmd in ("ny", "new"):
            self.session_id = None
            self.turns = 0
            typer.echo("Ny session. Samtalshistoriken är glömd.")

        elif cmd == "attest":
            self._attest(args)

        elif cmd == "status":
            backend = {
                None: "ej avgjord (laddas vid första frågan)",
                True: f"server ({self.server_url})",
                False: "lokal",
            }[self.use_server]
            typer.echo(f"  backend:  {backend}")
            typer.echo(f"  session:  {self.session_id or 'ingen ännu'}")
            typer.echo(f"  turer:    {self.turns}")
            typer.echo(f"  källor:   {'på' if self.show_sources else 'av'}")
            typer.echo(f"  debug:    {'på' if self.show_debug else 'av'}")
            typer.echo(
                f"  modeller: {'laddade' if self._rag else 'ej laddade'}"
            )

        elif cmd in ("källor", "kallor", "sources"):
            self.show_sources = _on_off(args, self.show_sources)
            typer.echo(f"Källor: {'på' if self.show_sources else 'av'}")

        elif cmd == "debug":
            self.show_debug = _on_off(args, self.show_debug)
            # Debug på återställer också URD:s egen INFO-logg, som är
            # dämpad i läget för att inte stå mellan fråga och svar.
            import logging as _logging
            level = _logging.INFO if self.show_debug else _logging.WARNING
            for name in ("app.retrieval", "app.api", "app.grammar",
                         "app.predication", "app.attest"):
                _logging.getLogger(name).setLevel(level)
            typer.echo(f"Debug: {'på' if self.show_debug else 'av'}")

        else:
            typer.echo(f"Okänt kommando: .{cmd}   (.hjälp visar alla)")

        return True


def _on_off(args: list[str], current: bool) -> bool:
    """Tolka 'på'/'av'; utan argument växlas värdet."""
    if not args:
        return not current
    return args[0].lower() in ("på", "pa", "on", "true", "ja", "1")


def run(server_url: str, show_sources: bool, show_debug: bool) -> None:
    """Kör slingan tills användaren avslutar."""
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

    repl = Repl(server_url, show_sources, show_debug)
    typer.echo(BANNER)
    typer.echo("")

    while True:
        try:
            line = input(PROMPT)
        except EOFError:              # Ctrl-D
            typer.echo("")
            break
        except KeyboardInterrupt:     # Ctrl-C avbryter raden, inte läget
            typer.echo("")
            continue

        line = line.strip()
        if not line:
            continue

        if line.startswith("."):
            if not repl.command(line):
                break
            continue

        try:
            repl.ask(line)
        except KeyboardInterrupt:
            # Avbrott mitt i en fråga ska inte fälla sessionen.
            typer.echo("\n(avbruten)")
        except Exception as e:
            typer.echo(f"Fel: {e}")

    typer.echo("Avslutar.")
