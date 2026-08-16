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
  .avsluta          avsluta (även Ctrl-D)

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
        self._warned_local = False
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
            # LOKALT LÄGE SAKNAR SAMTALSTILLSTÅND.
            #
            # Sessionshanteringen — QUD, aktiva dokument, rework-state —
            # ligger i api.py, inte i RagService. Lokalt läge har därför
            # aldrig haft samtalsminne, inte heller i `urd ask`.
            # Följdfrågor som "berätta mer" fungerar inte utan server.
            #
            # Sagt en gång per session, inte vid varje fråga.
            if not self._warned_local:
                typer.echo(
                    "Obs: utan server saknas samtalsminne — varje fråga\n"
                    "     står för sig. Starta 'urd serve' för QUD,\n"
                    "     följdfrågor och verification."
                )
                self._warned_local = True
            rag = self._rag_service()
            response = rag.answer(question)

        self.turns += 1
        from app.cli import _print_response
        _print_response(
            response,
            show_sources=self.show_sources,
            show_debug=self.show_debug,
        )

    # -- kommandon ---------------------------------------------------

    def command(self, line: str) -> bool:
        """Kör ett punktkommando. Returnerar False om läget ska avslutas."""
        parts = line[1:].strip().split()
        if not parts:
            return True
        cmd, args = parts[0].lower(), parts[1:]

        if cmd in ("avsluta", "quit", "exit", "q"):
            return False

        if cmd in ("hjälp", "hjalp", "help", "h", "?"):
            typer.echo(HELP)

        elif cmd in ("ny", "new"):
            self.session_id = None
            self.turns = 0
            typer.echo("Ny session. Samtalshistoriken är glömd.")

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
