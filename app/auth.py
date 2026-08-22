"""
Autentisering och identitet.

TOKEN SOM BÄR ETT SUBJEKT, INTE BARA EN NYCKEL.

En delad hemlighet för hela installationen är det första man tänker på
när tre personer ska få tillgång. Den är också det som måste ERSÄTTAS
i stället för utvidgas så snart behörighetsnivåer införs, eftersom den
inte svarar på frågan *vem* som frågar. Behörighetsarv från
dokumentkällan kräver ett subjekt med grupptillhörighet, och det
subjektet ska finnas från början även när alla tillhör samma grupp.

Det är därför Principal och inte bool.

TOKENS LAGRAS SOM HASH. Filen med användare ligger i instansens
tillstånd och versionshanteras aldrig, men den kan läsas av den som
kommer åt maskinen, säkerhetskopieras och råka hamna i en logg. En
sha256-summa räcker för ändamålet: token genereras en gång, visas en
gång, och kan därefter bara verifieras — inte återskapas.

Jämförelsen görs med konstant tid. Tidsskillnaden vid strängjämförelse
är en liten kanal, men den kostar ingenting att stänga.

GRUPPEN "*" BETYDER OAVGRÄNSAT och finns bara för lokal drift utan
autentisering, alltså en enanvändarmaskin där ingen åtkomstmodell är
meningsfull. Den ska aldrig kunna tilldelas en användare i filen — se
_validate_groups. Skillnaden mot en tom grupplista är avsiktlig: tom
lista betyder "får se ingenting", inte "får se allt".
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Gruppnamn som betyder "ingen avgränsning". Endast för lokal drift.
UNRESTRICTED = "*"

TOKEN_BYTES = 32


@dataclass(frozen=True)
class Principal:
    """
    Den som ställer frågan.

    name används i loggar och felmeddelanden. groups avgör vad
    principalen får se när behörighetsfiltrering är påslagen.
    """
    name: str
    groups: tuple[str, ...] = ()

    @property
    def unrestricted(self) -> bool:
        return UNRESTRICTED in self.groups

    def may_access(self, access_groups) -> bool:
        """
        Får principalen se ett dokument med dessa åtkomstgrupper?

        FRÅNVARANDE UPPGIFT BETYDER STÄNGT. Det bryter mot systemets
        mönster i övrigt, där en saknad uppgift behandlas som okänd och
        inte som en motsägelse — Attest väger inte ned en bindning för
        att avgränsningen saknas, och syntesen abstainar hellre än
        gissar. Här går asymmetrin åt andra hållet, eftersom kostnaden
        för fel är exponering och inte utebliven träff.

        Detta är ett medvetet undantag och ska inte "rättas" till
        konsekvens med resten av systemet.
        """
        if self.unrestricted:
            return True
        if not access_groups:
            return False
        return bool(set(self.groups) & set(access_groups))


# Principalen för lokal drift utan autentisering: en enanvändarmaskin
# där ingen åtkomstmodell är meningsfull.
LOCAL = Principal(name="local", groups=(UNRESTRICTED,))


@dataclass
class UserStore:
    """Användare inlästa ur instansens tillstånd."""
    by_token_hash: dict[str, Principal] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    path: Path | None = None
    # Filens tillstånd när den lästes: (mtime_ns, storlek). None när
    # filen saknades. Används bara för att upptäcka ändring — aldrig
    # för att avgöra om innehållet duger.
    stamp: tuple[int, int] | None = None

    @property
    def loaded(self) -> int:
        return len(self.by_token_hash)

    def verify(self, token: str | None) -> Principal | None:
        """
        Slå upp en token. Returnerar None när den inte känns igen.

        Jämförelsen görs mot alla poster med konstant tid, så att
        varken träff eller antal försök går att läsa ur svarstiden.
        """
        if not token:
            return None
        digest = hash_token(token)
        träff: Principal | None = None
        for stored, principal in self.by_token_hash.items():
            if hmac.compare_digest(stored, digest):
                träff = principal
        return träff


def hash_token(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def generate_token() -> str:
    """Ny token. Visas en gång; bara hashen lagras."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def _validate_groups(name: str, groups) -> tuple[list[str], list[str]]:
    fel: list[str] = []
    rena: list[str] = []
    for g in groups or []:
        if not isinstance(g, str) or not g.strip():
            fel.append(f"användare {name!r}: tom eller ogiltig grupp")
            continue
        g = g.strip()
        if g == UNRESTRICTED:
            # Får aldrig konfigureras. En användare som av misstag
            # tilldelas "*" skulle passera varje behörighetsfilter utan
            # att det syns någonstans.
            fel.append(
                f"användare {name!r}: gruppen {UNRESTRICTED!r} är reserverad "
                "för lokal drift och kan inte tilldelas"
            )
            continue
        rena.append(g)
    return rena, fel


def load_users(path: Path) -> UserStore:
    """
    Läs användarfilen. Saknad fil är inte ett fel — den betyder att
    ingen är upplagd ännu, och anroparen avgör vad det innebär.

    Formatet:

        users:
          - name: someone
            token_sha256: "..."
            groups: [institution]
    """
    store = UserStore(path=path)
    # Stämpeln tas FÖRE läsningen. Skrivs filen mitt under inläsningen
    # ser nästa kontroll en nyare stämpel och läser om — hellre en
    # överflödig omläsning än en ändring som aldrig upptäcks.
    store.stamp = _stamp(path)
    if not path.exists():
        return store

    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        store.errors.append(f"kunde inte läsa {path}: {e}")
        return store

    users = data.get("users")
    if users is None:
        store.errors.append(f"{path}: nyckeln 'users' saknas")
        return store
    if not isinstance(users, list):
        store.errors.append(f"{path}: 'users' måste vara en lista")
        return store

    for i, entry in enumerate(users, start=1):
        if not isinstance(entry, dict):
            store.errors.append(f"{path}: post {i} är inte en avbildning")
            continue
        name = str(entry.get("name") or "").strip()
        digest = str(entry.get("token_sha256") or "").strip().lower()
        if not name:
            store.errors.append(f"{path}: post {i} saknar name")
            continue
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            store.errors.append(
                f"{path}: användare {name!r} saknar giltig token_sha256"
            )
            continue
        groups, fel = _validate_groups(name, entry.get("groups"))
        store.errors.extend(fel)
        if digest in store.by_token_hash:
            store.errors.append(
                f"{path}: samma token_sha256 används av flera användare"
            )
            continue
        store.by_token_hash[digest] = Principal(name=name, groups=tuple(groups))

    return store


def _stamp(path: Path) -> tuple[int, int] | None:
    """Filens ändringstid och storlek, eller None om den inte finns."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def reload_if_changed(store: UserStore) -> tuple[UserStore, bool]:
    """
    Läs om användarfilen när den ändrats. Returnerar (store, ändrad).

    ÅTERKALLELSE MÅSTE SLÅ IGENOM UTAN OMSTART. En borttagen användare
    vars token fortsätter fungera tills någon startar om servern är
    inte en olägenhet utan ett säkerhetsfel: `urd auth remove` säger
    att åtkomsten upphört, och det ska vara sant.

    Kontrollen är ett stat-anrop per förfrågan — mikrosekunder mot en
    tur på flera sekunder. Innehållet läses bara när stämpeln skiljer
    sig.

    FELAKTIG FIL STÄNGER. Går filen inte att tolka blir resultatet en
    tom uppsättning, alltså avslag för alla, inte en fortsättning på
    den gamla. Det är samma asymmetri som gäller behörighet i övrigt:
    en frånvarande eller obegriplig uppgift betyder stängt, eftersom
    kostnaden för fel är exponering. Ett avslag är högljutt och
    åtgärdas på sekunder; en kvarlevande återkallad token upptäcks av
    ingen. Skrivningen från `urd auth` är atomär, så en halvskriven
    fil ska aldrig kunna orsaka detta.
    """
    if store.path is None:
        return store, False
    if _stamp(store.path) == store.stamp:
        return store, False
    return load_users(store.path), True


def bearer_token(header_value: str | None) -> str | None:
    """
    Plocka ut token ur ett Authorization-huvud.

    Både "Bearer <token>" och ett naket värde accepteras. Det senare
    för att curl-anrop och enkla skript inte ska falla på formatet —
    huvudet är ändå bara läsbart för den som redan har det.
    """
    if not header_value:
        return None
    value = header_value.strip()
    # Prefixet plockas bort FÖRE tomhetskontrollen. "Bearer " med tomt
    # värde ska ge None, inte strängen "Bearer" — annars avvisas ett
    # tomt huvud som en okänd token, vilket är rätt utfall av fel skäl
    # och ger vilseledande loggar.
    if value.lower() == "bearer" or value.lower().startswith("bearer "):
        return value[6:].strip() or None
    return value or None
    """
    Plocka ut token ur ett Authorization-huvud.

    Både "Bearer <token>" och ett naket värde accepteras. Det senare
    för att curl-anrop och enkla skript inte ska falla på formatet —
    huvudet är ändå bara läsbart för den som redan har det.
    """
    if not header_value:
        return None
    value = header_value.strip()
    # Prefixet plockas bort FÖRE tomhetskontrollen. "Bearer " med tomt
    # värde ska ge None, inte strängen "Bearer" — annars avvisas ett
    # tomt huvud som en okänd token, vilket är rätt utfall av fel skäl
    # och ger vilseledande loggar.
    if value.lower() == "bearer" or value.lower().startswith("bearer "):
        return value[6:].strip() or None
    return value or None


def is_loopback(host: str) -> bool:
    """
    Är bindningsadressen begränsad till maskinen själv?

    Detta är systemets enskilt viktigaste säkerhetsparameter: så länge
    den är sann är avsaknaden av autentisering ofarlig, och i samma
    stund den blir falsk är hela dokumentbeståndet nåbart för alla som
    når porten.
    """
    return host.strip().lower() in {
        "127.0.0.1", "localhost", "::1", "[::1]",
    }
