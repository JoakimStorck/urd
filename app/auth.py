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

LÖSENORD LAGRAS INTE SOM TOKENS. En sha256-summa är rätt för en
slumpad 256-bitarssträng — det finns ingenting att gissa. För ett
människovalt lösenord är samma konstruktion fel i kategori: den är
snabb, och snabbhet är precis vad den som fått tag i filen behöver.
Lösenord härleds därför med scrypt, som är minneshårt och därmed dyrt
även för den som har grafikkort. Parametrarna lagras i posten så att
de kan höjas utan att befintliga lösenord blir oläsbara.

TOKEN KAN VARA EN INBJUDAN. En post med enrollment: true får inte
autentisera vanliga anrop — den kan bara växlas in mot ett lösenord,
en gång, varefter token tas bort ur posten. Det gör den långa strängen
till något som bara behöver överleva överlämningen. Maskinkonton
(urd test, skript) saknar flaggan och fungerar oförändrat: lösenord
för människor, tokens för program.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Gruppnamn som betyder "ingen avgränsning". Endast för lokal drift.
UNRESTRICTED = "*"

TOKEN_BYTES = 32

# scrypt-parametrar. n=2**15, r=8, p=3 kostar 34 MB och ~0,27 s per
# härledning på utvecklingsmaskinen — inom OWASP:s rekommenderade band,
# och en kostnad som bara betalas vid inloggning, inte per anrop. p och
# inte högre n därför att p multiplicerar arbetet utan att multiplicera
# minnet; en server som loggar in flera samtidigt ska inte behöva
# hundratals megabyte.
#
# Kostnaden är också strypningens grund: varje felaktigt försök tar en
# kvarts sekund av angriparens tid, oavsett vad räknaren nedan gör.
#
# Parametrarna skrivs i posten och läses därifrån vid verifiering, så
# att de kan höjas här utan att äldre lösenord slutar fungera. Ett
# lösenord härlett med gamla parametrar verifieras med gamla; nästa
# gång det sätts används de nya.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 3
SCRYPT_DKLEN = 32
SALT_BYTES = 16

# Kortare än så är inte ett lösenord utan en gissningsövning. Längd
# framför teckenklasser är avsiktligt: sammansättningsregler driver
# fram förutsägbara mönster utan att öka entropin nämnvärt, vilket
# NIST SP 800-63B slog fast redan 2017. Av samma skäl finns ingen
# tvingad förnyelse.
PASSWORD_MIN_LENGTH = 12
# Ett tak finns bara för att en härledning av ett megabyte långt
# lösenord är en billig väg att belasta servern.
PASSWORD_MAX_LENGTH = 128


def hash_password(password: str) -> str:
    """
    Härled ett lösenord till en post som kan lagras.

    Formatet bär sina egna parametrar:

        scrypt$n=32768,r=8,p=3$<salt>$<hash>

    Salt är slumpat per lösenord, så att två användare med samma
    lösenord får skilda poster och en förberäknad tabell inte kan
    återanvändas.
    """
    salt = secrets.token_bytes(SALT_BYTES)
    dk = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return "$".join((
        "scrypt",
        f"n={SCRYPT_N},r={SCRYPT_R},p={SCRYPT_P}",
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    ))


def verify_password(password: str, record: str) -> bool:
    """
    Pröva ett lösenord mot en lagrad post.

    Jämförelsen görs med konstant tid. En oläsbar post ger False och
    inte ett undantag: den som kan skriva sönder posten ska inte kunna
    välja mellan att låsa ute och att släppa in.
    """
    parsed = _parse_password_record(record)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        dk = _scrypt(password, salt, n, r, p, dklen=len(expected))
    except (ValueError, MemoryError):
        # Parametrar som posten begär men maskinen inte klarar. Att
        # svara False är rätt: ett lösenord som inte KAN prövas är
        # inte ett lösenord som stämmer.
        return False
    return hmac.compare_digest(dk, expected)


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int,
            dklen: int = SCRYPT_DKLEN) -> bytes:
    # maxmem måste anges uttryckligen — OpenSSL:s standardtak ligger
    # under vad de här parametrarna behöver, och utan detta faller
    # anropet på ett fel som ser ut som ett felaktigt lösenord.
    behov = 128 * n * r * p
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
        dklen=dklen, maxmem=behov * 2 + 1024 * 1024,
    )


def _parse_password_record(record: str):
    """(n, r, p, salt, hash) ur en lagrad post, eller None."""
    if not isinstance(record, str):
        return None
    delar = record.split("$")
    if len(delar) != 4 or delar[0] != "scrypt":
        return None
    try:
        params = dict(
            bit.split("=", 1) for bit in delar[1].split(",")
        )
        n, r, p = int(params["n"]), int(params["r"]), int(params["p"])
        salt = base64.b64decode(delar[2], validate=True)
        expected = base64.b64decode(delar[3], validate=True)
    except Exception:
        return None
    if n < 2 or r < 1 or p < 1 or not salt or not expected:
        return None
    return n, r, p, salt, expected


# En post att pröva mot när användaren inte finns, så att ett okänt
# namn kostar lika mycket tid som ett känt. Utan den skulle svarstiden
# avslöja vilka konton som existerar — och en lista över anställda med
# konton i systemet är i sig en uppgift värd att skydda.
_DUMMY_RECORD = hash_password(secrets.token_urlsafe(32))


def password_problems(password: str, name: str = "") -> list[str]:
    """
    Vad som är fel med ett föreslaget lösenord. Tom lista = dugligt.

    Reglerna är avsiktligt få. Det som mäts är längd och att lösenordet
    inte är namnet, eftersom det är de två fel som faktiskt förekommer.
    Sammansättningsregler är medvetet uteslutna.
    """
    problem: list[str] = []
    if len(password) < PASSWORD_MIN_LENGTH:
        problem.append(
            f"lösenordet måste vara minst {PASSWORD_MIN_LENGTH} tecken"
        )
    if len(password) > PASSWORD_MAX_LENGTH:
        problem.append(
            f"lösenordet får vara högst {PASSWORD_MAX_LENGTH} tecken"
        )
    if password != password.strip():
        problem.append("lösenordet får inte börja eller sluta med blanksteg")
    if name and name.lower() in password.lower():
        problem.append("lösenordet får inte innehålla användarnamnet")
    if len(set(password)) < 5:
        problem.append("lösenordet är för enformigt")
    return problem


class Throttle:
    """
    Strypning av upprepade misslyckanden.

    Lösenord kan gissas på ett sätt en slumpad token inte kan, så
    inloggningsvägen behöver en spärr. Räknaren är per nyckel — api.py
    avgör vad nyckeln är, lämpligen namn och avsändaradress
    tillsammans, så att en angripare varken kan låsa ut en enskild
    användare genom att gissa i hens namn eller kringgå spärren genom
    att byta namn.

    Spärren växer: efter FAILURES_BEFORE_LOCK misslyckanden gäller
    BASE_LOCK sekunder, därefter dubbelt så länge för varje ytterligare
    misslyckande upp till MAX_LOCK. En lyckad inloggning nollställer.

    Tillståndet ligger i minnet och försvinner vid omstart. Det är en
    känd svaghet — den som kan starta om servern kan nollställa spärren
    — men en omstart kräver åtkomst till maskinen, och då är spärren
    inte det som skyddar.
    """

    FAILURES_BEFORE_LOCK = 5
    BASE_LOCK = 30.0
    MAX_LOCK = 900.0

    def __init__(self):
        self._state: dict[str, tuple[int, float]] = {}

    def locked_for(self, key: str) -> float:
        """Sekunder kvar av spärren, 0 om nyckeln är öppen."""
        misslyckanden, spärrad_till = self._state.get(key, (0, 0.0))
        kvar = spärrad_till - time.monotonic()
        return kvar if kvar > 0 else 0.0

    def record_failure(self, key: str) -> float:
        misslyckanden, _ = self._state.get(key, (0, 0.0))
        misslyckanden += 1
        if misslyckanden < self.FAILURES_BEFORE_LOCK:
            self._state[key] = (misslyckanden, 0.0)
            return 0.0
        över = misslyckanden - self.FAILURES_BEFORE_LOCK
        längd = min(self.BASE_LOCK * (2 ** över), self.MAX_LOCK)
        self._state[key] = (misslyckanden, time.monotonic() + längd)
        return längd

    def record_success(self, key: str) -> None:
        self._state.pop(key, None)


class SessionStore:
    """
    Sessioner som skapats genom inloggning med lösenord.

    EN SESSION ÄR INTE ETT KONTO. Sessionstoken bär samma principal som
    kontot, men den är kortlivad, kan återkallas ensam och lämnar
    aldrig serverns minne i annan form än strängen användaren fick. Att
    den försvinner vid omstart är avsiktligt: en glömd session ska inte
    överleva att maskinen startas om.

    TVÅ UTGÅNGAR. Den absoluta säger hur länge en inloggning får gälla
    över huvud taget; den overksamma säger hur länge den får ligga
    orörd. Bara den ena räcker inte — en absolut gräns ensam låter en
    övergiven session leva timmen ut, och en overksamhetsgräns ensam
    låter en session som används dagligen leva för alltid.

    Tokens lagras som hash, av samma skäl som kontotokens: den som får
    läsa serverns minne ska inte få en användbar nyckel på köpet.
    Uppslaget sker på hashen och inte genom jämförelse post för post —
    en slumpad 256-bitarssträng ger ingen tidskanal värd namnet, och
    sessionstabellen kan bli stor.
    """

    def __init__(self, ttl_seconds: float, idle_seconds: float):
        self.ttl = float(ttl_seconds)
        self.idle = float(idle_seconds)
        self._sessions: dict[str, dict] = {}

    def create(self, principal: Principal) -> tuple[str, float]:
        """Ny session. Returnerar (token, sekunder till absolut utgång)."""
        token = secrets.token_urlsafe(TOKEN_BYTES)
        nu = time.monotonic()
        self._sessions[hash_token(token)] = {
            "principal": principal,
            "expires_at": nu + self.ttl,
            "last_used": nu,
        }
        return token, self.ttl

    def verify(self, token: str | None) -> Principal | None:
        if not token:
            return None
        post = self._sessions.get(hash_token(token))
        if post is None:
            return None
        nu = time.monotonic()
        if nu > post["expires_at"] or nu - post["last_used"] > self.idle:
            self._sessions.pop(hash_token(token), None)
            return None
        post["last_used"] = nu
        return post["principal"]

    def revoke(self, token: str | None) -> bool:
        if not token:
            return False
        return self._sessions.pop(hash_token(token), None) is not None

    def revoke_for_name(self, name: str) -> int:
        """
        Avsluta alla sessioner för en användare.

        Anropas när lösenordet ändras. Ett byte av lösenord ska inte
        lämna kvar sessioner som skapades med det gamla — det är hela
        skälet att man byter.
        """
        nyckel = name.strip().lower()
        döm = [
            h for h, p in self._sessions.items()
            if p["principal"].name.strip().lower() == nyckel
        ]
        for h in döm:
            self._sessions.pop(h, None)
        return len(döm)

    def prune(self) -> int:
        nu = time.monotonic()
        döm = [
            h for h, p in self._sessions.items()
            if nu > p["expires_at"] or nu - p["last_used"] > self.idle
        ]
        for h in döm:
            self._sessions.pop(h, None)
        return len(döm)

    @property
    def active(self) -> int:
        return len(self._sessions)


def write_users(path: Path, users: list[dict]) -> None:
    """
    Skriv användarfilen atomärt med rättigheten 0600.

    Servern läser om filen så snart den ändrats, och en skrivning som
    först tomkör och sedan fyller ger ett fönster där en läsare ser
    halva innehållet — vilket med fail-closed betyder avslag för alla
    under någon millisekund. Skriv till granne och byt in med
    os.replace, som är atomärt inom samma filsystem.

    Rättigheten sätts på temporärfilen FÖRE inbytet, så att den
    färdiga filen aldrig existerar med vidare rättigheter.

    Bor här och inte i cli.py därför att både administratörens
    kommandon och inloggningsvägen skriver samma fil, och två
    implementationer av samma skrivning är en garanti för att den ena
    blir fel.
    """
    import os
    import yaml as _yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".ny")
    tmp.write_text(
        _yaml.safe_dump({"users": users}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def read_users_raw(path: Path) -> list[dict]:
    """Posterna som de står i filen, utan tolkning."""
    if not path.exists():
        return []
    import yaml as _yaml
    data = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("users") or [])


def set_password(path: Path, name: str, password: str,
                 consume_enrollment: bool = False) -> bool:
    """
    Sätt lösenord för en användare i filen. False när namnet saknas.

    consume_enrollment tar samtidigt bort token_sha256 och
    enrollment-flaggan: en inbjudan som växlats in ska inte kunna
    växlas in igen. Att ta bort token är hela förbrukningen — posten
    har därefter bara ett lösenord, och det är precis vad som avses.
    """
    users = read_users_raw(path)
    nyckel = name.strip().lower()
    hittad = False
    for post in users:
        if str(post.get("name") or "").strip().lower() != nyckel:
            continue
        hittad = True
        post["password_scrypt"] = hash_password(password)
        if consume_enrollment:
            post.pop("token_sha256", None)
            post.pop("enrollment", None)
    if hittad:
        write_users(path, users)
    return hittad


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
class UserRecord:
    """
    En post i användarfilen, som den lästes.

    token_digest kan saknas (användaren har växlat in sin inbjudan),
    password kan saknas (maskinkonto eller ännu inte inväxlad
    inbjudan). Att båda saknas är ett fel och posten släpps — se
    load_users.
    """
    principal: Principal
    token_digest: str | None = None
    password: str | None = None
    # Sant när token bara får användas för att växlas in mot ett
    # lösenord, aldrig för att autentisera vanliga anrop.
    enrollment: bool = False


class UserStore:
    """Användare inlästa ur instansens tillstånd."""

    def __init__(self):
        self.by_token_hash: dict[str, Principal] = {}
        self.by_name: dict[str, UserRecord] = {}
        self.errors: list[str] = []
        self.path: Path | None = None
        # Filens tillstånd när den lästes: (mtime_ns, storlek). None när
        # filen saknades. Används bara för att upptäcka ändring — aldrig
        # för att avgöra om innehållet duger.
        self.stamp: tuple[int, int] | None = None

    @property
    def loaded(self) -> int:
        return len(self.by_name)

    def verify(self, token: str | None) -> Principal | None:
        """
        Slå upp en token. Returnerar None när den inte känns igen.

        Jämförelsen görs mot alla poster med konstant tid, så att
        varken träff eller antal försök går att läsa ur svarstiden.

        Inbjudningstokens ingår INTE här. En stulen inbjudan ska kunna
        växlas in — vilket är högljutt, eftersom den rätta ägaren då
        inte kan växla in sin — men aldrig läsa dokument.
        """
        if not token:
            return None
        digest = hash_token(token)
        träff: Principal | None = None
        for stored, principal in self.by_token_hash.items():
            if hmac.compare_digest(stored, digest):
                träff = principal
        return träff

    def enrollment_for(self, token: str | None) -> UserRecord | None:
        """Posten vars inbjudan denna token är, om någon."""
        if not token:
            return None
        digest = hash_token(token)
        träff: UserRecord | None = None
        for record in self.by_name.values():
            if not record.enrollment or not record.token_digest:
                continue
            if hmac.compare_digest(record.token_digest, digest):
                träff = record
        return träff

    def verify_password(
        self, name: str, password: str, throttle: "Throttle | None" = None,
        key: str | None = None,
    ) -> Principal | None:
        """
        Pröva namn och lösenord. None när något inte stämmer.

        OKÄNT NAMN KOSTAR LIKA MYCKET SOM KÄNT. Prövningen görs mot en
        attrappost när namnet inte finns, så att svarstiden inte
        avslöjar vilka konton som existerar. Anroparen får samma svar i
        båda fallen och ska ge samma besked utåt.

        Strypningen är anroparens ansvar att kontrollera FÖRE anropet
        (locked_for); här registreras bara utfallet, så att räknaren
        följer verkligheten även om anroparen glömmer sin kontroll.
        """
        nyckel = key or name.strip().lower()
        record = self.by_name.get(name.strip().lower())
        lagrad = record.password if record and record.password else _DUMMY_RECORD
        ok = verify_password(password, lagrad)
        # Ett konto utan lösenord kan aldrig logga in, hur väl
        # attrappen än skulle råka stämma.
        if record is None or not record.password:
            ok = False
        if throttle is not None:
            if ok:
                throttle.record_success(nyckel)
            else:
                throttle.record_failure(nyckel)
        return record.principal if ok and record else None


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
            token_sha256: "..."          # valfri: inbjudan eller maskintoken
            enrollment: true             # token får BARA växlas in
            password_scrypt: "scrypt$..." # valfri: sätts vid inväxling
            groups: [institution]

    En post måste bära minst ett av token_sha256 och password_scrypt.
    Bär den ingetdera kan den aldrig autentisera, och att låta den
    ligga kvar tyst vore att visa en användare i 'urd auth list' som
    inte kan logga in.
    """
    store = UserStore()
    store.path = path
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
        lösenord = entry.get("password_scrypt")
        enrollment = bool(entry.get("enrollment"))
        if not name:
            store.errors.append(f"{path}: post {i} saknar name")
            continue
        if digest and (
            len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            store.errors.append(
                f"{path}: användare {name!r} har ogiltig token_sha256"
            )
            continue
        if lösenord is not None and _parse_password_record(lösenord) is None:
            # Fail closed: en oläsbar lösenordspost gör hela användaren
            # ogiltig i stället för att tyst falla tillbaka på token.
            store.errors.append(
                f"{path}: användare {name!r} har oläsbar password_scrypt"
            )
            continue
        if not digest and not lösenord:
            store.errors.append(
                f"{path}: användare {name!r} saknar både token_sha256 och "
                "password_scrypt och kan aldrig logga in"
            )
            continue
        if enrollment and not digest:
            store.errors.append(
                f"{path}: användare {name!r} är märkt enrollment men "
                "saknar token att växla in"
            )
            continue
        groups, fel = _validate_groups(name, entry.get("groups"))
        store.errors.extend(fel)
        nyckel = name.lower()
        if nyckel in store.by_name:
            store.errors.append(f"{path}: namnet {name!r} förekommer flera gånger")
            continue
        if digest and digest in store.by_token_hash:
            store.errors.append(
                f"{path}: samma token_sha256 används av flera användare"
            )
            continue
        principal = Principal(name=name, groups=tuple(groups))
        store.by_name[nyckel] = UserRecord(
            principal=principal,
            token_digest=digest or None,
            password=lösenord,
            enrollment=enrollment,
        )
        # Inbjudningstokens ingår inte i tokenuppslaget: de får växlas
        # in, inte användas.
        if digest and not enrollment:
            store.by_token_hash[digest] = principal

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
