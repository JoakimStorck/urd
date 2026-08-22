#!/usr/bin/env python3
"""
Skulle en REN KLON av repot fungera?

Kör från instansens rot:

    python3 scripts/verify_import.py
    python3 scripts/verify_import.py --quiet     # bara utfall och fel

Skriptet finns därför att repot den 22 augusti 2026 saknade app/auth.py
i två veckor, medan pushade commits importerade den. Instansen
fungerade — filen låg i arbetskatalogen, ospårad — så inget lokalt test
kunde upptäcka det. Samtidigt läste koden tre konfignycklar som en ren
installation inte hade. En klon av main gick alltså inte att starta,
och ingenting sa ifrån.

Tre kontroller, alla billiga och alla deterministiska:

1. SPÅRNING  — ligger något i app/ som git inte känner till? Det är
   den enda kontrollen som fångar felet PÅ MASKINEN DÄR DET UPPSTÅR,
   innan pushen. De två övriga fångar det först i en ren klon.

2. IMPORT    — går varje modul under app/ att importera? Fångar saknade
   filer, obundna modulnivånamn och trasiga importkedjor. py_compile
   gör det INTE: den kompilerar varje fil för sig och ser aldrig att
   ett namn saknas i en annan modul.

3. KONFIG    — läser koden någon settings-nyckel som Settings inte
   definierar? Fångar den andra halvan av auth-fallet: nycklar som
   levde i den lokala .urd/config.json men aldrig i koden.

4. NAMN      — används något namn som aldrig binds i filen? Fångar en
   funktion som raderats medan anropet står kvar. Namnet slås upp
   först när koden KÖRS, så varken py_compile eller importkontrollen
   ser det.

Vad skriptet INTE gör: det kör ingenting. Verkningslösa flaggor,
oåtkomlig kod och felaktiga returtyper kräver funktionella prov, och
de skrivs per patch. Det här är golvet, inte taket.

Tunga beroenden ersätts med attrapper. Det görs ALLTID, även när de
riktiga finns installerade, så att utfallet blir detsamma i en bar
klon som i en full .venv — och så att kontrollen aldrig laddar modeller,
öppnar vektorlagret eller talar med Ollama. Attrapperna registreras
före första importen och tar därmed företräde.

Exitkod 0 om allt är rent, annars 1.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import pkgutil
import re
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class _AttrappMeta(type):
    """
    Attributåtkomst på KLASSEN, inte bara på instanser.

    Koden gör Distance.COSINE och PayloadSchemaType.KEYWORD utan att
    instansiera något. Utan metaklass faller __getattr__ aldrig ut för
    de anropen, och importkontrollen larmar om ett fel som inte finns.
    """

    def __getattr__(cls, name):
        return _Attrapp()


class _Attrapp(metaclass=_AttrappMeta):
    """
    Står in för vad som helst: anropas, indexeras, itereras, attribut.

    Toleransen är avsiktlig. En snäv attrapp ger FALSKLARM — importen
    faller på attrappen och ser ut som ett fel i koden. Uppmätt:
    app.api instansierar retrievallagret vid import, som i sin tur
    anropar embed_query("test") och indexerar resultatet. Priset för
    toleransen är att skriptet inte kontrollerar signaturer mot de
    riktiga biblioteken; det är rätt avvägning för en importkontroll.
    """

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return _Attrapp()

    def __getattr__(self, name):
        return _Attrapp()

    def __getitem__(self, key):
        return _Attrapp()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


def _attrapp_modul(name: str, **attribut) -> None:
    m = types.ModuleType(name)
    m.__path__ = []          # så att undermoduler kan registreras
    for k, v in attribut.items():
        setattr(m, k, v)
    sys.modules.setdefault(name, m)


def installera_attrapper() -> None:
    """Tunga eller nätverksberoende bibliotek som aldrig ska köras här."""
    for name in ("stanza", "docling", "requests", "uvicorn"):
        _attrapp_modul(name)
    _attrapp_modul("rank_bm25", BM25Okapi=_Attrapp)
    _attrapp_modul("qdrant_client", QdrantClient=_Attrapp)
    _attrapp_modul(
        "qdrant_client.models",
        **{k: _Attrapp for k in (
            "Filter", "FieldCondition", "MatchValue", "MatchAny",
            "PointStruct", "VectorParams", "Distance", "SearchParams",
            "HnswConfigDiff", "PayloadSchemaType", "Range", "IsEmptyCondition",
        )},
    )
    _attrapp_modul(
        "sentence_transformers", SentenceTransformer=_Attrapp, CrossEncoder=_Attrapp
    )
    _attrapp_modul(
        "ollama",
        Client=_Attrapp,
        chat=_Attrapp(),
        generate=_Attrapp(),
        ResponseError=type("ResponseError", (Exception,), {}),
    )


# --------------------------------------------------------------------
# 1. Spårning
# --------------------------------------------------------------------

def kontrollera_sparning() -> list[str]:
    """
    Filer under app/ som git inte känner till.

    Genererat och lokalt undantas: cache, kompilat och instansdata.
    users.yaml nämns uttryckligen — den hör hemma i .urd/, och en kopia
    under app/ är antingen död eller en strökopia av inloggningsuppgifter.
    """
    try:
        ut = subprocess.run(
            ["git", "ls-files", "app"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return [f"kunde inte fråga git om spårade filer: {e}"]

    sparade = {ROOT / rad for rad in ut.splitlines() if rad}
    fel = []
    for p in sorted((ROOT / "app").rglob("*")):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        if p not in sparade:
            fel.append(f"ospårad fil i app/: {p.relative_to(ROOT)}")
    return fel


# --------------------------------------------------------------------
# 2. Import
# --------------------------------------------------------------------

def kontrollera_import(tyst: bool) -> tuple[list[str], list[str]]:
    """
    Importera varje modul under app/ och skilj FEL från ARTEFAKT.

    ImportError och NameError är kodens egna fel: en modul som saknas i
    repot, ett namn som aldrig bands på modulnivå. Det är precis de två
    felklasser skriptet finns för, och de faller ut före allt annat —
    api.py:s `from app import auth` ligger på rad 12, långt före något
    arbete.

    Andra undantag är regelmässigt attrappartefakter. api.py bygger
    retrievallagret VID IMPORT, vilket öppnar vektorlagret och läser
    hela kollektionen; ingen attrapp kan härma det ärligt, och att
    försöka vore att bygga in falsk trygghet. De rapporteras som
    upplysning, inte som fel — men de rapporteras, för att en modul som
    arbetar vid import är värd att veta om (den kan inte importeras
    medan servern håller Qdrant-låset).
    """
    import app

    fel, artefakter, klara = [], [], []
    for m in sorted(pkgutil.iter_modules(app.__path__), key=lambda x: x.name):
        namn = f"app.{m.name}"
        try:
            importlib.import_module(namn)
            klara.append(m.name)
        except (ImportError, NameError) as e:
            fel.append(f"{namn}: {type(e).__name__}: {e}")
        except Exception as e:
            artefakter.append(f"{namn}: arbetar vid import ({type(e).__name__}: {e})")
    if not tyst:
        print(f"  {len(klara)} moduler importerade")
    return fel, artefakter


# --------------------------------------------------------------------
# 3. Konfignycklar
# --------------------------------------------------------------------

_SETTINGS = re.compile(r"settings\.([a-z_][a-z0-9_]*)")


def kontrollera_konfig() -> list[str]:
    from app.config import settings

    fel = []
    for kat in ("app", "scripts"):
        for p in sorted((ROOT / kat).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            for i, rad in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                kod = rad.split("#", 1)[0]
                for nyckel in _SETTINGS.findall(kod):
                    if not hasattr(settings, nyckel):
                        fel.append(
                            f"{p.relative_to(ROOT)}:{i}: settings.{nyckel} "
                            "finns inte i Settings"
                        )
    return fel


# --------------------------------------------------------------------
# 4. Obundna namn
# --------------------------------------------------------------------

_BUILTINS = set(dir(__builtins__)) | {
    "__name__", "__file__", "__doc__", "__package__", "__builtins__",
    "self", "cls",
}


def _bundna_namn(träd) -> set[str]:
    """
    Varje namn som binds någonstans i filen.

    Medvetet GROVT: bindningar samlas utan hänsyn till räckvidd, vilket
    överskattar mängden och därmed ger färre falsklarm. Kontrollen ska
    hitta namn som inte finns NÅGONSTANS — det var den felklassen som
    slank igenom.
    """
    bundna: set[str] = set()
    for nod in ast.walk(träd):
        if isinstance(nod, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef, ast.Lambda)):
            # Lambda saknar namn men BINDER sina parametrar. Utan den
            # raden larmade kontrollen på "x" i varje sorteringsnyckel.
            if not isinstance(nod, ast.Lambda):
                bundna.add(nod.name)
            args = getattr(nod, "args", None)
            if args is not None:
                for a in (args.posonlyargs + args.args + args.kwonlyargs):
                    bundna.add(a.arg)
                for a in (args.vararg, args.kwarg):
                    if a is not None:
                        bundna.add(a.arg)
        elif isinstance(nod, ast.Name) and isinstance(nod.ctx, (ast.Store, ast.Del)):
            bundna.add(nod.id)
        elif isinstance(nod, (ast.Import, ast.ImportFrom)):
            for alias in nod.names:
                bundna.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(nod, ast.ExceptHandler) and nod.name:
            bundna.add(nod.name)
        elif isinstance(nod, (ast.Global, ast.Nonlocal)):
            bundna.update(nod.names)
    return bundna


def kontrollera_obundna_namn() -> list[str]:
    """
    Namn som används men aldrig binds i filen.

    Fångar den felklass som varken py_compile eller importkontrollen
    ser: en funktion som raderas medan anropet står kvar. Namnet slås
    upp först när koden KÖRS, så modulen importeras utan invändning och
    felet visar sig i drift.

    Uppmätt 2026-08-22: patch 0038 raderade _parenthesis_kind genom en
    slarvig textersättning. py_compile godtog filen, importkontrollen
    godtog modulen, och urd attest-build skulle ha fallit på NameError
    vid nästa körning.
    """
    fel = []
    for kat in ("app", "scripts"):
        for p in sorted((ROOT / kat).rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            try:
                träd = ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as e:
                fel.append(f"{p.relative_to(ROOT)}: kunde inte tolkas: {e}")
                continue
            bundna = _bundna_namn(träd) | _BUILTINS
            saknade = {
                n.id for n in ast.walk(träd)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                and n.id not in bundna
            }
            for namn in sorted(saknade):
                fel.append(
                    f"{p.relative_to(ROOT)}: {namn!r} används men binds aldrig"
                )
    return fel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="bara utfall och fel")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    installera_attrapper()

    alla: list[tuple[str, list[str]]] = []

    if not args.quiet:
        print("Spårning i git")
    alla.append(("spårning", kontrollera_sparning()))

    if not args.quiet:
        print("Import av app-moduler")
    importfel, artefakter = kontrollera_import(args.quiet)
    alla.append(("import", importfel))

    if not args.quiet:
        print("Konfignycklar")
    alla.append(("konfig", kontrollera_konfig()))

    if not args.quiet:
        print("Obundna namn")
    alla.append(("namn", kontrollera_obundna_namn()))

    antal = sum(len(f) for _, f in alla)
    print("")
    if artefakter:
        for a in artefakter:
            print(f"  Upplysning: {a}")
        print("")
    if antal == 0:
        print("Rent: en klon av repot går att importera.")
        return 0

    for namn, fel in alla:
        for f in fel:
            print(f"  FEL [{namn}] {f}")
    print("")
    print(f"{antal} problem. En ren klon av repot är inte körbar.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
