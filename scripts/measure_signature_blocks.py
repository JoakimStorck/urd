"""
Mät hur många rollbindningar som står i underskriftsblock och inte
fångas av uttaget.

HYPOTESEN. Beslutsdokumentens mest auktoritativa rollbindning står i
underskriftsblocket: namn på en rad, roll på nästa. Blocket saknar
finit verb och överlever sällan sentence_like_lines, som filtrerar
före dependensparsningen. Beståndets tydligaste uppgift om vem som
innehar en roll skulle därmed vara systematiskt osynlig för Attest.

Observationen som väckte frågan: "Vem är prefekt?" besvaras med vad en
prefekt ÄR, ur arbetsordning och delegationsordning, men namnger
ingen — trots att beslutsdokument undertecknade av prefekten finns i
beståndet.

VAD SKRIPTET MÄTER, OCH INTE. Det räknar närhet mellan personnamn och
rollord i chunktexten, och jämför mot vad Attest faktiskt utvunnit ur
samma dokument. Närhet är inte samma sak som bindning — ett protokoll
kan nämna ett namn och ett rollord i samma stycke utan att påstå
något om relationen. Utdata är därför KANDIDATER att handklassa, inte
ett facit.

Rollvokabulären hämtas ur Attest självt: de objekt som redan
förekommer i identitetsobservationer är beståndets egna rollord, och
den listan behöver inte skrivas för hand.

Körs med servern avstängd (Qdrant släpper in en process i taget):

    python -m scripts.measure_signature_blocks
    python -m scripts.measure_signature_blocks --show 40
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from app.qdrant_store import QdrantStore, StorageLockedError
from app import attest
from app.grammar import sentence_like_lines

# Samma mönster som retrievalens namnuttag: minst två versalinledda
# ord. Ett ensamt versalord är oftast en rubrik eller ett rollord.
# Varje led är antingen ett vanligt namnled eller en initial. Att
# tillåta initialer skiljer sig från retrievalens namnuttag, som
# kräver gemener efter versalen — och det uttaget missar därför
# "E Efternamn", vilket är en egen liten lucka värd att notera.
_LED = r"(?:[A-ZÅÄÖ][a-zåäöé\-]+|[A-ZÅÄÖ]\.?)"
# MELLANSLAG, INTE \s: ett namn korsar aldrig en radbrytning. Med \s+
# svalde mönstret rollen på nästa rad och gjorde "Namn / Prefekt" till
# ett enda fyra ords namn — precis det underskriftsblock vi letar
# efter blev osynligt för mätningen.
NAME = re.compile(rf"\b{_LED}(?:[^\S\n]+{_LED}){{1,3}}\b")

# Hur nära rollordet måste stå namnet för att räknas som kandidat.
# Underskriftsblock har dem på intilliggande rader; 60 tecken rymmer
# det utan att svälja ett helt stycke.
WINDOW = 60


def role_vocabulary(conn) -> set[str]:
    """Beståndets egna rollord, ur Attests identitetsobservationer."""
    rows = conn.execute(
        "SELECT DISTINCT object FROM observations WHERE kind = 'identitet'"
    ).fetchall()
    ord_ = set()
    for (obj,) in rows:
        if not obj:
            continue
        text = obj.strip().lower()
        # Flerordiga roller behålls hela; enordiga är de som fungerar
        # i ett underskriftsblock.
        if 3 <= len(text) <= 40:
            ord_.add(text)
    return ord_


def existing_bindings(conn) -> set[tuple[str, str, str]]:
    """(source_path, namnnyckel, rollnyckel) som redan är utvunna."""
    rows = conn.execute(
        "SELECT source_path, subject_key, object_key FROM observations"
        " WHERE kind = 'identitet'"
    ).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=20,
                    help="Antal exempel att visa (default 20).")
    ap.add_argument("--category", default=None,
                    help="Begränsa till dokument vars sökväg innehåller detta.")
    args = ap.parse_args()

    try:
        store = QdrantStore(vector_size=1024)
    except StorageLockedError as e:
        print(e)
        return 2

    conn = attest.connect()
    roles = role_vocabulary(conn)
    known = existing_bindings(conn)
    print(f"Rollvokabulär ur Attest: {len(roles)} distinkta rollord.")
    print(f"Redan utvunna identitetsbindningar: {len(known)}.\n")

    role_re = re.compile(
        r"\b(" + "|".join(sorted((re.escape(r) for r in roles), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )

    träffar = 0
    täckta = 0
    otäckta: list[tuple[str, str, str, str, bool, str]] = []
    dok_med_träff: set[str] = set()
    dok_med_lucka: set[str] = set()
    rollräknare: Counter = Counter()
    riktningsräknare: Counter = Counter()
    nådde_parsern = 0

    for chunk in store.iter_all_chunks():
        path = chunk.metadata.source_path
        if args.category and args.category.lower() not in path.lower():
            continue
        text = chunk.text
        # Rader som filtret släpper igenom — allt annat är osynligt
        # för uttaget oavsett vad det står.
        synliga = set(sentence_like_lines(text))

        for m in NAME.finditer(text):
            rå = m.group(0)
            # VERSALA ROLLORD ÄR OMÖJLIGA ATT SKILJA FRÅN NAMNLED på
            # formen allena: "Prefekt Anna Andersson" matchar som ett
            # tre ords namn. Rollord skalas därför av från båda
            # ändarna, och att de satt där är i sig uppgiften om
            # vilken konstruktion det rör sig om.
            led = rå.split()
            roll_före = None
            roll_efter = None
            while led and led[0].lower() in roles:
                roll_före = led.pop(0)
            while led and led[-1].lower() in roles:
                roll_efter = led.pop()
            if len(led) < 2:
                # Kvar blev inget namn — bara rollord.
                continue
            namn = " ".join(led)
            # Rollen kan stå på båda sidor: underskriftsblocket har
            # den EFTER namnet ("Namn / Prefekt"), appositionen FÖRE
            # ("Prefekt Namn"). Båda riktningarna räknas, och vilken
            # det var noteras — det är just skillnaden mellan den
            # konstruktion uttaget redan bär och den vi misstänker
            # faller bort.
            # Rollen kan sitta i namnfrasen (avskalad ovan) eller på
            # en närliggande rad.
            if roll_efter:
                roll, riktning = roll_efter, "efter"
            elif roll_före:
                roll, riktning = roll_före, "före"
            else:
                efter = role_re.search(text[m.end(): m.end() + WINDOW])
                if efter:
                    roll, riktning = efter.group(1), "efter"
                else:
                    start = max(0, m.start() - WINDOW)
                    kand = list(role_re.finditer(text[start: m.start()]))
                    if not kand:
                        continue
                    roll, riktning = kand[-1].group(1), "före"
            träffar += 1
            dok_med_träff.add(path)
            rollräknare[roll.lower()] += 1
            riktningsräknare[riktning] += 1

            nyckel = (path, attest._key(namn), attest._key(roll))
            if nyckel in known:
                täckta += 1
                continue

            dok_med_lucka.add(path)
            rad = next(
                (l for l in text.splitlines() if namn in l), ""
            ).strip()
            syns = rad in synliga
            if syns:
                nådde_parsern += 1
            otäckta.append((path, namn, roll, rad[:110], syns, riktning))

    print(f"Namn med rollord inom {WINDOW} tecken: {träffar} träffar "
          f"i {len(dok_med_träff)} dokument.")
    print(f"  redan utvunna av Attest:  {täckta}")
    print(f"  INTE utvunna:             {len(otäckta)}"
          f"  i {len(dok_med_lucka)} dokument")
    if otäckta:
        andel = 100 * nådde_parsern / len(otäckta)
        print(f"\nAv de outvunna nådde {nådde_parsern} ({andel:.0f} %) "
              f"igenom radfiltret;")
        print(f"  {len(otäckta) - nådde_parsern} "
              f"({100 - andel:.0f} %) sållades bort före parsning.")
        print("")
        print("  TALEN SKILJER TVÅ ÅTGÄRDER. Sållas raderna bort sitter")
        print("  felet i sentence_like_lines. Passerar de men ger ändå")
        print("  ingen observation är formen problemet: namn och roll")
        print("  står på var sin rad, ingen enskild mening bär båda, och")
        print("  meningar utan finit verb hoppas över vid parsning. Då")
        print("  hjälper inget filter — det krävs en konstruktion som")
        print("  känner igen underskriftsblockets FORM.")

    print(f"\nRollens läge: {riktningsräknare.get('efter', 0)} efter namnet "
          f"(underskriftsform), {riktningsräknare.get('före', 0)} före "
          f"(appositionsform).")

    print("\nVanligaste rollord i kandidaterna:")
    for roll, n in rollräknare.most_common(12):
        print(f"  {n:>5}  {roll}")

    print(f"\nExempel på outvunna kandidater (första {args.show}):")
    print("  [P] = raden nådde parsern, [-] = filtrerades bort före\n")
    for path, namn, roll, rad, syns, riktning in otäckta[: args.show]:
        mark = "P" if syns else "-"
        print(f"  [{mark}] {path.rsplit('/', 1)[-1]}  (roll {riktning} namnet)")
        print(f"      {namn}  ->  {roll}")
        print(f"      {rad!r}\n")

    print("Närhet är inte bindning. Listan är kandidater att handklassa,")
    print("inte ett facit — ett protokoll kan nämna namn och rollord i")
    print("samma stycke utan att påstå något om relationen.")

    store.close()
    return 0 if träffar else 1


if __name__ == "__main__":
    sys.exit(main())
