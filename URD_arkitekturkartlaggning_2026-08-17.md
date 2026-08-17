# URD — arkitekturkartläggning

**Datum:** 2026-08-17 · **Kod:** commit `72875a9` ("Läs dokumentvikt ur payloaden") · **Underlag:** hela kodbasen (≈14 800 rader i `app/` och `scripts/`), läst mot white paper v2.8 och åtgärdsplanerna.

Dokumentet är ett orienteringsdokument för den som ska arbeta i koden. Där koden och dokumentationen säger olika saker gäller koden; avvikelserna samlas i sista avsnittet.

---

## 1. Systemform i en mening

URD är en QUD-styrd dokumentassistent i sex lager: **ingest och representation** (Docling → sektioner → chunkar → evidensobjekt → Qdrant), **hybrid retrieval** (E5-embeddings + BM25 → cross-encoder på sannolikhetsskala → dokumentexpansion → evidensläsning → operationsstyrt urval), **samtalsmodell** (intentklassificering, QUD, driftskydd, rework-vägar, sessionstillstånd — allt i `RagService.converse`), **syntes och vakter** (direktsyntes → källvakt mot kontexten → korpuskontroll mot beståndet), **predikations- och attestlager** (Stanza-baserad dragutvinning; SQLite-index över grammatiska observationer som signal i urval, boost och kontroll), samt **gränssnitt** (CLI med interaktivt läge, FastAPI-server, klientläge, webb-UI).

Ansvarsfördelningen är genomgående ren: retrievalpolicyn bor i `retrieval.py`, samtalslogiken i `converse`, grammatiken i `grammar.py` utan URD-begrepp, kopplingen till URD i `predication.py`, aggregeringen i `attest.py`, och HTTP-lagret (`api.py`, 145 rader) är en tunn mappning av request → `converse` → response.

### Modulkarta

| Modul | Rader | Ansvar |
|---|---|---|
| `retrieval.py` | 2185 | BM25Index, Reranker, RagService: `answer`, `rework`, `converse`, attestsignal, urval |
| `cli.py` | 2153 | Typer-CLI: serve/stop/connect, ingest/reindex/enrich, stats, config, attest-*, ask, test, interaktivt läge |
| `grammar.py` | 1414 | Grammatisk dragutvinning ur svensk text (Stanza), fri från URD-begrepp |
| `ingest.py` | 902 | Docling-extraktion, sektionering, chunkning, kontextprefix, evidensobjekt, metadata |
| `attest.py` | 692 | Observationslager i SQLite: byggning, relevansmodell, uppslag |
| `rework.py` | 478 | Elaboration (direktprompt + shingle-klippning) och verification (tvåstegs med findings) |
| `intent.py` | 445 | LLM-klassificering av yttring: intent, substyle, question_operation |
| `qdrant_store.py` | 391 | Två kollektioner (chunkar + evidens), sökning med source_path-filter, CRUD |
| `config.py` | 365 | Settings med prioritet env > `.urd/config.json` > defaults |
| `config_validation.py` | 355 | Schemakontroll av instansens YAML-filer; `urd config validate`, startlogg, `/health` |
| `predication.py` | 337 | Kopplar drag till chunkar/fråga/svar; klassificering belagd/tvetydig/motsagd/obelagd (skuggläge) |
| `corpus_guard.py` | 337 | Korpuskontroll av rollbindningar + uppräkning av rollinnehavare |
| `synonyms.py` | 321 | Synonymexpansion för BM25, böjningstolerant |
| `repl.py` | 307 | Interaktivt läge med punktkommandon, lat uppstart, server-eller-lokalt |
| `session_state.py` | 269 | ConversationState, SessionStore, `select_active_hits` |
| `synthesis.py` | 251 | Direktsyntesen: källformatering, operationsblock, DIRECT_SYNTHESIS_PROMPT |
| `concepts.py` | 184 | Begreppsmodell med broader-relationer för BM25-expansion |
| `source_guard.py` | 175 | Siffervakt + källreferensvakt, deterministisk |
| `followup.py` | 175 | Omskrivning av följdfråga med sanering + vokabulärvalidering |
| `connect_api.py` | 167 | Klientläge: lokal UI-servering, proxy av `/chat` och `/document` |
| `qud_drift.py` | 157 | Driftmätning: fråga-mot-passage (primär) och fråga-mot-fråga (fallback) |
| `api.py` | 145 | FastAPI: `/`, `/health`, `/refresh`, `/document`, `/chat` |
| `question_rules.py` | 111 | Regelbaserad föroperation (entity_aggregation, entity_lookup, comparison, aggregation) |
| `social.py` | 103 | Social/meta-hantering utan retrieval |
| `llm.py` | 96 | Ollama-adapter: temperature 0, explicit num_ctx och think, trunkeringsvarning |
| `morphology.py` | 70 | Prefixbaserad böjningsheuristik (`is_inflection_of`, `VALID_ENDINGS`) |
| `embeddings.py` | 58 | E5-prefix: `query:` i embed_query, `passage:` i embed_texts |

Därtill `schemas.py` (Pydantic-modeller), `synthesis_types.py` (delad returtyp), `preprocess_llm.py` (enrich-extraktion), `quiet.py` (dämpning av bibliotek), `question_operations.py` (YAML-laddning av operationspolicyer).

---

## 2. Ingest och representation

### Extraktion och sektionering

`ingest.py` läser `.pdf`, `.docx` och `.xlsx` via en modulglobal Docling-`DocumentConverter`. Export sker till markdown med fallback till ren text. Misslyckad extraktion skiljs från tomt konverteringsresultat (`RawDocument.error`), och `ingest_path_with_evidence` returnerar felorsaken så att ett dokument utanför indexet aldrig blir osynligt.

Sektioner byggs ur markdown-rubriker (`split_markdown_sections`); ett dokument utan rubriker faller tillbaka på styckesindelning. Ovanpå detta ligger **rubrikhierarkin ur avsnittsnumreringen**: Docling ger inte pålitliga rubriknivåer (i anställningsordningen ligger 188 rubriker på nivå 2, varav 158 semantiskt icke-unika — "Behörighet" 16 gånger), så föräldrakedjan härleds ur numreringen (8.5.2 → 8.5 → 8), i första hand ur sektioner med matchande nummer, i andra hand ur dokumentets egen innehållsförteckning (igenkänd på punktledare). Saknas numret i båda källorna utelämnas nivån — hellre ingen kedja än en gissad.

### Chunkning och kontextprefix

`chunk_text` klipper på menings- eller radgräns i intervallet (chunk_size/2, chunk_size], med hårt klipp som fallback, och gränsmedvetet överlapp (nästa chunk börjar på hel mening/rad). Storlekarna läses ur `settings.chunk_size`/`chunk_overlap` (1200/150) — den gamla åtgärd 5.1-anmärkningen om hårdkodade värden är alltså åtgärdad.

Varje chunk indexeras med ett **kontextuellt prefix**: `Dokument: <titel>\nAvsnitt: <hela rubrikkedjan>\n---\n` följt av texten. Rubrikkedjan gör att "7.2 Behörighet" och "8.2 Behörighet" är olika texter för embedding och cross-encoder.

### Evidensobjekt

Fyra typer identifieras per stycke: figur, tabell, punktlista, numrerad lista (regex-baserade detektorer på markdown-form). Varje `EvidenceObject` bär `evidence_text` (märkt `[Tabell]`/`[Punktlista]`/…), `supporting_before`/`supporting_after` (grannstyckena), och `referring_passages` (upp till fyra stycken i samma sektion som refererar objektet — "följande", "tabell 3", figurnummer). Evidensobjekten lagras i en egen Qdrant-kollektion (`<collection>__evidence`); vid uppslag byggs deras söktext av evidens + stödtext + referenser, och sektionstiteln suffixas med `[<evidence_type>]`.

### Metadata: datum, diarienummer, dokumenttyp och normativ tyngd

`extract_document_header_info` läser **innehållet först, filnamnet sedan** (Joakims direktiv: datalagret kan flytta till databas). Prioritetsordnade mönster i de första 3000 tecknen: Revised/Reviderad > Beslutsdatum > Fastställd/Beslutad/Gäller från; därefter filnamnets `rev20250909` respektive `20250311_`. Diarienummer ur "Diarienummer C 2025/1205". Hellre `None` än gissning.

`infer_document_type` härleder `document_type` och `document_weight` (norm | guidance | record) **deterministiskt ur sökvägen** via `.urd/document_types.yaml` (repomall i `app/document_types.yaml`), med filnamnsmatchning, sökvägsmatchning och filnamnshintar. Detta ersätter i praktiken enrich som källa till dokumenttyp — enrich är ett LLM-anrop per sektion och kördes sällan, vilket var skälet till att fältet var null på varje träff och att normkälleregeln i syntesprompten länge saknade underlag. Sökvägshärledningen har företräde; `semantic.document_type` är komplement. Senaste commit (`72875a9`) stängde sista länken: `document_weight` läses nu tillbaka ur Qdrant-payloaden.

Enrich (`urd enrich` + `preprocess_llm.py`) extraherar per sektion: document_type, keywords, roles, actions, time_markers, applies_to, summary — med en separat, mindre Ollama-modell (`preprocess_ollama_model`). Metadatan skrivs tillbaka i chunk-payloaden; `semantic_enriched`/`semantic_version`/`semantic_source_hash` spårar status.

### Indexintegritet

Fingerprint är sha1 av sökväg + storlek + mtime. Ingest-loopen hittar nya/ändrade/försvunna dokument genom att jämföra disk mot `QdrantStore.get_indexed_documents`, och upptäcker dokument med chunkar men utan evidensobjekt (`get_evidence_source_paths`) för backfill. Känd backlogpunkt: backfillen försöker om samma 53 evidenslösa dokument vid varje ingest (behöver försöks-manifest).

---

## 3. Retrievalkedjan

Hela kedjan ligger i `RagService.answer`. Stegen i ordning:

**Söktext i två varianter.** `search_text` (kandidatinsamling: embedding + BM25) kan vara QUD-konkatenerad eller en omskriven retrievalfråga; `rerank_text` är **alltid den rena frågan**. Detta är den empiriskt funna separationen: mMARCO-tränade cross-encoders kollapsar på metatext av formen "(Huvudfråga i samtalet: …)". QUD-ankaret påverkar vad som tas upp i poolen, aldrig hur det bedöms.

**1. Semantisk sökning.** `Embedder` sätter `query:`/`passage:`-prefix automatiskt när modellnamnet innehåller "e5" och normaliserar vektorerna. Global sökning (limit 15) körs alltid; `preferred_source_paths` (broadening-ankare, attestdokument) ger en **kompletterande ankrad pool** (limit 8) som mergas in — aldrig ett hårt filter. (Qdrants `Filter(should=...)` finns kvar i `QdrantStore.search` men används numera bara för den ankrade delpoolen och för evidenssökningen, där avgränsning till valda dokument är avsikten.)

**Attestsignal för entitetsfrågor.** Vid `question_operation == "entity_lookup"` och `attest_selection` på: `_attest_source_paths` plockar kandidattermer ur frågan (allt utom frågeord/funktionsord, ≥4 tecken) plus personnamn på formen två versalinledda ord, slår upp åt **båda hållen** (`lookup_object` för rolltermer, `lookup_subject` för namn), och mappar de bäst belagda bindningarnas källfiler till sökvägar via BM25-indexet. Dokumenten läggs till som preferens och `relevance_by_path` sparas till boost-steget. Motivering i koden: cross-encodern mäter aboutness och kan inte skilja passagen som *predicerar* rollen från den som bara nämner den.

**2. BM25.** `BM25Index` byggs vid uppstart från alla chunkar i Qdrant (`_build_bm25_index`; `refresh_index` efter ingest, servern nås via `POST /refresh`). Söktexten breddas med tre slags tillägg som **bara** påverkar BM25: operationstermer ur `question_operations.yaml` (för relation_membership även broader-begreppens etiketter), synonymexpansion (`synonyms.py`, symmetrisk inom grupp, böjningstolerant via `morphology.is_inflection_of`), och broader-expansion ur begreppsmodellen ("adjungerad lektor" står under "adjungerad lärare").

**3. Merge + tvåspårig comparison.** Kandidatpoolerna dedupliceras på chunk_id. För comparison med ≥2 matchade begrepp körs varsitt kompletterande spår per begrepp (embedding på `rerank_text + label`, BM25 på label + operationstermer) — enkelspårig retrieval hämtar annars bara den sida frågan råkar likna.

**4. Cross-encoder-rerank på sannolikhetsskala.** `Reranker.rerank` filtrerar boilerplate (sändlistor, bilagerubriker, `<!-- image -->`, mikrochunkar), poängsätter (fråga, chunktext)-par och normaliserar logits genom sigmoid till (0,1). Golvet är `filter_floor` (default 0.5 = gamla logit-0). Debug bär både rå logit och sannolikhet per kandidat, inklusive bortfiltrerade — vid abstain är det enda spåret till varför.

**5. Dokumentexpansion.** Alla dokument med minst en chunk ≥ `expansion_min_prob` (0.55) expanderas: övriga chunkar hämtas ur BM25-indexets `_by_source` och rerankas i ett andra pass med lägre golv `expanded_min_prob` (0.27). Poolerna slås ihop och sorteras om.

**5b. Attestboost.** `_apply_attest_boost`: påslag = `attest_boost` (0.15) × bindningens relevans, additivt i sannolikhet, kapat vid 1.0. Att bara vidga poolen räckte inte (uppmätt 2026-08-16: dokumenten nådde poolen men rankades under protokollen); signalen måste påverka rangordningen. Attests egen rangordning fortplantas via relevansviktningen.

**6. Texturval.** `_dedup_and_select`: pass 1 tar allt ≥ `select_min_prob` (0.5) med dedup per (source_path, section_title), upp till `max_hits` (10); pass 2 fyller på till `min_desired_hits` (3) med hits ≥ 0.35 — osäkra men inte avfärdade, avsedda som elaborationsmaterial.

**7. Evidensläsning.** För de tre högst rankade dokumenten hämtas upp till 12 evidensobjekt (`search_evidence`, återanvänd query-vektor), rerankas mot den rena frågan med det lägre golvet, och boostas av textstöd: sektionsmatch mot en textchunk ≥0.5 ger +0.15, dokumentmatch +0.05 (`evidence_*_prob_boost`). Urvalet (`_select_evidence_hits`) kräver ≥0.5 efter boost, dedup per (dokument, sektion, typ), max 4. `_merge_with_evidence_precedence` ordnar sedan: evidensobjekt först, textträffar från samma dokument, övriga textträffar — principen "evidensobjekt först, stödtext därefter".

**8. Urval till syntes.** `_select_hits_for_synthesis`: golv `max(0.5, topp − 0.4)`, tak ur `_SYNTHESIS_MAX_HITS` per operation (direct_lookup 3, relation_membership/requirements 4, process 5, comparison 6, aggregation 8). För comparison säkrar `_ensure_comparison_balance` att varje jämfört begrepp täcks av minst en källa — kompletteringar får överskrida taket, en jämförelse utan båda sidorna är värdelös oavsett tak.

**Abstain** är ett fullt utbyggt svar: en fast formulering plus komplett debug (varje bedömd kandidat, bästa sannolikhet mot golvet, timing), och en loggrad som skiljer "inget relevant fanns" från "golvet var för högt".

---

## 4. Samtalsmodellen

All samtalslogik bor sedan 2026-08-16 i `RagService.converse` — `api.py` mappar bara HTTP. `answer()` finns kvar som den kontextlösa vägen (används av converse internt, diagnostik och skript).

### Klassificering

`intent.classify_utterance` kör ett LLM-anrop med QUD-text, historikfönster (`classification_history_turns`) och yttring, och parsar JSON till `Classification(intent, substyle, question_operation, reason)`. Fem intents: `new_main_question`, `related_to_qud` (substyles subquestion/broadening/narrowing_or_repair), `elaboration`, `verification_or_challenge`, `social_or_meta`. Frågeoperationen är en **separat axel** med åtta värden: direct_lookup, entity_lookup, entity_aggregation, relation_membership, comparison, requirements, process, aggregation.

Felläget är asymmetriskt och prompten lutar mot dokumentvägarna; fallback vid parse-/LLM-fel är `new_main_question` (garanterat källbaserat svar, ingen gammal QUD styr). Två skyddsregler efteråt: klasser som opererar på tidigare material utan QUD/historik tolkas om till new_main_question; elaboration/verification utan `active_hits` tolkas om till broadening (om QUD finns) eller new_main_question.

`question_rules.rule_based_operation` avgör **före** LLM:en för entydiga mönster, i prövningsordning: entity_aggregation ("vilka professorer finns"), entity_lookup ("vem är X", "vilken roll har X"), comparison, aggregation. Endast operationen överstyrs, aldrig intenten; `operation_source` (llm | rule_confirmed | rule_override) loggas.

### QUD och driftskydd

`ConversationState` bär: turhistorik (fönster = 2 × största kontextparametern), `active_doc_paths`, `active_answer_snippets`, `active_hits` (de faktiska chunkar som bar senaste dokumentsvaret, valda av `select_active_hits`: topphiten + upp till 2 till ≥0.5, inget toppdokumentlås), `consumed_hit_ids` (nollställs vid ny QUD), `last_answer`, och QUD som **ordagrann originaltext** med turindex. Rework-turer ersätter inte active_hits; sociala turer rör varken doc_paths eller snippets, så en följdfråga efter "tack" anknyter till senaste dokumenttur.

Driftskyddet (`qud_drift.measure_drift`) körs bara när klassificeringen är related_to_qud. Två nivåer: **dokumentbaserad drift** är primär när aktiva chunkar finns — högsta likheten yttring-mot-chunktext (fråga-mot-passage, exakt E5:s träningsregim; kalibreringen visade att fråga-mot-fråga inte kan skilja institutionell vokabulär från trådsamhörighet). QUD-likheten (fråga-mot-fråga) är fallback efter abstain. Båda måtten loggas alltid för efterkalibrering; trösklarna är `qud_drift_doc_threshold`/`qud_drift_threshold` (båda provisoriskt 0.80). Detekterad drift skriver om klassificeringen till new_main_question med spårbar reason.

### Dispatch och vägval

- **social_or_meta** → `social.handle_social`: kort LLM-svar utan retrieval, med explicit förbud mot nya faktapåståenden om dokumentinnehåll.
- **elaboration / verification_or_challenge** → `rework()` mot `state.active_hits` (se avsnitt 5).
- **new_main_question** → QUD sätts före retrieval; standardväg genom `answer()`.
- **related_to_qud** → QUD-ankare i search_text + samtalsbakgrund till syntesen (`qud_background_turns`). Vid **broadening** dessutom: `rewrite_followup` skriver om följdfrågan till en fristående retrievalfråga, och aktiva dokument skickas som preferens (ankrad delpool).

`rewrite_followup` bygger på två hårda lärdomar: omskrivningskontexten innehåller **aldrig** assistentens svarstext (källan till ärvda hallucinationer som "beslutskonstanta"), och resultatet saneras (prefix, källreferenser, citattecken) och **vokabulärvalideras** — innehållsord ≥5 tecken måste finnas i följdfrågan, QUD:n, tidigare användarfrågor eller dokumentnamnen, med böjningstolerans (prefix + längddiff ≤4). Förkastad omskrivning → originalfrågan med QUD-ankare.

### Kontextuell fallback vid abstain

Om första försöket abstainar, en tidigare QUD finns, och försöket antingen saknade QUD-ankare eller körde med omskriven fråga: retrieval körs om **en** gång med föregående QUD som ankare och samtalsbakgrund. Räddas svaret återställs dessutom QUD:n om turen felklassats som new_main_question. Fallbacken kan aldrig försämra utfallet (aktiveras bara när alternativet är tomt svar) och gör felaktiga driftöverridningar och klassificerarflippar återhämtningsbara.

---

## 5. Syntes och vakter

### Direktsyntesen

`synthesis.synthesize` är ett enda LLM-anrop. `DIRECT_SYNTHESIS_PROMPT` är organiserad i regelgrupper: korrekthet (endast uttryckligt stöd; tvådelade frågor besvaras i alla stödda delar — "ett halvt svar på en tvådelad fråga är ett fel"; delar utan stöd markeras), **källtyp och aktualitet** (normkällor väger tyngre än protokolluppgifter; nyare källa har företräde och båda datumen redovisas), relevans, och form (alltid svenska även ur engelska källor; `[Källa N]` efter varje påstående; inled direkt med svaret). Operationsspecifika block läggs till för relation_membership, comparison, requirements, process, aggregation.

Källhuvudena bär det som gör reglerna tillämpbara: `[Källa N] filnamn — sektionsrubrik (daterad ÅÅÅÅ-MM-DD) [dokumenttyp, normkälla/historisk uppgift]`. Samtalsbakgrunden skickas i ett eget block, uttryckligen märkt som tolkningsnyckel och inte källa.

### Rework: elaboration och verification

De två uppgifterna har medvetet olika arkitektur (`rework.py`):

**Elaboration** är en formuleringsuppgift → direktprompt, ingen mellanrepresentation (tvåstegsextraktion komprimerade bort detaljer). Materialet hämtas av `retrieve_for_elaboration`: alla chunkar ur aktiva dokument minus active_hits **och consumed_hit_ids** (konsumtionsspårningen ur white paperns önskelista finns alltså), rerankade mot QUD-frågan, golv 0.5, utan sektionsdedup. Tomt resultat → ärlig abstain (`ELABORATE_EMPTY_ANSWER`). Efteråt klipps upprepning **mekaniskt**: stycken vars 6-gram-shingles till ≥75 % finns i föregående svar stryks (korta stycken undantagna; skulle allt klippas behålls originalet). Känd lucka: klippningen jämför bara mot föregående svar, inte inom det egna svaret.

**Verification** är en klassificeringsuppgift → två steg. Steg 1 extraherar findings som JSON (claim, status supported/unclear/unsupported, source), med explicita förbud mot fabricerade granskningsobjekt och krav att insinuerade påståenden i användarens prövning granskas som egna findings. Steg 2 formulerar svaret ur findings-listan. Misslyckad parsning → ärlig abstain, **ingen** enstegsfallback — en granskning utan struktur vore bedräglig att presentera som granskning.

Källvisningen skiljer sig: elaboration visar de nya hits som bar tillägget; verification visar de ursprungliga.

### Källvakten (mot kontexten)

`source_guard.check_answer` är deterministisk strängbearbetning på millisekundnivå, körd på huvudvägen och elaborationsvägen. **Siffervakten**: alla flersiffriga tal i svaret (whitespace-normaliserade, exklusive listmarkörer och källhänvisningssiffror) måste förekomma i kontrollunderlaget. Underlaget är allt syntesen faktiskt såg: källtexterna plus källhuvudens metadata (datum som "daterad …", sektionsrubriker, filnamn), och på elaborationsvägen även föregående svar (legitim referenspunkt). **Källreferensvakten**: varje `[Källa N]` inom räckvidd; långa stycken utan referens räknas men fäller inte. Utfallet är trappat: full rapport i debug/JSONL alltid; synlig varningsrad i svaret bara vid obelagda tal. Matchningen är medvetet generös (substräng på normaliserade sifferföljder) — vaktens värde är att fånga tal som inte förekommer alls.

### Korpuskontrollen (mot beståndet)

`corpus_guard` prövar en annan fråga än källvakten: inte "är svaret troget sin kontext" utan "stämmer det med vad beståndet i övrigt belägger". Rollbindningar extraheras ur svarstexten med avsiktligt snäva mönster ("X är/har rollen som Y"; "som"-konstruktionen har företräde så att anställningsform mellan verb och roll inte tas för rollen), slås upp i Attest, och när svarets bindning är svag (tvetydig eller relevans < 0.15) medan en annan bindning för samma person är tydligt starkare (gap ≥ 0.3) **appenderas** ett tillägg — svaret skrivs aldrig om (aggregatet pekar ut, originaltexten bär), och formuleringen säger vad beståndet *belägger*, aldrig att svaret är fel (beläggning är inte sanning). Frånvaro av belägg i Attest ger inget fynd alls — uttaget missar ~vart femte fall och personer har flera roller.

Två Attest-drivna svarstillägg till: vid `entity_aggregation` appenderas **uppräkningen av rollinnehavare** (`format_role_holders`) — listan finns bara som sammanräkning över beståndet, och undantaget från "aggregatet formulerar aldrig" är preciserat: varje rad bär sina egna källor, datum, status (t.ex. "endast föreslagen"), scope och tvetydighetsflagga, och listan är märkt som sammanställning. Svaga belägg stryks inte — att visa beläggningen låter läsaren avgöra. Slutligen appenderas "Relaterade begrepp" ur begreppsmodellen, begränsat till begrepp som faktiskt kan beläggas i svarets dokument.

Ordningen i `answer()` är: syntes → källvakt → rolluppräkning → korpuskontroll → predikationsanalys (skugga) → relaterade begrepp.

---

## 6. Predikationslagret

Tudelningen är strikt: **`grammar.py` vet ingenting om URD** (tar text, returnerar drag), **`predication.py`** binder drag till chunkar, fråga och svar.

### Dragen och konstruktionerna (`grammar.py`)

Stanza (sv) parsas lat med felsäker init. Före parsning: kontextprefixet strippas, parentetiska appositioner extraheras med regex på **råtexten** (de sitter i tabellceller och rubriker som aldrig når dependensparsningen), och `sentence_like_lines` filtrerar bort tabeller, listor, metatext och för korta rader. Meningar utan finit verb och protokollboilerplate hoppas över.

Dragtyper: **identitet/åtskillnad**, **agens/patiens**, **modalitet** (ska/bör/får/kan → krav/rekommendation/tillåtelse/möjlighet), samt **kvantitet** (gränsriktning kring tal) — implementerad men avstängd (2 drag av 291 i mätningen; bär inte underhållskostnaden). Varje `Feature` bär konstruktion (relation), meningskontext, styrka (asserterad/presupponerad — appositionen *förutsätter* rollen och överlever negation; svagare evidens, och den säger EN innehavare, inte DEN) och tvetydighetsflagga.

Bärande konstruktioner, samtliga formulerade som satser om UD-syntax och samtliga formade av handklassade stickprov:

- **Titelidentitet** (`titel:nmod`/`titel:appos`): titel hänger under personnamn oavsett ordföljd. Krav: huvudordet PROPN med flat:name-barn (personnamn = minst två namnled — filtrerar "Rektor" och "Diarienummer C"), ingen case-markör mellan leden (skiljer titel från "rapporten om Andersson"), ingen disjunktion ("alternativt"/"respektive" — motsatsen till identitet). Samordnade titlar ("Prefekt och HR-expert Anna Andersson") ger drag för båda med `ambiguous=True` — tvetydigheten sitter i källan och redovisas i stället för att avgöras.
- **Tillsättning** (`tillsattning:<verb>`): person (obj, nsubj:pass, eller nsubj vid mottagande verb som "få") + roll som obl/nmod med markör till/som, även under uppdragssubstantiv ("får UPPDRAG som studierektor"). Verbet ger **status** (tillsatt/föreslagen/förlängd/avslutad) och `_role_scope` fångar avgränsningen ("för", "inom", "vid").
- **Parentetisk apposition** (`parentes:`): delas på form i tre relationer — roll, organisationstillhörighet (`tillhorighet`, kända org-suffix/-markörer), förkortning (`forkortning`, versalform). Alla tre är korrekta observationer men olika slags påståenden; en rollfråga får aldrig besvaras med en arbetsplats.
- **Agens**: verb + nsubj, med tre uppmätta skärpningar (tog bort ~40 % av felen): pronominella subjekt avvisas; passiv vänds vid obl:agent eller märks som **patiens** (kan aldrig läsas som vem som handlar); expl-konstruktioner ("det finns…") avvisas; direkt objekt före obl, obl märkt med `:obl`-suffix.
- **Kopula och predikativ borttagna** efter mätning (1/14 respektive 2/5) — mät-först-principen i praktiken; funktionerna som skulle behövas vid återinförande behålls.

`normalize_title` hanterar titelfrasernas form; morfologins gräns (Stanzas lemmatiserare gissar; SALDO är den utpekade åtgärden) ger splittrade belägg, inte fel bindningar.

### Kopplingen (`predication.py`)

Skuggläge: läser, jämför, skriver debug — ändrar ingenting; undantag fångas så att analysen aldrig kan sänka en fråga. Källdrag cachas per chunk-fingerprint (max 5000). Svarets drag klassificeras mot källornas som **belagd / tvetydig / motsagd / obelagd**; motsägelse prövas före stöd, och tvetydighet tystas inte av att en läsning skulle ge stöd. Motsägelsemönstren är de bekräftade felklasserna: identitet mot åtskillnad ("TB är prefekt" mot "prefekten uppdrog åt HR-specialist TB"), omvänd agens ("Vice rektor har delegerat till rektor"), kravnivå (ska mot bör), gränsriktning. Efter tre falsklarm i första mätningen skärptes matchningen: motsägelse kräver att **båda** leden matchar på innehållsord (böjningstolerant) och att källdraget har meningskontext att visa.

Aktiveringströskeln är dokumenterad (≥90 % precision på hundrastickprov), liksom vaktens asymmetri när den aktiveras: **avstå men aldrig bekräfta** — ett uttagsfel ska kosta ett falsklarm, aldrig ett falskt godkännande, vilket också skyddar mot cirkularitet när samma parser väljer in källan och granskar svaret.

---

## 7. Attest

`attest.py` är observationslagret: SQLite (`.urd/attest.db`) med tabellerna `observations` (subject/object med ortografinormaliserade nycklar, relation, kind, construction, ambiguous, **status**, **scope**, strength, mening, källa, datum, fingerprint) och `documents` (per dokument: fingerprint, antal, byggtid).

**Byggning** (`urd attest-build`) är inkrementell per dokument: oförändrat fingerprint hoppas över, ändrat dokuments bidrag raderas och ersätts, försvunna dokument städas. Därför lagras observationer och inte färdiga frekvenser — omräkning i stället för omextraktion. Lagrade kinds: identitet, agens, patiens, modalitet, forkortning, tillhorighet. Åtskillnad lagras **inte** (kvadratisk volym, säger inget om beläggning; behövs bara som motsägelse och finns då via predication). Förkortning/tillhörighet lagras men är inte identiteter — beståndets egen tvåspråkiga ordlista (~475 relationer) respektive arbetsplatser.

**Nyckelprincipen**: `_key` normaliserar hur ett ord *skrivs*, aldrig *vilket* ord det är — gemener, bindestreck och punkter bort, mellanslag kvar. "biträdande lektor" förblir skild från "lektor"; förkortningar kopplas inte till långformer (det är parentesuttagets jobb, ur beståndets egen praxis). Källans ordalydelse bevaras alltid i subject/object.

**Uppslag**: `lookup_subject` ("vad är X?") och `lookup_object` ("vem är X?"). `_matches_terms` kräver att nyckeln är värdets **avslutande ordföljd**: "prefekt" matchar "tf prefekt" men aldrig "proprefekt" eller "prefektbeslut" (delsträngsmatchning gav tolv kandidater varav sju brus). Böjningstolerant fallback bara när exakt uppslag ger noll. Grupper vars subjekt är samma person i olika namnform slås ihop vid uppslag med `_same_person` — en **strukturell** regel, inte ett avståndsmått: första och sista namnled identiska (med böjningstolerans för genitiv), skillnad endast i mellanled. "A Lind"/"A Maria Lind" är samma person; "A Lund"/"A Lundgren" är det inte, fast redigeringsavståndet är detsamma.

**Relevansmodellen** (`compute_relevance`) ersatte den borttagna `role_is_unique` (antagandet om en innehavare per uppdrag var falskt — fyra prefekter, flera proprefekter, tiotals studierektorer, antal utan regel). Tre åtskilda komponenter så att en rangordning går att förklara:

- **Styrka**: vägt dokumentantal (tvetydiga × 0.25, endast-föreslagna × 0.5, dokument räknas — inte observationer: samma mall arton gånger i sju dokument är sju belägg) genom `w²/(w²+2)`, kalibrerad mot Joakims omdöme: 1 → 0.33, 2 → 0.67, 3 → 0.82. Aldrig noll, aldrig ett.
- **Aktualitet**: halveringstid 730 dagar mätt mot **beståndets horisont**, inte dagens datum — slutar dokumenten i mars är allt därefter okänt. Okänt datum ger 0.5: varken gynnas eller straffas.
- **Relevans** = styrka × aktualitet. Kandidater med enbart tvetydiga belägg kan aldrig rankas överst — hårt sorteringsvillkor, inte bara en vikt.

Utdata heter `documents` och `observations`, aldrig `confidence` — Attest mäter korpus, inte verkligheten.

**Inkoppling i kedjan**, fyra ställen: (1) preferens + boost i urvalet för entity_lookup, (2) uppräkning för entity_aggregation, (3) korpuskontrollen av rollbindningar, (4) CLI/REPL-uppslag (`urd attest-lookup`, `.attest`). Kända avgränsningar: `scope` lagras och visas men är inte sökbar (utpekad nästa uppgift), och tre konstruktioner (tillsättning, avgränsning, namnsammanvägning) är ännu ogranskade i stickprov.

---

## 8. Gränssnitt

**CLI** (`app/cli.py`, Typer): `urd` utan argument startar det **interaktiva läget** (`repl.py`) — en tolk där sessionen lever mellan frågor, med punktkommandon (.hjälp, .status, .ny, .källor, .debug, .attest, .stopp, .avsluta), lat modellinladdning, och automatiskt val mellan körande server (HTTP) och lokal `RagService` — samtalsminnet fungerar likadant i båda eftersom det bor i `converse`. Kommandon: `serve`, `stop` (PID-fil), `connect`, `reset-index`, `ingest`, `reindex`, `enrich`, `backfill-enrich-status`, `stats`, `config` (inkl. `validate`), `attest-build`, `attest-sample` (handklassningsunderlag), `attest-lookup`, `ask` (kontextlös engångsfråga), `test`. `StorageLockedError` fångas centralt i `main()` — inbäddad Qdrant släpper in en process i taget, och det förutsägbara felet möts med ett begripligt meddelande i stället för traceback.

**`urd test`** kör sekvensbaserade batterier ur `.urd/questions.json` (fallback `test/questions.example.json`) med fälten should_find_sources/min_sources, should_abstain, should_detect_drift, expected_intent, expected_docs (i syntesens källor), expected_docs_in_retrieval (hit@k), answer_must_contain/must_not_contain, answer_numbers_must_be_sourced — och skriver JSONL-diagnostikspår per tur för `scripts/compare_test_runs.py`.

**HTTP-API** (`api.py`): `/` (webb-UI ur `app/static/`), `/health` (konfigfilstatus + aktiv LLM-konfiguration), `/refresh` (BM25-omladdning efter ingest), `/document` (originaldokument med path-traversal-skydd), `/chat` (→ `converse`). Vid uppstart: konfigvalidering med en statusrad per YAML-fil, och en modellrad (modell, think, num_ctx, enrich-modell) — det som 2026-08-14 kostade en hel testkörning att inte se.

**Klientläget** (`connect_api.py`, `urd connect`): servar samma UI lokalt och proxar `/chat` och `/document` till upstream (via `URD_UPSTREAM_SERVER`, satt av CLI ur `--server` eller config). `/health` rapporterar både klientens och upstreams status. Ingen modell, inget index, ingen cache-mekanism ännu — utvecklingsplanens steg om UI-cache och grundläggande säkerhet (token, TLS) är inte byggda.

---

## 9. Konfiguration

**Prioritet:** miljövariabel > `.urd/config.json` > hårdkodad default. `.urd/config.json` skapas med defaults om den saknas. Booleska värden tolkas strikt (`parse_bool`, delad med `urd config set`); ogiltiga värden loggas i stället för att tyst falla — lärdomen från synonymfilen.

**YAML-filer** (repomallar i `app/`, instansens i `.urd/`, tyst-men-synlig fallback via `config_validation`):

| Fil | Styr | Konsumeras av |
|---|---|---|
| `synonyms.yaml` | Synonymgrupper (12 handskrivna) | BM25-expansion |
| `concepts.yaml` | Begrepp med labels + broader | BM25-broader-expansion, comparison-spår, relaterade begrepp |
| `question_operations.yaml` | expansion_terms per operation | BM25-tillägg (`preferred_section_terms` laddas men konsumeras aldrig — se avsnitt 10) |
| `document_types.yaml` | Sökvägs-/filnamnsregler → typ + vikt | Ingest-metadata, normkälleregeln |

**Centrala configflaggor och trösklar** (defaults): `ollama_model` (repodefault mistral-nemo; instansen kör gemma4:12b), `llm_num_ctx` 8192, `llm_think` false (uppmätt: påslaget resonemang gav 107,6 s/tur och sämre svar), `predication_enabled` (skugglagret), `attest_selection` (signal + boost + korpuskontroll + uppräkning), `attest_boost` 0.15; sannolikhetsskalans `select_min_prob` 0.5, `expansion_min_prob` 0.55, `expanded_min_prob` 0.27, `evidence_section_prob_boost` 0.15, `evidence_document_prob_boost` 0.05, `max_hits` 10, `min_desired_hits` 3; driftskyddets `qud_drift_threshold` 0.80 och `qud_drift_doc_threshold` 0.80 (provisorisk, kalibreras ur JSONL); kontextfönstren `qud_background_turns` 1, `social_history_turns` 4, `classification_history_turns` 2. Gamla logit-skalade nycklar är avsiktligt borttagna så att kvarvarande värden i äldre config.json inte tolkas som sannolikheter. Två golv är avsiktligt konstanter i koden, inte config: `_BACKFILL_MIN_PROB` 0.35 och `_SYNTHESIS_MAX_HITS`.

`config_validation.py` speglar loadernas faktiska förväntningar och rapporterar på tre ställen: `urd config validate`, startloggen, `/health` — svaret på att en trasig YAML-rad en gång stängde av hela synonymexpansionen osynligt.

---

## 10. Avvikelser mellan dokumentation och kod, och småobservationer

Där dokumentationen och koden säger olika saker gäller koden. Följande är värt att notera:

1. **README beskriver ett äldre system.** "Two-stage synthesis" anges som huvudväg (koden kör direktsyntes sedan länge) och claimslagret som "planned but not yet implemented" (avfört i white paper v2.8 till förmån för evidensobjekt, och i praktiken efterträtt av Attest). Redan noterad som åtgärd 5.3, fortfarande ogjord — och README är det första en extern läsare ser, vilket väger tyngre med kommersiellt intresse.

2. **White paper v2.8 saknar en tredjedel av systemet.** Predikationslagret, Attest, korpuskontrollen, rolluppräkningen, source_guard, dokumenttyps-/normviktsmodellen, den kontextuella fallbacken, det dokumentbaserade driftmåttet och converse-flytten finns inte i texten; "Rework-state återvinner material" står som olöst fast konsumtionsspårningen är byggd; sessionshanteringen beskrivs i API-lagret. Detta dokument är avsett som underlag för revideringen.

3. **Stale docstring i `qud_drift.py`**: "Beslutet att överrida klassificeringen fattas av api-lagret" — det fattas numera i `RagService.converse`. Kosmetiskt men vilseledande för en ny läsare.

4. **`_attest_source_paths` returnerar fel typ i felfallen**: deklarerar `tuple[dict[str, float], dict]` men returnerar `[], debug` vid ImportError och anslutningsfel. Ofarligt i praktiken (anroparen truthiness-testar), men en typkontroll skulle flagga det, och en framtida anropare som gör `.get` på returvärdet i felfallet kraschar.

5. **`config_validation.KNOWN_OPERATIONS` saknar `entity_lookup` och `entity_aggregation`** som finns i `intent._VALID_OPERATIONS` och i regellagret. Endast varningsnivå — men det är exakt den sortens listsynk som kommentaren i `intent.py` varnar för, efter att en osynkad lista en gång tyst avaktiverade attestsignalen.

6. **`preferred_section_terms` laddas och valideras men konsumeras aldrig** (åtgärd 5.4, kvarstår). `relation_pair_ids` är däremot borta — den delen är städad. `settings.chunk_size`/`chunk_overlap` är numera kopplade till `chunk_text` (gamla åtgärd 5.1-anmärkningen är delvis åtgärdad; `top_k` är borttagen).

7. **`SessionStore` evictar fortfarande aldrig** (fas 4 punkt 15) — minnesväxt i långkörande server, förvärrad av att `active_hits` numera bär hela chunktexter per session.

8. **Repodefaulten `ollama_model: mistral-nemo`** släpar efter instansens modellbyte till gemma4:12b. Medvetet eller inte — en ny instans får Nemo, vars språkbrus är dokumenterat i åtgärdsplanen.

9. **Boilerplate-städning**: `._.DS_Store` ligger fortfarande i repots rot (noterad redan i kodanalysen 2026-08-11).

10. **Evidensbackfill-manifestet** saknas fortfarande: 53 dokument utan evidensobjekt konverteras om vid varje ingest.

Styrkor värda att bevara, oförändrade sedan kodanalysen och snarast förstärkta: debug-spårbarheten genom hela kedjan (varje steg lämnar räknare, kandidatlistor och timing i debug/JSONL, även vid abstain), den principfasta abstain-designen med den nya återhämtningsvägen, den konsekventa sannolikhetsskalan, och den strikta lagerseparationen grammar → predication → attest → retrieval som gör varje del mätbar för sig.
