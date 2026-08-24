# White paper: lokal AI-baserad dokumentassistent för interna styrdokument

## Version 3.0 — deliberation som arkitektur

---

## Bakgrund och syfte

URD (Unified Retrieval and Deliberation) är en lokalt driven dokumentassistent
för interna styrdokument. Syftet är att göra ett levande bestånd av regler,
rutiner, protokoll och beslutsordningar användbart för dem som faktiskt
arbetar med dokumenten: administratörer, lärare och andra kunskapsarbetare som
ställer frågor utifrån arbetsuppgifter snarare än utifrån dokumentstrukturen.

Version 2.8 beskrev ett system vars retrievalkedja fungerade och vars
samtalsstyrning bar grundflödet i en dialog. Sedan dess har tre saker hänt som
motiverar en ny version. Systemet har fått ett beståndsobservationslager —
Attest — som mekaniskt registrerar vad dokumenten betygar om roller, personer
och begrepp. Systemet har fått en instans- och säkerhetsmodell med
användarkonton, lösenord och sessioner. Och framför allt: en serie mätningar
har visat exakt var gränsen går mellan det systemet har och det namnet lovar.
URD har retrieval. Deliberation — det som ska göra hämtat material till ett
avvägt svar — finns i spridda delar men inte som arkitektur.

Detta dokument beskriver nuläget med den ärlighet projektet kräver av sig
självt, och beskriver deliberationslagret som design: principer, mekanism och
införandeväg, med en uttrycklig deklaration av vad som är byggt, vad som är
halvbyggt och vad som ännu bara är formulerat.

---

## Mål

URD har fyra sammanhängande mål.

**Begriplighet för användaren.** Frågor ska kunna ställas i användarens eget
arbetsspråk, utifrån situationer i verksamheten, utan kännedom om
dokumentbeståndets struktur eller terminologi.

**Korrekthet i svaret.** Svar ska bygga på vad dokumenten faktiskt säger.
Modellen får inte fabricera, inte blanda ihop roller, inte vända logik.
Osäkerhet ska kommuniceras hellre än döljas, och "jag hittar inget stöd" är
ett fullgott svar när källstöd saknas. Korrektheten är asymmetrisk: ett
felaktigt avstående kostar en omfråga, ett felaktigt påstående kostar
förtroende och i värsta fall ett felaktigt beslut i verksamheten.

**Aktualitet.** Beståndet är levande. Systemet ska hantera förändring utan
att gamla versioner blandas ihop med aktuella — och utan att historien
raderas: en fråga om vad som gällde förr ska kunna besvaras.

**Verifierbarhet.** Varje svar ska kunna knytas till sitt underlag. Med
version 3.0 utvidgas målet: inte bara källorna utan systemets egna
avgöranden ska lämna spår som går att granska i efterhand. Ett system som
väger och väljer måste kunna visa hur det vägde och varför det valde.

Två bivillkor ligger under målen. **Lokal drift** är ett absolut krav:
dokument och modeller lever hos organisationen, inte hos externa
leverantörer. **Öppen teknikstack** prioriteras: byggstenarna ska vara
inspekterbara och utbytbara.

---

## Systemets form

URD består i dag av sex samverkande delar:

1. dokumentingest och indexering,
2. hybrid retrieval med dokumentintern evidensläsning,
3. Attest — beståndsobservation med grammatisk utvinning,
4. QUD-styrd samtalsstyrning med separata svarsvägar,
5. syntes med mekaniska vakter,
6. instansmodell med autentisering och klientläge.

Delarna fungerar. Det version 3.0 handlar om är det som saknas mellan dem:
delarna producerar var och en kunskap om vad som får sägas — men de möts
aldrig. Deliberationslagret, beskrivet nedan, är inte en sjunde komponent
utan platsen där de sex befintliga konfronteras med varandra.

---

## Dokumentingest och representation

URD läser dokument från disk, extraherar text, delar den i sektioner och
chunkar med meningsmedveten delning, och berikar chunkarna med dokumenttitel,
sektionsrubrik och metadata. Textchunkar indexeras med kontextuellt prefix
och med de instruktionsprefix embeddingmodellen är tränad för.

**Dokumentdatum och diarienummer** extraheras ur dokumenthuvudet, med
filnamnskonventionen som reserv — innehåll före filnamn, eftersom datalagret
ska kunna flytta från filer till databas. Datum exponeras i källformatering
och syntesprompt, med regeln att nyare källa har företräde vid motstridiga
uppgifter och att båda datumen då redovisas.

**Innehållshash.** Varje dokument bär en hash över sitt sorterade
chunkinnehåll, skild från den sökvägsberoende fingerprinten. Mätning mot
beståndet visade att ungefär tio procent av dokumenten ligger i flera kopior
under olika sökvägar — 23 dokument i 50 exemplar. Kopiorna städas inte:
sökvägen bär information om dokumenttyp och normativ ställning, och kommer
att bära behörighet. Men beläggning räknas per innehåll, aldrig per sökväg.
En kopia är inte ett ytterligare belägg.

**Evidensobjekt** — figurer, tabeller, punktlistor, numrerade listor —
extraheras som egna informationsenheter med stödtext och referenser, lagras i
egen kollektion och rerankas dokumentinternt efter textretrieval, enligt
principen "evidensobjekt först, stödtext därefter, direkt text om inget
tydligt objekt bär frågan".

---

## Hybrid retrieval

Retrievalkedjan består av semantisk sökning, BM25 och flerspråkig
cross-encoder-reranking, med dokumentexpansion för dokument där någon chunk
rankat högt. Sedan version 2.8 har kedjan fått:

**Poäng på sannolikhetsskala.** Cross-encoderns logits normaliseras med
sigmoid före allt urval. Boostar verkar i sannolikhetsrummet med tak, och
urvalsreglerna är tolkningsbara golv i stället för kvoter på en skala där
kvoter saknar mening.

**Operationsstyrt urval.** Frågans operation styr hur brett underlag
syntesen får: aggregering och jämförelse får fler och bredare källor än
direkta uppslag.

**Dokumentbaserad driftkontroll.** Ämnesdrift mäts fråga-mot-passage mot de
aktiva dokumentens innehåll, efter att kalibrering visade att
fråga-mot-fråga-likhet inte kan skilja delad institutionell vokabulär från
äkta trådsamhörighet. Elliptiska följdfrågor som abstainar får en andra
chans med QUD-kontext.

**Synonymexpansion** är aktiv, med konfigvalidering som gör tyst
felkonfiguration omöjlig — en trasig konfigfil rapporteras vid start och i
hälsokontrollen i stället för att tyst stänga av funktionen.

En känd och uppmätt gräns kvarstår: retrievalen är **årsblind**. Ett årtal i
frågan är i praktiken ett ord bland andra; en fråga om ett visst år hämtar
dokument från andra år med hög poäng. Detta behandlas under
deliberationslagret som en fråga om tillåtlighet, inte om viktning.

---

## Attest: beståndsobservation

Attest är ett SQLite-baserat lager som vid byggtillfället läser hela
beståndet och registrerar grammatiska observationer: vem som betecknas med
vilken titel, vilka konstruktioner som binder namn till roller, vilka termer
som förklaras eller förkortas, vem som utses till vad. Utvinningen sker med
dependensparsning för löpande text och med textnära konstruktioner för det
som aldrig når en parser — tabellceller, rubriker, underskriftsblock.

Tre egenskaper definierar lagret.

**Observationerna är typade.** En bindning genom ett underskriftsblock, en
titel i apposition, ett tillsättningsverb och ett namn som råkar stå nära
ett rollord är olika konstruktioner och lagras som sådana. Skälens art
bevaras — det är detta som gör en senare bedömning möjlig utan att allt
reducerats till en siffra.

**Beläggning är inte sanning.** Attest mäter hur ofta något skrivits i
beståndet, viktat med aktualitet — inte om det stämmer. Lagret får styra
vart systemet tittar och vilka passager som reserveras för syntesen; det
får aldrig självt bära ett påstående i ett svar. Aggregatet pekar,
originaltexten bär.

**Klasserna är epistemiskt åtskilda.** En mätning över beståndets
parentetiska appositioner visade att den klass som kallades identitet till
sex sjundedelar bestod av något annat: par av namn på samma sak. Beståndet
visade sig innehålla en tvåspråkig termordlista — närmare 150
översättningspar och drygt 120 förkortningar, dokumentens egen variation
mellan svenska och engelska förvaltningstermer — som låg felklassad som
personbindningar och därmed kandiderade på "vem är"-frågor. Efter
omklassningen är termekvivalens en egen klass: ett uppslag på vem som
innehar en roll kan aldrig returnera ordlistan, och ordlistan är samtidigt
åtkomlig för det den faktiskt är — den synonymväg som version 2.8 pekade ut
som "dokumentens egen variation", nu realiserad som data.

Arbetet med omklassningen etablerade också en metod: mät fördelningen innan
regeln skrivs, beskriv felets form och inte dess instanser, och deklarera
regelns kända gränser i koden. Det ortografiska namnpredikat som skiljer
personnamn från termer är dokumenterat användbart för att utesluta namn,
inte för att fastställa dem — en främmandespråkig titel utan igenkännbart
huvudord passerar, och det står skrivet där regeln står.

---

## Syntes och vakter

Huvudvägen är direkt syntes från källor med detaljbevarande prompt: listor
återges i sin helhet, belopp och roller bevaras ordagrant, svaret börjar
med det mest specifika, varje påstående bär källhänvisning, svaret ges på
svenska även när källorna är engelska. Normkälleregeln — normkällor väger
tyngre än historiska protokolluppgifter — och aktualitetsregeln ligger i
prompten.

Efter syntesen verkar mekaniska vakter utan LLM-anrop. **Källvakten**
verifierar att varje tal i svaret förekommer i det underlag som skickades
till syntesen och att källreferenser pekar på faktiskt skickade källor.
**Attest-korpuskontrollen** prövar rollpåståenden mot beståndets
observationer. **Upprepningsklippning** stryker stycken som återger
föregående svar. Utfallet är i dag flaggor och loggning — det trappade
skarpa läget, där ett underkänt svar ersätts, hör till deliberationslagret.

Rework-vägarna elaboration och verification kvarstår som i version 2.8, med
källvakt även på elaborationsvägen och med skärpt verification-extraktion
efter att mätning visade att granskningen kunde fabricera sina egna
granskningsobjekt.

---

## QUD-styrd samtalsstyrning

Samtalsmodellen är oförändrad i sin struktur sedan version 2.8: en aktiv
huvudfråga, klassificering av varje yttring i samtalsroller, separata
svarsvägar, driftskydd. Broadening söker numera globalt med de aktiva
dokumenten som kompletteringspool i stället för som hårt filter, och
omskrivningen av följdfrågor är sanerad och vokabulärvaliderad så att
modellens egna formuleringar aldrig kan styra retrieval.

De kända klassificeringsgränserna kvarstår och är uppmätta: enstaka
intentmissar på gränsen mellan elaboration, relaterad fråga och ny
huvudfråga. Svarskvaliteten är numera god även när etiketten blir fel,
vilket flyttar frågan från kvalitetsproblem till latens- och
exakthetsarbete.

---

## Instans, autentisering och säkerhet

URD:s säkerhetsmodell beskrivs i ett eget kapitel i repot; här återges
formen. Beståndet innehåller personalärenden och individbeslut, och ett
verktyg som *svarar på frågor* över ett sådant bestånd är känsligare än
katalogen det läser — sammanställningen är känsligare än delarna.

Autentiseringen bygger på en användarfil med grupper, atomära skrivningar
och omedelbar omläsning: en återkallelse slår igenom utan omstart, och en
trasig fil betyder avslag för alla — fail closed. Människor och program
skiljs åt: en människa får en **inbjudan** som växlas in en gång mot ett
självvalt lösenord (minneshård härledning, strypning av gissningsförsök,
identiska svar och identisk svarstid för okänt namn och fel lösenord);
program får långlivade maskinkonton. Inloggning ger en session med dubbla
utgångar — absolut och vid overksamhet — som lever i serverns minne och dör
med den. Lösenordsvägen vägrar mekaniskt över okrypterad förbindelse som
inte är loopback. Klienten visar vem servern anser att man är, och en
session som skapats i det interaktiva läget avslutas när läget avslutas.

Det strukturella steg som återstår är **behörighet i indexet**: ett
behörighetsfält per dokument och ett filter i retrievalkedjan, så att
åtkomst avgörs innan material når rankningen. Frånvarande behörighet
betyder stängt — det medvetna undantaget från systemets epistemologi i
övrigt, där frånvaro av belägg aldrig är ett belägg. Attest-aggregatet
måste filtreras på samma villkor, annars läcker beståndsobservationen det
retrievalfiltret skyddar.

---

## Deliberation som arkitektur

### Ordet

Engelskans *deliberation* rymmer fyra saker som svenskan delar upp:
**samråd** — flera röster som hörs; **avvägning** — skäl som vägs mot
varandra; **bedömning** — en prövning som ser till skälens art och inte
bara deras mått; **vägval** — att något faktiskt avgörs. URD behåller den
engelska termen och deklarerar uppdelningen, eftersom uppdelningen är ett
mått: systemet har i dag det första, delar av det andra, och nästan inget
av de två sista.

### Utgångsläget, uppmätt

Systemet äger redan sju oberoende kunskapskällor om vad som får sägas:
frågans QUD och intent, frågans operation, Attests observationer,
dokumentdatum med normkälleregeln, reservationskanalen, vakternas
efterkontroll, och avståendet. Mätningarna visar att de aldrig möts.
Klassificeraren vet att en fråga efterfrågar en innehavare — syntesen
förpliktas aldrig av det, och frågan besvaras med en rollbeskrivning ur
ett normdokument som rankat högt. Rätt namn på den preciserade frågan bärs
av underskriftsblock som vinner rerankingen med tredje decimalen, och
vilket block som vinner växlar mellan körningar: systemet vet inte, det
läser sig till det, och det läser olika. Ett svar band rätt person till
rollen med motiveringen att personen *redogör för ärenden*. Gemensamt:
systemet har ingen representation av vad det påstår. Det producerar text
och kontrollerar ytan med strängoperationer — och samma brist finns i
mätningen själv, där en negativ assertion missade sitt eget fall när
modellen skrev ett pronomen i stället för namnet.

### Principerna

Deliberationens principer finns i ett eget dokument; här återges de i
sammandrag, i tre grupper.

**Vad en källa kan bära.** Källslaget bestämmer vad källan kan svara på:
ett normdokument säger vad som ska gälla, ett beslut vad som beslutats,
ett protokoll vad som sades, en utlysning att en plats var obesatt vid ett
datum — och ingen rankning kan göra det ena till det andra. En källa som
inte kan säga något om frågan röstar inte; tystnad bär ingen information,
med behörighet som deklarerat undantag. Struktur väger tyngre än närhet:
ett underskriftsblock binder ett namn till en roll genom dokumentets form,
ett namn intill ett rollord binder ingenting. Aggregatet pekar,
originaltexten bär.

**Hur källor väger mot varandra.** Tre nivåer av styrka: duplicering —
samma innehåll under flera sökvägar — räknas noll; upprepning — samma
bindning i många oberoende dokument över tid — är persistens och verklig
evidens; samstämmighet *över källslag* är starkast, eftersom ett beslut,
ett protokoll och ett underskriftsblock kan fela oberoende av varandra.
Normkälla går före redogörelse i sak. Aktualitet är två skilda mekanismer:
för händelsekällor en färskhetsskala, för normkällor ett
giltighetstillstånd — en arbetsordning är inte "lite gammal", den gäller
tills den ersätts. Aktualitet rangordnar och raderar inte: företrädaren
förblir svarbar som historia. Och bedömning går före tröskel: skillnaden
mellan två närliggande poäng får inte avgöra det skälens art kan avgöra.

**Hur avvägningen mynnar i ett vägval.** Frågans ram — tid, organisation,
omfång — avgör vilka belägg som alls är tillåtliga, innan något vägs;
belägg utanför ramen når inte vågen. Operationens löfte avgör vad som
räcker: att retrieval lyckades är inget belägg för att frågan kan
besvaras. Motsägelse i beståndet är ett fynd och redovisas; systemets egen
osäkerhet är det inte — ett svar som räknar upp vad som fanns och överlåter
slutsatsen har inte avvägt, det har delegerat avvägningen till den som
frågade. Grunden bärs i satsen, inte bredvid den: "undertecknar
institutionens beslut som prefekt, senast vid ett angivet datum" är ett
vägval; en osäkerhetsrapport intill ett svar är det inte. Och avståendet
är självt ett vägval — systemet har vägt och funnit grunden otillräcklig.

### Vägvalets slutna mängd

En deliberation som kan mynna i vad som helst är ingen deliberation.
Utfallen är därför få, namngivna och **slutna per operation**: operationen
definierar inte bara vad som räcker utan vilka utfall som finns. För en
fråga om vem som fyller en plats: *namnger* — *namnger med förbehåll*, där
slutledningsregeln redovisas i satsen — *beskriver men namnger inte* —
*motstridigt* — *avstår*. För en fråga om vad som gäller: *återger regeln*
— *regel saknas* — *motstridiga regler* — *avstår*. Slutenheten är det som
gör lagret prövbart: varje svar ska gå att hänföra till ett utfall, och
testbatteriet kan mäta om det stämmer.

### Mekanismen: ett åtagande genom kedjan

Deliberationslagret är inte en ny komponent instucken i kedjan utan ett
objekt som löper genom den: ett **åtagande** som öppnas när frågan förstås
och stängs när svaret prövats. Mekanismen är en monoton avsmalning av
frihet — varje steg tar bort möjligheter, inget steg nedströms återinför
dem.

**Inramning.** Frågan ger operation, ram och därmed utfallsmängd.
Klassificeraren finns; ramextraktionen är deterministisk — årtal,
organisationsnamn ur begreppsmodellen.

**Insamling.** Retrieval och Attest som i dag, med två skillnader: ramen
verkar som tillåtlighetsfilter, inte rankningssignal, och varje belägg bär
sitt skäls art framåt — källslag, konstruktion, datum — i stället för att
reduceras till en poäng.

**Överläggning.** Källorna möts i en uppslagning: en beslutstabell över
operation, tillgängliga källslag, konstruktioner och rammatchning som ger
en utfallsklass och de belägg som bär den. Ingen modell. Tabellen ligger i
konfiguration bredvid operationerna, granskningsbar och redigerbar. Poäng
skiljer likar åt inom en klass men väljer aldrig klass.

**Författande.** Utfallsklassen avgör vem som skriver. Namngivande utfall
författas av modellen, bunden till de bärande beläggen och med kravet att
grunden bärs i satsen. Icke-namngivande utfall — beskriver-men-namnger-
inte, motstridigt, avstår — författas av systemet ur metadata, som dagens
abstain-mall. Högst ett modellanrop, ibland noll: modellens frihet är en
funktion av beläggens styrka.

**Prövning.** Svaret parsas med samma grammatik som beståndet, och dess
bindningspåståenden jämförs med åtagandet: påstår svaret det som
beslutades, och inget mer? En rollbindning med pronomensubjekt är
overifierbar och därmed flaggbar — det fångar den felklass där bindningen
glider förbi strängkontroller genom omskrivning. Vid underkänt kan svaret
ersättas med det systemförfattade utfallet: vakternas trappade skarpa
läge.

Åtagandet loggas per tur i diagnostikspåret. Därmed gäller
verifierbarhetsmålet till slut även systemets egna avgöranden: frihet in,
ansvar ut, och varje steg lämnar ett läsbart spår.

### Provet på att lagret är verkligt

En överläggning som aldrig kan komma fram till något annat än vad modellen
ändå hade skrivit är dekoration. Kriteriet: kan lagret producera ett
utfall som skiljer sig från generationens? I dag kan bara avståendet och
vakternas flaggor det. Med författarskapsregeln blir svaret ja på mekanisk
väg — ett flytande substitutionssvar kan bytas mot en systemförfattad
mening om att beståndet beskriver rollen men inte namnger innehavaren.

### Införandeväg

Lagret införs under mätning, inte genom omläggning. Överläggning,
författande och prövning körs först **tyst**: beslutstabellen avgör utfall
per fråga, loggen jämför med vad systemet faktiskt svarade, och
divergensen mäts över testbatteriet. Divergens noll betyder att lagret är
dekoration. Hög divergens på fel ställen betyder felkalibrerad tabell. Hög
divergens på rätt ställen — substitutionssvaren — är beviset för att slå
på det, utfallsklass för utfallsklass. Samma disciplin som gällt allt
annat: mätning före makt.

### Självdeklaration

**Finns:** avvägningens kvantitativa del — styrka gånger aktualitet,
sigmoidnormaliserade poäng, normkälleregeln, dokumentdatum — samt två
verkliga utfall: avståendet och vakternas flaggor.

**Finns delvis:** källslag sätts deterministiskt men konsumeras inte av
någon avvägning; reservationskanalen reserverar men är frågeokänslig;
aktualitet redovisas men styr varken retrieval eller urval; operationen
styr retrievalbredd men förpliktar inte svaret.

**Finns inte:** platsen där källorna möts, representationen som håller ett
påstående med sitt skäl, utfallsmängden, författarskapsregeln, prövningen
av det egna svaret. Deliberationslagret är i version 3.0 en design med
uppmätt motivering — inte en implementerad komponent.

---

## Vad som fungerar

Grundflödet i dialogen: öppnande fråga, elaboration inom aktiva dokument,
broadening till näraliggande och till nya dokument, sociala markörer utan
statsförstöring, ärligt avstående när materialet är uttömt. Retrieval
hittar konsekvent rätt dokument när frågan delar terminologi med beståndet;
synonymexpansionen överbryggar de kartlagda varianterna; svaren kommer på
svenska även ur engelska källor; belopp och listor återges exakt och
källvakten går grön över batteriet. Attest-hygienen är uppmätt ren:
beläggning räknas per innehåll, termekvivalenser kandiderar inte på
personfrågor, och beståndets tvåspråkiga ordlista är åtkomlig som data.
Autentiseringskedjan är i drift: inbjudan, inväxling, lat inloggning,
identitet i prompten, automatisk utloggning. Testbatteriet om arton
sekvenser går med samma fyra kända avvikelser genom fjorton på varandra
följande patchar — mätbarheten håller.

---

## Vad som ännu inte fungerar

Redovisningen följer projektets regel att misslyckas tydligt.

**Substitutionssvaret.** En fråga om vem som fyller en plats besvaras med
en beskrivning av platsen, utan markering och utan abstain. Klassificeraren
vet vad frågan är; ingenting förpliktar svaret. Detta är
deliberationslagrets tydligaste uppmätta motivering.

**Rätt svar av fel skäl.** Den preciserade rollfrågan får rätt namn — buret
av underskriftsblock som vinner rerankingen med tredje decimalen, växlande
mellan körningar. Svaret är korrekt men grunden är närhet, inte struktur.

**Årsblind retrieval.** Tidsramen i en fråga påverkar inte vilka belägg som
hämtas. Ramprincipen är formulerad; ingenting upprätthåller den.

**Reservationskanalen är frågeokänslig och dess golv felkalibrerat.**
Kanalen reserverar samma passager oavsett vad frågan efterfrågar, och
golvet sattes mot styrkor som var uppblåsta innan beläggning räknades per
innehåll — legitima kandidater ligger nu under det.

**Mätningen har ett eget mätfel.** Negativa assertioner prövar strängar,
inte påståenden: ett svar band en person till en roll via pronomen och
passerade sin förbudslista. Ett testfall mäter ingenting eftersom det
förväntade namnet inte finns i beståndet — uppdraget är vakant — och ska
skrivas om till en vakansassertion. Batteriet behöver entitetsbindande
kontroller, vilket är samma grammatikprövning som deliberationens sista
steg.

**Övrigt känt:** intern upprepning inom ett och samma svar; halvöversatta
rolltermer ur engelska källor; enstaka intentmissar på kända gränser; en
böjningstolerant uppslagsfallback vars genomsläpp inte är uppmätt.

---

## Framåtsyftande arbete

**Deliberationslagret enligt införandevägen** — åtagandeobjekt, tyst
beslutstabell, divergensmätning, skarpt läge per utfallsklass. Detta är
huvudlinjen, och den binder samman tre arbeten som annars skulle byggas som
parallella halvlösningar: svarskontraktet, den tidsmedvetna svarsbildningen
och ramfiltret i retrieval är tre konsumenter av samma åtagande.

**Underskriftskonstruktionen.** Strukturell rollhärledning ur
underskriftsblock — den starkaste källan för innehavarfrågor — byggd med
strukturmetoden: skriptet föreslår ur formen, människan godkänner,
resultatet blir konfiguration. Rollvokabulären hämtas inte ur Attest;
aggregatet får inte definiera det som ska rätta aggregatet.

**Svarskontraktets entitetsbredd.** Kontraktet ska pröva svarets art mot
operationens löfte per entitetsslag — person, organ, dokument, ämne — och
frågan om operationsmodellens indelning hålls öppen tills den avgörs i
modellen, inte i koden.

**Behörighetsfältet** i index och retrievalkedja, före övrig utbyggnad som
kräver omindexering. **Anslutningsloggning** för frågevägarna och
**TLS-certifikat** hör till samma driftsättningsfas, liksom giltighetstid
på konton.

**Mätarbeten:** omkalibrering av reservationsgolvet mot den korrigerade
styrkefördelningen; mätning av uppslagsfallbacken; vakansassertion och
entitetsbindande negativfall i batteriet; versionsdetektering av normkällor
via innehållshash och kategori, som förutsättning för
giltighetsbedömningen.

**Avgränsade förbättringar** kvarstår från version 2.8 där de ännu inte
byggts: dokumentkommandofamiljen ovanpå ett gemensamt statuslager,
kontrollerad klientuppdatering, synkadapter mot dokumentkällan som en
isolerad och raderbar modul, samt modellutvärdering på extern
beräkningsnod med kvantisering som första isolerad variabel.

---

## Vad som inte är huvudlinje

**Tvåstegssyntes i huvudvägen** förblir avvisad, nu med skarpare skäl:
deliberation är källorna som möts, inte modellen som hörs två gånger. Två
generationssteg är samma röst i följd och tillför ingen oberoende kunskap.
Deliberationslagret behåller högst ett modellanrop per svar, ibland noll.

**Kunskapsgraf** förblir experiment, inte huvudlinje. Attest är den
medvetna mellanformen: typade observationer utan anspråk på att vara en
sanningsmodell.

**Claimslager** som separat indexerad entitet förblir ersatt — av
evidensobjekten för struktur och av Attest för bindningar.

**Finjustering** förblir komplement för språkdräkt i ett senare skede,
aldrig faktabärare. Modellbyte utvärderas mot batteriet som en configrad,
inte som arkitektur.

---

## Sammanfattning

URD är en QUD-styrd lokal dokumentassistent med hybrid retrieval, ett typat
beståndsobservationslager, direktsyntes med mekaniska vakter, en
instansmodell med fullständig autentisering — och en uppmätt lucka där
namnets andra hälft ska sitta. Version 3.0:s bidrag är att luckan nu har en
form: principer som går att falsifiera, en sluten utfallsmängd per
operation, ett åtagandeobjekt genom kedjan med författarskap efter
beläggens styrka, och en införandeväg som ger lagret makt först när dess
avgöranden uppmätt skiljer sig från generationens på rätt ställen.

Systemets centrala fråga har därmed flyttat ett steg till. Version 2.8
konstaterade att den flyttat från att hitta rätt dokument till att läsa
rätt information i rätt dokument. Version 3.0 flyttar den från att läsa
till att avgöra: att det som lästs vägs, att vägningen mynnar i ett
vägval, och att vägvalet syns — i satsen, i loggen och i mätningen.
