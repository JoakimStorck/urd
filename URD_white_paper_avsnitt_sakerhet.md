# Säkerhet och behörighet

*Avsnitt avsett att infogas i white paper vid revideringen till v3.0. Skrivet i dokumentets egen stil, för den som ska arbeta i koden.*

---

URD:s säkerhetsläge bestäms av två egenskaper som drar åt olika håll. Systemet kör helt lokalt — modeller, index och dokument lever hos organisationen, och ingen del av frågeflödet passerar organisationens gräns. Samtidigt är dokumentbeståndet i sig känsligt: styrdokument, protokoll och beslut, inklusive personalärenden. En assistent som besvarar frågor om beståndet är i praktiken en läsbehörighet till hela beståndet, och den insikten är utgångspunkten för allt som följer.

Systemet utför inga åtgärder. Det har inga verktyg, kan inte skriva, skicka eller ändra något. Konsekvensen av en lyckad attack är informationsläckage, inte förändring. Det begränsar skadan men gör den inte mindre allvarlig.

## Skyddet vilar idag på en enda egenskap

Både server och klient binder loopback som standard, och så länge de gör det är hotbilden teoretisk. Filutlämningen är skyddad mot sökvägsmanipulation genom uppslag av den absoluta sökvägen före kontroll mot dokumentkatalogen, vilket fångar både relativa mönster och symboliska länkar. Sessionsidentiteter genereras som UUID4 och är inte gissningsbara. Vektorlagret körs inbäddat och lyssnar inte på nätverk.

Vad som däremot saknas är åtkomstkontroll i varje form: ingen autentisering, ingen kryptering, ingen anslutningsloggning, och ingen koppling mellan sessionstillstånd och användare. Bristerna är ofarliga under loopback och verkliga i samma stund som klientläget används av någon annan — och klientläget förutsätter per definition att servern lyssnar på nätverket.

Det gör bindningsadressen till systemets enskilt viktigaste säkerhetsparameter, och den enda realistiska katastrofen är att den ändras utan att någon tänker på följden. Därför varnar startloggen och hälsokontrollen när bindningen inte är loopback och autentisering saknas.

## Behörighet är retrieval, inte presentation

Den bärande principen för behörighetsmodellen är att **ett svar inte går att censurera i efterhand**. När syntesen väl läst en passage är innehållet ute; även en sammanfattning läcker. Filtrering måste därför ske innan någon passage når vare sig rangordning eller syntes, och det betyder att behörighet hör hemma i sökkedjan och inte i något lager ovanpå den.

Varje textstycke bär därför en behörighetsuppgift i sin metadata, satt vid synkronisering, och sökningen filtrerar på användarens grupptillhörighet före cross-encoderns bedömning.

**Aggregatlagret är den svåra delen och den lätt förbisedda.** Attest räknar samman uppgifter över hela beståndet — hur många dokument som binder en person till en roll, vilka roller som finns, vilka dokument som saknar belägg. Ett aggregat som räknar dokument användaren inte får se läcker deras existens och ibland deras innehåll: "beståndet binder X till rollen i tre dokument, senast i maj" är en uppgift även när dokumenten är stängda. Attestuppslag måste därför filtreras på samma grupptillhörighet som retrieval, vilket kräver att observationernas källsökväg kan knytas till en behörighet.

## Mappstrukturen approximerar, filen avgör

SharePoint ärver behörigheter nedåt genom bibliotek och mappar, och att spegla mappstrukturen i den lokala dokumentytan är därför rätt utgångspunkt. Speglingen är värdefull även oberoende av behörigheter: sökvägen är redan systemets källa till dokumenttyp och normativ tyngd via `document_types.yaml`, och en struktur som följer källan gör den härledningen stabil när nya dokument tillkommer.

Men arvet kan brytas per fil, och delningslänkar skapar behörigheter som inte hör till någon mapp alls. Mappstrukturen är alltså en god approximation som **felar åt fel håll**: ett dokument med snävare behörighet hamnar i en mapp systemet tror är öppen. Den faktiska behörigheten läses därför per fil vid synkronisering, och avviker den från mappens gäller filens — avvikelsen loggas, eftersom den betyder att någon medvetet avgränsat dokumentet.

Går behörigheten inte att läsa synkas filen inte alls. Ett dokument utanför beståndet är ett känt bortfall; ett dokument med gissad behörighet är en tyst läcka. Det är samma asymmetri som styr resten av systemet, tillämpad på synkroniseringen.

Modellen är medvetet grovkornig: ett fåtal grupper som täcker verkliga fall, inte en spegling av källans fulla arvsstruktur. En korrekt spegling vore obegriplig att granska, och obegripliga behörigheter blir felkonfigurerade.

## Ett undantag från systemets epistemik

Genomgående i URD gäller att en frånvarande uppgift är okänd och inte en motsägelse. Attest väger inte ned en bindning för att avgränsningen saknas, korpuskontrollen rapporterar inte frånvaro som fynd, och syntesen abstainar hellre än gissar.

**Behörighet är undantaget.** Ett dokument utan behörighetsuppgift betraktas som stängt. Asymmetrin går åt andra hållet här eftersom kostnaden för fel är exponering och inte utebliven träff, och undantaget ska vara uttryckligen kommenterat där det implementeras — annars ser det ut som en inkonsekvens och städas bort av någon som har systemets övriga mönster i huvudet.

## Basnivån måste vara strukturellt komplett

Behörighetsnivåer kommer att efterfrågas, men de behöver inte byggas färdiga i första omgången. Vad som däremot måste finnas från början är tre strukturella egenskaper, eftersom de inte går att lägga till i efterhand utan att röra hela kedjan.

Behörighetsfältet ska finnas i indexet även när värdet är detsamma för allt: att införa fältet senare kräver omindexering av hela beståndet, medan ett bättre värde i ett befintligt fält bara kräver en ny synkronisering. Filtreringen ska ligga i sökkedjan även när villkoret släpper igenom allt, eftersom det är strukturen som är svår att flytta. Och autentiseringen ska producera ett **subjekt** med grupptillhörighet, inte bara en delad nyckel — en nyckel utan identitet måste ersättas när behörighet införs, medan en identitetsbärande nyckel kan utvidgas.

Med de tre på plats blir varje förfining additiv: filnivå som avviker från mappen blir ett annat värde i samma fält, fler grupper blir fler värden i samma lista, och aggregatfiltrering blir samma villkor applicerat på källsökvägen.

Behörighetsgrupperna hör hemma i YAML, på samma plats och av samma skäl som dokumenttypsreglerna: konfiguration framför kod.

## Indirekt promptinjektion

Dokumenten är indata till en språkmodell. Den som kan placera en fil i den synkade dokumentytan kan skriva text som riktar sig till modellen snarare än till läsaren. Risken är i dag liten eftersom skrivbehörigheten till dokumentytan är snäv, och den växer med SharePoint-integrationen.

Skadan är begränsad av att systemet inte kan utföra något — utfallet är vilseledande text, inte en åtgärd. Men för en assistent vars enda syfte är att svara korrekt om styrdokument är vilseledande text den värsta tänkbara skadan.

Det befintliga skyddet är arkitektoniskt och betydande: varje påstående bär källhänvisning, originaldokumentet kan öppnas direkt ur svaret, och verifierbarhet är ett av systemets fyra huvudmål snarare än ett tillägg. En manipulerad uppgift är därmed spårbar till sitt dokument. Ytterligare åtgärder bör vänta tills problemet är uppmätt — samma princip som styr allt annat utvecklingsarbete i systemet.

## Diagnostiken hör innanför åtkomstkontrollen

Spårnings- och inspektionsfunktionerna exponerar indexets interna struktur, inklusive chunktexter och observationslagrets innehåll. De är avsedda för utveckling och felsökning, men de går genom samma server som besvarar frågor.

När åtkomstkontroll införs ska de omfattas från början, inte läggas till efteråt. Och när behörighetsfiltrering finns måste den gälla även dem — en diagnostikvy som visar chunkar användaren inte får läsa är en läcka oavsett hur teknisk den ser ut.
