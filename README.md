# Projekt z predmetu Inteligentná analýza údajov

Projekt bol zameraný na čistenie a predspracovanie dát a vytvorenie modelu schopného predikovať stav Chronickej Lymfatickej Leukémie (CLL) u pacientov, na základe výsledkov krvných testov.

Oficiálny text zadania (Nguyen).

Úvod

Cieľom projektu je osvojiť si prehľad fungovania v dátovej vede, základné koncepty a techniky analýzy dát, pochopia, ako fungujú a získajú intuíciu pre ich vhodnú aplikáciu so strojovým učením. Taktiež získajú predstavu, aké otázky vieme pomocou analýzy dát zodpovedať a budeme schopní aplikovať a vyhodnotiť základné prístupy strojového učenia. Dôraz je kladený na analýzu a predspracovanie dát, použitie metód strojového učenia, spôsoby ich vyhodnotenia a porovnania.
 
Projekt sa vypracúva vo dvojiciach. Pri riešení sa používa programovací jazyk Python a dostupné knižnice pre dátovú vedu ako pandas, numpy, scipy, scikit-learn, atd.. V každej fáze sa odovzdáva vykonateľný Jupyter Notebook, ktorý obsahuje všetky vykonané transformácie nad dátami s vhodnou dokumentáciou. Odovzdaný notebook musí obsahovať nielen kód, ale aj jeho výsledky (vypočítané hodnoty, výpisy, vizualizácie a pod.) spolu s komentárom k získaným výsledkom a z toho plynúce rozhodnutia pre ďalšie kroky dátového procesu. Schopnosť dobre komunikovať a prezentovať relevantné výsledky sa predstavuje významnú zložku hodnotenia.
 
Pri každej fáze v odovzdanom notebooku tiež uveďte percentuálny podiel práce členov dvojice.

Zadanie (the quest)

Každá dvojica bude pracovať s pridelenou dátovou sadou (3. týždeň).
Vašou úlohou je vedieť predikovať závislú hodnotu (indikator)
Budete sa musieť pritom vysporiadať s viacerými problémami, ktoré sa v dátach nachádzajú ako formáty dát, chýbajúce, vychýlené hodnoty a pod.

Dáta (the data)

Dataset'2021: Chronic Lymphocytic Leukemia (CLL)

Dátová sada predstavuje zdravotné záznamy pacientov s chronickou leukémiou (angl. Chronic Lymphocytic Leukemia - CLL). Každá dátová sada obsahuje údaje v dvoch súboroch

osobné údaje
laboratórne vyšetreni

Pacienti s chronickou leukémiou chodia na pravidelné vyšetrenie, bežne raz za mesiac. Choroba sa lieči ale sa nevylieči. Keď je stav zlepšený, tak môžu chodiť v intervale 3 alebo 6 mesiacov. Laboratórne vyšetrenie sa robí odberom, potom testy v laboratoriach, hodnoty sa sumarizujú posielajú naspäť lekárom. Každé takéto vyšetrenie má rozsiahle výsledky, ktoré sa hodnotia dlho. Nemocnica vedie aj centrálne záznamy o pacientoch (osobné údaje), kontakty s poisťovňou aj ďalšie potrebné údaje.

 
V záznamoch laboratórnych vyšetrení je závislá premenná (indikator) indikujúca stavu choroby pacienta. Ide o indikáciu či patient potrebuje ďalšie laboratórne vyšetrenie v preddefinovanom intervale alebo ten interval sa dá predĺžiť na základe výsledkov laboratorných testov aj osobných údajov pacienta.

Fáza 1 - prieskumná analýza (v 6. týždni): 15% = 15 bodov

V tejto fáze sa od Vás očakáva:
 
1. Základný opis dát spolu s ich charakteristikami (5 bodov)

Pre dosiahnutie plného počtu bodov uveďte
počet záznamov,
počet atribútov,
ich typy,
pre zvolené významné atribúty ich distribúcie, základné deskriptívne štatistiky a pod.

2. Párová analýza dát (5 bodov)

Preskúmajte vzťahy medzi zvolenými dvojicami atribútov.
Identifikujte závislostí medzi dvojicami atribútov a závislosti medzi predikovanou premennou a ostatnými premennými.

3. Formulácia a štatistické overenie hypotéz o dátach (2 body)

Sformulujte dve hypotézy o dátach v kontexte zadanej predikčnej úlohy. 
Sformulované hypotézy overte vhodne zvoleným štatistickým testom.

4. Identifikácia problémov v dátach s navrhnutým riešením (3 body)

Identifikujte problémy v dátach napr.: nevhodná štruktúra dát, duplicitné záznamy, nejednotné formáty, chýbajúce hodnoty, vychýlené hodnoty. V dátach sa môžu nachádzať aj iné, tu nevymenované problémy.
Navrhnuté riešenie prvotne realizujte na dátach. Problém s dátami môžete riešiť iteratívne v každej fáze aj vo všetkých fázach podľa potreby.

Správa sa odovzdáva v 6. týždni semestra

Na cvičení, dvojica svojmu cvičiacemu odprezentuje vykonanú prieskumnú analýzu v Jupyter Notebooku.
Správu elektronicky odovzdá jeden člen z dvojice do systému AIS do nedele 31.10.2021 23:59.

Fáza 2 - predspracovanie údajov (v 9. týždni): 20% = 20 bodov

V tejto fáze sa od Vás očakáva:

Realizujte predspracovanie údajov pre strojové učenie. Výsledkom bude upravená dátová sada vo formáte csv alebo tsv, kde jedno pozorovanie musí byť opísané jedným riadkom.
V 3. fáze budeme pracovať s algoritmami strojového učenia, ktorých implementácia v scikit-learn podporuje len numerické dáta. To vedie k tomu že treba niečo spraviť s nenumerickými dátami.
Replikovateľnosť predspracovania na trénovacej a testovacej množine dát.

Keď sa predspracovaním mohol zmeniť tvar a charakteristiky dát (počet atribútov, distribúcie hodnôt a pod.), je možné že treba znovu zrealizovať podstatné časti prieskumnej analýzy a opakovane podľa Vašej potreby. Bodovanie znovu za EDA už nebudeme, zmeny ale zdokumentujte. Problém s dátami môžete riešiť iteratívne v každej fáze aj vo všetkých fázach podľa vlastnej potreby.
 
1. Integrácia a čistenie dát (5b)

Transformujte dáta na vhodný formát pre strojové učenie t.j. jedno pozorovanie musí byť opísané jedným riadkom a každý atribút musí byť v numerickom formáte. 
Pri riešení chýbajúcich hodnôt (missing values) vyskúšajte rôzne stratégie z nasledujúcich podskupín:
odstránenie pozorovaní s chýbajúcimi údajmi
nahradenie chýbajúcej hodnoty mediánom, priemerom, pomerom (ku korelovanému atribútu), alebo pomocou lineárnej regresie resp. kNN
Podobne postupujte aj pri riešení vychýlených hodnôt (outlier detection):
odstránenie vychýlených (odľahlých) pozorovaní
nahradenie vychýlenej hodnoty hraničnými hodnotami rozdelenia (5% resp. 95%)

2. Realizácia predspracovania dát (5b)

Transformované dáta pre strojové učenie si rozdeľuje na trénovaciu a testovaciu množinu (train and test dataset) podľa vami preddefinovaným pomerom. Naďalej pracujte len s trénovacím datasetom.
Transformujte atribútov dát pre strojové učenie podľa dostupných techník (minimálne 2 techniky) ako scaling, transformers a ďalšie.
Zdôvodnite Vašu voľby/rozhodnutie pre realizáciu (t.j. zdokumentovanie)

3. Výber atribútov pre strojové učenie (5b)

Zistite ktoré atribúty (features) vo vašich dátach pre strojové učenie sú informatívne k atribútu “indikator”.
Zoradíte tie atribúty v poradí podľa dôležitosti. 
Zdôvodnite Vašu voľby/rozhodnutie pre realizáciu (t.j. zdokumentovanie) 

4. Replikovateľnosť predspracovania (5b)

Upravte váš kód realizujúci predspracovanie trénovacej množiny tak, aby ho bolo možné bez ďalších úprav znovu použiť na predspracovanie testovacej množiny (napr. pomocou funkcie/í)
Očakáva sa aj využitie možnosti sklearn.pipeline

Správa sa odovzdáva v 9. týždni semestra
Na cvičení, dvojica svojmu cvičiacemu odprezentuje vykonanú prácu v Jupyter Notebooku.
Správu elektronicky odovzdá jeden člen z dvojice do systému AIS do nedele 21.11.2021 23:59.

Fáza 3 - strojové učenie (v 12. týždni): 20% = 20 bodov

Pri dátovej analýze nemusí byť naším cieľom získať len znalosti obsiahnuté v aktuálnych dátach, ale aj natrénovať model, ktorý bude schopný robiť rozumné predikcie pre nové pozorovania pomocou strojového učenia.
V tejto fáze sa od Vás očakáva:
 
1. Manuálne vytvorenie a vyhodnotenie rozhodovacích pravidiel pre klasifikáciu (5b)

Naimplementujte 1R algorithm (1R or OneR), ktorý je jednoduchá klasifikácia t.j. rozhodnutie na základe jedného atribútu. Môžete implementovať komplikovanejšie t.j. zahŕňajúce viacero atribútov (ich kombinácie).
Pravidlá by v tomto kroku mali byť vytvorené manuálne na základe pozorovaných závislostí v dátach. Vyhodnoťte klasifikátor pomocou metrík accuracy, precision a recall.

2. Natrénovanie a vyhodnotenie klasifikátora strojového učenia (5b)

Na trénovanie využite minimálne jeden stromový algoritmus strojového učenia v scikit-learn.
Vizualizujte natrénované pravidlá.
Vyhodnoťte natrénovaný model pomocou metrík accuracy, precision a recall
Porovnajte natrénovaný klasifikátor s Vašimi manuálne vytvorenými pravidlami z prvého kroku.

3. Optimalizácia - hyperparameter tuning (5b)

Preskúmajte hyperparametre Vášho zvoleného klasifikačného algoritmu v druhom kroku a vyskúšajte ich rôzne nastavenie tak, aby ste minimalizovali overfitting (preučenie) a optimalizovali výsledok. 
Vysvetlite, čo jednotlivé hyperparametre robia/ Pri nastavovaní hyperparametrov algoritmu využite krížovú validáciu (cross validation) na trénovacej množine.

4. Vyhodnotenie vplyvu zvolenej stratégie riešenia na klasifikáciu (5b)

Vyhodnotíte Vami zvolené stratégie riešenia projektu z hľadiska classification accuracy: 

Stratégie riešenia chýbajúcich hodnôt a outlierov;
Scaling resp. transformer či zlepší accuracy klasifikácie;
Výber atribútov a výber algoritmov;
Hyperparameter tuning resp. ensemble learning.
Ktorý spôsob z každého hore-uvedených bodov sa ukázal ako vhodnejší pre daný problém? Vyhodnotenie podložíte dôkazmi.

Správa sa odovzdáva v 12. týždni semestra

Na cvičení, dvojica svojmu cvičiacemu odprezentuje vykonanú prácu v Jupyter Notebooku.
Správu elektronicky odovzdá jeden člen z dvojice do systému AIS do nedele 12.12.2021 23:59.
