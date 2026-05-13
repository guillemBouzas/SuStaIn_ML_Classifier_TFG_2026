# SuStaIn_ML_Classifier_TFG_2026

## Estudi multimodal per a l’estratificació de pacients en diferents estadis de la malaltia d’Alzheimer

Aquest repositori de GitHub conté el codi principal desenvolupat en el marc del meu Treball de Final de Grau.

A continuació es descriu el contingut de cadascuna de les carpetes del repositori:

- **Classifiers** &rarr; Aquesta carpeta conté els models de classificació per a la detecció de positivitat amiloide. Dos dels models utilitzen com a característiques les volumetries obtingudes amb l’atles SynthSeg, mentre que els altres dos empren les volumetries derivades de l’atles NextBrain. Les característiques utilitzades en cada model es poden identificar a través del nom del fitxer corresponent.

  Per a cada atles s’han desenvolupat dues variants del model: una basada en una compensació manual entre les classes positiva i negativa, i una altra basada en la tècnica de sobremostreig SMOTE. La metodologia emprada en cada cas també es pot identificar mitjançant el nom del fitxer.

- **SuStaIn** &rarr; Aquesta carpeta inclou el codi necessari per a la construcció dels models de progressió de la malaltia. En concret, s’hi poden trobar tres models multimodals que integren biomarcadors estructurals derivats de les volumetries dels atles SynthSeg i NextBrain, així com biomarcadors d’Amyloid PET i Tau PET.

  El nom de cada model comença amb el nombre de característiques utilitzades durant l’entrenament. Addicionalment, també s’hi inclouen dos models basats exclusivament en regions de l’hipocamp segmentades amb l’atles NextBrain. Aquests models es poden identificar perquè el nom del fitxer conté `hp`.

- **Evaluation** &rarr; Aquesta carpeta conté el codi utilitzat per a l’avaluació de la consistència i la robustesa dels grups obtinguts mitjançant els models SuStaIn. L' script inclòs permet analitzar si les estratificacions generades presenten coherència entre els diferents biomarcadors i configuracions utilitzades.

- **Data Cleaners** &rarr; Aquesta carpeta conté el codi emprat per al preprocessament inicial dels fitxers CSV utilitzats posteriorment en els diferents models descrits anteriorment.

  A l’interior de la carpeta s’hi troben tres subcarpetes diferenciades segons la procedència de les dades o l’atles utilitzat: **A4**, **ADNI** i **NextBrain**.

Per identificar la base de dades utilitzada en cada script, cal tenir en compte que únicament els fitxers que contenen el prefix `adni` al nom fan ús de la base de dades ADNI. La resta de scripts treballen amb dades provinents de la base de dades A4.
