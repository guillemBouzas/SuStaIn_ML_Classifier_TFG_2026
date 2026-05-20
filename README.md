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

## Dependències

Aquest projecte ha estat desenvolupat en Python i requereix les següents llibreries:

### Llibreries estàndard
- os
- json
- math

### Manipulació de dades
- pandas
- numpy

### Visualització
- matplotlib
- seaborn

### Machine Learning
- scikit-learn
  - svm
  - RandomForestClassifier
  - KNeighborsClassifier
  - VotingClassifier
  - LogisticRegression
  - HistGradientBoostingClassifier
  - LinearRegression
  - MLPClassifier
  - LabelEncoder
  - StandardScaler
  - confusion_matrix
  - accuracy_score
  - precision_score
  - RocCurveDisplay
  - PrecisionRecallDisplay
  - model_selection

### Estadística
- scipy
  - stats
  - linear_sum_assignment

### Models de progressió de malaltia
- pySuStaIn

### Models estadístics
- statsmodels

### Instal·lació

Es recomana instal·lar totes les dependències amb:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels
```

```bash
pip install git+https://github.com/ucl-pond/pySuStaIn
```

## Aclariments

Per a la redacció i desenvolupament del codi d’aquest treball s’han utilitzat eines d’intel·ligència artificial generativa, com ara ChatGPT i Google Gemini, com a suport durant el procés de programació.

Tot i això, tot el codi present en aquest repositori ha estat revisat personalment i n’entenc el funcionament i la implementació. Així mateix, cal destacar que el disseny metodològic, el plantejament dels algorismes i les decisions experimentals han estat desenvolupats principalment per mi, o bé basats en documentació i tutorials oficials utilitzats com a referència, com és el cas del tutorial proporcionat per a la utilització de SuStaIn.

Finalment, cal indicar que alguns comentaris en anglès presents dins dels scripts relacionats amb els models SuStaIn provenen directament d’aquests tutorials oficials i s’han mantingut amb l’objectiu de preservar la claredat i facilitar la comprensió del funcionament original del codi.
