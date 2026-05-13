#------------------------------------------------------------------------------------------------------------------------------------------
# Imports
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------------
# PREPARACIÓ DEL DATASET
# Carreguem les dades
df1 = pd.read_csv('info_subjecte.csv', sep = ',')
df2 = pd.read_csv('nextbrain_L_vols.csv', sep = ',')
df3 = pd.read_csv('nextbrain_R_vols.csv', sep = ',')
df4 = pd.read_csv('A4_synthseg_vols.csv', sep = ',')

# No necessitarem aquestes columnes
df1.drop(columns=['PTGENDER', 'PTRACE', 'APOE_E4_COUNT', 'PTEDUCAT'], inplace=True)
df2.drop(columns = ['Unnamed: 0', 'session'], inplace = True)
df3.drop(columns = ['Unnamed: 0', 'session'], inplace = True)
df4 = df4[['subject', 'total intracranial', 'Unnamed: 0']] # Les úniques columnes que m'interessen d'aquest

# Ens quedarem amb només una run de cada pacient per així evitar tenir pacients duplicats
df4 = df4[df4['Unnamed: 0'] == 0]
df4.drop(columns = ['Unnamed: 0'], inplace = True) # Ja no em fa falta

# Ajuntarem els df del nextbrain en un de sol
columnes = df2.columns.tolist()
dfNextbrain = df2.copy()

for col in columnes:
    if col != 'subject':
        dfNextbrain[col] = df2[col] + df3[col]

# Comprovem que s'hagi sumat correctament
#print(dfNextbrain['white_matter_of_forebrain'].head())

# Unir dataframes
df_total = pd.merge(df1, dfNextbrain, left_on='BID', right_on='subject')
df_total = pd.merge(df_total, df4, on='subject', how='inner')

# Scikit-learn no accepta NaNs, també els eliminem
df_total = df_total.dropna()

# Mides del dataset "original"
print(f'El dataset original és de {df_total.shape}')

# Encoding de la variable objectiu (negatiu = 0, positiu = 1)
le = LabelEncoder()
y = le.fit_transform(df_total['SCORE'])

# Variables predictores
X = df_total.drop(columns = ['BID', 'subject'])
#------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------------
# PRIMERA NORMALITZACIÓ DE LES DADES
dividir_total = True # Per decidir si utilitzem la variable del volum intracranial per dividir

if dividir_total:
    # Comencem la normalització dividint pel volum intracranial
    # Fem una llista amb les columnes que hem de normalitzar (excloem total intracranial, PTAGE i SCORE que no s'han de normalitzar)
    cols_to_normalize = X.columns.drop(['total intracranial', 'PTAGE', 'SCORE'])

    # Dividim les columnes per la referència (total intracranial)
    X[cols_to_normalize] = X[cols_to_normalize].div(X['total intracranial'], axis=0)

X_negativos = X[X['SCORE'] == 'negative']

# El mateix d'abans, definim quines columnes s'han de normalitzar
cols_to_correct = X.columns.drop(['PTAGE', 'SCORE', 'total intracranial'])

# Bucle per ajustar el model per cada columna
for col in cols_to_correct:
    model = LinearRegression(fit_intercept=True)

    # Entrenem només amb els negatius
    X_train = X_negativos[['PTAGE']]
    y_train = X_negativos[col]

    model.fit(X_train, y_train)

    # Predim per TOT el dataset complet (positius i negatius)
    # Volem saber el volum esperat per la seva edat
    valores_predichos = model.predict(X[['PTAGE']])

    # Calculem el residu: Valor Real - Valor Predit
    # Sobreescrivim amb el valor predit la columna
    X[col] = X[col] - valores_predichos
# ------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------
# ÚLTIMS PASSOS PER A TENIR EL DATASET LLEST
# La meva variable predictora ja neta
X.drop(columns=['PTAGE', 'SCORE', 'total intracranial'], inplace=True)

# --- EL FIX PER L' ERROR DE 'c_contiguous' ---
# Convertim X a un array de numpy pur i ens assegurem de que sigui C-contiguous
X = np.ascontiguousarray(X.values)
# -----------------------------------------------

# Pot ser que hi hagi algun total intracranial que sigui 0
# Retorna True si hi ha al menys un NaN
hay_nan = np.isnan(X).any()

if hay_nan:
    # Crear una màscara de las files que NO tenen NaN
    mask = ~np.isnan(X).any(axis=1)

    # Filtrar els arrays amb la mateixa màscara (Tornem a eliminar NaN que hagin aparegut en aquest preprocés)
    X = X[mask]
    y = y[mask]

# Mirem la mida del dataset al final
print(f'El dataset que utilitzem finalment és de {X.shape}')
# ------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------
# PREPARACIÓ DE L'ENTRENAMENT I MODELS

# Llibreries extres
from sklearn.model_selection import StratifiedKFold  # Per entrenar fent folds
from imblearn.over_sampling import SMOTE  # Generador de pacients sintètics

# Definim l'objecte de validació creuada
# n_splits=5 significa que entrenarem 5 vegades (80% tren / 20% test cada vegada)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sm = SMOTE(random_state=42)  # El "generador" de pacientes sintéticos

# Llistes per guardar els resultados de cada "fold"
resultados_acc = []
resultados_sens = []
resultados_espec = []
resultados_prec = []


print("Iniciant validació creuada amb SMOTE (Balanceig 50/50)...")

# Aquí preparem l'ensamble
# Models
clf1 = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')  # Lo de probaility = True és pel voting soft
clf2 = RandomForestClassifier(n_estimators=100, class_weight='balanced')
clf3 = KNeighborsClassifier(n_neighbors=5, weights='distance')
logreg = LogisticRegression(random_state=16, max_iter=5000, class_weight='balanced')
gradboost = HistGradientBoostingClassifier(class_weight='balanced')
# Xarxa neuronal petita
# (10, 8) significa 10 neurones en la 1ª capa i 8 en la 2ª.
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(10, 8),
    max_iter=1000,
    early_stopping=True,  # Per no sobreentrenar
    validation_fraction=0.1,  # Proporció de les dades d'entrenament que s'utilitzen com a validació
    random_state=42,
    activation='relu'  # Funció d'activació estàndard
)

# Ensamble
modelos = [('svm', clf1), ('rf', clf2), ('knn', clf3), ('gradboost', gradboost), ('logreg', logreg), ('mlp', mlp_clf)]
ensemble = VotingClassifier(estimators=modelos, voting='soft')
# ------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------
# EL BUCLE D'ENTRENAMENT
# skf.split(X, y) genera els índexs para cada ronda
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    # Separar les dades del fold
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    # -- NORMALITZACIÓ --
    scaler = StandardScaler()
    mask_sanos = (y_train_fold == 0)
    scaler.fit(X_train_fold[mask_sanos])

    X_train_norm = scaler.transform(X_train_fold)
    X_test_ready = scaler.transform(X_test_fold)

    # Apliquem SMOTE a les dades d'entrenament
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train_norm, y_train_fold)

    # -- ENTRENAMENT --
    # Ara l'entrenament estarà equilibrat
    ensemble.fit(X_train_resampled, y_train_resampled)

    # -- EVALUACIÓ --
    preds = ensemble.predict(X_test_ready)
    cm_fold = confusion_matrix(y_test_fold, preds)
    tn, fp, fn, tp = cm_fold.ravel()

    # Mètriques
    acc = accuracy_score(y_test_fold, preds)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    espec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = precision_score(y_test_fold, preds)

    resultados_acc.append(acc)
    resultados_sens.append(sens)
    resultados_espec.append(espec)
    resultados_prec.append(prec)


    # Aquí estem parlant del valors de train
    print(f"Fold {i + 1}: Original: {len(y_train_fold)} mostres -> Amb SMOTE: {len(y_train_resampled)} mostres")

    # -- GUARDEM IMATGES
    # Creem una carpeta on guardar les figures
    if not os.path.isdir('figures_SMOTE'):
        os.mkdir('figures_SMOTE')

    # La corba ROC
    plt.figure(0)
    RocCurveDisplay.from_estimator(ensemble, X_test_ready, y_test_fold)
    plt.savefig(f'figures_SMOTE/roc_fold {i + 1}.svg')

    # La precision-recall curve
    plt.figure(1)
    PrecisionRecallDisplay.from_estimator(ensemble, X_test_ready, y_test_fold)
    plt.savefig(f'figures_SMOTE/precision_recall_fold {i + 1}.svg')
    # --
# ------------------------------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------
# RESULTATS FINALS
print("\n" + "=" * 30)
print("   RESUM FINAL DEL MODEL")
print("=" * 30)
print(f"Accuracy Mitjana:      {np.mean(resultados_acc):.4f}")
print(f"Sensibilitat Mitjana:  {np.mean(resultados_sens):.4f} (Capacitat detecció positius)")
print(f"Especificitat Mitjana: {np.mean(resultados_espec):.4f} (Capacitat detecció negatius)")
print(f"Precisió Mitjana: {np.mean(resultados_prec):.4f} (Capacitat per no etiquetar com a positiu un valor negatiu)")
print("=" * 30)
# ------------------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------------------------------
# ESTALVI
# Aquest apartat busca mirar l'estalvi que es guanyaria en utilitzar el model
# Fòrmules extretes de "Study of early stages of Alzheimer’s disease using magnetic resonance imaging" d'Adrià Casamitjana

ro = 0.2 # prevalença esperada de pacients amyloid positius en la població general
Cpet = 3000 # preu en euros d'un PET
Cmri = 700 # preu en euros d'un MRI
Ns = df_total.shape[0] # Nombre de pacients que vull observar, els del meu estudi


Cs = Ns * (Cpet + Cmri) # Cost de l'estudi si no comptéssim amb el model

K = Ns * ro # Nombre ideal de pacients que volem reclutar (idealment serien els que el model ha marcat com True Positive)
R = np.mean(resultados_sens) # Sensibilitat
P = np.mean(resultados_prec) # Precisió


Cp = K/ro * (Cmri/R + ro*Cpet/P) # Cost de l'estudi amb el model


estalvi = ((Cs - Cp)/Cs)*100 # Percentatge d'estalvi


#print(Ns)
print("\n" + "=" * 30)
print("   SAVINGS DEL MODEL")
print("=" * 30)
print(f"Sense el model l'estudi costaria: {Cs:.4f}")
print(f"Amb el model l'estudi costaria: {Cp:.4f}")
print(f"Hem estalviat un: {estalvi:.4f}")
print("=" * 30)
# ------------------------------------------------------------------------------------------------------------------------------------------
