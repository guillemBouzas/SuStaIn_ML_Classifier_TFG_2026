# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Load libraries
import os
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pySuStaIn
import statsmodels.formula.api as smf
from scipy import stats
import sklearn.model_selection
from sklearn.preprocessing import LabelEncoder
import math
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PREPARACIÓ DEL DATASET
# Load the data 
# The data needs to be in the same directory as your notebook

# Obrir els dataframes i combinar-los
df1 = pandas.read_csv('info_subjecte.csv', sep = ',')
df2 = pandas.read_csv('A4_synthseg_vols.csv', sep = ',')
df3 = pandas.read_csv('nextbrain_only_CA_grouped.csv', sep = ',')

df2 = df2[['subject', 'total intracranial', 'Unnamed: 0']] # Les úniques columnes que m'interessen d'aquest
# Ens quedarem amb només una run per pacient, no volem pacients repetits
df2 = df2[df2['Unnamed: 0'] == 0]


data = pandas.merge(df1, df2, left_on='BID', right_on='subject')
# Al segmentar amb NextBrain dos pacients no es van fer, per consistència els eliminem tot i treballar amb Synthseg
ids_to_remove = {'B69162123', 'B11343379'}
data = data[~data['subject'].isin(ids_to_remove)]

# Treure les columnes que no necessitem
data.drop(columns = ['BID', 'PTRACE', 'PTEDUCAT', 'PTGENDER'], inplace = True, errors='ignore')

# -- Arreglem les dades de les regions de l'hipocamp

# -- Ajuntarem el dentate gyrus en una sola variable
cols_dentate = df3.filter(like='dentate_gyrus').columns
df3['DG'] = df3[cols_dentate].sum(axis=1)
df3 = df3.drop(columns=cols_dentate)
# --

# -- El mateix amb el subiculum
cols_subiculum = df3.filter(like='subiculum').columns
df3['subiculum'] = df3[cols_subiculum].sum(axis=1)
df3 = df3.drop(columns=cols_subiculum)
# --

# -- Ajuntem els df
data = pandas.merge(data, df3, on= 'subject', how='inner')
# --

# -- Farem una primera normalització dividint pel total intracranial del synthseg

# Creem una llista amb les columnes per normalitzar
cols_to_normalize = data.columns.drop(['PTAGE',
                                       'SCORE',
                                       'APOE_E4_COUNT',
                                       'subject',
                                       'Unnamed: 0'])

# Dividim aquestes columnes per la referència
data[cols_to_normalize] = data[cols_to_normalize].div(data['total intracranial'], axis=0)

data.drop(columns=['total intracranial', 'subject', 'Unnamed: 0'], inplace=True)

data.dropna(inplace = True)

# Comprovem les dimensions finals del nostre df
print(f"Finalment, amb totes les modificacions, el nostre algorisme queda amb les següents dimensions: {data.shape}")

# Convertim els positius i negatius en 1 i 0
le = LabelEncoder()
y = le.fit_transform(data['SCORE'])
data['SCORE'] =  y

# Podem mirar a l'inici de tots quants pacients tenim de cada per comparar-ho amb el resultat del susTAIN
conteo_score = data['SCORE'].value_counts().sort_index()
print(conteo_score)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# store our biomarker labels as a variable
biomarkers = data.columns[3:]
print(biomarkers)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creem una carpeta on guardar les figures

if not os.path.isdir('figures'):
    os.mkdir('figures')
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# first a quick look at the patient and control distribution for one of our biomarkers

# Observarem totes les distribucions en una sola quadrícula
n_cols = 5
n_rows = math.ceil(len(biomarkers) / n_cols)

# Fem la figura gran perquè es vegi bé
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()  # Convertimos la matriz de ejes en una lista plana para iterar fácil

for i, biomarker in enumerate(biomarkers):
    sns.kdeplot(
        data=data,
        x=biomarker,
        hue='SCORE',
        fill=True,  
        common_norm=False,  
        ax=axes[i]  
    )
    axes[i].set_title(biomarker, fontsize=12)
    axes[i].set_xlabel('')  

# Esborrem alguns espais que no necessitem
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.title('Distribucions abans de normalitzar')
plt.savefig('figures/biomarkers_distributions_before_normalization.svg', dpi=300)
plt.close()  # Per netejar les figures
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# now we perform the normalization

# make a copy of our dataframe (we don't want to overwrite our original data)
zdata = pandas.DataFrame(data,copy=True)

# for each biomarker
for biomarker in biomarkers:
    mod = smf.ols('Q("%s") ~ PTAGE + APOE_E4_COUNT'%biomarker,  # fit a model finding the effect of age and headsize on biomarker
                  data=data[data.SCORE==0] # fit this model *only* to individuals in the control group
                 ).fit() # fit model    
    #print(mod.summary())
    
    # get the "predicted" values for all subjects based on the control model parameters
    predicted = mod.predict(data[['PTAGE', 'APOE_E4_COUNT',biomarker]]) 
    
    # calculate our zscore: observed - predicted / SD of the control group residuals
    w_score = (data.loc[:,biomarker] - predicted) / mod.resid.std()
    
    #print(np.mean(w_score[data.SCORE==0]))
    #print(np.std(w_score[data.SCORE==0]))
    
    # save zscore back into our new (copied) dataframe
    zdata.loc[:,biomarker] = w_score
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Scatterplot abans de normalitzar

# Ho mirarem pels biomarcadors
cols_to_melt = [b for b in biomarkers if b in data.columns]

# Això crea un nou DataFrame on cada fila té: PTAGE, SCORE, el nom del biomarcador i el seu valor.
data_long_age = data.melt(id_vars=['PTAGE', 'SCORE'],
                           value_vars=cols_to_melt,
                           var_name='Biomarker',
                           value_name='Value')

# Creem la quadrícula (FacetGrid)
# col="Biomarker" crea un subplot per cada biomarcador diferent.
# col_wrap=5 farà que cada 5 gràfics salti de línia.
# hue="SCORE" coloreja els punts segons el SCORE.
g = sns.FacetGrid(data_long_age, col="Biomarker", hue="SCORE",
                  col_wrap=5, sharex=True, sharey=False, height=4, aspect=1.2)

# Mapegem el scatterplot a cada quadre
# 'PTAGE' anirà a l'eix X, i 'Value' (el valor del biomarcador) a l'eix Y.
# 'sharex=True' (per defecte a true quan els eixos X són els mateixos)
# 'sharey=False' és CRUCIAL perquè els diferents biomarcadors tenen escales molt diferents.
g.map(sns.scatterplot, "PTAGE", "Value", alpha=0.6)

# Ajustos estètics i títols
g.add_legend() # Afegeix la llegenda del SCORE
g.set_titles("{col_name}") # Posa el nom del biomarcador com a títol de cada sub-gràfic
g.set_axis_labels("PTAGE", "Valor")

# Títol general per a tota la figura
plt.subplots_adjust(top=0.95) # Deixa un espai per al títol
g.fig.suptitle('Relació entre PTAGE i Biomarcadors per SCORE (abans de normalitzar)', fontsize=16)

# Guardar la imatge
plt.savefig('figures/all_biomarkers_vs_ptage_scatterplot_grid_before_normalization.svg', dpi=300, bbox_inches='tight')
plt.close()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# El mateix que abans pero ara després de normalitzar

cols_to_melt = [b for b in biomarkers if b in zdata.columns]

zdata_long_age = zdata.melt(id_vars=['PTAGE', 'SCORE'],
                           value_vars=cols_to_melt,
                           var_name='Biomarker',
                           value_name='Value')

g = sns.FacetGrid(zdata_long_age, col="Biomarker", hue="SCORE",
                  col_wrap=5, sharex=True, sharey=False, height=4, aspect=1.2)

g.map(sns.scatterplot, "PTAGE", "Value", alpha=0.6)

g.add_legend() 
g.set_titles("{col_name}") 
g.set_axis_labels("PTAGE", "Valor")

plt.subplots_adjust(top=0.95) 
g.fig.suptitle('Relació entre PTAGE i Biomarcadors per SCORE (després de normalitzar)', fontsize=16)

plt.savefig('figures/all_biomarkers_vs_ptage_scatterplot_grid_after_normalization.svg', dpi=300, bbox_inches='tight')
plt.close()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Tornem a observar les distribucions un cop normalitzades les dades
n_cols = 5
n_rows = math.ceil(len(biomarkers) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()  

for i, biomarker in enumerate(biomarkers):
    sns.kdeplot(
        data=zdata,
        x=biomarker,
        hue='SCORE',
        fill=True,  
        common_norm=False,  
        ax=axes[i]  
    )
    axes[i].set_title(biomarker, fontsize=12)
    axes[i].set_xlabel('')  

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.title('Distribucions després de normalitzar')
plt.savefig('figures/biomarkers_distributions_after_normalization.svg', dpi=300)
plt.close()  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CLIPPING
# Resulta que per alguna raó han sortit alguns Z_scores enormement alts i enormement baixos
# Això feia que al aplicar l'algorisme hi hagués una probabilitat que s'apropava tant a 0 que l'ordinador interpretava com a 0
# Limitem els valores al rang [-5, 5] perquè coincideixin Z_vals i Z_max
# Per a l'algorisme no ens cal diferenciar exactament que és enormement gran o petit, només que és un extrem, i això ja ho representa el rang
zdata = np.clip(zdata, -5, 5)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Prepare the susTAIN inputs

N = len(biomarkers)         # number of biomarkers

SuStaInLabels = biomarkers
Z_vals = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
Z_max  = np.array([5]*N)  

# Input the settings for z-score SuStaIn
# To make the tutorial run faster I've set 
# N_startpoints = 10 and N_iterations_MCMC = int(1e4)
# I recommend using N_startpoints = 25 and 
# N_iterations_MCMC = int(1e5) or int(1e6) in general though

N_startpoints = 10
N_S_max = 4 # Aquí per canviar quant subtipus li posem (N_S_max = control + n_subtipus)
N_iterations_MCMC = int(1e4)
output_folder = os.path.join(os.getcwd(), 'WorkshopOutput')
dataset_name = 'WorkshopOutput'

# Initiate the SuStaIn object
sustain_input = pySuStaIn.ZscoreSustain(
                              zdata[biomarkers].values,
                              Z_vals,
                              Z_max,
                              SuStaInLabels,
                              N_startpoints,
                              N_S_max, 
                              N_iterations_MCMC, 
                              output_folder, 
                              dataset_name, 
                              False)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Run susTAIN !!

# make the output directory if it's not already created
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

samples_sequence,   \
samples_f,          \
ml_subtype,         \
prob_ml_subtype,    \
ml_stage,           \
prob_ml_stage,      \
prob_subtype_stage  = sustain_input.run_sustain_algorithm()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate subtypes

# for each subtype model
for s in range(N_S_max):
    # load pickle file (SuStaIn output) and get the sample log likelihood values
    pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
    pk = pandas.read_pickle(pickle_filename_s)
    samples_likelihood = pk["samples_likelihood"]
    
    # plot the values as a line plot
    plt.figure(0)
    plt.plot(range(N_iterations_MCMC), samples_likelihood, label="subtype" + str(s))
    plt.legend(loc='upper right')
    plt.xlabel('MCMC samples')
    plt.ylabel('Log likelihood')
    plt.title('MCMC trace')

plt.savefig('figures/MCMC_trace_eval_subtypes.svg', dpi=300)
plt.close() 

for s in range(N_S_max):
    # load pickle file (SuStaIn output) and get the sample log likelihood values
    pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
    pk = pandas.read_pickle(pickle_filename_s)
    samples_likelihood = pk["samples_likelihood"]
    
    # plot the values as a histogram plot
    plt.figure(1)
    plt.hist(samples_likelihood, label="subtype" + str(s))
    plt.legend(loc='upper right')
    plt.xlabel('Log likelihood')  
    plt.ylabel('Number of samples')  
    plt.title('Histograms of model likelihood')

plt.savefig('figures/Hist_model_likelihood_eval_subtypes.svg', dpi=300)
plt.close() 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Let's plot positional variance diagrams to interpret the subtype progressions

# Aquesta part l'haurem d'adaptar si volem fer l'anàlisi amb dos subtypes i no només amb 1, quan ho vulgui fer amb dos subtypes agafar codi del tutorial
s = 2 # 1 split = 2 subtypes, 2 split = 3 subtypes
M = len(zdata)

# get the sample sequences and f
pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pandas.read_pickle(pickle_filename_s)
samples_sequence = pk["samples_sequence"]
samples_f = pk["samples_f"]

# Configuracions del plot
plt.rc('font', size=6)
fig, ax = plt.subplots(figsize=(18, 18))

# use this information to plot the positional variance diagrams
tmp=pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1,2), biomarker_labels=biomarkers)

for axis in plt.gcf().axes:
    axis.tick_params(axis='x', rotation=90, labelsize=3)
    axis.tick_params(axis='y', labelsize=3)
    
    # No mostrem tots els números de l'eix X per no saturar
    ticks = axis.get_xticks()
    axis.set_xticks(ticks[::2]) # Ensenyem un de cada dos nombres

# Guardar la imatge
plt.tight_layout()
plt.savefig('figures/positional_variance.svg', dpi=300, bbox_inches='tight')
plt.close() 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Subtype and stage individuals

# let's take a look at all of the things that exist in SuStaIn's output (pickle) file
print(pk.keys())

# Aquesta part també la tenim adaptada pel nostre cas en el que només tenim un subtipus extra
# Ens afegeix columnes al nostre df:
#   - El subtipus assignat
#   - La certesa (probabilitat) que té el model de que pertany a aquest subtipus
#   - L'etapa de la malaltia (stage)
#   - La certesa de que aquest pacient estigui en aquesta etapa

# Aquí per el cas de tres subtipus
s = 2
pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pandas.read_pickle(pickle_filename_s)

for variable in ['ml_subtype', # the assigned subtype
                 'prob_ml_subtype', # the probability of the assigned subtype
                 'ml_stage', # the assigned stage 
                 'prob_ml_stage',]: # the probability of the assigned stage
    
    # add SuStaIn output to dataframe
    zdata.loc[:,variable] = pk[variable]

# let's also add the probability for each subject of being each subtype
for i in range(s):
    zdata.loc[:,'prob_S%s'%i] = pk['prob_subtype'][:,i]

print(zdata.head())
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORTANT!!! The last thing we need to do is to set all "Stage 0" subtypes to their own subtype
# We'll set current subtype (0 and 1) to 1 and 0, and we'll call "Stage 0" individuals subtype 0.

# make current subtypes (0 and 1) 1 and 2 instead
zdata.loc[:,'ml_subtype'] = zdata.ml_subtype.values + 1

# convert "Stage 0" subjects to subtype 0
zdata.loc[zdata.ml_stage==0,'ml_subtype'] = 0

print(zdata.ml_subtype.value_counts())
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Ens serveix per comprovar si els individus control han estat assignats al subtipus 0

sns.displot(x='ml_stage',hue='SCORE',data=zdata,col='ml_subtype')
plt.savefig('figures/subtype_dist_hist.svg', dpi=300, bbox_inches='tight')
plt.clf() 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Dibuixem quan de segur està el model de predir que un individu pertany a un subtipus o un altre segons en quina etapa de la malaltia es trobi

sns.pointplot(x='ml_stage',y='prob_ml_subtype', # input variables
              hue='ml_subtype',                 # "grouping" variable
            data=zdata[zdata.ml_subtype>0]) # only plot for Subtypes 1 and 2 (not 0)
plt.ylim(0,1) 
plt.axhline(0.5,ls='--',color='k') # plot a line representing change (0.5 in the case of 2 subtypes)
plt.savefig('figures/certesa_subtype_stage.svg', dpi=300, bbox_inches='tight')
plt.clf() 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Evaluate relationships

# Plotting relationship between a biomarker and SuStaIn stage across subtypes

# Anem a fer-ho també per l'edat, el codi canviarà degut a que l'edat 'real' ha quedat clipejada abans

# Assegurem que zdata només tingui files que existeixen a data (després de la neteja)
# Això sincronitza ambdós DataFrames per si de cas
zdata = zdata[zdata.index.isin(data.index)]

# Assignem l'edat real
# Fem servir .loc per buscar exactament els mateixos IDs (índexs) que hi ha a zdata
var_real = 'AGE_REAL'
zdata[var_real] = data.loc[zdata.index, 'PTAGE']

# El gràfic lmplot
# Filtrem ml_subtype > 0 per evitar els "no assignats" si n'hi hagués
g = sns.lmplot(x='ml_stage', y=var_real, hue='ml_subtype',
               data=zdata[zdata.ml_subtype > 0],
               height=6, aspect=1.2, scatter_kws={'alpha': 0.5})

ax = g.ax

# Càlcul dels estadístics per cada subtipus
# Suposem que els teus subtipus estan guardats a zdata.ml_subtype.unique()
subtypes = [s for s in zdata.ml_subtype.unique() if s > 0]

for i, subtype in enumerate(subtypes):
    # Extraiem les dades filtrades per subtipus
    subset = zdata[zdata.ml_subtype == subtype]

    x_vals = subset['ml_stage'].values
    y_vals = subset[var_real].values

    # Comprovem que hi hagi prou dades i que no siguin constants (per evitar errors de Pearson)
    if len(x_vals) > 1 and y_vals.std() > 0:
        r, p = stats.pearsonr(x_vals, y_vals)
        text_label = f'S{subtype}: r={r:.3f}, p={p:.3f}'

        # Posicionament del text a la cantonada del gràfic
        ax.text(0.05, 0.95 - (i * 0.05), text_label,
                transform=ax.transAxes,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.title('Relació entre l\'estadi del model i l\'edat real')
plt.savefig('figures/relationship_age_stage.svg', dpi=300, bbox_inches='tight')
plt.close()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# we can also look at differences in each biomarker across subtypes

'''
És bàsicament un anàlisis estadístic:
- t (Estadístico t): Indica qué tan grande es la diferencia entre los grupos. Cuanto más alejado de cero (positivo o negativo), mayor es la diferencia.
- p (p-valor): Indica si esa diferencia es estadísticamente significativa. Generalmente, si p < 0.05, la separación que hizo el modelo es válida para ese biomarcador.
'''

from scipy import stats
results = pandas.DataFrame(index=biomarkers)
for biomarker in biomarkers:
    t,p = stats.ttest_ind(zdata.loc[zdata.ml_subtype==0,biomarker],
                         zdata.loc[zdata.ml_subtype==1,biomarker],)
    results.loc[biomarker,'t'] = t
    results.loc[biomarker,'p'] = p
    
print(results)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Representa els resultats de la t-student en un gràfic de calor

sns.heatmap(pandas.DataFrame(results['t']),square=True,annot=True,
           cmap='RdBu_r')
plt.savefig('figures/t_student_biomarkers.svg', dpi=300, bbox_inches='tight')
plt.close() # Per netejar les figures
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# plot an example variable:

# Boxplots

vars_to_plot = ['AGE_REAL']

# Creem la figura
fig, axes = plt.subplots(1, 5, figsize=(18, 6))

for i, var in enumerate(vars_to_plot):
    # Verifiquem si la variable existeix per evitar errors
    if var in zdata.columns:
        sns.boxplot(x='ml_subtype', y=var, data=zdata, ax=axes[i], palette='Set2')
        # També podem afegir els punts individuals (stripplot) per veure la dispersió real
        sns.stripplot(x='ml_subtype', y=var, data=zdata, ax=axes[i],
                      color='black', alpha=0.3, size=4)

        axes[i].set_title(f'Distribució de:\n{var.replace("_", " ").title()}', fontsize=14)
        axes[i].set_xlabel('Subtipus ML')
        axes[i].set_ylabel('Valor')
    else:
        axes[i].text(0.5, 0.5, f'Variable "{var}"\nno trobada',
                     ha='center', va='center')

# Ajustar l'espai entre gràfics perquè no se solapin els eixos
plt.tight_layout()

# Guardar la figura 
plt.savefig('figures/relevant_boxplots.svg', dpi=300, bbox_inches='tight')
plt.close()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Aquí mirarem quants positius i negatius hi ha de cada subtype

# Recorrem tots els subtipus
for subtype in sorted(zdata['ml_subtype'].unique()):
    # Filtrem el dataframe pel subtipus i en mirem l'SCORE
    scores_subtipo = zdata.loc[zdata['ml_subtype'] == subtype, 'SCORE'].values
    conteo = zdata.loc[zdata['ml_subtype'] == subtype, 'SCORE'].value_counts()
    
    print(f"\n--- Subtipud {subtype} ---")
    print(f"Quantitat de subjectes: {len(scores_subtipo)}")
    print(f"Valors de SCORE: {conteo}")
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Ens guardarem zdata per fer una avaluació del resultat de diferents experiments
zdata.to_csv('Experiment.csv', index=False, sep=',', encoding='utf-8')
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
