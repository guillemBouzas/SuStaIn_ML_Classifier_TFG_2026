# -- imports --
from scipy.optimize import linear_sum_assignment  # Per trobar l'aparellament òptim
import pandas as pd
import numpy as np

# -- Carreguem les diferents runs que hem fet amb un model de sustain --

r1 = pd.read_csv('non_imputed/experiments_hp_biomarker/Experiment_0.csv', sep =',')
r2 = pd.read_csv('non_imputed/experiments_hp_biomarker/Experiment_1.csv', sep =',')
r3 = pd.read_csv('non_imputed/experiments_hp_biomarker/Experiment_2.csv', sep =',')
r4 = pd.read_csv('non_imputed/experiments_hp_biomarker/Experiment_3.csv', sep =',')

# -- algorisme per a l'avaluació --

runs = [r1,r2,r3,r4] # els guardem en un sol vector

matrix = [] # ens guardarem una matriu d'accuracies que ens interessin

for i in range(len(runs)):
    accuracies = []
    si = [runs[i][runs[i]['ml_subtype'] == s].index.values for s in [1, 2, 3]]

    for j in range(len(runs)):
        sj = [runs[j][runs[j]['ml_subtype'] == s].index.values for s in [1, 2, 3]]

        # Creem una "matriu de costos" interna (3x3) entre els subtipus de Run I i Run J
        cost_matrix = np.zeros((3, 3))
        for row in range(3):
            for col in range(3):
                inter = len(set(si[row]) & set(sj[col]))
                unio = len(set(si[row]) | set(sj[col]))
                # Guardem Jaccard (fem servir 1 - jaccard perquè l'algorisme minimitza)
                cost_matrix[row, col] = inter / unio if unio > 0 else 0

        # Trobem l'assignació òptima (quina parella va amb quina per maximitzar la suma)
        # Fem servir linear_sum_assignment sobre la matriu negativa per maximitzar
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        # L'accuracy de la comparació és la mitjana dels Jaccards d'aquestes parelles òptimes
        final_acc = cost_matrix[row_ind, col_ind].mean()
        accuracies.append(final_acc)

    matrix.append(accuracies)

print(pd.DataFrame(matrix))

'''
COM AVALUEM LA SIMILITUD ENTRE SUBTIPUS:

1. Construcció de la Matriu de Similitud (Jaccard)

L'algorisme construeix una taula (matriu) de 3x3 (per 3 subtipus). 
Cada cel·la de la matriu conté l'Índex de Jaccard entre un subtipus de la Run I i un subtipus de la Run J.


2. Resolució del Problema d'Assignació (Linear Sum Assignment)

Aquest és el punt clau per a la simetria. Imaginem que:

- El subtipus 1 de la Run A s'assembla un 80% al subtipus 2 de la Run B.
- El subtipus 2 de la Run A s'assembla un 75% també al subtipus 2 de la Run B.

ambdós subtipus de A es voldrien aparellar amb el mateix de B. 
L'algorisme de Linear Sum Assignment (conegut com a algorisme Hongarès) busca la configuració global òptima. 
Obliga a fer parelles úniques (1 a 1) de manera que la suma total de les similituds sigui la màxima possible.

3. Càlcul de l'Accuracy Final

Una vegada l'algorisme ha decidit quina és la "parella oficial" de cada subtipus en l'altra run:

- Extraiem els valors de similitud d'aquestes parelles escollides.
- En fem la mitjana (o el mínim, si volguéssim ser molt estricte).
'''