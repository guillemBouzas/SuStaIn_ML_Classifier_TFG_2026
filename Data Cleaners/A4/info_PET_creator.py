# Script per ordenar la informació de cada pacient del PET amyloid
import pandas as pd

# Obrim el df
nombre_archivo_1 = 'Imaging\A4_PETSUVR_PRV2_27Jan2026.csv' 

df = pd.read_csv(nombre_archivo_1, sep=',')

# Eliminem les columnes que no necessitem per res

df = df[['BID', 'brain_region', 'suvr_cer', 'centiloid']]

# Ara reestructurarem el df de la manera desitjada

# Pivotem el DataFrame per les regions cerebrals
df_pivot = df.pivot(index='BID', columns='brain_region', values='suvr_cer') # Aquí aconseguim convertir les 8 files de cada pacient en un pacient amb 8 columnes

# Rescatem el valor de 'centiloid'
# Fem servir groupby per obtenir l'únic valor no nul per cada pacient
df_centiloid = df.groupby('BID')['centiloid'].first()

# Unim resultats
df_final = df_pivot.join(df_centiloid).reset_index()

# Guardem el df net
df_final.to_csv('info_PET.csv', index=False, sep=',', encoding='utf-8')

