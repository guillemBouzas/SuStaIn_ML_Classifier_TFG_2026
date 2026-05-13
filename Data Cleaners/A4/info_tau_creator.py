# Script per ordenar la informació de cada pacient del TAU 
import pandas as pd

# Obrim el df
nombre_archivo_1 = 'Imaging\TAUSUVR_27Jan2026.csv' 

df = pd.read_csv(nombre_archivo_1, sep=',')

# Ens carreguem les columnes que no necessitem o que tenen molts NaN

df = df.drop(columns = ['update_stamp', 'Mean_non_WM_hypointensities', 'bi_vessel', 'Mean_Right_vessel', 'Mean_Left_vessel'])

# Eliminar totes las columnes que comencen con 'Volume_'
df_final = df.drop(columns=[col for col in df.columns if col.startswith('Volume_')])

# Guardem el df net
df_final.to_csv('info_tau.csv', index=False, sep=',', encoding='utf-8')