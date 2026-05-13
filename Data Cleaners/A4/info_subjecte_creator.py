# Script per combinar la info de PETVADATA, SUBJINFO i PTDEMOG
import pandas as pd

# Obrim els csv que necessitem

nombre_archivo_1 = 'Imaging\A4_PETVADATA_PRV2_27Jan2026.csv' 
nombre_archivo_2 = 'Assessments\A4_PTDEMOG_PRV2_22Jan2026.csv'
nombre_archivo_3 = 'Assessments\A4_SUBJINFO_PRV2_22Jan2026.csv'

# Obrim els dataframes
df1 = pd.read_csv(nombre_archivo_1, sep=',')
df2 = pd.read_csv(nombre_archivo_2, sep=',')
df3 = pd.read_csv(nombre_archivo_3, sep=',')


# Comencem amb PETVADATA

# Ens quedem només amb les columnes de BID i l'agregat del test
df1 = df1[['BID','SCORE']]


# Comencem amb els demographics

df2 = df2[['BID', 'PTGENDER', 'PTAGE', 'PTRACE', 'PTEDUCAT']]


# Comencem amb subjinfo

# Definim el "traductor" de valors, ho fem comptabilitzant quants al·lels E4 té el pacient
mapeo_apoe = {
    'E2/E2': 0,
    'E2/E3': 0,
    'E3/E3': 0,
    'E2/E4': 1,
    'E3/E4': 1,
    'E4/E4': 2
}


# Creem la nova columna basada en l'original
df3['APOE_E4_COUNT'] = df3['APOEGN'].map(mapeo_apoe)

# Seleccionem només les  columnes que necessitem
df3 = df3[['BID', 'APOE_E4_COUNT']]


# Ajuntem els tres dataframes en un de sol, ajuntarem per BID

df_final = df1.merge(df2, on='BID', how='outer')

df_final = df_final.merge(df3, on='BID', how='outer')


# Ens assegurem que tingui el format que volem 
print(df_final.head())

print(df_final.shape)
print(df_final.isnull().sum()) # Per comprovar quants nan tenim per columna


# Guardem el df net
df_final.to_csv('info_subjecte.csv', index=False, sep=',', encoding='utf-8')
