#%%
import numpy as np
import os

#%%
def corregir_unidades(archivo_salida):
    nombre = archivo_salida
    with open(f'{nombre}', 'r') as f:
        lineas = f.readlines()

    with open(f'{nombre}', 'w') as f:
        f.write(lineas[0]) # Título
        # Corregir línea de Rango
        rango = lineas[1].split()
        f.write(f"{float(rango[0])*10} {float(rango[1])*10}\n")
        
        for i in range(2, len(lineas)):
            partes = lineas[i].split()
            if len(partes) == 2: # Si es una fila de datos (lambda n/k)
                f.write(f"{float(partes[0])*10}\t{partes[1]}\n")
            else: # Si es la línea de conteo de puntos
                f.write(lineas[i])


def crear_in3(archivo_entrada, nombre_material, archivo_salida,unidad = 'nm'):
    # 1. Cargar los datos (suponiendo que tienen 3 columnas: lambda, n, k)
    datos = np.loadtxt(archivo_entrada,skiprows=1)
    #para .nkv
    #datos = np.loadtxt(archivo_entrada, comments=';', usecols=(0, 1, 2))
    if unidad == 'um':
        lams = datos[:, 0] * 1000
    elif unidad == 'nm':
        lams = datos[:, 0]
    n_vals = datos[:, 1]
    k_vals = datos[:, 2]
    
    n_puntos = len(lams)
    l_min, l_max = lams[0], lams[-1]
    
    # 2. Escribir el archivo con el formato de Optical
    with open(archivo_salida, 'w') as f:
        # Encabezado
        f.write(f'"{nombre_material}"\n')
        f.write(f'{l_min:.1f} {l_max:.1f}\n')
        
        # Bloque de n
        f.write(f'{n_puntos}\n')
        for i in range(n_puntos):
            f.write(f'{lams[i]:.4f}\t{n_vals[i]:.4f}\n')
            
        # Bloque de k
        f.write(f'{n_puntos}\n')
        for i in range(n_puntos):
            f.write(f'{lams[i]:.4f}\t{k_vals[i]:.4f}\n')
    corregir_unidades(archivo_salida)
            
    print(f"✅ Archivo {archivo_salida} creado con éxito.")
    

    

#%%
# nombre_archivo_porosa = 'T1_porosa.in3'
# # Crea un archivo vacío abriéndolo y cerrándolo en una línea
# open(nombre_archivo_porosa, 'a').close()

# nombre_archivo_densa = 'T1_densa.in3'
# # Crea un archivo vacío abriéndolo y cerrándolo en una línea
# open(nombre_archivo_densa, 'a').close()

# ruta_poroso = r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\tmm-Rodrigo\indices\T1_porosa.txt'
# ruta_densa = r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\tmm-Rodrigo\indices\T1_densa.txt'

# crear_in3(f'{ruta_poroso}', 'T1 porosa experimental', f'{nombre_archivo_porosa}')
# crear_in3(f'{ruta_densa}', 'T1 densa experimental', f'{nombre_archivo_densa}')
nombre_archivo = 'Rutilo.in3'
open(nombre_archivo,'a').close()
ruta =r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\TMM-R\tmm-Rodrigo\indices\TiO2Palik.nk'
crear_in3(f'{ruta}','Si_optical_nk',f'{nombre_archivo}',unidad='um')

#%%
# nombre_archivo_GaAs = 'GaAs_Palik.in3'
# # Crea un archivo vacío abriéndolo y cerrándolo en una línea
# open(nombre_archivo_GaAs, 'a').close()
# ruta_GaAs = r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\tmm-Rodrigo\indices\GaAs_Palik.nk'
# crear_in3(f'{ruta_GaAs}','GaAs Palik',f'{nombre_archivo_GaAs}')
# %%
