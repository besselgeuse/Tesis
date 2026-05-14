#%%
import numpy as np
from scipy.interpolate import interp1d
def in3_a_nk_interpolado(archivo_in3, archivo_salida_nk, skip_n, filas_n, skip_k, filas_k):
    # 1. Extraer bloques n y k
    datos_n = np.loadtxt(archivo_in3, skiprows=skip_n, max_rows=filas_n)
    lams_n, n_vals = datos_n[:, 0], datos_n[:, 1]
    
    datos_k = np.loadtxt(archivo_in3, skiprows=skip_k, max_rows=filas_k)
    lams_k, k_vals_temp = datos_k[:, 0], datos_k[:, 1]
    
    # 2. Unir y ordenar todas las longitudes de onda únicas
    lams_union = np.unique(np.concatenate((lams_n, lams_k)))
    
    # 3. Funciones de interpolación
    # Para n: extrapolamos si se sale un poco del rango
    interp_n = interp1d(lams_n, n_vals, kind='linear', fill_value="extrapolate")
    # Para k: rellenamos con 0.0 si no existe en su rango original
    interp_k = interp1d(lams_k, k_vals_temp, kind='linear', fill_value=0.0, bounds_error=False)
    
    # 4. Calcular los nuevos arreglos
    n_final = interp_n(lams_union)
    k_final = interp_k(lams_union)
    
    # 5. Guardar el archivo
    datos_nk = np.column_stack((lams_union, n_final, k_final))
    datos_nk[:,0] /= 10.
    np.savetxt(archivo_salida_nk, datos_nk, fmt='%.6f', delimiter='\t', header='Wavelength\tn\tk', comments='; ')
    print(f"Archivo guardado: {archivo_salida_nk}")
   
#%%
# Ejemplo de uso (basado en tu archivo silicon.in3):
# in3_a_nk('silicon.in3', 'silicon_nuevo.nk', skip_n=3, filas_n=259, skip_k=262, filas_k=321)
ruta_in3 = r'D:\archivos\TESIS\optical-master\n\TiO2Palik.in3'
archivo_salida = "TiO2Palik.nk"

in3_a_nk_interpolado(ruta_in3,archivo_salida,3,71,75,8)
#%%