#%%
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

Dirección_tmm_Rodrigo = r'./'

# Crear carpeta del día actual
today_str = datetime.now().strftime('%Y/%m/%d')
files_dir = f'./Files/{today_str}'
os.makedirs(files_dir, exist_ok=True)

sys.path.append(f'{Dirección_tmm_Rodrigo}')

from tmm_utils_Rodrigo import (load_interp, RT_with_cache, load_fn, plot_js,
                               cauchy_fn, constant_fn, brugg_fn, stack2tmm, load_interp_in3,n_eff
                               )
# %%
# Primero el Ti (para el que usaron valores experimentales)
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
f_T1 = 0.58

ruta_materiales = f'{Dirección_tmm_Rodrigo}\indices'

materials = {
    'air':(constant_fn(1.0),constant_fn(0.0)),
    'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows = 1)),
    'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows = 1)),
    'T1_densa': (modelo_T1, constant_fn(0.0)),
    'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
    'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows = 1)),
    'MgF2' : (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv',skiprows=1)),
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1)),
    'SiO2' : (load_interp(f'{ruta_materiales}/nkdata/SiO2_Palik.nk',skiprows=1,unit='um')),
    'Si' : (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nk', skiprows = 1)),
    'Aluminum' : (load_interp_in3(f'{ruta_materiales}/nkdata/optical/Aluminum.in3', unit='A')),
    'T1_densa_in3' : (load_interp_in3(f'{ruta_materiales}/nkdata/optical/T1_densa.in3', unit='A')),
    'GaAs_Palik_in3' : (load_interp_in3(f'{ruta_materiales}/nkdata/optical/GaAs_Palik.in3', unit='A')),
}
n_sio2 = materials['SiO2'][0]
n_rutilo = materials['Rutilo'][0]
k_sio2 = materials['SiO2'][1]
k_rutilo = materials['Rutilo'][1]

# 2. Creamos la función de índice de refracción efectivo
n_mezcla_sio2_tio2 = brugg_fn(n_sio2, n_rutilo, 0.5)
k_mezcla_sio2_tio2 = brugg_fn(k_sio2,k_rutilo,0.5)
materials['SiO2_TiO2_brugge'] = (n_mezcla_sio2_tio2,k_mezcla_sio2_tio2)
# # Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#asigno el TiO2 rutilo de la tesis de Marcela
#materials['T1_densa'] = (load_interp(f'{ruta_materiales}/TiO2Palik_Marcela.nk',skiprows=1))
materials['Rutilo_poroso'] = (brugg_fn(n_rutilo,constant_fn(1.0),0.5),brugg_fn(k_rutilo,constant_fn(1.0),0.5))


#stack optico de la celda
sustrato = "GaAs"

if sustrato == "GaAs":
    stack_SiO2_TiO2 = [
                    [np.inf, 'air', 'i'],
                    [50, 'Rutilo', 'c'],
                    [25, 'SiO2', 'c'],
                    [np.inf, 'GaAs', 'i'],
        ]
else:
    stack_SiO2_TiO2 = [
                    [np.inf, 'air', 'i'],
                    #[1,'Rutilo_poroso','c'],
                    [31, 'Rutilo', 'c'],
                    #[1,'SiO2_TiO2_brugge','c'],
                    [12, 'SiO2', 'c'],
                    [np.inf, 'Si', 'i'],
        ]


# %%
# Ahora defino las constantes para calcular la Js_max
step = 1.0
lams = np.arange(300, 901, step)
 
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)

if sustrato == "GaAs":
    IQE = load_fn(f'{ruta_materiales}/IQEGaAs2.txt',delimiter=" ")(lams)
else:
    IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt',delimiter=",")(lams)/100

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams

# 8/4 cambie la funcion trapz por trapezoid
jmax_am0 = np.trapezoid(am0*weight, dx=step)
jmax_am15 = np.trapezoid(am15*weight, dx=step)
print(jmax_am0)
print(jmax_am15)
# %%
# En esta parte calcula la R y T para todo el espectro de
# longitudes de onda definido en lams y para todas las combinaciones
# de grosores para la bicapa de TiO2 especificado en 
# thicks_SiO2_TiO2
print('celda')

thicks_SiO2_TiO2 = {#1:np.arange(0.1,1,0.1),#indice superior
                   1:np.arange(10,121,1),
                   #3:np.arange(0.1,1,0.1),
                   2:np.arange(10,101,1),
                   }
# file = './transfermatrix/RAT_tio2/T1_air_20-120_20-100_300_900.pickle'
#file = './Files/GaAs_R1_300-900nm_14/4.pickle'
file = f'{files_dir}/RT_Si_SiO2_Rutilo_0º_pol_p.txt'
#file = f'./Files/2026/04/30/RT_Si_SiO2_Rutilo.txt'
RT = RT_with_cache(file,stack_SiO2_TiO2, materials, lams,pol='p',th_0=0, thicks=thicks_SiO2_TiO2)
ref, trans = RT
# El codigo de tmm_utils no me esta funcionando porque lams es un array
# y la función calculate_RT que a su vez usa tmm.coh_tmm espera un unico
# valor de lambda, entonces lo que voy a hacer es cambiar la función
# calculate_RT para agregar un ciclo for en lams y solucionar este problema

#%%
js_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
js_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T

T_js_am0 =  np.trapezoid(trans*am0*weight, dx=step).T


plot_js('T1 no encapsulada. AM0. (1-R)', thicks_SiO2_TiO2[1], thicks_SiO2_TiO2[2], 
        js_am0)
plot_js('T1 no encapsulada. AM0. (T)', thicks_SiO2_TiO2[1], thicks_SiO2_TiO2[2], 
        T_js_am0)

#%%
# Ahora grafico la R_max en funcion de lambda
maxj = js_am0.max()
maxind = np.unravel_index(js_am0.argmax(),js_am0.shape)
#print(maxind)
R_max = ref[maxind[1],maxind[0]]#[indice_sup,indice_inf]
R_especifica = ref[25,0]
#print(R_max)
print(f"{thicks_SiO2_TiO2[1][maxind[1]]}, grosor del medio 1") #superior
print(f"{thicks_SiO2_TiO2[2][maxind[0]]}, grosor del medio 2") #inferior


plt.figure()
plt.plot(lams,R_max,'o-',ms=2,color="green",label="reflectancia optimizada")
#plt.plot(lams,R_max_negativa,'o-',ms=2,color="red",label="reflectancia optimizada invertida")
plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
plt.ylabel("reflectancia R")
plt.grid()
plt.legend()
plt.show()
#%%
# Ahora guardo estos datos
# Apilamos las variables como columnas
datos_a_guardar = np.column_stack((lams, R_max))

# Guardamos en un archivo .txt
np.savetxt(f'{files_dir}/Reflectancia_Optima_Si_SiO2_Rutilo_tmm.txt', datos_a_guardar, 
           header='Wavelength(nm) Reflectancia', 
           fmt='%.4f', 
           delimiter='\t')

#%%
# Ahora comparo con los datos del optical
# NOTA: Si da error al cargar el archivo de optical, es porque se movió a otra carpeta de fecha.
# Puedes copiarlo manualmente a la carpeta de hoy, o cambiar la ruta a la fecha correspondiente.
datos_tmm = np.loadtxt(f"./Files/2026/04/30/Reflectancia_Optima_Si_SiO2_Rutilo_tmm.txt", dtype=float, unpack=True, skiprows=1)
try:
    datos_optical = np.loadtxt(f"./Files/2026/04/30/RT_Si_SiO2_Rutilo_optical.txt", dtype=float, unpack=True, skiprows=4)
except FileNotFoundError:
    print(f"ADVERTENCIA: No se encontró RT_Si_SiO2_Rutilo_optical.txt en {files_dir}.")
    print("Por favor, copia el archivo a esa carpeta o ajusta la ruta.")
    datos_optical = np.array([[], []])

#defino las variables para graficar
lambdas_tmm = datos_tmm[0]
R_tmm = datos_tmm[1]
lambdas_optical = datos_optical[0]
R_optical = datos_optical[1]/100

datos_tesis_Marcela = np.loadtxt("./Files/2026/05/05/Reflectancia_Si_SiO2(12.5)_TiO2(31.5).csv",delimiter=",",skiprows=1)

plt.figure()
#plt.plot(lambdas_tmm,R_tmm,'o-',ms=4,label="Reflectancia optimizada con tmm-s",color="Blue")
#plt.plot(lambdas_tmm,R_especifica,'o-',ms=2,color="green",label="Reflectancia espesores Marcela tmm-s")
plt.plot(lams,ref,'-',ms=2,color="green",label="Reflectancia espesores Marcela tmm-s")
plt.plot(datos_tesis_Marcela[:,0],datos_tesis_Marcela[:,1]/100,'o-',ms=2,color="blue",label="Reflectancia fig 3 cap 4 Tesis de Marcela")
#if len(datos_optical) > 0:
    #plt.plot(lambdas_optical,R_optical,label="Reflectancia con optical (espesores Marcela)",color="Orange")
plt.grid()
plt.legend()
plt.xlabel(r"Longitud de onda ($\lambda$) [nm]")
plt.ylabel("Reflectancia")
plt.show()

#%%


#%%
# En esta parte quiero comparar las Jsc conseguida con mis parametros optimizados
# y los de Simon, para eso voy a hallar los indices correspondientes a sus espesores
# tabulados
indice_superior = 25
indice_inferior = 0
print(fr"J_sc Marcela: {js_am0[indice_inferior, indice_superior]}")
print(fr"J_sc optima: {js_am0[maxind]}")


# %%
# Aca voy a queres comparar los archivos .in3 del T1_densa con los del modelo experimental para corroborar que la función load_interp_in3
# funciona correctamente
lams = np.arange(300, 901, 1)
T1_densa_in3 = materials['T1_densa_in3'][0](lams)
T1_densa_exp = materials['T1_densa'][0](lams)

plt.figure()
plt.plot(lams,T1_densa_in3,'o-',ms=4,label="T1_densa_in3",color="Blue")
plt.plot(lams,T1_densa_exp,'o-',ms=4,label="T1_densa_exp",color="Orange")
plt.grid()
plt.legend()
plt.show()
# %%
#Ahora una segunda prueba para el GaAs_Palik

lams = np.arange(300, 901, 1)
GaAs_Palik_in3 = materials['GaAs_Palik_in3'][0](lams)
GaAs_Palik_exp = materials['GaAs'][0](lams)

plt.figure()
plt.plot(lams,GaAs_Palik_in3,'o-',ms=4,label="GaAs_Palik_in3",color="Blue")
plt.plot(lams,GaAs_Palik_exp,'o-',ms=4,label="GaAs_Palik",color="Orange")
plt.grid()
plt.legend()
plt.show()
#Todo ok
# %%
#aca voy a calcular la diferencia en cuadrados minimos entre 2 curvas de datos
import numpy as np
from scipy.interpolate import interp1d

def indices_minima_4d(y_datos, matriz_simulada_4d):
    """
    Calcula el error cuadrático medio (MSE) entre un vector y una matriz de 4 dimensiones,
    devolviendo el valor mínimo del MSE y los índices (i, j, k, l).
    
    Parámetros:
    -----------
    y_datos : array_like
        Curva de datos experimental/objetivo (1D).
    matriz_simulada_4d : ndarray
        Matriz de datos simulada con dimensiones (N1, N2, N3, N4, long_onda).
    """
    # 1. Calculamos la diferencia y el MSE a lo largo del último eje (longitud de onda)
    mse_matrix = np.mean((matriz_simulada_4d - y_datos) ** 2, axis=-1)
    
    # 2. Obtenemos el índice plano del valor mínimo en el arreglo 4D
    min_index_flat = np.argmin(mse_matrix)
    
    # 3. Convertimos el índice plano a coordenadas (i, j, k, l)
    i, j, k, l = np.unravel_index(min_index_flat, mse_matrix.shape)
    
    # 4. Obtenemos el valor mínimo del error (MSE)
    min_error = mse_matrix[i, j, k, l]
    
    return min_error, i, j, k, l
# %%
datos_tesis_Marcela = np.loadtxt("./Files/2026/05/05/Reflectancia_Si_SiO2(12.5)_TiO2(31.5).csv",delimiter=",",skiprows=1)
datos_tesis_Marcela_interpolados = interp1d(
    datos_tesis_Marcela[:, 0], 
    datos_tesis_Marcela[:, 1], 
    bounds_error=False, 
    fill_value="extrapolate" # Opcional, o puedes rellenar con el valor del borde: fill_value=(datos_tesis_Marcela[0, 1], datos_tesis_Marcela[-1, 1])
)(lams)/100
#%%
min_error, i, j, k, l = indices_minima_4d(datos_tesis_Marcela_interpolados, ref)
indices = (i,j,k,l)
print(indices)
plt.figure()
plt.plot(lams,ref[indices],label="reflectancia mas cercana en mse")
plt.plot(lams,datos_tesis_Marcela_interpolados,label="reflectancia medida")
plt.grid()
plt.legend()
plt.xlabel("longitud de onda")
plt.ylabel("reflectancia")
# %%

