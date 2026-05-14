# 13 de abril, en este codigo voy a realizar las simulaciónes de la celda R1 encapsulada despues de haber contrastado las simulaciones sin encapsular
# contra las del optical y obtenido resultados identicos
#%%
import sys
#import os
import numpy as np
import matplotlib.pyplot as plt
import ast

Dirección_tmm_Rodrigo = r'./'

sys.path.append(f'{Dirección_tmm_Rodrigo}')

from tmm_utils_Rodrigo import (load_interp, RT_with_cache, load_fn, plot_js,
                               cauchy_fn, constant_fn, brugg_fn, stack2tmm,
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
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1))
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#%%
stack_T1_glass = [
                  [np.inf, 'air', 'i'],
                  [100, 'MgF2', 'c'],
                  [300_000, 'vidrio', 'i'],
                  [50, 'T1_porosa', 'c'],
                  [50, 'T1_densa', 'c'],
                  [25, 'InGaP', 'c'],
                  [np.inf, 'GaAs', 'i'],
    ]

# %%
# Ahora defino las constantes para calcular la Js_max
step = 1.0
lams = np.arange(300, 901, step)
 
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
IQE = load_fn(f'{ruta_materiales}/IQEGaAs2.txt',delimiter=" ")(lams)

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams

# 8/4 cambie la funcion trapezoid por trapezoid
jmax_am0 = np.trapezoid(am0*weight, dx=step)
jmax_am15 = np.trapezoid(am15*weight, dx=step)

#%%
print('T1 encapsulado')

thicks_T1_glass = {3:np.arange(10,151,1),
                   4:np.arange(20,101,1),
                   }

# file = './transfermatrix/RAT_tio2/T1_air_20-120_20-100_300_900.pickle'
file = './Files/GaAs_R1_encap_300-900.pickle'
RT = RT_with_cache(file,stack_T1_glass, materials, lams, thicks=thicks_T1_glass)
ref, trans = RT

#%%
js_T1_glass_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
js_T1_glass_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T

T_js_T1_glass_am0 =  np.trapezoid(trans*am0*weight, dx=step).T
T_js_T1_glass_am15 =  np.trapezoid(trans*am15*weight, dx=step).T

espesores_simon = [87,54]

plot_js('T1 encapsulada. AM0. (1-R)', thicks_T1_glass[3], thicks_T1_glass[4], 
        js_T1_glass_am0,espesores_simon,"optimo Simon")
plot_js('T1 encapsulada. AM10. (T)', thicks_T1_glass[3], thicks_T1_glass[4], 
        T_js_T1_glass_am0,espesores_simon,"optimo Simon")
#%%
# Ahora grafico la R_max en funcion de lambda
maxj = js_T1_glass_am0.max()
maxind = np.unravel_index(js_T1_glass_am0.argmax(),js_T1_glass_am0.shape)
print(maxind)
#%%
R_max = ref[maxind[1],maxind[0]] #estan invertidos por culpa del meshgrid en js_plot
#print(R_max)
print(f"{thicks_T1_glass[3][maxind[1]]}, grosor del medio 1") #poroso
print(f"{thicks_T1_glass[4][maxind[0]]}, grosor del medio 2") #denso


plt.figure()
plt.plot(lams,R_max,'o-',ms=2,color="green",label="reflectancia optimizada")
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
np.savetxt('Reflectancia_Optima_encap.txt', datos_a_guardar, 
           header='Wavelength(nm) Reflectancia', 
           fmt='%.4f', 
           delimiter='\t')
#%%
# Ahora comparo con los datos del optical
files = r'./Files'
datos_tmm = np.loadtxt(f"{files}/Reflectancia_optima_encap.txt", dtype=float, unpack=True, skiprows=1)
datos_optical = np.loadtxt(f"{files}/RT_R1_encap_optical_nVidrio.txt", dtype=float, unpack=True, skiprows=1)

#defino las variables para graficar
lambdas_tmm = datos_tmm[0]
R_tmm = datos_tmm[1]
lambdas_optical = datos_optical[0]
R_optical = datos_optical[1]/100

plt.figure()
plt.plot(lambdas_tmm,R_tmm,'o-',ms=4,label="Reflectancia con tmm",color="Blue")
plt.plot(lambdas_optical,R_optical,label="Reflectancia con optical",color="Orange")
plt.grid()
plt.legend()
plt.xlabel(r"Longitud de onda ($\lambda$) [nm]")
plt.ylabel("Reflectancia")
plt.show()
# %%
# En esta parte quiero comparar las Jsc conseguida con mis parametros optimizados
# y los de Simon, para eso voy a hallar los indices correspondientes a sus espesores
# tabulados

indice_poroso = np.where(thicks_T1_glass[3] == espesores_simon[0])[0]
indice_denso = np.where(thicks_T1_glass[4] == espesores_simon[1])[0]

print(fr"J_sc de Simon: {js_T1_glass_am0[indice_denso, indice_poroso]}")
print(fr"J_sc mia: {js_T1_glass_am0[maxind]}")
#%%