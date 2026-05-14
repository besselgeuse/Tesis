# 14 de abril, en este codigo voy a realizar las simulaciónes de una capa antirreflectante de TiO2 sobre
# un sustrato de Si
#%%
import sys
#import os
import numpy as np
import matplotlib.pyplot as plt
import ast

Dirección_tmm_Rodrigo = r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\tmm-Rodrigo'

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
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1)),
    'Si': (load_interp(f'{ruta_materiales}/Silicio_Palik.nk',skiprows=1))
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#%%
stack_T1_Si = [
                  [np.inf, 'air', 'i'],
                  [50, 'T1_densa', 'c'],
                  [np.inf, 'Si', 'i'],
    ]
# %%
# Ahora defino las constantes para calcular la Js_max
step = 1.0
lams = np.arange(300, 901, step)
 
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
IQE = load_fn(f'{ruta_materiales}/IQEGaAs2.txt')(lams)

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams

# 8/4 cambie la funcion trapezoid por trapezoid
jmax_am0 = np.trapezoid(am0*weight, dx=step)
jmax_am15 = np.trapezoid(am15*weight, dx=step)

#%%
print('TiO2 sobre Si')

thicks_T1_Si = {1:np.arange(10,151,1),
                   }
# file = './transfermatrix/RAT_tio2/T1_air_20-120_20-100_300_900.pickle'
file = './Files/TiO2_Si_encap_300-900.pickle'
RT = RT_with_cache(file,stack_T1_Si, materials, lams, thicks=thicks_T1_Si)
ref, trans = RT

#%%
js_T1_Si_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
js_T1_Si_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T

T_js_T1_Si_am0 =  np.trapezoid(trans*am0*weight, dx=step).T
T_js_T1_Si_am15 =  np.trapezoid(trans*am15*weight, dx=step).T

espesores_simon = [87,54]

plot_js('T1 encapsulada. AM0. (1-R)', thicks_T1_Si[3], thicks_T1_Si[4], 
        js_T1_Si_am0,espesores_simon,"optimo Simon")
plot_js('T1 encapsulada. AM10. (T)', thicks_T1_Si[3], thicks_T1_Si[4], 
        T_js_T1_Si_am0,espesores_simon,"optimo Simon")
#%%
# Ahora grafico la R_max en funcion de lambda
maxj = js_T1_Si_am0.max()
maxind = np.unravel_index(js_T1_Si_am0.argmax(),js_T1_Si_am0.shape)
print(maxind)
#%%
R_max = ref[maxind[1],maxind[0]] #estan invertidos por culpa del meshgrid en js_plot
#print(R_max)
print(f"{thicks_T1_Si[3][maxind[1]]}, grosor del medio 1") #poroso
print(f"{thicks_T1_Si[4][maxind[0]]}, grosor del medio 2") #denso


plt.figure()
plt.plot(lams,R_max,'o-',ms=2,color="green",label="reflectancia optimizada")
plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
plt.ylabel("reflectancia R")
plt.grid()
plt.legend()
plt.show()
