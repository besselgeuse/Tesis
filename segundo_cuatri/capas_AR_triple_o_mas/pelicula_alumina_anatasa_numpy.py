"""
14 de agosto de 2026.
voy a recalcular la reflectancia de la película de al2o3/anatasa/si con numpy usando la definicion de tmm.
"""
#%%
import sys
#import os
import numpy as np
import matplotlib.pyplot as plt
import ast

Dirección_tmm_Rodrigo = r'../'

sys.path.append(f'{Dirección_tmm_Rodrigo}')

from tmm_utils_Rodrigo import (load_interp, RT_with_cache, load_fn, plot_js,
                               cauchy_fn, constant_fn, brugg_fn, stack2tmm,
                               calculate_RT_torch)
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
    'Al2O3': (load_interp(f'{ruta_materiales}/nkdata/optical/Al2O3.nkv',skiprows=1)),
    'SiO2': (load_interp(f'{ruta_materiales}/nkdata/optical/SiO2.nkv',skiprows=1)),
    'Si': (load_interp(f'{ruta_materiales}/nkdata/Silicio_Palik.nk',skiprows=1)),
    'anatasa': (load_interp(f'{ruta_materiales}/nkdata/anatase.nkv',skiprows=1)),
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#%%
stack_AA = [
                  [np.inf, 'air', 'i'],
                  [64,'Al2O3','c'],
                  [39,'anatasa','c'],
                  [np.inf, 'Si', 'i'],
    ]
#%%
# Ahora defino las constantes para calcular la Js_max
step = 1.0
lams = np.arange(300, 901, step)
thicks = {
    1:np.arange(10,100,1),
    2:np.arange(10,100,1)
}
 
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt',delimiter=",")(lams)/100

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams
jmax_am0 = np.trapezoid(am0*weight, dx=step)
jmax_am15 = np.trapezoid(am15*weight, dx=step)
#%%
File = r'../Files/R_AA_numpy.txt'
RT = RT_with_cache(None,stack_AA, materials, lams, pol='s', th_0=0.0, thicks=thicks)
ref, trans = RT
jsc_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
#%%
# Ahora grafico la R_max en funcion de lambda
maxj = jsc_am0.max()
maxind = np.unravel_index(jsc_am0.argmax(),jsc_am0.shape)
print(maxj)
print(maxind)
#%%
R_max = ref[maxind[1],maxind[0]] #estan invertidos por culpa del meshgrid en js_plot
#print(R_max)
print(f"{thicks[1][maxind[1]]}, grosor del medio 1") #Al2O3
print(f"{thicks[2][maxind[0]]}, grosor del medio 2") #rutilo


plt.figure()
plt.plot(lams,R_max,'o-',ms=2,color="green",label="reflectancia optimizada")
plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
plt.ylabel("reflectancia R")
plt.grid()
plt.legend()
plt.show()
plt.show()
#%%
"""
Hasta ahora se que el Silicio no es el problema, probablemente sea el archivo de anatasa
ya que la simulación de anatasa y rutilo me da distinto a la que conseguí en CNEA.

actualización 20/8
Parece ser que lo solucioné, el problema si era el Silicio despues de todo.
Se debía a estar usando el archivo de filmetrix en lugar del del Palik, es raro
puesto que los habia graficado y eran iguales.
"""
#%%
lams = np.arange(300, 901, step)
Si_Palik = load_interp(f'{ruta_materiales}/nkdata/Silicio_Palik.nk',unit='um',skiprows=1)[0](lams)
Si_Filmetrix = load_interp(f'{ruta_materiales}/nkdata/optical/Si.nkv',skiprows=1)[0](lams)
 
plt.figure()
plt.plot(lams,Si_Palik,'o-',ms=4,color="green",label="Si Palik")
plt.plot(lams,Si_Filmetrix,'o-',ms=2,color="red",label="Si Filmetrix")
plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
plt.ylabel("reflectancia R")
plt.grid()
plt.legend()
plt.show()
""""
PEOR AUN, UN HORROR.
Todo este tiempo estaba usando el Si_Palik pero con la unit='nm' y debería ser 'um'.
Corrigiendo esto efectivamente el Si_Palik es igual al del Filmetrix por lo que mis ajustes originales
estaban correctos.
"""