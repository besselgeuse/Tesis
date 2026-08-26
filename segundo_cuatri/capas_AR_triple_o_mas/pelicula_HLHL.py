"""
9 de agosto de 2026.
voy a realizar un ajuste de cuatro capas AR por dos razones:
1. Quiero retomar el uso del optimizador con torch que hice ya que a diferencia del de Simon
que usa numpy, este me permite optimizar 3 o mas cantidad de capas.
2. Segun el paper que lei los sistemas de capas AR con indice high-low-high-low podrían ser
una buena capa antireflectante para el Silicio.
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
    'Si': (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nkv',skiprows=1))
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#%%
stack_HLHL = [
                  [np.inf, 'air', 'i'],
                  [(0,80),'Al2O3','c'],
                  [(0,80),'T1_densa','c'],
                  [(0,80),'Al2O3','c'],
                  [(0,80),'T1_densa','c'],
                  [np.inf, 'Si', 'i'],
    ]
#%%
# Ahora defino las constantes para calcular la Js_max
step = 1.0
lams = np.arange(300, 901, step)
 
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt',delimiter=",")(lams)/100

e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)

weight = const*IQE*lams
weight_am0 = weight * am0
weight_am15 = weight * am15
#%%
thickness, R_curve = calculate_RT_torch(stack_HLHL, materials, lams, weights=weight_am0, pol='s', th_0=0.0, num_starts=2000, num_epochs=100, lr=1.0, use_cuda=True)
# %%
print(thickness)
import matplotlib.pyplot as plt

plt.plot(lams,R_curve)
plt.show()
# %%
