#%%
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tmm_utils_Rodrigo import load_fn,calculate_RT,cauchy_fn,brugg_fn,load_interp
#%%
ruta_materiales='./indices'
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
modelo_TiO2 = cauchy_fn(2.274,2.781,7.359) #muestra numero 2 de TiO2 fabricada por sputtering
f_T1 = 0.58
materials = {
        'air': lambda: (constant_fn(1.0), constant_fn(0.0)),
        'SiO2': lambda: load_interp(f'{ruta_materiales}/nkdata/SiO2.nkv', skiprows=1),
        'Si': lambda: load_interp(f'{ruta_materiales}/nkdata/optical/Si.nk', skiprows=1),
    }
for i in np.arange(0,1,0.1):
    materials[f'TiO2_poroso_{i}'] = lambda: (brugg_fn(modelo_TiO2, constant_fn(1.0), i), load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)[1])
    materials[f'SiO2_poroso_{i}'] = lambda: (brugg_fn(materials['SiO2'], constant_fn(1.0), i), materials['SiO2'][1])
#%%
# Primero una lista que va a tener todas las combinaciones de stacks posibles
# identificadas con un numero de id
stacks = []
espesores = np.arange(10,200,10)
n_o = 0
for i in materials.keys():
    for j in materials.keys():
        stacks.append([
            [np.inf,'air','i'],
            [,i,'c'],
            [,j,'c'],
            [np.inf,'Si','i']
        ]
            )
#%%