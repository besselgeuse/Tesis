"""
14 de abril de 2026
voy a ajustar los espesores de una celda de TiO2 con y sin nanotubos, encima
de la misma voy a poner alumina con un 50% de porosidad.
"""
#%%
import sys
import os
import re
import ast
import numpy as np
import torch
import matplotlib.pyplot as plt

Dirección_tmm_Rodrigo = r'../'
if Dirección_tmm_Rodrigo not in sys.path:
    sys.path.append(Dirección_tmm_Rodrigo)

from tmm_utils_Rodrigo import (
    load_interp, autorange_fit_ellipsometry_torch, load_fn, plot_js,
    cauchy_fn, constant_fn, brugg_fn, stack2tmm, calculate_RT_torch, transform_I_to_psi_delta,
    load_interp_in3, leer_datos_guardados, interpolar_todo, guardar_resultados_txt
)
#%%

modelo_T1 = cauchy_fn(2.338, 1.906, 0.824)   # muestra T1 del paper (R-1)
f_T1 = 0.58

ruta_materiales = f'{Dirección_tmm_Rodrigo}/indices'

materials = {
    'air': (constant_fn(1.0), constant_fn(0.0)),
    'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows=1)),
    'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows=1)),
    'T1_densa': (modelo_T1, constant_fn(0.0)),
    'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
    'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)),
    'MgF2': (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv', skiprows=1)),
    'vidrio': (load_interp(f'{ruta_materiales}/glass_thales.nk', skiprows=1)),
    'Al2O3': (load_interp(f'{ruta_materiales}/nkdata/optical/Al2O3.nkv', skiprows=1)),
    'SiO2': (load_interp(f'{ruta_materiales}/nkdata/optical/SiO2.nkv', skiprows=1)),
    'Si': (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nkv', skiprows=1)),
    'anatasa': (load_interp(f'{ruta_materiales}/nkdata/anatase.nkv',unit='um', skiprows=1)),
    'aluminio': (load_interp_in3(f'{ruta_materiales}/nkdata/optical/aluminum.in3', unit='A'))
}

materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
materials['Al2O3_porosa'] = (brugg_fn(materials['Al2O3'][0], constant_fn(1.0), 0.5), materials['Al2O3'][1])
#%%
stack_HLHL = [
                  [np.inf, 'air', 'i'],
                  [(0,100),'Al2O3_porosa','c'],
                  [(0,100),'T1_porosa','c'],
                  [(0,100),'T1_densa','c'],
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
thickness, R_curve = calculate_RT_torch(stack_HLHL,
                                    materials,
                                    lams,
                                    weights=weight_am0, pol='s',
                                    th_0=0.0,
                                    num_starts=2000, 
                                    num_epochs=150, 
                                    lr=1.0, 
                                    use_cuda=True)
# %%
jmax_am0 = np.trapezoid(weight_am0, dx=step)
jsc = jmax_am0 - np.trapezoid(R_curve*weight_am0, dx=step)

print(f'grosores: {thickness}')
print(f'corriente: {jsc} mA/cm2')

#%%
import matplotlib.pyplot as plt
plt.plot(lams,R_curve)
plt.grid()
plt.legend()
plt.show()
# %%
def guardar_curva_R(lams,R_curve,path):

    with open(path,'w') as f:
        f.write(f'grosores: {thickness}\n')
        f.write(f'corriente: {jsc} mA/cm2\n')
        f.write('wavelength\tReflectance\n')
        for i in range(len(lams)):
            f.write(f'{lams[i]}\t{R_curve[i]}\n')
    return None
#%%
guardar_curva_R(lams,R_curve,f'../Files/reflectancias_optimas/R_curve_Al2O3_porosa_T1_porosa_T1_densa_Si.txt')
    
# %%
