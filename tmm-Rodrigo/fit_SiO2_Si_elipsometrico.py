#%%
# -*- coding: utf-8 -*-
"""
Ejemplo de uso de calculate_RT_torch para optimizar espesores
de un antirreflejo SiO2/TiO2 sobre GaAs usando PyTorch Autograd.

Este script:
1. Carga los materiales GaAs, SiO2, TiO2 
2. Construye la lista de índices de refracción como tensores de PyTorch
3. Ejecuta calculate_RT_torch para encontrar espesores óptimos
4. Grafica la reflectancia optimizada vs longitud de onda
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Importar utilidades del TMM
sys.path.append('./')
from tmm_utils_Rodrigo import (
    load_interp, cauchy_fn, constant_fn, brugg_fn, calculate_RT_torch, load_fn, fit_ellipsometry_torch
)
#%%
# ============================================================
# 1. DEFINICIÓN DE MATERIALES
# ============================================================
ruta_materiales = r'.'

# Primero el Ti (para el que usaron valores experimentales)
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
f_T1 = 0.58
modelo_SiO2_Daniel = cauchy_fn(1.46,0.0,0.0)
ruta_materiales = f'{ruta_materiales}\indices'

materials = {
    'air':(constant_fn(1.0),constant_fn(0.0)),
    'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows = 1)),
    'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows = 1)),
    'T1_densa': (modelo_T1, constant_fn(0.0)),
    'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
    'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows = 1)),
    'SiO2' : (load_interp(f'{ruta_materiales}/nkdata/SiO2_Palik.nk',skiprows=1,unit='um')),
    'MgF2' : (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv',skiprows=1)),
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1)),
    'Si': (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nk', skiprows = 1)),
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
materials['SiO2_Daniel'] = (modelo_SiO2_Daniel, constant_fn(0.0))
materials['SiO2_Si_brugge'] = (brugg_fn(modelo_SiO2_Daniel, constant_fn(1.0), 0.5), constant_fn(0.0))
#%%
# ============================================================
# 3. CARGAR DATOS EXPERIMENTALES Y CONSTRUIR n_list
# ============================================================
# Cargar datos experimentales de elipsometría
wl_exp, psi_deg, delta_deg = np.loadtxt('./Files/2026/05/19/SiO2_Si_Sputtering.txt', skiprows=5, unpack=True)

# Convertir ángulos de grados a radianes
psi = np.radians(psi_deg)
delta = np.radians(delta_deg)

# Definir parámetros elipsométricos experimentales Is e Ic
Is_exp = np.sin(2 * psi) * np.sin(delta)
Ic_exp = np.sin(2 * psi) * np.cos(delta)

# Pasarlos a formato tensor de PyTorch
Is_exp_torch = torch.tensor(Is_exp, dtype=torch.float64)
Ic_exp_torch = torch.tensor(Ic_exp, dtype=torch.float64)

# EN LUGAR DE INTERPOLAR: Evaluamos el modelo teórico exactamente 
# en las longitudes de onda experimentales. Las funciones de interpolación 
# de los materiales (n_fn, k_fn) se encargarán del resto automáticamente.
lams_np = wl_exp

# Orden de capas: [superestrato, capa1, sustrato] (air / SiO2 / Si)
layer_names = ['air','SiO2_Si_brugge','SiO2_Daniel','Si']

n_list_np = []
for name in layer_names:
    n_fn, k_fn = materials[name]
    n_complex = n_fn(lams_np) + 1j * k_fn(lams_np)
    n_list_np.append(n_complex)

n_list_np = np.array(n_list_np)  # [num_layers, num_wl]

# Convertir a tensores de PyTorch
n_list_torch = torch.tensor(n_list_np, dtype=torch.complex128)
lams_torch = torch.tensor(lams_np, dtype=torch.float64)

print("=" * 60)
print("Stack: air / SiO2 / Si")
print(f"Rango espectral: {lams_np[0]:.0f} - {lams_np[-1]:.0f} nm ({len(lams_np)} puntos)")
print(f"Forma de n_list: {n_list_torch.shape}")
print("=" * 60)
#%%
# ============================================================
# 4. DEFINIR LÍMITES DE ESPESORES Y EJECUTAR OPTIMIZACIÓN
# ============================================================
# Límites para las capas finitas [min_nm, max_nm]:
d_bounds = [
    (0.1, 5.0),   # SiO2 Si brugge
    (0.1, 15.0),   # SiO2 Daniel
]

# air / SiO2 / Si
# air / SiO2 / Si
c_list = ['i', 'c','c', 'i']

# Modelos paramétricos de dispersión por capa (None para estáticas, 'cauchy' para SiO2)
layer_models = [None, 'bruggeman' ,'cauchy', None]

# En elipsometría, la medida NUNCA se hace a incidencia normal (0 grados)
# porque las polarizaciones s y p son degeneradas (r_p = -r_s), haciendo que
# Is_teo sea siempre 0 e Ic_teo sea siempre -1 para CUALQUIER espesor.
# El ángulo de incidencia determinado físicamente para el mejor ajuste es 71.0 grados.
angulo_incidencia_deg = 71.0 
th_0_rad = np.radians(angulo_incidencia_deg)

# Ejecutar la optimización conjunta (espesores + parámetros de dispersión)
best_thicknesses, best_Is, best_Ic, best_params = fit_ellipsometry_torch(
    n_list=n_list_torch,
    d_bounds=d_bounds,
    lams=lams_torch,
    Is_exp=Is_exp_torch,
    Ic_exp=Ic_exp_torch,
    c_list=c_list,
    th_0=th_0_rad,
    num_starts=1000,     # 2000 semillas para exploración de mínimos globales
    num_epochs=150,      # 250 épocas para convergencia fina
    lr=1.5,              # Learning rate
    use_cuda=True,       # Usar CUDA para alta velocidad en paralelo
    layer_models=layer_models,
    layer_names=layer_names
)
# %%
# ============================================================
# 5. GRAFICAR RESULTADOS
# ============================================================
fig,ax = plt.subplots(2,1,figsize=(10, 6))

# # Construir etiqueta descriptiva
sio2_label = f"SiO2={best_thicknesses[1]:.1f} nm"
# sio2_key = 'SiO2_Daniel' if 'SiO2_Daniel' in best_params else 'SiO2'
# if sio2_key in best_params:
#     sio2_p = best_params[sio2_key]
#     sio2_label += f" (A={sio2_p['A']:.3f}, B={sio2_p['B']:.3f}, C={sio2_p['C']:.3f})"

ax[0].plot(lams_np, best_Is, 'b-', linewidth=2, 
         label=f'Torch Autograd: Is ({sio2_label})')
ax[1].plot(lams_np, best_Ic, 'r-', linewidth=2, 
         label=f'Torch Autograd: Ic ({sio2_label})')
ax[0].plot(lams_np, Is_exp , 'k--', linewidth=1.5, 
         label='Is (Exp)')
ax[1].plot(lams_np, Ic_exp , 'm--', linewidth=1.5, 
         label='Ic (Exp)')
ax[0].set_xlabel(r'Longitud de onda $\lambda$ [nm]', fontsize=12)
ax[0].set_ylabel('I', fontsize=12)
ax[0].set_title('Fit elipsometrico de SiO2/Si', fontsize=14)
ax[0].grid(True, alpha=0.3)
ax[0].legend(fontsize=11)
ax[0].set_xlim(450, 900)
ax[0].set_ylim(0, None)
ax[1].set_xlabel(r'Longitud de onda $\lambda$ [nm]', fontsize=12)
ax[1].set_ylabel('I', fontsize=12)
ax[1].set_title('Fit elipsometrico de SiO2/Si', fontsize=14)
ax[1].grid(True, alpha=0.3)
ax[1].legend(fontsize=11)
ax[1].set_xlim(450, 900)
#ax[1].set_ylim(0, None)

fig.tight_layout()
#plt.savefig('./Files/reflectancia_optimizada_torch.png', dpi=150)
plt.show()
# %%
#Ahora vamos a ver si el índice de SiO2 que nos dio es coherente con el índice de SiO2 de Palik 
#Vamos a graficar n y k de SiO2 de Palik y el índice de SiO2 que nos dio el fit
A = best_params['SiO2_Daniel']['A']
B = best_params['SiO2_Daniel']['B']
C = best_params['SiO2_Daniel']['C']

SiO2_Daniel_n = cauchy_fn(A,B,C)(lams_np)
SiO2_Palik_n = materials['SiO2'][0](lams_np)

plt.figure(figsize=(10, 6))
plt.plot(lams_np, SiO2_Daniel_n, 'b-', linewidth=2, label='SiO2_Daniel')
plt.plot(lams_np, SiO2_Palik_n, 'r-', linewidth=2, label='SiO2_Palik')
plt.xlabel('Longitud de onda [nm]', fontsize=12)
plt.ylabel('Indice de refracción', fontsize=12)
plt.title('Indice de refracción de SiO2', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.show()

# %%
#comparación con los del ajuste que hizo el deltapsi2
A = 2.4159710
B = 101.5135000
C = -198.7589000
SiO2_Daniel_n = cauchy_fn(A,B,C)(lams_np)
wl_exp, psi_deg, delta_deg = np.loadtxt('./Files/2026/05/19/SiO2_Si_Sputtering.txt', skiprows=5, unpack=True)
#Ahora a partir del n quiero conseguir el psi y el delta del modelo y compararlo con el experimental

