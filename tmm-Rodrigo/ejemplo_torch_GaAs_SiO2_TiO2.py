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
    load_interp, cauchy_fn, constant_fn, brugg_fn, calculate_RT_torch,load_fn
)
#%%
# ============================================================
# 1. DEFINICIÓN DE MATERIALES
# ============================================================
ruta_materiales = r'.'

# Primero el Ti (para el que usaron valores experimentales)
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
f_T1 = 0.58

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
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1))
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])
#%%
# ============================================================
# 2. DEFINICIÓN DEL RANGO ESPECTRAL Y CONSTANTES
# ============================================================
step = 1.0
lams_np = np.arange(300, 901, step)  # 300-900 nm
am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams_np)
am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams_np)
e = 1.60218e-19 # A.s
h = 6.6226E-34 # J·s
c = 2.9979e17 # nm/s

const = e/(h*c)
sustrato = "GaAs"
if sustrato == "GaAs":
    IQE = load_fn(f'{ruta_materiales}/IQEGaAs2.txt',delimiter=" ")(lams_np)
else:
    IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt',delimiter=",")(lams_np)/100
weight = const*IQE*lams_np
jmax_am15 = np.trapezoid(am15*weight, dx=step)
jmax_am0 = np.trapezoid(am0*weight, dx=step)
print(jmax_am0)
print(jmax_am15)
#%%
# ============================================================
# 3. CONSTRUIR n_list COMO TENSOR DE PYTORCH
# ============================================================
# Stack óptico: air / Rutilo(TiO2) / SiO2 / GaAs
# Orden de capas: [superestrato, capa1, capa2, sustrato]
layer_names = ['air','MgF2','vidrio','T1_porosa', 'T1_densa','InGaP', 'GaAs']

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
print("Stack: air / MgF2 / vidrio / T1_porosa / T1_densa / InGaP / GaAs")
print(f"Rango espectral: {lams_np[0]:.0f} - {lams_np[-1]:.0f} nm ({len(lams_np)} puntos)")
print(f"Forma de n_list: {n_list_torch.shape}")
print("=" * 60)
#%%
# ============================================================
# 4. DEFINIR LÍMITES DE ESPESORES Y EJECUTAR OPTIMIZACIÓN
# ============================================================
# Límites para las capas finitas [min_nm, max_nm]:

d_bounds = [
    (10.0, 100.0),   # MgF2
    (300000.0, 300000.0),   # vidrio
    (10.0, 120.0),   # T1 porosa
    (10.0, 100.0),   # T1 densa
    (10.0, 100.0),   # InGaP
]

# air / MgF2 / vidrio / T1_porosa / T1_densa / InGaP / GaAs
c_list = ['i', 'c', 'i', 'c', 'c', 'c', 'i']

print("\nLímites de espesores:")
print(f"  MgF2: {d_bounds[0][0]:.0f} - {d_bounds[0][1]:.0f} nm")
print(f"  vidrio: {d_bounds[1][0]:.0f} - {d_bounds[1][1]:.0f} nm")
print(f"  T1 denso: {d_bounds[2][0]:.0f} - {d_bounds[2][1]:.0f} nm")
print(f"  T1 poroso: {d_bounds[3][0]:.0f} - {d_bounds[3][1]:.0f} nm")
print(f"  InGaP: {d_bounds[4][0]:.0f} - {d_bounds[4][1]:.0f} nm")


# Definir el vector de pesos (flujo de fotones útiles AM1.5g * IQE) en PyTorch
# Esto orientará al optimizador a maximizar Jsc en lugar de reflectancia cruda
weights_torch = torch.tensor(weight * am0, dtype=torch.float64)

# Ejecutar la optimización
best_thicknesses, best_R_curve = calculate_RT_torch(
    n_list=n_list_torch,
    d_bounds=d_bounds,
    lams=lams_torch,
    c_list=c_list,
    weights=weights_torch, # <--- Pasamos el vector de pesos
    pol='s',
    th_0=0.0,
    num_starts=500,     # 500 semillas para balancear velocidad/exploración
    num_epochs=200,    # 200 épocas
    lr=1.5,            # Learning rate
    use_cuda=True      # Usar CUDA
)
#%%
# ============================================================
# 5. GRAFICAR RESULTADOS
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(lams_np, best_R_curve * 100, 'b-', linewidth=2, 
         label=f'Torch Autograd: MgF2={best_thicknesses[0]:.1f} nm, vidrio={best_thicknesses[1]:.1f} nm, T1_densa={best_thicknesses[2]:.1f} nm, T1_porosa={best_thicknesses[3]:.1f} nm, InGaP={best_thicknesses[4]:.1f} nm')
plt.xlabel(r'Longitud de onda $\lambda$ [nm]', fontsize=12)
plt.ylabel('Reflectancia R [%]', fontsize=12)
plt.title('Optimización de antirreflejo MgF2/vidrio/T1_porosa/T1_densa/InGaP sobre GaAs\n(PyTorch Autograd)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(300, 900)
plt.ylim(0, None)
plt.tight_layout()
#plt.savefig('./Files/reflectancia_optimizada_torch.png', dpi=150)
plt.show()

print(f"\nGráfico guardado en: ./Files/reflectancia_optimizada_torch.png")

# %%
# 6. CALCULAR Js
j_opt_am15 = jmax_am15 - np.trapezoid(best_R_curve * am15 * weight, dx=step)
j_opt_am0 = jmax_am0 - np.trapezoid(best_R_curve * am0 * weight, dx=step)
print(f"\nCorriente de cortocircuito optimizada para am15: Jsc = {j_opt_am15:.4f} mA/cm²")
print(f"\nCorriente de cortocircuito optimizada para am0: Jsc = {j_opt_am0:.4f} mA/cm²")

# Gráfica final
plt.figure(figsize=(10, 6))
plt.plot(lams_np, best_R_curve * 100, 'b-', linewidth=2, 
         label=f'Torch Autograd: Jsc={j_opt_am15:.3f} mA/cm²')
plt.xlabel(r'Longitud de onda $\lambda$ [nm]', fontsize=12)
plt.ylabel('Reflectancia R [%]', fontsize=12)
plt.title('Antirreflejo optimizado /MgF2/vidrio/T1_densa/T1_porosa/InGaP sobre GaAs (PyTorch)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(300, 900)
plt.ylim(0, None)
plt.tight_layout()
plt.show()

# %%
