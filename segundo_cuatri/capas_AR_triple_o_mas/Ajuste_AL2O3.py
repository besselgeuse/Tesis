"""
19 de agosto de 2026
En este código voy a usar la función de tmm_utils_Rodrigo autorange_fit_ellipsometry_torch
para ajustar las mediciones realizadas el 18 de agosto de las muestras de Al2O3 anodizadas
por Patricia el viernes 14 de agosto. Las mediciones están en la carpeta 2026.08.18, la cual 
a su vez se encuentra en la carpeta Files.
"""

#%% IMPORTS Y RUTAS
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
#%% DICCIONARIO DE MATERIALES

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
    'anatasa': (load_interp(f'{ruta_materiales}/nkdata/anatase.nkv', skiprows=1)),
    'aluminio': (load_interp_in3(f'{ruta_materiales}/nkdata/optical/aluminum.in3', unit='A'))
}

materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])

#%% CARGA DE DATOS Y CONFIGURACIÓN DEL STACK
DATA_dir = r'../Files/2026.08.25/Al2O3_nanotubes.9-med1.txt'

wl_exp,psi_exp,delta_exp,Ic_exp_raw,Is_exp_raw = np.loadtxt(DATA_dir,skiprows=61,usecols=(0,1,2,3,4),max_rows=1334-61,unpack=True)

# Recalcular Is e Ic trigonométricamente a partir de Psi y Delta (en grados)
psi_rad = np.radians(psi_exp)
delta_rad = np.radians(delta_exp)
Is_exp = np.sin(2 * psi_rad) * np.sin(delta_rad)
Ic_exp = np.sin(2 * psi_rad) * np.cos(delta_rad)

wl_exp,psi_exp,delta_exp,Ic_exp,Is_exp = interpolar_todo(wl_exp,psi_exp,delta_exp,Ic_exp,Is_exp)
# Definición del stack óptico y modelos de dispersión
layer_names = ['air', 'Al2O3_nanotube', 'Al2O3', 'Si']
layer_models = [
    None,
    {'model': 'bruggeman', 'f_bounds': (0, 0.9)},
    {'model': 'cauchy', 'bounds': [(1, 5), (-25, 25), (-25, 25)]},
    None
]

# Rangos de espesores iniciales para las capas finitas (nm)
# Ajustados al régimen físico de ~900 nm totales (~450 nm nanotubos + ~450 nm Al2O3)
d_bounds = [(0.0, 150.0), (0.0, 150.0)]

# n_max_limits: límites máximos para n en cada capa (excepto aire/substraído)
n_max_limits = [None, None, 2.0, None]

# CONSTRUCCIÓN DE TENSORES Y OPTIMIZACIÓN AUTORANGE
n_list_np = []
for i, name in enumerate(layer_names):
    if layer_models[i] is not None:
        n_list_np.append(np.ones_like(wl_exp, dtype=complex))
    else:
        n_fn, k_fn = materials[name]
        n_list_np.append(n_fn(wl_exp) + 1j * k_fn(wl_exp))

n_list_torch = torch.tensor(np.array(n_list_np), dtype=torch.complex128)
lams_torch = torch.tensor(wl_exp, dtype=torch.float64)
Is_exp_torch = torch.tensor(Is_exp, dtype=torch.float64)
Ic_exp_torch = torch.tensor(Ic_exp, dtype=torch.float64)
th_0_rad = np.radians(69.95)

#%%
best_thicknesses, best_Is_curve, best_Ic_curve, best_params = autorange_fit_ellipsometry_torch(
    n_list=n_list_torch,
    d_bounds=d_bounds,
    lams=lams_torch,
    Is_exp=Is_exp_torch,
    Ic_exp=Ic_exp_torch,
    th_0=th_0_rad,
    num_starts=200,
    num_epochs=250,
    lr=1.5,
    use_cuda=True,
    layer_models=layer_models,
    layer_names=layer_names,
    n_max_limits=n_max_limits,
    max_attempts=15
)

# Reconstruir Psi y Delta calculadas
psi_fit, delta_fit = transform_I_to_psi_delta(best_Is_curve, best_Ic_curve)

print("\n" + "=" * 60)
print(f"Espesores óptimos encontrados: {best_thicknesses} nm")
if best_params:
    print(f"Parámetros de dispersión del Al2O3: {best_params['Al2O3']}")
print("=" * 60)

#%%
#celda para cargar datos optimizados ya guardados
path = "../Files/2026.08.25/Al2O3_nanotubes.7-med1_fit_results.txt"
data_file, wl_exp, Is_exp, best_Is_curve, Ic_exp, best_Ic_curve,psi_exp, psi_fit, delta_exp, delta_fit,best_thicknesses, best_params = leer_datos_guardados(path)

#%% GRAFICADO DE RESULTADOS
material = 'Al2O3'
A, B, C = best_params[material]['A'],best_params[material]['B'], best_params[material]['C']
indice_Alumina = cauchy_fn(A,B,C)(wl_exp)
f_air = best_params['Al2O3_nanotube']['f_air']
d_Al2O3 = best_thicknesses[1]
d_Al2O3_nanotube = best_thicknesses[0]

indice_Alumina_nanotube = brugg_fn(cauchy_fn(A,B,C), constant_fn(1.0), f_air)(wl_exp)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(wl_exp, Is_exp, 'b.', label='$I_s$ exp', alpha=0.6)
ax1.plot(wl_exp,best_Is_curve,'b-',label='$I_s$ fit', linewidth=1.8)
ax2.plot(wl_exp, Ic_exp, 'r.', label='$I_c$ exp', alpha=0.6)
ax2.plot(wl_exp,best_Ic_curve,'r-',label='$I_c$ fit', linewidth=1.8)
ax1.set_xlabel('Longitud de onda [nm]')
ax1.set_ylabel('Componentes $I_s$')
ax1.set_ylim(np.min(best_Is_curve)-0.1,np.max(best_Is_curve)+0.1)
ax1.grid(True)
ax1.legend()
ax2.set_xlabel('Longitud de onda [nm]')
ax2.set_ylabel('Componentes $I_c$')
ax2.set_ylim(np.min(best_Ic_curve)-0.1,np.max(best_Ic_curve)+0.1)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,1,figsize=(10,8))
ax = axes
ax.plot(wl_exp,indice_Alumina)
ax.set_xlabel('Longitud de onda [nm]')
ax.set_ylabel('Índice de refracción')
ax.set_title(f'Índice de refracción de los nanotubos de {material}')
ax.grid(True)
plt.show()

print("mejores espesores:")
print(best_thicknesses)

print("mejores parametros:")
print(best_params)

#%% GUARDAR RESULTADOS EN TXT
salida_txt = os.path.splitext(DATA_dir)[0] + '_fit_results.txt'
guardar_resultados_txt(
    salida_txt, DATA_dir, wl_exp, Is_exp, best_Is_curve, Ic_exp, best_Ic_curve,
    psi_exp, psi_fit, delta_exp, delta_fit, best_thicknesses, best_params
)
# %%
# %% CÁLCULO MANUAL DE I_s, I_c, PSI Y DELTA PARA ESPESORES GRUESOS (~970 nm)
from tmm_utils_Rodrigo import coh_tmm_torch_batched

# Parámetros del ajuste del elipsómetro / teóricos
A_elip, B_elip, C_elip = 1.578, -9.1729, 17.8583  # Modelo Cauchy para Al2O3
f_air_elip = 0.985                               # Fracción de aire en la capa porosa (nanotubo)

# Espesores de prueba (nm) que suman ~970 nm (ej. 500 nm nanotubo + 470 nm Al2O3 denso)
d_nanotube_test = 5.0
d_al2o3_test = 970.0
d_total_test = d_nanotube_test + d_al2o3_test
print(f"Calculando respuesta óptica para espesores gruesos: d_nanotube = {d_nanotube_test} nm, d_Al2O3 = {d_al2o3_test} nm (Total = {d_total_test} nm)")

# Construcción de índices de refracción
n_air_calc = torch.ones_like(lams_torch, dtype=torch.complex128)
n_al2o3_calc = torch.tensor(cauchy_fn(A_elip, B_elip, C_elip)(wl_exp), dtype=torch.complex128)
n_nanotube_calc = torch.tensor(brugg_fn(cauchy_fn(A_elip, B_elip, C_elip), constant_fn(1.0), f_air_elip)(wl_exp), dtype=torch.complex128)
n_si_calc = n_list_torch[3]  # Silicio del stack cargado

# Stack 3D para PyTorch TMM: [1, 4, num_wl]
n_stack_test = torch.stack([n_air_calc, n_nanotube_calc, n_al2o3_calc, n_si_calc], dim=0).unsqueeze(0)
d_stack_test = torch.tensor([[float('inf'), d_nanotube_test, d_al2o3_test, float('inf')]], dtype=torch.float64, device=lams_torch.device).unsqueeze(-1).expand(1, 4, len(wl_exp))

# Cálculo TMM para polarizaciones s y p
res_s_test = coh_tmm_torch_batched('s', n_stack_test, d_stack_test, th_0=th_0_rad, lam_vac=lams_torch)
res_p_test = coh_tmm_torch_batched('p', n_stack_test, d_stack_test, th_0=th_0_rad, lam_vac=lams_torch)

# Reflectancias R_s y R_p
R_s_test = (res_s_test['r'][0] * res_s_test['r'][0].conj()).real.cpu().numpy()
R_p_test = (res_p_test['r'][0] * res_p_test['r'][0].conj()).real.cpu().numpy()

# Coeficiente elipsométrico rho = r_p / r_s
rho_test = torch.conj(res_p_test['r'][0] / res_s_test['r'][0])
psi_calc_rad = torch.atan(torch.abs(rho_test))
delta_calc_rad = torch.angle(rho_test)

# Componentes Is e Ic calculadas
Is_calc_thick = (torch.sin(2 * psi_calc_rad) * torch.sin(delta_calc_rad)).cpu().numpy()
Ic_calc_thick = (torch.sin(2 * psi_calc_rad) * torch.cos(delta_calc_rad)).cpu().numpy()

# Ángulos Psi y Delta calculados (en grados)
psi_calc_deg = np.degrees(psi_calc_rad.cpu().numpy())
delta_calc_deg = np.degrees(delta_calc_rad.cpu().numpy())
delta_calc_deg = np.mod(delta_calc_deg, 360)

# --- GRAFICADO DE RESULTADOS PARA ESPESOR GRUESO (~970 nm) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# 1. Reflectancias Rs y Rp
axes[0].plot(wl_exp, R_s_test, 'b-', label='$R_s$ simulada', linewidth=1.5)
axes[0].plot(wl_exp, R_p_test, 'r-', label='$R_p$ simulada', linewidth=1.5)
axes[0].set_title(f'Reflectancias $R_s$ y $R_p$ (Espesor Total = {d_total_test:.1f} nm)')
axes[0].set_xlabel('Longitud de onda [nm]')
axes[0].set_ylabel('Reflectancia')
axes[0].grid(True)
axes[0].legend()

# 2. Componentes Is e Ic
axes[1].plot(wl_exp, Is_exp, 'b.', label='$I_s$ exp', alpha=0.5)
axes[1].plot(wl_exp, Is_calc_thick, 'b-', label=f'$I_s$ calc ({d_total_test:.0f} nm)', linewidth=1.8)
axes[1].plot(wl_exp, Ic_exp, 'r.', label='$I_c$ exp', alpha=0.5)
axes[1].plot(wl_exp, Ic_calc_thick, 'r-', label=f'$I_c$ calc ({d_total_test:.0f} nm)', linewidth=1.8)
axes[1].set_title(f'Componentes $I_s$ e $I_c$ para Espesor Grueso ({d_total_test:.1f} nm)')
axes[1].set_xlabel('Longitud de onda [nm]')
axes[1].set_ylabel('Intensidad Modulada')
axes[1].grid(True)
axes[1].legend()

# 3. Parámetros Psi y Delta
axes[2].plot(wl_exp, psi_exp, 'g.', label='$\\Psi$ exp', alpha=0.5)
axes[2].plot(wl_exp, psi_calc_deg, 'g-', label=f'$\\Psi$ calc ({d_total_test:.0f} nm)', linewidth=1.8)
axes[2].plot(wl_exp, delta_exp, 'm.', label='$\\Delta$ exp', alpha=0.5)
axes[2].plot(wl_exp, delta_calc_deg, 'm-', label=f'$\\Delta$ calc ({d_total_test:.0f} nm)', linewidth=1.8)
axes[2].set_title(f'Parámetros Elipsométricos $\\Psi$ y $\\Delta$ (Espesor Total = {d_total_test:.1f} nm)')
axes[2].set_xlabel('Longitud de onda [nm]')
axes[2].set_ylabel('Grados [°]')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()

# %% CÁLCULO MANUAL USANDO tmm.coh_tmm (VERSIÓN NUMPY) Y COMPARACIÓN CON TORCH
import tmm_core as tmm

# Construcción del vector de índices NumPy por capa y por longitud de onda
n_air_np = np.ones_like(wl_exp, dtype=complex)
n_al2o3_np = cauchy_fn(A_elip, B_elip, C_elip)(wl_exp) + 0j
n_nanotube_np = brugg_fn(cauchy_fn(A_elip, B_elip, C_elip), constant_fn(1.0), f_air_elip)(wl_exp) + 0j
n_si_np = n_list_torch[3].cpu().numpy()  # Silicio

# n_list de forma [num_capas, num_wl] y d_list de capas
n_list_np_stack = np.array([n_air_np, n_nanotube_np, n_al2o3_np, n_si_np])
d_list_np_stack = [np.inf, d_nanotube_test, d_al2o3_test, np.inf]

# Cálculo TMM coherente (NumPy) para polarización s y p
res_s_np = tmm.coh_tmm('s', n_list_np_stack, d_list_np_stack, th_0_rad, wl_exp)
res_p_np = tmm.coh_tmm('p', n_list_np_stack, d_list_np_stack, th_0_rad, wl_exp)

# Reflectancias R_s y R_p (NumPy)
R_s_np = res_s_np['R']
R_p_np = res_p_np['R']

# Coeficiente elipsométrico rho = r_p / r_s
rho_np = np.conj(res_p_np['r'] / res_s_np['r'])
psi_np_rad = np.arctan(np.abs(rho_np))
delta_np_rad = np.angle(rho_np)

# Componentes Is e Ic calculadas (NumPy)
Is_calc_coh_np = np.sin(2 * psi_np_rad) * np.sin(delta_np_rad)
Ic_calc_coh_np = np.sin(2 * psi_np_rad) * np.cos(delta_np_rad)

# Ángulos Psi y Delta (en grados)
psi_np_deg = np.degrees(psi_np_rad)
delta_np_deg = np.mod(np.degrees(delta_np_rad), 360)

# --- COMPARACIÓN DE PRECISIÓN ENTRE NUMPY (coh_tmm) Y PYTORCH (coh_tmm_torch_batched) ---
diff_max_Is = np.max(np.abs(Is_calc_coh_np - Is_calc_thick))
diff_max_Ic = np.max(np.abs(Ic_calc_coh_np - Ic_calc_thick))
diff_max_psi = np.max(np.abs(psi_np_deg - psi_calc_deg))

print("\n================ COMPARACIÓN coh_tmm VS coh_tmm_torch_batched ================")
print(f"Diferencia máxima absoluta en Is : {diff_max_Is:.2e}")
print(f"Diferencia máxima absoluta en Ic : {diff_max_Ic:.2e}")
print(f"Diferencia máxima absoluta en Psi: {diff_max_psi:.2e} °")
print("================================================================================")

# --- GRAFICADO DE RESULTADOS (tmm.coh_tmm NumPy) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# 1. Reflectancias Rs y Rp (NumPy)
axes[0].plot(wl_exp, R_s_np, 'b-', label='$R_s$ (coh_tmm NumPy)', linewidth=1.5)
axes[0].plot(wl_exp, R_p_np, 'r-', label='$R_p$ (coh_tmm NumPy)', linewidth=1.5)
axes[0].set_title(f'Reflectancias $R_s$ y $R_p$ con coh_tmm (Espesor Total = {d_total_test:.1f} nm)')
axes[0].set_xlabel('Longitud de onda [nm]')
axes[0].set_ylabel('Reflectancia')
axes[0].grid(True)
axes[0].legend()

# 2. Componentes Is e Ic (NumPy)
axes[1].plot(wl_exp, Is_exp, 'b.', label='$I_s$ exp', alpha=0.5)
axes[1].plot(wl_exp, Is_calc_coh_np, 'b-', label=f'$I_s$ coh_tmm ({d_total_test:.0f} nm)', linewidth=1.8)
axes[1].plot(wl_exp, Ic_exp, 'r.', label='$I_c$ exp', alpha=0.5)
axes[1].plot(wl_exp, Ic_calc_coh_np, 'r-', label=f'$I_c$ coh_tmm ({d_total_test:.0f} nm)', linewidth=1.8)
axes[1].set_title(f'Componentes $I_s$ e $I_c$ con coh_tmm ({d_total_test:.1f} nm)')
axes[1].set_xlabel('Longitud de onda [nm]')
axes[1].set_ylabel('Intensidad Modulada')
axes[1].grid(True)
axes[1].legend()

# 3. Parámetros Psi y Delta (NumPy)
axes[2].plot(wl_exp, psi_exp, 'g.', label='$\\Psi$ exp', alpha=0.5)
axes[2].plot(wl_exp, psi_np_deg, 'g-', label=f'$\\Psi$ coh_tmm ({d_total_test:.0f} nm)', linewidth=1.8)
axes[2].plot(wl_exp, delta_exp, 'm.', label='$\\Delta$ exp', alpha=0.5)
axes[2].plot(wl_exp, delta_np_deg, 'm-', label=f'$\\Delta$ coh_tmm ({d_total_test:.0f} nm)', linewidth=1.8)
axes[2].set_title(f'Parámetros Elipsométricos $\\Psi$ y $\\Delta$ con coh_tmm ({d_total_test:.1f} nm)')
axes[2].set_xlabel('Longitud de onda [nm]')
axes[2].set_ylabel('Grados [°]')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()
# %%
"""
pasos a seguir:
1. Realizar simulaciones de los parametros  Is e Ic para espesores gruesos y finitos, de esta 
manera se va a ver las oscilaciones excesivas que aparecen por culpa de la interferencia
2. Realizar los ajustes que me faltan de las mediciones 2,3,4,5,7,9
3. Documentar en el word los resultados.
4. Escanear la imagen de las muestras para determinar sus dimensiones.
"""