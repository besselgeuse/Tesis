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
import numpy as np
import torch
import matplotlib.pyplot as plt

Dirección_tmm_Rodrigo = r'../'
if Dirección_tmm_Rodrigo not in sys.path:
    sys.path.append(Dirección_tmm_Rodrigo)

from tmm_utils_Rodrigo import (
    load_interp, autorange_fit_ellipsometry_torch, load_fn, plot_js,
    cauchy_fn, constant_fn, brugg_fn, stack2tmm, calculate_RT_torch, transform_I_to_psi_delta,
    load_interp_in3
)

#%% FUNCIONES AUXILIARES DE CARGA DE DATOS Y TRANSFORMACIÓN
def interpolar_todo(wl_exp=None,psi_exp=None,delta_exp=None,Ic_exp=None,Is_exp=None):
    from scipy.interpolate import interp1d
    def interpolar_datos(lams, datos=None):
        if datos is not None:
            return interp1d(lams,datos,kind='linear',fill_value='extrapolate')
        else:
            return None
    mask_wl = np.where((wl_exp >= 440) & (wl_exp <= 830))
    wl_exp = wl_exp[mask_wl]
    psi_exp = psi_exp[mask_wl]
    delta_exp = delta_exp[mask_wl]
    Ic_exp = Ic_exp[mask_wl]
    Is_exp = Is_exp[mask_wl]

    psi_exp_interp = interpolar_datos(wl_exp,psi_exp)
    delta_exp_interp = interpolar_datos(wl_exp,delta_exp)
    Ic_exp_interp = interpolar_datos(wl_exp,Ic_exp)
    Is_exp_interp = interpolar_datos(wl_exp,Is_exp)

    return wl_exp, psi_exp_interp(wl_exp), delta_exp_interp(wl_exp), Ic_exp_interp(wl_exp), Is_exp_interp(wl_exp)    


def guardar_resultados_txt(salida_path, data_file, wl_exp, Is_exp, Is_fit, Ic_exp, Ic_fit, 
                           psi_exp, psi_fit, delta_exp, delta_fit, 
                           best_thicknesses, best_params):
    """Guarda los resultados del ajuste en un archivo TXT."""
    with open(salida_path, 'w', encoding='utf-8') as f:
        f.write("# ========================================================\n")
        f.write("# RESULTADOS DE AJUSTE ELIPSOMÉTRICO (AUTORANGE + TMM)\n")
        f.write("# ========================================================\n")
        f.write(f"# Archivo de origen: {os.path.basename(data_file)}\n")
        f.write(f"# Espesores óptimos (nm): {list(best_thicknesses)}\n")
        if best_params:
            f.write(f"# Error mínimo (Chi2 Red / MSE): {best_params.get('chi2_min', 0.0):.6e}\n")
            f.write("# Parámetros de dispersión optimizados:\n")
            for name, params in best_params.items():
                if name != 'chi2_min' and not isinstance(name, int):
                    f.write(f"#   Capa '{name}': {params}\n")
        f.write("# ========================================================\n")
        f.write("# Wavelength_nm\tIs_exp\tIs_fit\tIc_exp\tIc_fit\tPsi_exp_deg\tPsi_fit_deg\tDelta_exp_deg\tDelta_fit_deg\n")
        
        for i in range(len(wl_exp)):
            f.write(f"{wl_exp[i]:.4f}\t{Is_exp[i]:.6f}\t{Is_fit[i]:.6f}\t"
                    f"{Ic_exp[i]:.6f}\t{Ic_fit[i]:.6f}\t{psi_exp[i]:.4f}\t"
                    f"{psi_fit[i]:.4f}\t{delta_exp[i]:.4f}\t{delta_fit[i]:.4f}\n")
    print(f"--> Resultados guardados exitosamente en: {salida_path}")

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
DATA_dir = r'../Files/2026.08.25/Al2O3_nanotubes.8-med1.txt'

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
    {'model': 'bruggeman', 'f_bounds': (0.1, 0.7)},
    {'model': 'cauchy', 'bounds': [(1, 5), (-10, 10), (-10, 10)]},
    None
]

# Rangos de espesores iniciales para las capas finitas (nm)
# Ajustados al régimen físico de ~900 nm totales (~450 nm nanotubos + ~450 nm Al2O3)
d_bounds = [(0.0, 100.0), (900.0, 1000.0)]

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
    num_starts=400,
    num_epochs=150,
    lr=1.5,
    use_cuda=True,
    layer_models=layer_models,
    layer_names=layer_names,
    n_max_limits=n_max_limits,
    max_attempts=1
)

# Reconstruir Psi y Delta calculadas
psi_fit, delta_fit = transform_I_to_psi_delta(best_Is_curve, best_Ic_curve)

print("\n" + "=" * 60)
print(f"Espesores óptimos encontrados: {best_thicknesses} nm")
if best_params:
    print(f"Parámetros de dispersión del Al2O3: {best_params['Al2O3']}")
print("=" * 60)

#%% GRAFICADO CIENTÍFICO DE RESULTADOS
material = 'Al2O3'
A, B, C = best_params[material]['A'],best_params[material]['B'], best_params[material]['C']
indice_Alumina = cauchy_fn(A,B,C)(wl_exp)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
ax1 = axes[0]
ax2 = axes[1]

ax1.plot(wl_exp, Is_exp, 'b.', label='$I_s$ exp', alpha=0.6)
ax1.plot(wl_exp,best_Is_curve,'b-',label='$I_s$ fit', linewidth=1.8)
ax2.plot(wl_exp, Ic_exp, 'r.', label='$I_c$ exp', alpha=0.6)
ax2.plot(wl_exp,best_Ic_curve,'r-',label='$I_c$ fit', linewidth=1.8)
ax1.set_xlabel('Longitud de onda [nm]')
ax1.set_ylabel('Componentes $I_s$')
ax1.set_ylim(0.6,1.2)
ax1.grid(True)
ax1.legend()
ax2.set_xlabel('Longitud de onda [nm]')
ax2.set_ylabel('Componentes $I_c$')
ax2.set_ylim(-0.04,0.2)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,1,figsize=(10,8))
ax = axes
ax.plot(wl_exp,indice_Alumina)
ax.set_xlabel('Longitud de onda [nm]')
ax.set_ylabel('Índice de refracción')
ax.set_title(f'Índice de refracción de {material}')
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
