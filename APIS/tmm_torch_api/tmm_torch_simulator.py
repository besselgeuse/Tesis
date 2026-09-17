

def ejecutar_simulacion(stack:list, thicks:list, c_list:list, sustrato:str = 'GaAs', pol:str = 's', th_0:float = 0.0, num_starts:int = 500, num_epochs:int = 200, lr:float = 1.5, use_cuda:bool = True):
    
    import sys
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import json

    # Importar utilidades del TMM
    sys.path.append('./')
    from tmm_utils_Rodrigo import (
        load_interp, cauchy_fn, constant_fn, brugg_fn, calculate_RT_torch, load_fn
    )

    # ============================================================
    # 1. DEFINICIÓN DE MATERIALES
    # ============================================================
    ruta_materiales = r'.'

    # Primero el Ti (para el que usaron valores experimentales)
    modelo_T1 = cauchy_fn(2.338, 1.906, 0.824)   #muestra T1 del paper (R-1)
    f_T1 = 0.58

    ruta_materiales = f'{ruta_materiales}/indices'

    materials = {
        'air': (constant_fn(1.0), constant_fn(0.0)),
        'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows=1)),
        'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows=1)),
        'T1_densa': (modelo_T1, constant_fn(0.0)),
        'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
        'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)),
        'SiO2': (load_interp(f'{ruta_materiales}/nkdata/SiO2_Palik.nk', skiprows=1, unit='um')),
        'MgF2': (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv', skiprows=1)),
        'vidrio': (load_interp(f'{ruta_materiales}/glass_thales.nk', skiprows=1))
    }

    # Asigno k rutilo a las capas experimentales
    materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
    materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])

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
    
    if sustrato == "GaAs":
        IQE = load_fn(f'{ruta_materiales}/IQEGaAs2.txt', delimiter=" ")(lams_np)
    else:
        IQE = load_fn(f'{ruta_materiales}/IQE_Si.txt', delimiter=",")(lams_np)/100
    weight = const*IQE*lams_np
    jmax_am15 = np.trapezoid(am15*weight, dx=step)
    jmax_am0 = np.trapezoid(am0*weight, dx=step)
    print(jmax_am0)
    print(jmax_am15)

    # ============================================================
    # 3. CONSTRUIR n_list COMO TENSOR DE PYTORCH
    # ============================================================
    # Stack óptico completo: [superestrato, ...capas_finitas..., sustrato]
    full_stack = ['air'] + stack + [sustrato]
    full_c_list = ['i'] + list(c_list) + ['i']

    n_list_np = []
    for name in full_stack:
        n_fn, k_fn = materials[name]
        n_complex = n_fn(lams_np) + 1j * k_fn(lams_np)
        n_list_np.append(n_complex)

    n_list_np = np.array(n_list_np)  # [num_layers, num_wl]

    # Convertir a tensores de PyTorch
    n_list_torch = torch.tensor(n_list_np, dtype=torch.complex128)
    lams_torch = torch.tensor(lams_np, dtype=torch.float64)
    print("hola mundo")
    d_bounds = []
    for thick in thicks:
        d_bounds.append((thick['min'], thick['max']))

    print("=" * 60)
    print("Stack completo: ", full_stack)
    print(f"Rango espectral: {lams_np[0]:.0f} - {lams_np[-1]:.0f} nm ({len(lams_np)} puntos)")
    print(f"Forma de n_list: {n_list_torch.shape}")
    print("=" * 60)

    # ============================================================
    # 4. DEFINIR LÍMITES DE ESPESORES Y EJECUTAR OPTIMIZACIÓN
    # ============================================================
    # Definir el vector de pesos (flujo de fotones útiles AM0 * IQE) en PyTorch
    weights_torch = torch.tensor(weight * am0, dtype=torch.float64)
    print("adios mundo")
    # Ejecutar la optimización
    best_thicknesses, best_R_curve = calculate_RT_torch(
        n_list=n_list_torch,
        d_bounds=d_bounds,
        lams=lams_torch,
        c_list=full_c_list,
        weights=weights_torch, # <--- Pasamos el vector de pesos
        pol=pol,
        th_0=th_0,
        num_starts=num_starts,     # 500 semillas para balancear velocidad/exploración
        num_epochs=num_epochs,    # 200 épocas
        lr=lr,            # Learning rate
        use_cuda=use_cuda      # Usar CUDA
    )

    # Convertir best_R_curve a numpy si es un tensor
    if isinstance(best_R_curve, torch.Tensor):
        best_R_curve_np = best_R_curve.detach().cpu().numpy()
    else:
        best_R_curve_np = best_R_curve

    j_opt_am15 = jmax_am15 - np.trapezoid(best_R_curve_np * am15 * weight, dx=step)
    j_opt_am0 = jmax_am0 - np.trapezoid(best_R_curve_np * am0 * weight, dx=step)

    return best_thicknesses, best_R_curve_np, j_opt_am15, j_opt_am0, lams_np

