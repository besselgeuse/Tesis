import sys
import os
import hashlib
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

# Importar utilidades del TMM
sys.path.append('./')
from tmm_utils_Rodrigo import (
    load_interp, cauchy_fn, constant_fn, brugg_fn, calculate_RT_torch, load_fn, fit_ellipsometry_torch
)

# ============================================================
# DEFINICIÓN DE MATERIALES
# ============================================================
ruta_materiales = r'.'

# Primero el Ti (para el que usaron valores experimentales)
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
f_T1 = 0.58
modelo_SiO2 = cauchy_fn(1.46,0.0,0.0) #esto solo es para inicializar el modelo de SiO2
ruta_materiales = f'{ruta_materiales}/indices'

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
materials['modelo_SiO2'] = (modelo_SiO2, constant_fn(0.0))
materials['SiO2_Si_brugge'] = (brugg_fn(modelo_SiO2, constant_fn(1.0), 0.5), constant_fn(0.0))

# ============================================================

def _compute_cache_key(data_path, skiprows, layer_names, layer_models, d_bounds, 
                       theta_0, num_starts, num_epochs, lr, use_cuda):
    """Genera un hash único basado en los parámetros de entrada del ajuste."""
    # Convertir layer_models a formato serializable (los dicts ya lo son)
    models_serializable = []
    for m in layer_models:
        if m is None:
            models_serializable.append(None)
        elif isinstance(m, dict):
            models_serializable.append(m)
        else:
            models_serializable.append(str(m))
    
    key_data = {
        'data_path': os.path.abspath(data_path),
        'skiprows': skiprows,
        'layer_names': layer_names,
        'layer_models': models_serializable,
        'd_bounds': [list(b) if isinstance(b, (tuple, list)) else b for b in d_bounds],
        'theta_0': theta_0,
        'num_starts': num_starts,
        'num_epochs': num_epochs,
        'lr': lr,
        'use_cuda': use_cuda,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def ajuste_elipsometrico(data_path, Data_R=None, skiprows=0, layer_names=None, layer_models=None, d_bounds=None,
                         theta_0=69.5, num_starts=100, num_epochs=150, lr=1.5, 
                         use_cuda=False, cache_path=None, force_recalc=False):    
    """
    Genera un ajuste elipsometrico de un stack de materiales.

    Args:
        data_path (str): Ruta del archivo de datos experimentales (.txt con wl, psi, delta).
        skiprows (int): Número de filas a saltar en el archivo de datos.
        layer_names (list): Lista de nombres de las capas. Para capas con modelo paramétrico
            (cauchy, bruggeman, etc.) el nombre es solo una etiqueta (no necesita existir en 
            el diccionario de materiales). Para capas sin modelo (None), el nombre DEBE existir 
            en el diccionario de materiales.
        layer_models (list): Lista de modelos de dispersión por capa. None para capas estáticas,
            'cauchy' para Cauchy transparente, 'cauchy_absorbent' para Cauchy con absorción,
            'bruggeman' para Bruggeman EMA, o un diccionario para configuración avanzada:
            {'model': 'cauchy', 'bounds': [(A_min,A_max), (B_min,B_max), (C_min,C_max)]}
            {'model': 'bruggeman', 'f_air': 0.42} o {'model': 'bruggeman', 'f_bounds': (0.3, 0.7)}
        d_bounds (list): Límites de los espesores para las capas finitas (excluyendo superestrato y sustrato).
        theta_0 (float): Ángulo de incidencia en grados. Por defecto 69.5 grados.
        num_starts (int): Número de semillas para la optimización. Por defecto 100.
        num_epochs (int): Número de épocas para la optimización. Por defecto 150.
        lr (float): Learning rate. Por defecto 1.5.
        use_cuda (bool): Usar CUDA para optimización. Por defecto False.
        cache_path (str): Ruta de la carpeta de caché. Si se proporciona, los resultados 
            se guardan/cargan automáticamente. Por defecto None (sin caché).
        force_recalc (bool): Si True, ignora la caché y recalcula. Por defecto False.

    Returns:
        tuple: (best_thicknesses, best_Is, best_Ic, best_params, wl_exp)
    """
    
    # ── VERIFICAR CACHÉ ──
    cache_file = None
    if cache_path is not None and not force_recalc:
        os.makedirs(cache_path, exist_ok=True)
        cache_key = _compute_cache_key(data_path, skiprows, layer_names, layer_models,
                                        d_bounds, theta_0, num_starts, num_epochs, lr, use_cuda)
        cache_file = os.path.join(cache_path, f"fit_{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            print(f"=== Cargando resultados desde caché: {cache_file} ===")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            return (cached['best_thicknesses'], cached['best_Is'], cached['best_Ic'], 
                    cached['best_params'], cached['wl_exp'], cached['Is_exp'], cached['Ic_exp'])
    
    # ── CARGAR DATOS EXPERIMENTALES ──
    wl_exp, psi_deg, delta_deg = np.loadtxt(data_path, skiprows=skiprows, unpack=True)

    # Filtrar longitudes de onda mayores a 830 nm (ruido experimental)
    mask = wl_exp <= 830.0
    wl_exp = wl_exp[mask]
    psi_deg = psi_deg[mask]
    delta_deg = delta_deg[mask]

    # Convertir ángulos de grados a radianes
    psi = np.radians(psi_deg)
    delta = np.radians(delta_deg)

    # Definir parámetros elipsométricos experimentales Is e Ic
    Is_exp = np.sin(2 * psi) * np.sin(delta)
    Ic_exp = np.sin(2 * psi) * np.cos(delta)

    # Pasarlos a formato tensor de PyTorch
    Is_exp_torch = torch.tensor(Is_exp, dtype=torch.float64)
    Ic_exp_torch = torch.tensor(Ic_exp, dtype=torch.float64)

    lams_np = wl_exp

    # ── CONSTRUIR n_list ──
    # Mejora 1: Para capas con modelo paramétrico, usamos un dummy (n=1, k=0).
    # El optimizador sobreescribirá estos valores con el modelo (Cauchy, etc.).
    # Solo las capas con layer_models[i] == None buscan en el diccionario materials.
    n_list_np = []
    for i, name in enumerate(layer_names):
        if layer_models[i] is not None:
            # Capa paramétrica: dummy placeholder (será sobreescrito)
            n_list_np.append(np.ones_like(lams_np, dtype=complex))
        else:
            # Capa estática: buscar en el diccionario de materiales
            if name not in materials:
                raise ValueError(
                    f"El material '{name}' no existe en el diccionario de materiales. "
                    f"Materiales disponibles: {list(materials.keys())}"
                )
            n_fn, k_fn = materials[name]
            n_complex = n_fn(lams_np) + 1j * k_fn(lams_np)
            n_list_np.append(n_complex)

    n_list_np = np.array(n_list_np)  # [num_layers, num_wl]

    # Convertir a tensores de PyTorch
    n_list_torch = torch.tensor(n_list_np, dtype=torch.complex128)
    lams_torch = torch.tensor(lams_np, dtype=torch.float64)

    print("=" * 60)
    print(f"Stack: {layer_names}")
    print(f"Rango espectral: {lams_np[0]:.0f} - {lams_np[-1]:.0f} nm ({len(lams_np)} puntos)")
    print(f"Forma de n_list: {n_list_torch.shape}")
    print("=" * 60)

    th_0_rad = np.radians(theta_0)

    # Ejecutar la optimización conjunta (espesores + parámetros de dispersión)
    best_thicknesses, best_Is, best_Ic, best_params = fit_ellipsometry_torch(
        n_list=n_list_torch,
        d_bounds=d_bounds,
        lams=lams_torch,
        Is_exp=Is_exp_torch,
        Ic_exp=Ic_exp_torch,
        Data_R=Data_R,
        th_0=th_0_rad,
        num_starts=num_starts,
        num_epochs=num_epochs,
        lr=lr,
        use_cuda=use_cuda,
        layer_models=layer_models,
        layer_names=layer_names
    )
    
    # ── GUARDAR EN CACHÉ ──
    if cache_path is not None:
        os.makedirs(cache_path, exist_ok=True)
        if cache_file is None:
            cache_key = _compute_cache_key(data_path, skiprows, layer_names, layer_models,
                                            d_bounds, theta_0, num_starts, num_epochs, lr, use_cuda)
            cache_file = os.path.join(cache_path, f"fit_{cache_key}.pkl")
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'best_thicknesses': best_thicknesses,
                'best_Is': best_Is,
                'best_Ic': best_Ic,
                'best_params': best_params,
                'wl_exp': wl_exp,
                'Is_exp': Is_exp,
                'Ic_exp': Ic_exp,
            }, f)
        print(f"=== Resultados guardados en caché: {cache_file} ===")

    return best_thicknesses, best_Is, best_Ic, best_params, wl_exp, Is_exp, Ic_exp

def transform_I_to_psi_delta(Is, Ic):
    """Convierte parámetros Is, Ic a psi (grados) y delta (grados)."""
    psi = 0.5 * np.degrees(np.arcsin(np.sqrt(Is**2 + Ic**2)))
    delta = np.degrees(np.arctan2(Is, Ic))
    
    return psi, delta 