
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:45:17 2020

@author: simon

Utilities to calculate RAT of optical stacks with tmm_vec
"""
# 8/4 cambie los imports de Simon para que se pueda elegir la ruta del tmm_core

import numpy as np
import os
import pickle
import sys
import importlib.util
import torch
import torch.optim as optim

# 1. Definimos la ruta exacta al archivo tmm_core.py de Simon

tmm_path = r".\tmm_core.py"

# 2. Cargamos el módulo manualmente desde esa dirección
spec = importlib.util.spec_from_file_location("tmm", tmm_path)
tmm = importlib.util.module_from_spec(spec)
sys.modules["tmm"] = tmm # Esto asegura que otros scripts vean esta versión
spec.loader.exec_module(tmm)

from functools import partial
from itertools import product
from scipy.interpolate import interp1d
from openpyxl import load_workbook
import matplotlib.pyplot as plt


def cauchy_fn(A,B,C):
    def fn(lams):
        lams = lams*1.0
        return A+B*np.power(10,4)/np.power(lams,2)+C*np.power(10,9)/np.power(lams,4)
    return fn


def constant_fn(val):
    return partial(np.full_like, fill_value=val)

def n_eff(n1,n2,f):
    """
    Modelo de bruggeman. f es la fraccion de la especie 2.
    """
    if f == 0: return n1
    if f == 1: return n2
    e1 = n1**2
    e2 = n2**2
    #f = 1-f2
    omega = (1-f)*(e2-2*e1)+f*(e1-2*e2)
    return (np.sqrt(np.sqrt(omega**2 + 8*e1*e2)-omega))/2


def brugg_fn(n_a, n_b, f_b):
    def fn(lams):
        n1 = n_a(lams)
        n2 = n_b(lams)
        return n_eff(n1, n2, f_b)
    return fn

def load_fn(filename, skiprows=1,delimiter = "\t"):
    
    wv, I = np.loadtxt(filename, skiprows=skiprows,delimiter=f"{delimiter}" ,unpack=True)
    return interp1d(wv, I,fill_value=(I[0],I[-1]), bounds_error=False)


def load_interp(filename, comments=';', skiprows=0, unit='nm'):
    '''loads optical data and outputs interpolation functions for n and k. 
    Output functions take wv in nm. Keep default arguments to load .nkv files
    '''
    scale = {'nm':1.0, 'um':1000.0}
    
    wv, n, k = np.loadtxt(filename, comments=comments, 
                      skiprows=skiprows, unpack=True)
    
    wv *= scale[unit]
    
    n_fn = interp1d(wv, n,fill_value=(n[0],n[-1]), bounds_error=False)
    k_fn = interp1d(wv, k,fill_value=(k[0],k[-1]), bounds_error=False)
    
    return n_fn, k_fn


def load_interp_in3(filename, unit='A'):
    '''loads optical data and outputs interpolation functions for n and k from .in3 files.
    Output functions take wv in nm. Default unit is Angstroms ('A').
    '''
    scale = {'nm': 1.0, 'um': 1000.0, 'A': 0.1}
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != '']
    
    n_pts = 0
    k_pts = 0
    
    wv_n = []
    n_vals = []
    wv_k = []
    k_vals = []
    
    idx = 0
    # Buscar el número de puntos de n
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 1:
            try:
                n_pts = int(parts[0])
                idx += 1
                break
            except ValueError:
                pass
        idx += 1
        
    for _ in range(n_pts):
        parts = lines[idx].split()
        wv_n.append(float(parts[0]))
        n_vals.append(float(parts[1]))
        idx += 1
        
    # Buscar el número de puntos de k
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 1:
            try:
                k_pts = int(parts[0])
                idx += 1
                break
            except ValueError:
                pass
        idx += 1
        
    for _ in range(k_pts):
        parts = lines[idx].split()
        wv_k.append(float(parts[0]))
        k_vals.append(float(parts[1]))
        idx += 1
        
    wv_n = np.array(wv_n)
    wv_k = np.array(wv_k)
    n_vals = np.array(n_vals)
    k_vals = np.array(k_vals)
    
    if unit in scale:
        wv_n *= scale[unit]
        wv_k *= scale[unit]
        
    n_fn = interp1d(wv_n, n_vals, fill_value=(n_vals[0], n_vals[-1]), bounds_error=False)
    k_fn = interp1d(wv_k, k_vals, fill_value=(k_vals[0], k_vals[-1]), bounds_error=False)
    
    return n_fn, k_fn


def stack2tmm(stack, materials, lams, add_inf=True):
    '''Transforms a stack into three lists that can be fed into the 'tmm' package.
    BY DEFAULT ADDS inf AIR LAYERS'''
    n_list = []
    d_list = []
    c_list = []
    if add_inf:
        n_list.append([1.]*len(lams))
        d_list.append(np.inf)
        c_list.append('i')
        
    for layer in stack:
        d_list.append(layer[0])
        mat = materials[layer[1]]
        n_list.append(mat[0](lams) + mat[1](lams)*1.0j)
        c_list.append(layer[2])
        
    if add_inf:  
        n_list.append([1.]*len(lams))
        d_list.append(np.inf)
        c_list.append('i')
    
    return n_list, d_list, c_list

   
def RT_with_cache(file, stack, materials, lams, pol='s', th_0=0, thicks=None):
    if file is None:
        return calculate_RT(stack, materials, lams, pol, th_0, thicks)
    else:
    
        if os.path.exists(file):
            with open(file,'rb') as f:
                print('loading from cache\n',)
                RT = pickle.load(f)
        else:
            RT = calculate_RT(stack, materials, lams, pol, th_0, thicks)
            with open(file,'wb') as f:
                print('saving to cache\n')
                pickle.dump(RT, f)
        return RT


def calculate_RT(stack, materials, lams, pol='s', th_0=0, thicks=None):
    '''calulate R and T of a stack, for all combinations of thicks.'''
    n_list, d_list, c_list = stack2tmm(stack, materials, lams, add_inf=False)
    n_lams = len(lams)
    inc = 'i' in c_list[1:-1] #check if  any of the finite layers is incocherent
    if inc:
        def f(pol, n_list, d_list, th_0, lams, c_list):
            return tmm.inc_tmm(pol, n_list, d_list, c_list, th_0, lams)
    else:
        def f(pol, n_list, d_list, th_0, lams, c_list=None):
            return tmm.coh_tmm(pol, n_list, d_list, th_0, lams)
            
    if thicks is None:
        #case with only one thickness
        RAT = f(pol, n_list, d_list, th_0, lams, c_list)
        return RAT['R'], RAT['T']
        
    else:
        #case with varying thicknesses
        keys = thicks.keys()
        vals = thicks.values()
        sizes =  [len(val) for val in vals]
        params = product(*vals) #cartesian product of all thicknesses
        total = np.prod(sizes)
        # out = [np.nan]*total
        Rs = np.empty((total,n_lams))
        Ts = np.empty((total,n_lams))
        
        for i,par in enumerate(params):
            if not i%500: print(f'case {i}/{total}')
            
            for key,t in zip(keys, par):
                d_list[key] = t
            
            RAT = f(pol, n_list, d_list, th_0, lams, c_list)
            Rs[i] = RAT['R']
            Ts[i] = RAT['T']
            
        print('####finished####',)
        return Rs.reshape(*sizes,n_lams), Ts.reshape(*sizes,n_lams)

              



def calculate_RT_torch(n_list, d_bounds, lams, c_list=None, weights=None, pol='s', th_0=0.0, num_starts=50, num_epochs=150, lr=2.0, use_cuda=False):
    """
    Optimiza el espesor de un stack de capas delgadas usando PyTorch Autograd.
    
    Parámetros:
    - n_list: Tensor complejo de PyTorch (torch.complex128) con los índices [num_capas, num_lams].
    - d_bounds: Lista que define los límites o valores fijos para cada capa finita.
                Puede contener:
                - Tupla/Lista (min, max): para optimizar esa capa dentro de dichos límites.
                - Float/Int o Tupla (val, val): para mantener esa capa con un espesor constante fijo.
    - lams: Tensor con las longitudes de onda.
    - c_list: Lista opcional indicando 'c' (coherente) o 'i' (incoherente) para cada capa.
              Si se provee, utiliza la versión inc_tmm_torch para cálculos gruesos.
    - weights: Tensor opcional con los pesos (por ejemplo, espectro solar * IQE * wavelength)
               para realizar una optimización por reflectancia ponderada (Jsc).
    - pol: Polarización, 's' o 'p'.
    - th_0: Ángulo de incidencia (radianes).
    - num_starts: Cantidad de semillas aleatorias para evitar mínimos locales.
    - num_epochs: Número de iteraciones del Descenso de Gradiente.
    - lr: Learning rate (tasa de aprendizaje) para el optimizador Adam.
    - use_cuda: Si es True, utiliza la GPU para acelerar los cálculos (si está disponible).
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en dispositivo: {device}")
    
    # Mover tensores al dispositivo
    n_list = n_list.to(device)
    lams = lams.to(device)
    
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float64, device=device)
        else:
            weights = weights.to(device)
        # Normalizar pesos para asegurar promedio ponderado correcto
        weights = weights / weights.sum()
    
    num_layers = n_list.shape[0]
    num_finite_layers = num_layers - 2
    num_wl = n_list.shape[1]
    
    if len(d_bounds) != num_finite_layers:
        raise ValueError(f"d_bounds debe tener longitud igual al número de capas finitas ({num_finite_layers}), pero tiene {len(d_bounds)}.")
    
    # 1. ANALIZAR CAPAS OPTIMIZABLES Y FIJAS
    is_optimizable = []
    opt_bounds = []  # Límites (min, max) para optimizables
    fixed_values = {}  # Mapa de idx -> valor_fijo para no optimizables
    
    for idx, b in enumerate(d_bounds):
        if isinstance(b, (tuple, list)):
            d_min, d_max = float(b[0]), float(b[1])
            if d_min == d_max:
                is_optimizable.append(False)
                fixed_values[idx] = d_min
            else:
                is_optimizable.append(True)
                opt_bounds.append((d_min, d_max))
        else:
            is_optimizable.append(False)
            fixed_values[idx] = float(b)
            
    num_opt_layers = sum(is_optimizable)
    print(f"Capas a optimizar: {num_opt_layers} | Capas fijas: {num_finite_layers - num_opt_layers}")
    
    if num_opt_layers == 0:
        raise ValueError("No se definieron capas optimizables en d_bounds (todas son fijas).")
        
    # 2. INICIALIZACIÓN MULTI-START EN EL ESPACIO NO ACOTODO (LOGIT)
    # Inicializamos d_initial en el espacio físico dentro de [d_min, d_max],
    # y luego lo mapeamos a d_opt no acotado para usar el truco del Sigmoide.
    d_initial = torch.zeros((num_starts, num_opt_layers), dtype=torch.float64, device=device)
    opt_idx = 0
    for idx, opt in enumerate(is_optimizable):
        if opt:
            d_min, d_max = opt_bounds[opt_idx]
            d_init_phys = torch.empty(num_starts, device=device).uniform_(d_min, d_max)
            # Mapeo inverso de sigmoide (logit)
            p = (d_init_phys - d_min) / (d_max - d_min)
            p = torch.clamp(p, 1e-7, 1.0 - 1e-7)  # Evitar desbordes numéricos
            d_initial[:, opt_idx] = torch.log(p / (1.0 - p))
            opt_idx += 1
            
    d_opt = d_initial.clone().detach().requires_grad_(True)
    
    # 3. OPTIMIZADOR
    optimizer = optim.Adam([d_opt], lr=lr)
    
    print(f"Iniciando optimización con Autograd ({num_starts} semillas en paralelo)...")
    
    # Columna de infinitos para sustrato/superestrato: [num_starts, 1]
    inf_col = torch.full((num_starts, 1), float('inf'), dtype=torch.float64, device=device)
    
    # Helper para construir d_fisico a partir de d_opt y las capas fijas
    def reconstruct_d_fisico(d_opt_tensor):
        d_fisico_cols = []
        o_idx = 0
        for idx, opt in enumerate(is_optimizable):
            if opt:
                d_min, d_max = opt_bounds[o_idx]
                # Mapeo sigmoide: garantiza d_min <= espesor <= d_max estrictamente
                col = d_min + (d_max - d_min) * torch.sigmoid(d_opt_tensor[:, o_idx])
                d_fisico_cols.append(col)
                o_idx += 1
            else:
                val = fixed_values[idx]
                col = torch.full((d_opt_tensor.shape[0],), val, dtype=torch.float64, device=device)
                d_fisico_cols.append(col)
        return torch.stack(d_fisico_cols, dim=1)
    
    # 4. BUCLE DE OPTIMIZACIÓN (sin loop sobre semillas)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        d_fisico = reconstruct_d_fisico(d_opt)  # [num_starts, num_finite_layers]
        
        # Construir d_list BATCHED: [num_starts, num_layers]
        d_full = torch.cat([inf_col, d_fisico, inf_col], dim=1)
        # Expandir a [num_starts, num_layers, num_wl]
        d_full_3d = d_full.unsqueeze(-1).expand(-1, -1, num_wl)
        
        # UNA SOLA llamada para TODAS las semillas
        if c_list is None:
            result = tmm.coh_tmm_torch(pol=pol, n_list=n_list, d_list=d_full_3d,
                                        th_0=th_0, lam_vac=lams)
        else:
            result = tmm.inc_tmm_torch(pol=pol, n_list=n_list, d_list=d_full_3d, c_list=c_list,
                                        th_0=th_0, lam_vac=lams)
            
        R = result['R']  # [num_starts, num_wl]
        
        # Loss (promedio simple o ponderado por weights)
        if weights is None:
            loss_per_start = R.mean(dim=-1)
        else:
            loss_per_start = (R * weights).sum(dim=-1)
            
        loss = loss_per_start.sum()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            mejor_R_actual = loss_per_start.min().item()
            if weights is None:
                print(f"Epoch {epoch}/{num_epochs} | Mejor Reflectancia Media: {mejor_R_actual*100:.2f}%")
            else:
                print(f"Epoch {epoch}/{num_epochs} | Mejor Reflectancia Media Ponderada: {mejor_R_actual*100:.2f}%")

    # 5. EXTRACCIÓN DEL MÍNIMO GLOBAL
    with torch.no_grad():
        d_final_fisico = reconstruct_d_fisico(d_opt)
        d_full = torch.cat([inf_col, d_final_fisico, inf_col], dim=1)
        d_full_3d = d_full.unsqueeze(-1).expand(-1, -1, num_wl)
        if c_list is None:
            result = tmm.coh_tmm_torch(pol=pol, n_list=n_list, d_list=d_full_3d,
                                        th_0=th_0, lam_vac=lams)
        else:
            result = tmm.inc_tmm_torch(pol=pol, n_list=n_list, d_list=d_full_3d, c_list=c_list,
                                        th_0=th_0, lam_vac=lams)
        R_final = result['R']
    
    if weights is None:
        loss_final = R_final.mean(dim=-1)
    else:
        loss_final = (R_final * weights).sum(dim=-1)
        
    best_idx = loss_final.argmin()
    
    best_thicknesses = d_final_fisico[best_idx].cpu().detach().numpy()
    best_R_curve = R_final[best_idx].cpu().detach().numpy()
    
    print(f"\n¡Optimización completada!")
    print(f"Espesores óptimos encontrados: {best_thicknesses} nm")
    if weights is None:
        print(f"Reflectancia media óptima: {loss_final[best_idx].item()*100:.2f}%")
    else:
        print(f"Reflectancia media ponderada óptima: {loss_final[best_idx].item()*100:.2f}%")
    
    return best_thicknesses, best_R_curve

def load_stack(filename, materials):
    '''Loads stack from Xlsx used by IR-S in Matlab. 'Materials' is a dict 
    where new materials will be added. Only materials not already present will 
    be added. YOU SHOULD CHECK that different nk's have different material names
    '''
    
    wb = load_workbook(filename = filename)
    ws = wb['MJSC Definition']
    
    stack = []
    start_row = 19
    end_row = 35
    
    for row in range(start_row, end_row+1):
        
        mat = ws.cell(row=row, column=3).value #C
        nk = ws.cell(row=row, column=5).value  #E
        thick = ws.cell(row=row, column=7).value #G
        
        if mat not in materials:
            # print('working on '+mat)
            materials[mat] = (load_interp('./nkdata/optical/' + nk))
        
        stack.append([thick, mat, 'c'])

    return  stack, materials

def plot_js(titulo, e_porosa, e_densa, js,espesores_comparación = None,label_comparación = "experimental",label_simulado = "optimo simulado"):
    print(f'caso: {titulo}')
    X, Y = np.meshgrid(e_porosa, e_densa)
    maxj = js.max()
    maxind = np.unravel_index(js.argmax(),js.shape)
    x_opt = X[maxind]
    y_opt = Y[maxind]
    print(f'e_superior: {x_opt}, e_inferior:{y_opt}')
    print(f'Max Jsc = {maxj:.4f}\n')
    Z = 100*js/maxj
    

    fig,ax=plt.subplots(1,1)
    colors =['blue','navy','indigo','purple','red','orangered','orange','w']
    cp = ax.contourf(X, Y, Z,levels=[70,80,85,90,95,98,99, 100], colors=colors)
    cbar =fig.colorbar(cp) # Add a colorbar to a plot
    # 8/4 le agregue este label
    cbar.set_label('Jsc_Normalizada [%]', rotation=270, labelpad=15)
    # 8/4 le agregue este scatter
    ax.scatter(x_opt, y_opt, color='white', edgecolors='black', s=100, marker='*', label=f'{label_simulado}')
    #ahora el valor experimental de Simon
    if espesores_comparación != None: 
        espesores_Simon = espesores_comparación
        ax.scatter(*espesores_Simon, color='green', edgecolors='black', s=100, marker='*', label=f'{label_comparación}')
    
    ax.set_title(titulo)
    ax.set_xlabel('Espesor superior [nm]')
    ax.set_ylabel('Espesor inferior [nm]')
    ax.legend()
    plt.show()
    
