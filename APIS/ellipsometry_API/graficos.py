#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import ast
import sys
import importlib.util

# 1. Cargar tmm_core de manera segura
spec = importlib.util.spec_from_file_location("tmm", "./tmm_core.py")
tmm = importlib.util.module_from_spec(spec)
sys.modules["tmm"] = tmm
spec.loader.exec_module(tmm)

from tmm_utils_Rodrigo import cauchy_fn, n_eff, load_interp, brugg_fn, constant_fn

# Ruta del archivo de resultados
results_path = './Datos-04-6/resultados_stack_6.txt'
exp_path = './Datos-04-6/1TIO2-2.txt'
#%%
# 2. Parsear de forma robusta e inmune a problemas de codificación ASCII/UTF-8
def parse_stack_info(filepath):
    layers = []
    current_layer = {}
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        in_stack_info = False
        for line in f:
            line = line.strip()
            
            # Búsqueda segura basada en subcadenas ASCII en minúsculas
            if "informaci" in line.lower() and "stack" in line.lower():
                in_stack_info = True
                equals_count = 0
                continue
            if in_stack_info and "=====" in line:
                equals_count += 1
                if equals_count > 1:
                    if current_layer:
                        layers.append(current_layer)
                    break
                continue
            
            if in_stack_info:
                if line.lower().startswith("material:"):
                    if current_layer:
                        layers.append(current_layer)
                    current_layer = {"material": line.split(":", 1)[1].strip()}
                elif "modelo" in line.lower() and "dispers" in line.lower():
                    current_layer["dispersion_model"] = line.split(":", 1)[1].strip()
                elif "espesor:" in line.lower():
                    esp_str = line.split(":", 1)[1].replace("nm", "").strip()
                    current_layer["thickness"] = float(esp_str)
                elif "par" in line.lower() and "model" in line.lower():
                    param_str = line.split(":", 1)[1].strip()
                    try:
                        current_layer["params"] = ast.literal_eval(param_str)
                    except Exception as e:
                        print(f"Error al parsear parámetros: {e}")
                        current_layer["params"] = {}
                        
    return layers

# Obtener información de las capas
layers = parse_stack_info(results_path)

print("Capas detectadas en el ajuste:")
for idx, layer in enumerate(layers):
    print(f"Capa {idx}: {layer['material']} | Modelo: {layer['dispersion_model']} | Espesor: {layer.get('thickness', 'N/A')} nm")
    if 'params' in layer:
        print(f"  Parámetros: {layer['params']}")

# 3. Cargar el espectro experimental de reflectancia (1TIO2-2.txt)
R_exp = np.loadtxt(exp_path, skiprows=5, unpack=True)
wl_exp = np.linspace(190, 900, len(R_exp))

# 4. Reconstruir n_list, d_list y c_list para el cálculo de reflectancia
n_list = []
d_list = []
c_list = []

# Encontrar los parámetros de Cauchy de antemano para usar de base en Bruggeman
cauchy_layer = None
for lay in layers:
    if lay.get("dispersion_model") == "cauchy":
        cauchy_layer = lay
        break

for idx, layer in enumerate(layers):
    mat_name = layer["material"]
    model = layer.get("dispersion_model")
    
    # c_list: Puntas son 'i' (incoherente), en medio son 'c' (coherente)
    if idx == 0 or idx == len(layers) - 1:
        c_list.append('i')
        d_list.append(np.inf)
    else:
        c_list.append('c')
        d_list.append(layer["thickness"])
        
    # n_list: Cálculo del índice de refracción complejo
    if mat_name == "air":
        n_layer = np.ones_like(wl_exp, dtype=complex)
    elif mat_name == "Si":
        n_Si_fn, k_Si_fn = load_interp('./indices/nkdata/optical/Si.nk', skiprows=1)
        n_layer = n_Si_fn(wl_exp) + 1j * k_Si_fn(wl_exp)
    elif model == "cauchy":
        params = layer["params"]
        A = params.get('A')
        B = params.get('B')
        C = params.get('C')
        n_fn = cauchy_fn(A, B, C)
        n_layer = n_fn(wl_exp).astype(complex)
    elif model == "bruggeman":
        params = layer.get("params", {})
        f_air = params.get("f_air", 0.5)
        linked_name = params.get("linked_to")
        
        # Buscar la capa Cauchy que corresponda a linked_name
        linked_cauchy_layer = None
        for lay in layers:
            if lay.get("material") == linked_name and lay.get("dispersion_model") == "cauchy":
                linked_cauchy_layer = lay
                break
                
        if linked_cauchy_layer is None:
            # Fallback a la primera capa Cauchy encontrada
            for lay in layers:
                if lay.get("dispersion_model") == "cauchy":
                    linked_cauchy_layer = lay
                    break
                    
        if linked_cauchy_layer is None:
            raise ValueError("No se encontró una capa de Cauchy que sirva como base para la capa de Bruggeman.")
            
        c_params = linked_cauchy_layer["params"]
        A_c = c_params.get('A')
        B_c = c_params.get('B')
        C_c = c_params.get('C')
        n_c_fn = cauchy_fn(A_c, B_c, C_c)
        
        n_b_fn = constant_fn(1.0)
        n_eff_fn = brugg_fn(n_c_fn, n_b_fn, f_air)
        n_layer = n_eff_fn(wl_exp).astype(complex)
    else:
        # Intentar cargar desde el diccionario de materiales de fit_elipsometrico.py
        try:
            from fit_elipsometrico import materials
            if mat_name in materials:
                n_fn, k_fn = materials[mat_name]
                n_layer = n_fn(wl_exp) + 1j * k_fn(wl_exp)
            else:
                raise ValueError(f"Material '{mat_name}' no registrado.")
        except Exception as e:
            raise ValueError(f"Error al cargar material '{mat_name}': {e}")
            
    n_list.append(n_layer)

n_list = np.array(n_list)
d_list = np.array(d_list, dtype=float)

print("\nEstructura armada para cálculo TMM:")
print(f"n_list shape: {n_list.shape}")
print(f"d_list: {d_list}")
print(f"c_list: {c_list}")

# 5. Ejecutar la simulación TMM usando inc_tmm a incidencia normal (theta_0 = 0)
print("\nEjecutando simulación TMM (inc_tmm) a incidencia normal...")
tmm_results = tmm.inc_tmm('s', n_list, d_list, c_list, th_0=0, lam_vac=wl_exp)
R_calc = tmm_results['R']
T_calc = tmm_results['T']
print("listo")
#%%
# 6. Graficar comparación de reflectancias
plt.close('all')
plt.figure(figsize=(10, 6))
plt.plot(wl_exp, R_exp, 'k-', label='Reflectancia Experimental (1TIO2-2.txt)', linewidth=1.5)
plt.plot(wl_exp, R_calc, 'r--', label='Reflectancia Calculada (TMM)', linewidth=2)
plt.xlabel('Longitud de onda (nm)', fontsize=12)
plt.ylabel('Reflectancia', fontsize=12)
plt.title('Comparación de Reflectancia: Experimental vs Teórica (TMM)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlim([450,850])
plt.tight_layout()
plt.savefig('reflectancia_comparacion.png', dpi=300)
plt.show()
# %%
wl,best_Ic,best_Is,Ic_exp,Is_exp = np.loadtxt(results_path, skiprows=26, unpack=True,delimiter=",")
fig,ax = plt.subplots(2,1,figsize=(10,6))
ax[0].plot(wl,Ic_exp,'k-',label="Ic Experimental")
ax[0].plot(wl,best_Ic,'r--',label="Ic Calculada")
ax[0].legend()
ax[1].plot(wl,Is_exp,'k-',label="Is Experimental")
ax[1].plot(wl,best_Is,'r--',label="Is Calculada")
ax[1].legend()
plt.tight_layout()
plt.show()
# %%
#Ahora voy a querer graficar los indices de refracción
n_TiO2 = n_list[2,:]
plt.figure()
plt.plot(wl_exp,n_TiO2.real,'k-',label="n TiO2")
plt.xlim([np.min(wl),np.max(wl)])
plt.ylim([3,5])
plt.legend()
plt.show()
# %%
