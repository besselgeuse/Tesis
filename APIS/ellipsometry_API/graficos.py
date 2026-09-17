# region 1. Extraer datos de los archivos txt
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
results_path = './Datos/Datos-04-6/resultados_stack_7.txt'
exp_path = './Datos/Datos-04-6/1TIO2-2.txt'
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

# endregion
# %%
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
# region 2. Extraer datos directamente de la base de datos
#%% 
import models
from database import engine, SessionLocal
import numpy as np
import matplotlib.pyplot as plt
from tmm_utils_Rodrigo import cauchy_fn
#%%
db = SessionLocal()

# Query 
stack = db.query(models.Stack).filter(models.Stack.id == 96).first()

best_Is = np.array(stack.best_Is)
best_Ic = np.array(stack.best_Ic)
wl_exp = np.array(stack.wl_exp)
Is_exp = np.array(stack.Is_exp)
Ic_exp = np.array(stack.Ic_exp)
#%%
lams, err_psi, err_delta = np.loadtxt("./Datos/Datos-25-6/meds/SiO2_N4_Si/std/SiO2_N4_Si_std.txt", unpack=True, skiprows=1)

# 1. Convertir Is_exp e Ic_exp experimentales a psi_exp y delta_exp en radianes para las derivadas
psi_exp = 0.5 * np.arcsin(np.clip(np.sqrt(Is_exp**2 + Ic_exp**2), 0.0, 1.0))
delta_exp = np.arctan2(Is_exp, Ic_exp)

# 2. Interpolar desviaciones del archivo std (grados -> rad) a la grilla de wl_exp
from scipy.interpolate import interp1d
_, unique_idx = np.unique(lams, return_index=True)
unique_idx = np.sort(unique_idx)
lams_clean = lams[unique_idx]
err_psi_rad = np.radians(err_psi[unique_idx])
err_delta_rad = np.radians(err_delta[unique_idx])

err_psi_fn = interp1d(lams_clean, err_psi_rad, bounds_error=False, fill_value="extrapolate")
err_delta_fn = interp1d(lams_clean, err_delta_rad, bounds_error=False, fill_value="extrapolate")

err_psi_interp = err_psi_fn(wl_exp)
err_delta_interp = err_delta_fn(wl_exp)

# 3. Propagación analítica de errores
dIs_dpsi = 2 * np.cos(2 * psi_exp) * np.sin(delta_exp)
dIc_dpsi = 2 * np.cos(2 * psi_exp) * np.cos(delta_exp)
dIs_ddelta = Ic_exp
dIc_ddelta = -Is_exp

Is_err = np.sqrt((dIs_dpsi ** 2) * (err_psi_interp ** 2) + (dIs_ddelta ** 2) * (err_delta_interp ** 2))
Ic_err = np.sqrt((dIc_dpsi ** 2) * (err_psi_interp ** 2) + (dIc_ddelta ** 2) * (err_delta_interp ** 2))

#Voy a usar la desviación estándar como error en Ic y Is calculadas en la linea 512 y 513
#%%
#solo voy a graficar los datos experimentales y calculados, sin los errores
plt.plot(wl_exp,Ic_exp,'k-',label="Ic Experimental")
plt.plot(wl_exp,best_Ic,'r--',label="Ic Calculada")
plt.legend()
plt.show()
plt.plot(wl_exp,Is_exp,'k-',label="Is Experimental")
plt.plot(wl_exp,best_Is,'r--',label="Is Calculada")
plt.legend()
plt.show()
#%%
# 4. Graficar con fill_between para las bandas de error
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Ic
ax[0].plot(wl_exp, Ic_exp, 'k-', label="Ic Experimental")
ax[0].fill_between(wl_exp, Ic_exp - desv_Ic_interp(wl_exp), Ic_exp + desv_Ic_interp(wl_exp), color='red', alpha=0.3, label="Error Ic Exp")
ax[0].plot(wl_exp, best_Ic, 'r--', label="Ic Calculada")
ax[0].legend()
ax[0].grid(True)

# Is
ax[1].plot(wl_exp, Is_exp, 'k-', label="Is Experimental")
ax[1].fill_between(wl_exp, Is_exp - desv_Is_interp(wl_exp), Is_exp + desv_Is_interp(wl_exp), color='green', alpha=0.3, label="Error Is Exp")
ax[1].plot(wl_exp, best_Is, 'r--', label="Is Calculada")
ax[1].legend()
ax[1].grid(True)

ax[0].set_xlim([440, 830])
ax[1].set_xlim([440, 830])
plt.tight_layout()
plt.show()

# %%
#Parametros de la capa de Tio2
capa = 2
A = stack.layers[capa]["params"]["A"]
B = stack.layers[capa]["params"]["B"]
C = stack.layers[capa]["params"]["C"]
thickness = stack.layers[capa]["thickness"]
n_SiO2 = cauchy_fn(A,B,C)(wl_exp)

plt.figure(figsize=(10, 6))
plt.plot(wl_exp,n_SiO2.real,'k-',label="n alumina")
plt.xlim([np.min(wl_exp),np.max(wl_exp)])
plt.ylim([np.min(n_SiO2.real) - 0.1,np.max(n_SiO2.real) + 0.1])
plt.legend()
plt.show()
#%%
#En esta celda voy a graficar los parametros de cauchy ajustados por el ellipsometro directamente.
"""
\par  4) TiO2_trsansparente  A       =   1.6724000 \'fc     0.0078927
\par  5) TiO2_trsansparente  B       =  42.1256800 \'fc     0.8866562
\par  6) TiO2_trsansparente  C       = -26.8226100 \'fc     1.2132820
"""
A = 1.6724000
B = 42.1256800
C = -26.8226100

n_SiO2 = cauchy_fn(A,B,C)(wl_exp)

plt.figure(figsize=(10, 6))
plt.plot(wl_exp,n_SiO2.real,'k-',label="n TiO2")
plt.xlim([np.min(wl_exp),np.max(wl_exp)])
plt.ylim([np.min(n_SiO2.real) - 0.1,np.max(n_SiO2.real) + 0.1])
plt.legend()
plt.show()
# %%
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
# region 3. Graficos del elipsometro y varios
#%%
import numpy as np
import matplotlib.pyplot as plt
from tmm_utils_Rodrigo import cauchy_fn,load_interp
#%%
#Muestras de TiO2
muestra1 = np.loadtxt("Datos/Datos-09-6/TiO2_Si_SP_S1.txt", skiprows = 5,unpack=True,delimiter="\t")
muestra1_ajustada = np.loadtxt("Datos/Datos-09-6/TiO2_Si_SP_S1_fit.txt", skiprows = 5,unpack=True,delimiter="\t")
muestra2 = np.loadtxt("Datos/Datos-11-6/TiO2_Si_Sputtering_D2.txt", skiprows = 5,unpack=True,delimiter="\t")
muestra2_ajustada = np.loadtxt("Datos/Datos-11-6/TiO2_Si_Sputtering_D2_fit.txt", skiprows = 5,unpack=True,delimiter="\t")

wl1,psi1,delta1 = muestra1
wl2,psi2,delta2 = muestra2
wl1_a,psi1_a,delta1_a = muestra1_ajustada
wl2_a,psi2_a,delta2_a = muestra2_ajustada

fig,ax = plt.subplots(2,1,figsize=(10,6))
ax[0].plot(wl1,psi1,'k-',label="psi Experimental")
ax[0].plot(wl1_a,psi1_a,'r--',label="psi ajustado")
ax[0].legend()
ax[1].plot(wl1,delta1,'k-',label="delta Experimental")
ax[1].plot(wl1_a,delta1_a,'r--',label="delta ajustado")
ax[1].legend()
plt.tight_layout()
plt.show()
# %% 
# Mismo para la muestra 2
fig,ax = plt.subplots(2,1,figsize=(10,6))
ax[0].plot(wl2,psi2,'k-',label="psi Experimental")
ax[0].plot(wl2_a,psi2_a,'r--',label="psi ajustado")
ax[0].legend()
ax[1].plot(wl2,delta2,'k-',label="delta Experimental")
ax[1].plot(wl2_a,delta2_a,'r--',label="delta ajustado")
ax[1].legend()
plt.tight_layout()
plt.show()
# %%
"""
Para la muestra 1 tengo estos parametros:
\par  2) TiO2_trsansparente  A =  2.4072740 \'fc 0.0102419
\par  3) TiO2_trsansparente  B = -3.4768180 \'fc 0.6405784
\par  4) TiO2_trsansparente  C = 13.3268100 \'fc 0.9630267
"""
ruta_materiales = r"./indices"
Rutilo = load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)
n_1 = cauchy_fn(2.4072740,-3.4768180,13.3268100)(wl1)
n_rutilo = Rutilo[0](wl1)
plt.figure(figsize=(10, 6))
plt.plot(wl1,n_1.real,'k-',label="n TiO2")
plt.plot(wl1,n_rutilo,'r--',label="n Palik")
plt.xlim([np.min(wl1),np.max(wl1)])
plt.ylim([np.min(n_1.real) - 0.1,np.max(n_rutilo) + 0.1])
plt.legend()
plt.show()
# %%
"""
Para la muestra 2 tengo estos parametros:
\par  2) TiO2_trsansparente  A =  2.2738000 \'fc 0.0034419
\par  3) TiO2_trsansparente  B =  2.7818630 \'fc 0.2146606
\par  4) TiO2_trsansparente  C =  7.3597000 \'fc 0.3259595
"""
n_2 = cauchy_fn(2.2738000,2.7818630,7.3597000)(wl2)
plt.figure(figsize=(10, 6))
plt.plot(wl2,n_2.real,'k-',label="n TiO2")
plt.plot(wl2,n_rutilo,'r--',label="n Palik")
plt.xlim([np.min(wl2),np.max(wl2)])
plt.ylim([np.min(n_1.real) - 0.1,np.max(n_rutilo) + 0.1])
plt.legend()
plt.show()
# %%
#Ahora la comparación de los indices de las dos muestras
plt.figure(figsize=(10, 8))
plt.plot(wl1,n_1.real,'b-',label="n TiO2 Muestra 1")
plt.plot(wl2,n_2.real,'r-',label="n TiO2 Muestra 2")
plt.xlim([np.min(wl1),np.max(wl1)])
plt.ylim([np.min(n_1.real) - 0.1,np.max(n_2.real) + 0.1])
plt.legend()
plt.show()
#%%
import sys
import numpy as np
import importlib
from tmm_utils_Rodrigo import constant_fn,cauchy_fn,calculate_RT,brugg_fn,load_interp
import models
from database import engine, SessionLocal
import matplotlib.pyplot as plt
ruta_materiales = r"./indices"
# 1. Cargar tmm_core de manera segura
spec = importlib.util.spec_from_file_location("tmm", "./tmm_core.py")
tmm = importlib.util.module_from_spec(spec)
sys.modules["tmm"] = tmm
spec.loader.exec_module(tmm)
# Ahora voy a graficar las reflectancias y compararlas con la simulación usando el tmm.
#R_TiO2 = np.loadtxt("./Datos/mediciones_de_reflectancia/11-6/2TiO2N.txt")
#R_lams = np.linspace(190,900,len(R_TiO2))

db = SessionLocal()

# Query 
stack = db.query(models.Stack).filter(models.Stack.id == 95).first()


capa = 1
A = stack.layers[capa]["params"]["A"]
B = stack.layers[capa]["params"]["B"]
C = stack.layers[capa]["params"]["C"]
thickness = stack.layers[capa]["thickness"]

f_air = stack.layers[capa-1]["params"]["f_air"]
thickness_poroso = stack.layers[capa-1]["thickness"]

materials = {}
materials["TiO2_SP"] = (cauchy_fn(A,B,C),constant_fn(0.0))
materials["TiO2_SP_poroso"] = (brugg_fn(cauchy_fn(A,B,C),constant_fn(1.0),f_air), load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows=1)[1])
materials["Si"] = (load_interp(f'{ruta_materiales}/nkdata/optical/Si.nk', skiprows=1))
materials["air"] = (constant_fn(1.0),constant_fn(0.0))

stack = [[np.inf,"air", "i"],
         [thickness_poroso, "TiO2_SP_poroso", "c"],
         [thickness, "TiO2_SP", "c"],
         [np.inf, "Si", "i"]]

#calculo de reflectancia
RT = calculate_RT(th_0=0, stack=stack, materials=materials,lams = R_lams)
ref,trans = RT

plt.figure(figsize=(10, 6))
plt.plot(R_lams,R_TiO2,'k-',label="R Experimental")
plt.plot(R_lams,ref,'r--',label="R Calculada")
plt.xlim([430,850])
plt.ylim([0,0.6])
plt.legend()
plt.show()
#%%
#========================================================
#========================================================
#========================================================
#========================================================
#========================================================
#========================================================
# Region 4: trabajando con datos directos del elipsometro
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
dir_datos = './Datos/Datos-08-7/meds'
material = 'TiO2_recocido'
muestra = 'TiO2_recocido_SiO2(T4)'
dir_muestra = f'{dir_datos}/{material}'

# Cargar psi, delta, Ic e Is directamente de los archivos excluyendo el footer de resúmenes
psi = {}
delta = {}
Ic = {}
Is = {}
lams_dict = {}
from io import StringIO
for i in range(1, 4):
    lines = []
    with open(f'{dir_muestra}/{muestra}.M{i}.txt', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '# MINIMA:' in line:
                break
            lines.append(line)
            
    lams_dict[i], psi[i], delta[i], Ic[i], Is[i] = np.loadtxt(
        StringIO("".join(lines)), 
        skiprows=63, 
        usecols=(0, 1, 2, 3, 4), 
        unpack=True
    )
    plt.plot(lams_dict[i],Is[i],'o',label=f'Is M{i}')
    plt.plot(lams_dict[i],Ic[i],'o',label=f'Ic M{i}')
plt.legend()
plt.ylim(-1,1)
plt.xlim(430,850)
plt.show()

# Definimos una grilla de longitud de onda común (usaremos las lams de M1)
lams = lams_dict[1]

# Interpolar todas las mediciones a la grilla común lams para resolver tamaños inhomogéneos
from scipy.interpolate import interp1d
for i in range(1, 4):
    psi_fn = interp1d(lams_dict[i], psi[i], bounds_error=False, fill_value="extrapolate")
    Ic_fn = interp1d(lams_dict[i], Ic[i], bounds_error=False, fill_value="extrapolate")
    Is_fn = interp1d(lams_dict[i], Is[i], bounds_error=False, fill_value="extrapolate")
    
    psi[i] = psi_fn(lams)
    Ic[i] = Ic_fn(lams)
    Is[i] = Is_fn(lams)
    
    # Interpolar delta de forma circular usando cos/sin para evitar discontinuidades en 0/360
    delta_rad = np.radians(delta[i])
    cos_d_fn = interp1d(lams_dict[i], np.cos(delta_rad), bounds_error=False, fill_value="extrapolate")
    sin_d_fn = interp1d(lams_dict[i], np.sin(delta_rad), bounds_error=False, fill_value="extrapolate")
    
    cos_d_interp = cos_d_fn(lams)
    sin_d_interp = sin_d_fn(lams)
    delta[i] = np.degrees(np.arctan2(sin_d_interp, cos_d_interp)) % 360

# 1. Medias de Is e Ic (para el promedio vectorial/circular)
media_Ic = np.mean([Ic[i] for i in range(1, 4)], axis=0)
media_Is = np.mean([Is[i] for i in range(1, 4)], axis=0)
desv_Is = np.std([Is[i] for i in range(1, 4)], axis=0, ddof=1)
desv_Ic = np.std([Ic[i] for i in range(1, 4)], axis=0, ddof=1)
desv_Ic_interp = interp1d(lams, desv_Ic, bounds_error=False, fill_value="extrapolate")
desv_Is_interp = interp1d(lams, desv_Is, bounds_error=False, fill_value="extrapolate")


fig,ax = plt.subplots(2,1,figsize=(10,8))
ax[0].errorbar(lams, media_Is, yerr=desv_Is, fmt='o', label='Is_mean')
ax[1].errorbar(lams, media_Ic, yerr=desv_Ic, fmt='o', label='Ic_mean')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set_xlim(430,850)
ax[1].set_xlim(430,850)
ax[1].set_ylim(-1,1)
ax[0].set_ylim(-1,1)
plt.tight_layout()
plt.show()

# %%
plt.plot(lams,desv_Is,'ko',label='desv_Is')
plt.plot(lams,desv_Ic,'ro',label='desv_Ic')
plt.ylim(-0.05,1)
plt.xlim(430,850)
plt.legend()
plt.show()
#%%
# Convertir las medias vectoriales a psi y delta en grados
# Usamos np.clip para evitar que el ruido supere 1.0 en el arcsin
arg_arcsin = np.clip(np.sqrt(media_Ic**2 + media_Is**2), 0.0, 1.0)
media_psi = 0.5 * np.degrees(np.arcsin(arg_arcsin))
media_delta = np.degrees(np.arctan2(media_Is, media_Ic)) % 360
# 2. Desviación estándar de psi (directa en grados, estable y sin arcsin)
desv_psi = np.std([psi[i] for i in range(1, 4)], axis=0, ddof=1)
# 3. Desviación estándar de delta (con alineación circular para evitar discontinuidades)
delta_aligned = []
for i in range(1, 4):
    diff = delta[i] - media_delta
    # Corregir la discontinuidad 0/360 mapeando la diferencia a [-180, 180]
    diff = (diff + 180) % 360 - 180
    delta_aligned.append(media_delta + diff)
desv_delta = np.std(delta_aligned, axis=0, ddof=1)

fig,ax = plt.subplots(2,1,figsize=(10,8))
ax[0].errorbar(lams, media_psi, yerr=desv_psi, fmt='o', label='psi_mean')
ax[1].errorbar(lams, media_delta, yerr=desv_delta, fmt='o', label='delta_mean')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set_xlim(430,850)
ax[1].set_xlim(430,850)
# ax[0].set_ylim(-1,100)

# ax[1].set_ylim(-1,2)
plt.tight_layout()
plt.show()
#%%
# Ahora voy a querer guardar estos datos de media y sigma
np.savetxt(f'{dir_muestra}/mean/{muestra}_media.txt', np.c_[lams, media_psi, media_delta], header='lams, psi_mean, delta_mean', fmt='%.6f')
np.savetxt(f'{dir_muestra}/std/{muestra}_std.txt', np.c_[lams, desv_psi, desv_delta], header='lams, desv_psi, desv_delta', fmt='%.6f')
# %%
#verifico que la conversión es correcta
media_Ic_reconst = np.sin(np.radians(media_psi)*2)*np.cos(np.radians(media_delta))
media_Is_reconst = np.sin(np.radians(media_psi)*2)*np.sin(np.radians(media_delta))

fig,ax = plt.subplots(2,1,figsize=(10,8))
ax[0].plot(lams,media_Is_reconst,'ro',label='Is_reconst')
ax[0].plot(lams,media_Is,'ko',label='Is_mean')
ax[1].plot(lams,media_Ic_reconst,'ro',label='Ic_reconst')
ax[1].plot(lams,media_Ic,'ko',label='Ic_mean')
ax[0].grid()
ax[1].grid()
ax[0].legend()
ax[1].legend()
ax[0].set_xlim(430,850)
ax[1].set_xlim(430,850)
ax[1].set_ylim(-1,1)
ax[0].set_ylim(-1,1)
plt.tight_layout()
plt.show()
# %%
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
# Región 4 graficos variados
# %%
import matplotlib.pyplot as plt
import numpy as np
from tmm_utils_Rodrigo import load_interp
ruta_materiales = './indices'
materials = {
    'SiO2_nk' : (load_interp(f'{ruta_materiales}/nkdata/SiO2_Palik.nk',skiprows=1,unit='um')),
    'SiO2_nkv' : (load_interp(f'{ruta_materiales}/nkdata/SiO2.nkv',skiprows=1))
}

lams = np.linspace(300,900,1000)

plt.figure(figsize=(10,6))
plt.plot(lams,materials['SiO2_nkv'][0](lams).real,'b-',label="n SiO2 nkv")
plt.plot(lams,materials['SiO2_nk'][0](lams).real,'g--',label="n SiO2 nk")
plt.ylim(1,2)
plt.legend()
plt.show()


# %%
