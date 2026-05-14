# 20/4 EN ESTE CODIGO VOY A REALIZAR LA SIMULACIÓN DE TiO2 DENSO SOBRE SILICIO CRISTALINO
# Estoy usando la IQE del Si que extraje de la tesis novel uses for titanium dioxide for sillicon solar cells de B.S Richards
#%%
import sys
#import os
import numpy as np
import matplotlib.pyplot as plt

Dirección_tmm_Rodrigo = r'D:\archivos\TESIS\tmm-Rodrigo'

sys.path.append(f'{Dirección_tmm_Rodrigo}')

from tmm_utils_Rodrigo import (load_interp, RT_with_cache, load_fn, plot_js,
                               cauchy_fn, constant_fn, brugg_fn, stack2tmm,
                               )
#%%
modelo_T1 = cauchy_fn(2.338,1.906,0.824)   #muestra T1 del paper (R-1)
f_T1 = 0.58

ruta_materiales = f'{Dirección_tmm_Rodrigo}\indices'

materials = {
    'air':(constant_fn(1.0),constant_fn(0.0)),
    'GaAs': (load_interp(f'{ruta_materiales}/GaAs_Palik.nk', skiprows = 1)),
    'InGaP': (load_interp(f'{ruta_materiales}/InGaPSch.nk', skiprows = 1)),
    'T1_densa': (modelo_T1, constant_fn(0.0)),
    'T1_porosa': (brugg_fn(modelo_T1, constant_fn(1.0), f_T1), constant_fn(0.0)),
    'Rutilo': (load_interp(f'{ruta_materiales}/TiO2Palik.nk', unit='um', skiprows = 1)),
    'MgF2' : (load_interp(f'{ruta_materiales}/nkdata/optical/MgF2.nkv',skiprows=1)),
    'vidrio' : (load_interp(f'{ruta_materiales}/glass_thales.nk',skiprows=1)),
    'Si' : (load_interp(f'{ruta_materiales}/nkdata/Silicio_Palik.nk', skiprows = 1))
}

# Asigno k rutilo a las capas experimentales
materials['T1_densa'] = (materials['T1_densa'][0], materials['Rutilo'][1])
materials['T1_porosa'] = (materials['T1_porosa'][0], materials['Rutilo'][1])

#%%
stack_T1_Si = [
                  [np.inf, 'air', 'i'],
                  [50, 'T1_densa', 'c'],
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

# 8/4 cambie la funcion trapezoid por trapezoid
jmax_am0 = np.trapezoid(am0*weight, dx=step)
jmax_am15 = np.trapezoid(am15*weight, dx=step)
#%%
print('Si con TiO2')

thicks_T1_Si = {1:np.arange(10,151,1)}

# file = './transfermatrix/RAT_tio2/T1_air_20-120_20-100_300_900.pickle'
file = './Files/T1_Si_300-900.pickle'
RT = RT_with_cache(file,stack_T1_Si, materials, lams, thicks=thicks_T1_Si)
ref, trans = RT
# %%
js_T1_Si_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
js_T1_Si_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T

T_js_T1_Si_am0 =  np.trapezoid(trans*am0*weight, dx=step).T
T_js_T1_Si_am15 =  np.trapezoid(trans*am15*weight, dx=step).T

espesor_Si = thicks_T1_Si[1]
#%%
js_max_indice_am0 = js_T1_Si_am0.argmax()
js_max_indice_am15 = js_T1_Si_am15.argmax()

T_js_max_indice_am0 = T_js_T1_Si_am0.argmax()
T_js_max_indice_am15 = T_js_T1_Si_am15.argmax()

# Crear una figura con 2 subplots (1 fila, 2 columnas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 
# Gráfico 1: Corriente (Jsc)
ax1.plot(espesor_Si, js_T1_Si_am0, label='AM0')
ax1.plot(espesor_Si, js_T1_Si_am15, label='AM1.5')
ax1.scatter(espesor_Si[js_max_indice_am0],js_T1_Si_am0[js_max_indice_am0],label="maximo AM0")
ax1.scatter(espesor_Si[js_max_indice_am15],js_T1_Si_am15[js_max_indice_am15],label="maximo AM1.5")
ax1.set_title('(1 - Reflectancia) integrada')
ax1.set_xlabel('Espesor (nm)')
ax1.set_ylabel('Jsc (mA/cm²)')
ax1.legend()
ax1.grid()

# Gráfico 2: Transmitancia integrada
ax2.plot(espesor_Si, T_js_T1_Si_am0, label='AM0')
ax2.plot(espesor_Si, T_js_T1_Si_am15, label='AM1.5')
ax2.scatter(espesor_Si[T_js_max_indice_am0],T_js_T1_Si_am0[T_js_max_indice_am0],label="maximo AM0")
ax2.scatter(espesor_Si[T_js_max_indice_am15],T_js_T1_Si_am15[T_js_max_indice_am15],label="maximo AM15")
ax2.set_title('Transmitancia Integrada')
ax2.set_xlabel('Espesor (nm)')
ax2.legend()
ax2.grid()
plt.tight_layout()
plt.show()
# %%
# Impresión de resultados
print(f"Máximo Jsc AM0: {js_T1_Si_am0[js_max_indice_am0]:.2f} mA/cm² en espesor {espesor_Si[js_max_indice_am0]} nm") 
print(f"Máximo Jsc AM1.5: {js_T1_Si_am15[js_max_indice_am15]:.2f} mA/cm² en espesor {espesor_Si[js_max_indice_am15]} nm") 
print(f"Máxima Transmitancia AM0: {T_js_T1_Si_am0[T_js_max_indice_am0]:.2f} en espesor {espesor_Si[T_js_max_indice_am0]} nm") 
print(f"Máxima Transmitancia AM1.5: {T_js_T1_Si_am15[T_js_max_indice_am15]:.2f} en espesor {espesor_Si[T_js_max_indice_am15]} nm") 
#%%
plt.figure(figsize=[12,8])
plt.plot(lams,ref[js_max_indice_am0],label="reflectancia AM0")
plt.plot(lams,ref[js_max_indice_am15],label="reflectancia AM1.5")
plt.grid()
plt.legend()
plt.xlabel(r'longitud de onda ($\lambda$) [nm]')
plt.ylabel('reflectancia')

# %%
