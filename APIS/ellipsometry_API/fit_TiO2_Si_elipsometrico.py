#%%
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones del módulo local
from fit_elipsometrico import (
    ajuste_elipsometrico, transform_I_to_psi_delta
)
from tmm_utils_Rodrigo import cauchy_fn, n_eff
#%%

data_path = r'./Datos-28-5/TiO2_Si_Sputtering_sincinta.txt'
skiprows = 5
layer_names = ['air', 'T1_porosa', 'T1_densa', 'Si']
layer_models = [
    None,  # air (capa estática)
    {
        'model': 'bruggeman',
        'f_bounds': [
            (0., 1.)   # f_air
        ]
    },
    {
        'model': 'cauchy',
        'bounds': [
            (0., 30.),   # A (TiO2 densa)
            (-10., 50.),  # B
            (-100., 100.)   # C
        ]
    },
    None  # Si (capa estática)
]
d_bounds = [(0., 40.), (0., 80.)] 
theta_0 = 69.5
num_starts = 50
num_epochs = 50
lr = 1
use_cuda = True

best_thicknesses, best_Is, best_Ic, best_params, wl_exp, Is_exp, Ic_exp = ajuste_elipsometrico(
    data_path, skiprows, layer_names, layer_models, d_bounds, 
    theta_0, num_starts, num_epochs, lr, use_cuda)
    
#%%
#Ahora obtener las curvas teóricas de y psi y delta y graficarlas contra el ajuste
wl_exp,psi_exp,delta_exp = np.loadtxt(data_path,skiprows=skiprows,unpack=True)

# Convertir ángulos de grados a radianes
psi_exp = np.radians(psi_exp)
delta_exp = np.radians(delta_exp)

# Definir parámetros elipsométricos experimentales Is e Ic
Is_exp = np.sin(2 * psi_exp) * np.sin(delta_exp)
Ic_exp = np.sin(2 * psi_exp) * np.cos(delta_exp)

fig,ax = plt.subplots(2,1,figsize=(12,8))
ax[0].plot(wl_exp,Is_exp,label='Is_exp')
ax[0].plot(wl_exp,best_Is,label='Is_fit')
ax[1].plot(wl_exp,Ic_exp,label='Ic_exp')
ax[1].plot(wl_exp,best_Ic,label='Ic_fit')
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel('Wavelength [nm]')
ax[0].set_ylabel('Is')
ax[1].set_xlabel('Wavelength [nm]')
ax[1].set_ylabel('Ic')

plt.show()
# %%
# Ahora voy a graficar el n_fit
A, B, C = best_params['T1_densa']['A'], best_params['T1_densa']['B'], best_params['T1_densa']['C']
n_fit_densa = cauchy_fn(A, B, C)(wl_exp)

f_air = best_params['T1_porosa']['f_air']
n_fit_porosa = np.array([n_eff(nb, 1.0, f_air) for nb in n_fit_densa])

plt.figure(figsize=(10, 6))
plt.plot(wl_exp, n_fit_densa, 'r-', label='n_fit T1_densa (Cauchy)')
plt.plot(wl_exp, n_fit_porosa.real, 'b--', label='n_fit T1_porosa (Bruggeman)')
plt.xlabel('Wavelength [nm]')
plt.ylabel('n')
plt.title('Índices de Refracción Ajustados')
plt.legend()
plt.grid(True)
plt.show()
# %%
