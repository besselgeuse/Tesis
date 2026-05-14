#%%
import Stack_simulator as st
import numpy as np

stack = [[np.inf,'air','i'],
         [50,'T1_porosa','c'],
         [50,'T1_densa','c'],
         [25,'InGaP','c'],
         [np.inf,'GaAs','i'],]
ruta_IQE = './indices/IQEGaAs2.txt'

thicks = {
    1:np.arange(10,121,1),
    2:np.arange(10,101,1),
}

st.ejecutar_simulacion(stack,ruta_IQE,thicks)
#%%