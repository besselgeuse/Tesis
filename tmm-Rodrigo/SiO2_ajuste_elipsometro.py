#%%
import numpy as np 
import matplotlib.pyplot as plt
from tmm_utils_Rodrigo import cauchy_fn,constant_fn,load_interp
#%%
A = 2.4159710
B = 101.5135000
C = -198.7589000

SiO2_elipsometro = cauchy_fn(A,B,C)
SiO2_palik= load_interp(r"./indices/nkdata/SiO2_Palik.nk",skiprows=1,unit='um')
lams = np.arange(300,901,1)
# %%
plt.plot(lams,SiO2_elipsometro(lams),label = "n_SiO2_elipsometro")
plt.plot(lams,SiO2_palik[0](lams),label = "n_SiO2_palik")
plt.grid()
plt.xlabel("longitud de onda (nm)")
plt.ylabel("índice de refracción")
plt.legend()
# %%