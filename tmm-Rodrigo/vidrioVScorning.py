#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
ruta_vidrio = r'./indices/nkdata/optical/Si.nk'
ruta_corning = r'C:\Users\Maria Lujan\Desktop\Rodrigo\programas\optical-master\n\silicon.in3'

datos_vidrio = np.loadtxt(f'{ruta_vidrio}', skiprows=1)
data_n_corning = np.loadtxt(f'{ruta_corning}', skiprows=3, max_rows=259)
data_k_corning = np.loadtxt(f'{ruta_corning}', skiprows=263, max_rows=321)
#%%
lams_vidrio = datos_vidrio[:,0]
n_vidrio = datos_vidrio[:,1]
k_vidrio = datos_vidrio[:,2]

lams_n_corning = data_n_corning[:,0]/10
n_corning = data_n_corning[:,1]
lams_k_corning = data_k_corning[:,0]/10
k_corning = data_k_corning[:,1]
#%%
plt.figure()
plt.plot(lams_vidrio,n_vidrio,"purple",label = "n del Si.nk")
plt.plot(lams_n_corning,n_corning,'o--',ms= 4,color ="orange",label = "n del silicon.in3")
plt.legend()
plt.grid()
plt.xlim([200,1000])
plt.xlabel(r"longitudes de onda ($\lambda$) [nm]")
plt.ylabel("indices de refracciòn (n)")
# %%
plt.figure()
plt.plot(lams_vidrio,k_vidrio,"purple",label = "k del Si.nk")
plt.plot(lams_k_corning,k_corning,'o--',ms= 4,color ="orange",label = "k del silicon.in3")
plt.legend()
plt.grid()
plt.xlim([200,1000])
plt.xlabel(r"longitudes de onda ($\lambda$) [nm]")
plt.ylabel("indices de refracciòn (k)")
# %%
