#%%
import pvlib
import pandas as pd
import matplotlib.pyplot as plt

# Definir ubicación y tiempos
loc = pvlib.location.Location(-34.60, -58.38, tz='America/Argentina/Buenos_Aires')
tiempos = pd.date_range(start='2026-05-12 08:00', end='2026-05-12 18:00', freq='30min')

# Calcular posición solar
solpos = loc.get_solarposition(tiempos)
# 'apparent_zenith' es el ángulo respecto a la normal (el que usas en TMM)
angulos_incidencia = solpos['apparent_zenith']
#%%
print(angulos_incidencia)
print(solpos.head())

# %%
# Graficar ángulos en función de la hora
plt.figure(figsize=(10, 6))
plt.plot(range(len(angulos_incidencia)), angulos_incidencia.values, marker='o', linewidth=2, markersize=6)
plt.xlabel('Hora del día', fontsize=12)
plt.ylabel('Ángulo de incidencia (°)', fontsize=12)
plt.title('Ángulo de incidencia solar vs Hora - Buenos Aires (12/05/2026)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(range(len(angulos_incidencia)), [t.strftime('%H:%M') for t in angulos_incidencia.index], rotation=45)
plt.tight_layout()
plt.show()
# %%
print("hola mundo de git hub")
# %%
