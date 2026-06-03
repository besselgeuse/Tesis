#%%
import matplotlib.pyplot as plt
import numpy as np
import json
from database import sessionLocal
import models
from functions import Tiro_parabolico as tp
#%%
id_simulacion = 3

# 1. Abrimos sesión con la base de datos directamente
db = sessionLocal()

# 2. Buscamos el experimento
experimento = db.query(models.trajectory_data).filter(models.trajectory_data.id == id_simulacion).first()
db.close() # Cerramos la sesión porque ya tenemos los datos


# 3. Extraemos y convertimos los datos de JSON (texto) a Arrays de Numpy
x = np.array([json.loads(experimento.x)])[0]
y = np.array([json.loads(experimento.y)])[0]
theta = float(experimento.theta)
vi = float(experimento.v_inicial)

# 4. Graficamos
plt.figure(figsize=(10, 5))
plt.plot(x, y, label=f'Tiro (Vi={vi} m/s, Angulo={theta}°)', color='b', linewidth=2)

# Decoración del gráfico
plt.title(f'Simulación #{experimento.id}')
plt.xlabel('Distancia X (m)')
plt.ylabel('Altura Y (m)')
plt.axhline(0, color='black', linewidth=1) # Línea del suelo
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()
#%%
# 1. Creas el objeto (fila) con los datos que quieras
x,y,t = tp(30,50)
nueva_simulacion = models.trajectory_data(
    theta="30.0", 
    v_inicial="50.0", 
    t=json.dumps(t.tolist()), # Simulando el string del JSON
    x=json.dumps(x.tolist()), 
    y=json.dumps(y.tolist())
)

# 2. La añades a la sesión
db.add(nueva_simulacion)

# 3. Confirmas la transacción para que se guarde en MySQL
db.commit()

print(f"¡Fila añadida exitosamente! ID asignado: {nueva_simulacion.id}")

db.close()

# %%
