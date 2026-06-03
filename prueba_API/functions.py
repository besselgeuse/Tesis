# pyrefly: ignore [missing-import]
import numpy as np

#Voy a querer que esta función calcule la trayectoria en el tiempo dado un ángulo y velocidad inicial 

def Tiro_parabolico(th0,vi, g=9.8):
    t = np.linspace(0, 2 * vi * np.sin(th0 * np.pi / 180) / g, 100) #linspace entre el tiempo inicial y final con 100 puntos
    x = vi * t * np.cos(th0 * np.pi / 180)
    y = vi * t * np.sin(th0 * np.pi / 180) - 0.5 * g * t**2
    return x, y, t


