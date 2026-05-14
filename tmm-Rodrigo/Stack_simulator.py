import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from Materials_config import obtener_diccionario_materiales
# Primero configuramos la ruta
Dirección_tmm_Rodrigo = r'./'
if Dirección_tmm_Rodrigo not in sys.path:
    sys.path.append(Dirección_tmm_Rodrigo)

# Ahora importamos todo globalmente
from tmm_utils_Rodrigo import (load_interp, RT_with_cache, load_fn, 
                               plot_js, cauchy_fn, constant_fn, 
                               brugg_fn, stack2tmm)
def ejecutar_simulacion(stack,ruta_IQE,thicks_input,ruta_materiales = './indices',File_direction = 'None', ruta_catalogo='catalogo_de_materiales.txt'):
        """
        Ejecuta la simulación TMM y retorna los resultados.

        Returns:
            tuple: (ref, trans, lams, js_am0, js_am15, thicks, figs)
                - ref: Matriz de reflectancia
                - trans: Matriz de transmitancia
                - lams: Array de longitudes de onda
                - js_am0: Densidad de corriente AM0
                - js_am15: Densidad de corriente AM1.5
                - thicks: Diccionario de espesores
                - figs: Lista de figuras matplotlib generadas
        """

        # 1. Definición de materiales (como lo tenías)
        modelo_T1 = cauchy_fn(2.338, 1.906, 0.824)
        f_T1 = 0.58
        ruta_materiales = f'{ruta_materiales}'

        materials = obtener_diccionario_materiales(ruta_materiales,stack, ruta_catalogo=ruta_catalogo)

        stack_T1_glass = stack


        # Ahora defino las constantes para calcular la Js_max
        step = 1.0
        lams = np.arange(300, 901, step)

        am0 = load_fn(f'{ruta_materiales}/nkdata/am0.txt')(lams)
        am15 = load_fn(f'{ruta_materiales}/nkdata/am15g.txt')(lams)
        IQE = load_fn(f'{ruta_IQE}',delimiter=" ")(lams)

        e = 1.60218e-19 # A.s
        h = 6.6226E-34 # J·s
        c = 2.9979e17 # nm/s

        const = e/(h*c)

        weight = const*IQE*lams

        # 8/4 cambie la funcion trapezoid por trapezoid
        jmax_am0 = np.trapezoid(am0*weight, dx=step)
        jmax_am15 = np.trapezoid(am15*weight, dx=step)


        print('T1 encapsulado')

        thicks = thicks_input

        # file = './transfermatrix/RAT_tio2/T1_air_20-120_20-100_300_900.pickle'
        if File_direction == 'No_guardar':
               file = None
        elif File_direction == 'None':
               today_str = datetime.now().strftime('%Y/%m/%d')
               files_dir = f'./Files/{today_str}'
               os.makedirs(files_dir, exist_ok=True)
               tiempo_seguro = datetime.now().strftime("%Y%m%d_%H%M%S")
               file = f'{files_dir}/archivo_{tiempo_seguro}.pickle'
        else:
               file = File_direction

        # Si thicks está vacío, no hay optimización
        if thicks is not None and len(thicks) == 0:
               thicks = None

               
        RT = RT_with_cache(file,stack_T1_glass, materials, lams, thicks=thicks)
        ref, trans = RT

        figs = []
        n_thicks = len(thicks) if thicks is not None else 0

        if n_thicks == 0:
                # Sin optimización: ref y trans son arrays 1D (una sola configuración)
                js_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step)
                js_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step)
                print(f'Jsc AM0 = {js_am0:.4f}')
                print(f'Jsc AM1.5 = {js_am15:.4f}')

        elif n_thicks == 1:
                # 1 espesor: ref shape (N_espesores, n_lams)
                js_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
                js_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T
                T_js_am0 = np.trapezoid(trans*am0*weight, dx=step).T

                keys = list(thicks.keys())
                espesores = thicks[keys[0]]

                maxj = js_am0.max()
                max_idx = js_am0.argmax()
                espesor_opt = espesores[max_idx]
                print(f'Espesor óptimo capa {keys[0]}: {espesor_opt} nm')
                print(f'Max Jsc AM0 = {maxj:.4f}')

                fig1 = _plot_js_1d('Jsc vs Espesor - Reflectancia', espesores, js_am0, keys[0])
                figs.append(fig1)
                fig2 = _plot_js_1d('Jsc vs Espesor - Transmitancia', espesores, T_js_am0, keys[0])
                figs.append(fig2)

        elif n_thicks == 2:
                js_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
                js_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T
                T_js_am0 = np.trapezoid(trans*am0*weight, dx=step).T

                keys = list(thicks.keys())
                
                # Generar figuras de contorno sin plt.show()
                fig1 = _plot_js_2d('Stack simulado - Reflectancia', thicks[keys[0]], thicks[keys[1]], js_am0)
                figs.append(fig1)
                fig2 = _plot_js_2d('Stack simulado - Transmitancia', thicks[keys[0]], thicks[keys[1]], T_js_am0)
                figs.append(fig2)
                        
                maxj = js_am0.max()
                maxind = np.unravel_index(js_am0.argmax(),js_am0.shape)
                print(f"Espesor óptimo capa {keys[0]}: {thicks[keys[0]][maxind[1]]} nm")
                print(f"Espesor óptimo capa {keys[1]}: {thicks[keys[1]][maxind[0]]} nm")
                print(f'Max Jsc AM0 = {maxj:.4f}')

        else:
                # 3 o más espesores: no se grafica, solo se reporta el óptimo
                js_am0 = jmax_am0 - np.trapezoid(ref*am0*weight, dx=step).T
                js_am15 = jmax_am15 - np.trapezoid(ref*am15*weight, dx=step).T

                keys = list(thicks.keys())
                maxj = js_am0.max()
                maxind = np.unravel_index(js_am0.argmax(), js_am0.shape)
                for i, key in enumerate(keys):
                        print(f"Espesor óptimo capa {key}: {thicks[key][maxind[i]]} nm")
                print(f'Max Jsc AM0 = {maxj:.4f}')
                print('(3+ espesores: no se genera gráfico de contorno)')

        return ref, trans, lams, js_am0, js_am15, thicks, figs


def _plot_js_1d(titulo, espesores, js, capa_idx):
    """Genera una figura 1D de Jsc vs espesor para una sola capa optimizada."""
    maxj = js.max()
    max_idx = js.argmax()
    espesor_opt = espesores[max_idx]
    print(f'caso: {titulo}')
    print(f'Espesor óptimo: {espesor_opt} nm, Max Jsc = {maxj:.4f}\n')

    fig, ax = plt.subplots(1, 1)
    ax.plot(espesores, js, 'o-', ms=3, color='navy', label='Jsc')
    ax.axvline(x=espesor_opt, color='red', linestyle='--', alpha=0.7, label=f'Óptimo: {espesor_opt} nm')
    ax.scatter(espesor_opt, maxj, color='red', edgecolors='black', s=100, marker='*', zorder=5, label=f'Max Jsc = {maxj:.4f}')
    ax.set_title(titulo)
    ax.set_xlabel(f'Espesor capa {capa_idx} [nm]')
    ax.set_ylabel('Jsc [mA/cm²]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def _plot_js_2d(titulo, e_porosa, e_densa, js):
    """Genera una figura de contorno Jsc sin mostrarla (para embeber en la interfaz)."""
    X, Y = np.meshgrid(e_porosa, e_densa)
    maxj = js.max()
    maxind = np.unravel_index(js.argmax(), js.shape)
    x_opt = X[maxind]
    y_opt = Y[maxind]
    print(f'caso: {titulo}')
    print(f'e_superior: {x_opt}, e_inferior:{y_opt}')
    print(f'Max Jsc = {maxj:.4f}\n')
    Z = 100 * js / maxj

    fig, ax = plt.subplots(1, 1)
    colors = ['blue', 'navy', 'indigo', 'purple', 'red', 'orangered', 'orange', 'w']
    cp = ax.contourf(X, Y, Z, levels=[70, 80, 85, 90, 95, 98, 99, 100], colors=colors)
    cbar = fig.colorbar(cp)
    cbar.set_label('Jsc_Normalizada [%]', rotation=270, labelpad=15)
    ax.scatter(x_opt, y_opt, color='white', edgecolors='black', s=100, marker='*', label='optimo simulado')
    ax.set_title(titulo)
    ax.set_xlabel('Espesor superior [nm]')
    ax.set_ylabel('Espesor inferior [nm]')
    ax.legend()
    return fig


def graficar_reflectancia_optima(ref=None, lams=None, js_am0=None, thicks=None, archivo=None):
    """
    Grafica la reflectancia optimizada para un stack dado o cargada de un archivo txt.
    Retorna (R_max, fig, lams) para uso en la interfaz.

    Args:
        ref (np.array): Matriz de reflectancia
        lams (np.array): Array de longitudes de onda
        js_am0 (np.array): Matriz de densidad de corriente para AM0
        thicks (dict or None): Diccionario de espesores optimizados
        archivo (str): Ruta a archivo .txt para cargar (opcional)

    Returns:
        tuple: (R_max, fig, lams)
    """
    if archivo is not None:
        datos = np.loadtxt(archivo, dtype=float, unpack=True, skiprows=1)
        lams = datos[0]
        R_max = datos[1]
        label_plot = "reflectancia cargada"
    else:
        n_thicks = len(thicks) if thicks is not None else 0

        if n_thicks == 0:
            # Sin optimización: ref ya es 1D
            R_max = ref
            label_plot = "reflectancia (sin optimización)"
        elif n_thicks == 1:
            max_idx = js_am0.argmax()
            R_max = ref[max_idx]
            label_plot = "reflectancia optimizada"
        elif n_thicks == 2:
            maxind = np.unravel_index(js_am0.argmax(), js_am0.shape)
            print(maxind)
            R_max = ref[maxind[1], maxind[0]]  # invertidos por culpa del meshgrid
            label_plot = "reflectancia optimizada"
        else:
            maxind = np.unravel_index(js_am0.argmax(), js_am0.shape)
            print(maxind)
            R_max = ref.reshape(js_am0.shape + (len(lams),))[maxind]
            label_plot = "reflectancia optimizada"

    fig = plt.figure()
    plt.plot(lams,R_max,'o-',ms=2,color="green",label=label_plot)
    plt.xlabel(r"longitud de onda ($\lambda$) [nm]")
    plt.ylabel("reflectancia R")
    plt.grid()
    plt.legend()
    return R_max, fig, lams

def guardar_reflectancia(lams,R_max,files = './Files',archivo = None):
    datos_a_guardar = np.column_stack((lams, R_max))

    if files == './Files':
        today_str = datetime.now().strftime('%Y/%m/%d')
        files = f'./Files/{today_str}'
        os.makedirs(files, exist_ok=True)

    # Guardamos en un archivo .txt
    if archivo == None:
        tiempo_seguro = datetime.now().strftime("%Y%m%d_%H%M%S")
        archivo = f'Reflectancia_Optima_{tiempo_seguro}.txt'

    np.savetxt(f"{files}/{archivo}", datos_a_guardar, 
           header='Wavelength(nm) Reflectancia', 
           fmt='%.4f', 
           delimiter='\t')

def comparar_reflectancias(archivo1=None, archivo2=None, archivo1_unidad=1, archivo2_unidad=1):

    """
    Compara la reflectancia de dos archivos dados por sus rutas absolutas.
    Retorna la figura matplotlib para embeber en la interfaz.

    Args:
        archivo1 (str): Ruta completa del primer archivo
        archivo2 (str): Ruta completa del segundo archivo
        archivo1_unidad (float): Unidad del primer archivo (1 o 100%)
        archivo2_unidad (float): Unidad del segundo archivo (1 o 100%)

    Returns:
        fig: Figura matplotlib (o None si falta algún archivo)
    """
        
    if archivo1 == None or archivo2 == None:
        print("Por favor ingrese dos archivos para comparar")
        return None
    else:
        datos1= np.loadtxt(archivo1, dtype=float, unpack=True, skiprows=1)
        datos2 = np.loadtxt(archivo2, dtype=float, unpack=True, skiprows=1)
        #defino las variables para graficar
        lambdas1 = datos1[0]
        R1 = datos1[1]/archivo1_unidad
        lambdas2 = datos2[0]
        R2 = datos2[1]/archivo2_unidad
        fig = plt.figure()
        # Extract filename for the legend
        import os
        name1 = os.path.basename(archivo1)
        name2 = os.path.basename(archivo2)
        plt.plot(lambdas1,R1,'o-',ms=4,label=name1,color="Blue")
        plt.plot(lambdas2,R2,label=name2,color="Orange")
        plt.grid()
        plt.legend()
        plt.xlabel(r"Longitud de onda ($\lambda$) [nm]")
        plt.ylabel("Reflectancia")
        return fig
