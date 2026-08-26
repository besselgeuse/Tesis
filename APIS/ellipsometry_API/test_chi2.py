import sys
import os

# Asegurar path de importación
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fit_elipsometrico import ajuste_elipsometrico

def run_test():
    data_path = './Datos/Datos-25-6/meds/SiO2_N4_Si/mean/SiO2_N4_Si_media.txt'
    std_path = './Datos/Datos-25-6/meds/SiO2_N4_Si/std/SiO2_N4_Si_std.txt'
    
    # Definición de capas
    layer_names = ['air', 'SiO2', 'Si']
    layer_models = [None, 'cauchy', None]  # SiO2 con modelo Cauchy optimizable
    d_bounds = [(10.0, 100.0)]             # Espesor de SiO2 a optimizar (de 10 a 100 nm)
    
    print("=== Iniciando Test de Ajuste Elipsométrico con Chi2 Reducido ===")
    print(f"Cargando medias desde: {data_path}")
    print(f"Cargando std desde: {std_path}")
    
    try:
        best_thicknesses, best_Is, best_Ic, best_params, wl_exp, Is_exp, Ic_exp = ajuste_elipsometrico(
            data_path=data_path,
            std_path=std_path,
            skiprows=1,  # SiO2_N4_Si_media.txt tiene 1 fila de header
            layer_names=layer_names,
            layer_models=layer_models,
            d_bounds=d_bounds,
            theta_0=70.0,  # Ángulo del test
            num_starts=20,  # Pocas semillas para velocidad
            num_epochs=100, # Pocas épocas para velocidad
            lr=1.0,
            use_cuda=False,
            force_recalc=True  # Ignorar caché
        )
        
        print("\n=== ¡Test Completado Exitosamente! ===")
        print(f"Espesor óptimo encontrado: {best_thicknesses[0]:.2f} nm")
        print("Parámetros de Cauchy optimizados:")
        print(best_params)
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución del test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_test()
