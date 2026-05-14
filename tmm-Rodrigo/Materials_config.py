# materiales_config.py
# materiales_config.py
from tmm_utils_Rodrigo import load_interp, cauchy_fn, constant_fn, brugg_fn, load_interp_in3

def obtener_diccionario_materiales(ruta_materiales, stack, ruta_catalogo='catalogo_de_materiales.txt'):
    """
    Genera el diccionario dinámicamente extrayendo los nombres de los materiales desde el stack.
    """
    materials = {}
    
    # 1. EXTRACCIÓN INTELIGENTE DEL STACK
    materiales_necesarios = set([capa[1] for capa in stack])
    
    f_T1 = 0.58
    modelo_T1 = cauchy_fn(2.338, 1.906, 0.824)

    # 2. EL CATÁLOGO MAESTRO (Con lambdas para mantener la optimización)
    with open(ruta_catalogo, 'r', encoding='utf-8') as f:
        contenido = f.read()
        
    entorno = {
        'ruta_materiales': ruta_materiales,
        'load_interp': load_interp,
        'load_interp_in3': load_interp_in3,
        'constant_fn': constant_fn,
        'cauchy_fn': cauchy_fn,
        'brugg_fn': brugg_fn,
        'f_T1': f_T1,
        'modelo_T1': modelo_T1
    }
    
    catalogo = eval(contenido, entorno)

    # 3. CICLO FOR DINÁMICO
    for material in materiales_necesarios:
        if material in catalogo:
            # Ejecuta la función lambda y guarda el resultado
            materials[material] = catalogo[material]()
        else:
            print(f"⚠️ Advertencia: El material '{material}' no está definido en el catálogo.")

    return materials