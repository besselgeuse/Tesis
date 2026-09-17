# Explicación Línea por Línea: `fit_ellipsometry_torch`

Esta guía analiza detalladamente el código fuente de la función `fit_ellipsometry_torch` dentro del archivo `tmm_utils_Rodrigo.py`. El objetivo es entender qué hace cada bloque de código y cómo se estructuran las variables para la optimización conjunta en PyTorch.

---

## 1. Firma de la Función y Preparación de Tensores

Esta sección define qué parámetros recibe el algoritmo y cómo los adecúa para trabajar en PyTorch (ya sea en CPU o GPU).

```python
def fit_ellipsometry_torch(n_list, d_bounds, lams, Is_exp, Ic_exp, th_0=0.0, 
                           num_starts=50, num_epochs=150, lr=2.0, use_cuda=False, 
                           layer_models=None, layer_names=None):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Ejecutando en dispositivo: {device}")
    
    lams = lams.to(device)
    
    if not isinstance(Is_exp, torch.Tensor):
        Is_exp = torch.tensor(Is_exp, dtype=torch.float64, device=device)
    else:
        Is_exp = Is_exp.to(device)
        
    if not isinstance(Ic_exp, torch.Tensor):
        Ic_exp = torch.tensor(Ic_exp, dtype=torch.float64, device=device)
    else:
        Ic_exp = Ic_exp.to(device)
```

*   **Línea 558-560 (`def fit_ellipsometry_torch(...)`)**: Define la firma. Recibe la lista de índices ópticos (`n_list`), límites de espesores (`d_bounds`), longitudes de onda (`lams`), parámetros elipsométricos medidos (`Is_exp`, `Ic_exp`), el ángulo de incidencia (`th_0`) y configuraciones de optimización.
*   **Línea 583 (`device = torch.device(...)`)**: Configura el dispositivo. Si `use_cuda=True` y tienes drivers/GPU de NVIDIA disponibles, usará la GPU para procesar las semillas en paralelo. De lo contrario, usará la CPU.
*   **Línea 587 (`lams = lams.to(device)`)**: Mueve el tensor de longitudes de onda al dispositivo seleccionado.
*   **Línea 589-598 (`if not isinstance(Is_exp, torch.Tensor)...`)**: Verifica si los datos experimentales de $I_s$ e $I_c$ son arreglos de numpy o tensores. Si son numpy, los convierte a tensores de PyTorch de doble precisión (`float64`) y los carga en el dispositivo. Si ya son tensores, simplemente se asegura de moverlos al dispositivo actual.

---

## 2. Procesamiento de n_list y Dummies Paramétricos

Aquí se procesan los índices de refracción iniciales. Si hay capas cuyo índice se optimiza dinámicamente, se crea un tensor dummy (de unos) para rellenar la estructura temporalmente.

```python
    if isinstance(n_list, torch.Tensor):
        n_list = n_list.to(device)
        num_layers = n_list.shape[0]
        num_wl = n_list.shape[1]
    else:
        n_list_tensors = []
        for item in n_list:
            if isinstance(item, torch.Tensor):
                n_list_tensors.append(item.to(device))
            else:
                n_list_tensors.append(torch.ones_like(lams, dtype=torch.complex128, device=device))
        n_list = torch.stack(n_list_tensors, dim=0)
        num_layers = n_list.shape[0]
        num_wl = n_list.shape[1]
        
    num_finite_layers = num_layers - 2
    
    if len(d_bounds) != num_finite_layers:
        raise ValueError(f"d_bounds debe tener longitud {num_finite_layers}, pero tiene {len(d_bounds)}.")
```

*   **Línea 600-603 (`if isinstance(n_list, torch.Tensor)`)**: Si `n_list` ya es un único tensor 2D de dimensiones `[num_capas, num_wavelengths]`, se mueve al dispositivo directamente y se obtienen las dimensiones del sistema.
*   **Línea 604-613 (`else: ...`)**: Si es una lista de materiales (lo normal al usar capas paramétricas), se recorre elemento por elemento. Si es un material estático (cuyo índice ya es un tensor), se mueve a la GPU. Si es una capa paramétrica (como Cauchy, que viene representada por un dummy), se crea un tensor de números complejos de $1.0 + 0j$ (un "aire ficticio") de la misma longitud que `lams`. Luego, `torch.stack` los apila a todos para formar un único tensor consolidado.
*   **Línea 617 (`num_finite_layers = num_layers - 2`)**: Calcula el número de capas intermedias con espesor finito. Restamos 2 porque el superestrato (aire, capa 0) y el sustrato (capa última) son ópticamente infinitos y no tienen espesores a ajustar.
*   **Línea 619-620 (`if len(d_bounds) != num_finite_layers`)**: Validación. Verifica que el usuario haya especificado límites de espesor exactamente para las capas intermedias finitas.

---

## 3. Parseo de Modelos de Dispersión Paramétricos

Esta sección analiza qué modelos matemáticos (Cauchy, Cauchy Absorbente, Bruggeman) se asignan a cada capa y define los límites ópticos de búsqueda.

```python
    is_parametric = False
    disp_layers = []
    total_disp_params = 0
    
    if layer_models is not None:
        for idx, model in enumerate(layer_models):
            if model is not None:
                is_parametric = True
                
                if isinstance(model, dict):
                    model_type = model.get('model', 'cauchy')
                    bounds = model.get('bounds', None)
                    initial = model.get('initial', None)
                else:
                    model_type = model
                    bounds = None
                    initial = None
```

*   **Línea 623-625 (`is_parametric = False...`)**: Inicializa banderas y variables para rastrear si el ajuste incluye optimización de índices de refracción (`is_parametric`), información detallada de cada modelo (`disp_layers`) y el número total de variables del índice a optimizar (`total_disp_params`).
*   **Línea 627-629 (`if layer_models is not None...`)**: Comienza a iterar sobre la configuración de modelos de cada capa. Si la capa actual tiene un modelo (no es `None`), activa la bandera paramétrica.
*   **Línea 632-640 (`if isinstance(model, dict)...`)**: Permite flexibilizar la entrada. Si el modelo viene como un diccionario (ej: para especificar límites personalizados), extrae el nombre del modelo, límites (`bounds`) e inicialización (`initial`). Si viene simplemente como una cadena de texto (ej: `'cauchy'`), define límites por defecto.

---

## 4. Inicialización de Rangos por Modelo

Línea por línea se leen los parámetros correspondientes para cada uno de los modelos físicos implementados.

```python
                if model_type == 'cauchy':
                    num_p = 3
                    default_bounds = [(1.0, 3.0), (-1.0, 1.0), (-1.0, 1.0)]
                    default_initial = [1.5, 0.0, 0.0]
                elif model_type == 'cauchy_absorbent':
                    num_p = 6
                    default_bounds = [(1.0, 3.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)]
                    default_initial = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif model_type == 'bruggeman':
                    if isinstance(model, dict):
                        f_air_fixed = model.get('f_air', None)
                        f_bounds = model.get('f_bounds', None)
                    else:
                        f_air_fixed = None
                        f_bounds = None
                    
                    if f_bounds is not None:
                        num_p = 1
                        default_bounds = [tuple(f_bounds)]
                        default_initial = [sum(f_bounds) / 2.0]
                    else:
                        num_p = 0
                        default_bounds = []
                        default_initial = []
```

*   **Línea 642-646 (`if model_type == 'cauchy'`)**: Configura el modelo Cauchy clásico transparente. Requiere $3$ parámetros ($A, B, C$). Sus límites predeterminados de búsqueda son: $A \in [1.0, 3.0]$ (índice base), $B \in [-1.0, 1.0]$ y $C \in [-1.0, 1.0]$. Su semilla inicial típica es $A=1.5, B=0.0, C=0.0$.
*   **Línea 647-651 (`elif model_type == 'cauchy_absorbent'`)**: Configura el modelo Cauchy con absorción. Agrega $3$ parámetros más para representar la parte imaginaria $k(\lambda)$ (coeficientes $D, E, F$). Total: $6$ parámetros.
*   **Línea 652-660 (`elif model_type == 'bruggeman'`)**: Modelo de Bruggeman. Si es un diccionario, verifica si el usuario quiere optimizar la fracción de aire (se busca `f_bounds`) o dejarla fija en un valor constante (se busca `f_air`).
*   **Línea 661-670 (`if f_bounds is not None`)**: Si hay `f_bounds` (fracción de aire optimizable), el modelo agrega $1$ parámetro de búsqueda, con límites dados por `f_bounds`, inicializándolo en el punto medio. Si es fija, agrega $0$ parámetros a optimizar.

---

## 5. Consolidación de Información de Dispersión

Se asocian los límites finales y se crea un registro estructurado con la posición de cada variable en el vector general.

```python
                if bounds is None:
                    bounds = default_bounds
                if initial is None:
                    initial = default_initial
                    
                layer_info = {
                    'layer_idx': idx,
                    'model_type': model_type,
                    'param_bounds': bounds,
                    'param_initial': initial,
                    'num_params': num_p,
                    'start_idx': total_disp_params
                }
                if model_type == 'bruggeman':
                    if isinstance(model, dict) and 'f_bounds' in model:
                        layer_info['f_optimizable'] = True
                    elif isinstance(model, dict) and 'f_air' in model:
                        layer_info['f_air'] = model['f_air']
                        layer_info['f_optimizable'] = False
                    else:
                        layer_info['f_air'] = 0.5
                        layer_info['f_optimizable'] = False
                
                disp_layers.append(layer_info)
                total_disp_params += num_p
```

*   **Línea 674-677 (`if bounds/initial is None`)**: Si el usuario no especificó límites o valores iniciales personalizados, se adoptan los valores por defecto del modelo.
*   **Línea 679-686 (`layer_info = {...}`)**: Crea un diccionario con los metadatos de la capa. Incluye el índice físico de la capa (`layer_idx`), modelo, límites, semillas y el índice de inicio en el vector global de optimización (`start_idx`) para saber dónde empiezan sus variables correspondientes.
*   **Línea 688-696 (`if model_type == 'bruggeman'`)**: Define si la fracción de aire de la aproximación de medio efectivo es fija o variable, guardando su valor fijo (por defecto $0.5$) en caso de que no sea optimizable.
*   **Línea 698-699 (`disp_layers.append...`)**: Guarda la información en la lista y acumula el total de variables ópticas optimizables (`total_disp_params`).

---

## 6. Clasificación de Espesores Ópticos (Fijos y Optimizables)

El código analiza los límites de espesores proporcionados por el usuario para separar qué capas se mantendrán fijas y cuáles se ajustarán.

```python
    is_optimizable = []
    opt_bounds = []
    fixed_values = {}
    
    for idx, b in enumerate(d_bounds):
        if isinstance(b, (tuple, list)):
            d_min, d_max = float(b[0]), float(b[1])
            if d_min == d_max:
                is_optimizable.append(False)
                fixed_values[idx] = d_min
            else:
                is_optimizable.append(True)
                opt_bounds.append((d_min, d_max))
        else:
            is_optimizable.append(False)
            fixed_values[idx] = float(b)
            
    num_opt_layers = sum(is_optimizable)
```

*   **Línea 705-707 (`is_optimizable = []...`)**: Prepara listas para marcar si la capa es variable (`is_optimizable`), sus límites físicos de espesor (`opt_bounds`) y un diccionario para guardar los valores estáticos (`fixed_values`).
*   **Línea 709-720 (`for idx, b in enumerate(d_bounds)`)**: Itera sobre los límites de las capas finitas. Si el límite es un par `[d_min, d_max]` y ambos números son diferentes, se marca como optimizable y se guardan sus límites. Si son iguales (ej: `(50, 50)`) o el usuario ingresó un único número decimal (ej: `50.0`), se marca como no-optimizable (fijo) y se registra su espesor constante.
*   **Línea 722 (`num_opt_layers = sum(is_optimizable)`)**: Suma las banderas `True` para saber exactamente cuántos espesores serán optimizados por el algoritmo.

---

## 7. Inicialización Multi-Start de Espesores en Logits

Se realiza la siembra aleatoria paralela en el espacio libre de restricciones (logits) mediante la transformada sigmoide inversa.

```python
    d_initial = torch.zeros((num_starts, num_opt_layers), dtype=torch.float64, device=device)
    o_idx = 0
    for idx, opt in enumerate(is_optimizable):
        if opt:
            d_min, d_max = opt_bounds[o_idx]
            d_init_phys = torch.empty(num_starts, device=device).uniform_(d_min, d_max)
            p = (d_init_phys - d_min) / (d_max - d_min)
            p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
            d_initial[:, o_idx] = torch.log(p / (1.0 - p))
            o_idx += 1
            
    d_opt = d_initial.clone().detach().requires_grad_(True)
```

*   **Línea 726 (`d_initial = torch.zeros((num_starts...))`)**: Prepara una matriz en el dispositivo con dimensiones `[semillas, espesores_optimizables]` llena de ceros.
*   **Línea 729-731 (`if opt: ...`)**: Para cada espesor optimizable, genera una distribución uniforme aleatoria de valores de espesor físico para todas las semillas, acotada en su intervalo `[d_min, d_max]`.
*   **Línea 732 (`p = (d_init_phys - d_min) / (d_max - d_min)`)**: Normaliza los espesores aleatorios al rango $[0, 1]$.
*   **Línea 733 (`p = torch.clamp(p, 1e-7, 1.0 - 1e-7)`)**: Limita el rango para evitar la división por cero o logaritmo de cero en el paso siguiente.
*   **Línea 734 (`d_initial[:, o_idx] = torch.log(p / (1.0 - p))`)**: **Sigmoide inversa (Logit)**. Mapea el valor normalizado del rango $[0, 1]$ al rango ilimitado $(-\infty, \infty)$.
*   **Línea 737 (`d_opt = d_initial.clone().detach().requires_grad_(True)`)**: Duplica los logits iniciales en un tensor dedicado a la optimización, y activa `requires_grad_(True)` para indicarle a Autograd de PyTorch que debe calcular gradientes matemáticos para estas variables.

---

## 8. Inicialización Multi-Start de Parámetros Ópticos

Se realiza el mismo procedimiento de distribución aleatoria para los coeficientes Cauchy y fracciones de Bruggeman.

```python
    if is_parametric:
        p_initial = torch.zeros((num_starts, total_disp_params), dtype=torch.float64, device=device)
        for d_lay in disp_layers:
            start_idx = d_lay['start_idx']
            bounds = d_lay['param_bounds']
            initial = d_lay['param_initial']
            
            for k in range(d_lay['num_params']):
                p_min, p_max = float(bounds[k][0]), float(bounds[k][1])
                p_init_phys = torch.empty(num_starts, device=device).uniform_(p_min, p_max)
                p = (p_init_phys - p_min) / (p_max - p_min)
                p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
                p_initial[:, start_idx + k] = torch.log(p / (1.0 - p))
                
        p_opt = p_initial.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([d_opt, p_opt], lr=lr)
    else:
        optimizer = optim.Adam([d_opt], lr=lr)
```

*   **Línea 740-741 (`if is_parametric: ...`)**: Si se ajustan índices complejos, prepara una matriz `p_initial` de tamaño `[semillas, parametros_opticos]`.
*   **Línea 742-748 (`for d_lay in disp_layers: ...`)**: Recorre las capas con modelos y extrae la información de sus límites.
*   **Línea 751-755 (`for k in range(d_lay['num_params'])`)**: Para cada parámetro del modelo (ej: A, B, C de Cauchy), siembra valores aleatorios uniformes dentro de sus límites físicos, los normaliza en $[0, 1]$, los limita con `clamp` y calcula su equivalente en logit usando la fórmula de logit.
*   **Línea 757 (`p_opt = p_initial...requires_grad_(True)`)**: Convierte los logits de índices de refracción en tensores optimizables con gradiente activo.
*   **Línea 758-760 (`optimizer = optim.Adam(...)`)**: Crea el optimizador Adam de PyTorch. Si el ajuste es paramétrico, optimizará tanto los espesores (`d_opt`) como los índices (`p_opt`) con la tasa de aprendizaje (`lr`) elegida. De lo contrario, solo optimiza los espesores.

---

## 9. Función de Reconstrucción de Espesores Físicos

Esta función interna se encarga de rearmar el vector de espesores físicos a partir de las variables optimizables (logits) y fijas.

```python
    inf_col = torch.full((num_starts, 1), float('inf'), dtype=torch.float64, device=device)
    
    def reconstruct_d_fisico(d_opt_tensor):
        d_fisico_cols = []
        o_idx = 0
        for idx, opt in enumerate(is_optimizable):
            if opt:
                d_min, d_max = opt_bounds[o_idx]
                col = d_min + (d_max - d_min) * torch.sigmoid(d_opt_tensor[:, o_idx])
                d_fisico_cols.append(col)
                o_idx += 1
            else:
                val = fixed_values[idx]
                col = torch.full((d_opt_tensor.shape[0],), val, dtype=torch.float64, device=device)
                d_fisico_cols.append(col)
        return torch.stack(d_fisico_cols, dim=1)
```

*   **Línea 764 (`inf_col = torch.full(...)`)**: Prepara una columna de dimensiones `[num_starts, 1]` llena de infinito (`inf`). En el método de matriz de transferencia (TMM), las capas semi-infinitas exteriores (el aire inicial y el sustrato final) tienen un espesor físicamente infinito.
*   **Línea 766 (`def reconstruct_d_fisico(d_opt_tensor)`)**: Declara la función de reconstrucción para usarla en cada iteración del bucle.
*   **Línea 769-774 (`if opt: ...`)**: Si la capa es optimizable, aplica la sigmoide al tensor de logits (llevándolo a $[0, 1]$), lo escala multiplicando por el ancho del rango físico (`d_max - d_min`) y le suma el límite mínimo (`d_min`). Esto devuelve el espesor físico exacto.
*   **Línea 775-778 (`else: ...`)**: Si la capa es fija, crea una columna donde todas las semillas tienen el mismo valor de espesor constante especificado.
*   **Línea 779 (`return torch.stack(..., dim=1)`)**: Apila todas las columnas para reconstruir una matriz de espesores físicos de forma `[semillas, capas_finitas]`.

---

## 10. Reconstrucción de Índices de Refracción Complejos

Esta función calcula los tensores de índices complejos a partir del espectro cromático de cada modelo.

```python
    lams_2d = lams.unsqueeze(0)  # [1, num_wl]
    inv_lam2 = 1e4 / (lams_2d ** 2)
    inv_lam4 = 1e9 / (lams_2d ** 4)

    def reconstruct_n_list_batched(p_opt_tensor):
        layer_tensors = [None] * num_layers
        
        # 1. Copiar capas estáticas (no paramétricas)
        for idx in range(num_layers):
            is_this_parametric = False
            for d_lay in disp_layers:
                if d_lay['layer_idx'] == idx:
                    is_this_parametric = True
                    break
            if not is_this_parametric:
                layer_tensors[idx] = n_list[idx].unsqueeze(0).expand(num_starts, -1)
```

*   **Línea 781 (`lams_2d = lams.unsqueeze(0)`)**: Cambia la dimensión de `lams` de 1D a 2D (forma `[1, num_wl]`) para poder realizar operaciones matriciales y broadcasting por semilla.
*   **Línea 782-783 (`inv_lam2 = 1e4 / ...`)**: Precomputa los términos $10^4/\lambda^2$ y $10^9/\lambda^4$ que se usan en las ecuaciones de Cauchy. Hacerlo fuera del bucle ahorra cálculos repetitivos y acelera notablemente la optimización.
*   **Línea 785-787 (`def reconstruct_n_list_batched(...)`)**: Declara la función de reconstrucción para la lista de índices ópticos. Inicializa una lista vacía de tamaño igual al número de capas.
*   **Línea 790-797 (`# 1. Copiar capas estáticas...`)**: Recorre las capas. Si la capa no es paramétrica (es estática), toma su índice de refracción original `n_list[idx]` y lo expande agregando la dimensión de semillas, obteniendo una forma `[semillas, num_wl]`.

---

## 11. Aplicación de las Ecuaciones de Cauchy

Esta sección calcula la dispersión de las capas ópticas con los coeficientes paramétricos optimizados.

```python
        # 2. Calcular capas con Cauchy
        for d_lay in disp_layers:
            if d_lay['model_type'] in ['cauchy', 'cauchy_absorbent']:
                idx = d_lay['layer_idx']
                start_idx = d_lay['start_idx']
                bounds = d_lay['param_bounds']
                model_type = d_lay['model_type']
                
                p_phys_list = []
                for k in range(d_lay['num_params']):
                    p_min, p_max = float(bounds[k][0]), float(bounds[k][1])
                    p_logit = p_opt_tensor[:, start_idx + k]
                    p_phys = p_min + (p_max - p_min) * torch.sigmoid(p_logit)
                    p_phys_list.append(p_phys)
                    
                if model_type == 'cauchy':
                    A = p_phys_list[0].unsqueeze(1)
                    B = p_phys_list[1].unsqueeze(1)
                    C = p_phys_list[2].unsqueeze(1)
                    n_calc = A + B * inv_lam2 + C * inv_lam4
                    n_complex = n_calc.to(torch.complex128)
                elif model_type == 'cauchy_absorbent':
                    A = p_phys_list[0].unsqueeze(1)
                    B = p_phys_list[1].unsqueeze(1)
                    C = p_phys_list[2].unsqueeze(1)
                    D = p_phys_list[3].unsqueeze(1)
                    E = p_phys_list[4].unsqueeze(1)
                    F = p_phys_list[5].unsqueeze(1)
                    n_calc = A + B * inv_lam2 + C * inv_lam4
                    k_calc = D + E * inv_lam2 + F * inv_lam4
                    k_calc = torch.clamp(k_calc, min=0.0)
                    n_complex = torch.complex(n_calc, k_calc)
                    
                layer_tensors[idx] = n_complex
```

*   **Línea 800-806 (`for d_lay in disp_layers: ...`)**: Filtra las capas que usan modelos de Cauchy.
*   **Línea 807-812 (`for k in range(d_lay['num_params'])`)**: Mapea cada logit correspondiente al parámetro Cauchy al rango físico correcto aplicando la sigmoide: $p_{\text{phys}} = p_{\text{min}} + (p_{\text{max}} - p_{\text{min}})\sigma(p_{\text{logit}})$.
*   **Línea 814-819 (`if model_type == 'cauchy'`)**: Toma los parámetros físicos $A$, $B$ y $C$, les añade una dimensión de longitud de onda (unsqueeze) y calcula el índice óptico real usando la ecuación cromática de Cauchy. Lo convierte a tipo complejo de PyTorch (con parte imaginaria cero).
*   **Línea 820-830 (`elif model_type == 'cauchy_absorbent'`)**: Realiza el mismo cálculo para los coeficientes de dispersión reales ($A, B, C$) e imaginarios ($D, E, F$). Clampea el coeficiente de extinción ($k$) a valores no-negativos y fusiona ambas componentes en un tensor complejo de tipo `torch.complex`.
*   **Línea 832 (`layer_tensors[idx] = n_complex`)**: Asigna el índice calculado a la posición de la capa correspondiente.

---

## 12. Cálculo del Medio Efectivo de Bruggeman

Líneas dedicadas al procesamiento de capas mezcla que dependen de la dispersión de otra capa.

```python
        # 3. Capas dependientes (Bruggeman EMA)
        for d_lay in disp_layers:
            if d_lay['model_type'] == 'bruggeman':
                idx = d_lay['layer_idx']
                n_base = layer_tensors[idx + 1]
                n_air = torch.ones_like(n_base)
                
                if d_lay.get('f_optimizable', False):
                    start_idx = d_lay['start_idx']
                    p_min, p_max = float(d_lay['param_bounds'][0][0]), float(d_lay['param_bounds'][0][1])
                    p_logit = p_opt_tensor[:, start_idx]
                    f_air = p_min + (p_max - p_min) * torch.sigmoid(p_logit)
                    f_air = f_air.unsqueeze(1)
                else:
                    f_air = d_lay.get('f_air', 0.5)
                
                layer_tensors[idx] = n_eff_torch(n_base, n_air, f_air)

        return torch.stack(layer_tensors, dim=1)
```

*   **Línea 835-839 (`if d_lay['model_type'] == 'bruggeman'`)**: Encuentra la capa que usa Bruggeman. Por diseño físico, toma como base el material denso ubicado inmediatamente debajo (capa `idx + 1`) y define la segunda fase como aire ($n_{\text{air}} = 1.0 + 0j$).
*   **Línea 844-849 (`if d_lay.get('f_optimizable', False)`)**: Si la porosidad es variable, lee el logit correspondiente, le aplica la sigmoide para acotarlo entre los límites y le añade una dimensión para operar con broadcasting.
*   **Línea 850-851 (`else: f_air = ...`)**: Si la porosidad es constante, asigna su fracción directa sin gradientes.
*   **Línea 854 (`layer_tensors[idx] = n_eff_torch(...)`)**: Ejecuta el cálculo analítico de Bruggeman de forma vectorial usando una subfunción compatible con Autograd de PyTorch.
*   **Línea 857 (`return torch.stack(layer_tensors, dim=1)`)**: Consolida todos los índices ópticos en un único tensor 3D de dimensiones `[semillas, capas, wavelengths]` listo para la simulación óptica.

---

## 13. El Bucle de Optimización por Gradiente

Sección crítica del código donde se ejecuta el bucle de épocas, la simulación física electromagnética y la retropropagación de gradientes.

```python
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        d_fisico = reconstruct_d_fisico(d_opt)
        d_full = torch.cat([inf_col, d_fisico, inf_col], dim=1)
        d_full_3d = d_full.unsqueeze(-1).expand(-1, -1, num_wl)
        
        if is_parametric:
            n_list_3d = reconstruct_n_list_batched(p_opt)
            res_s = coh_tmm_torch_batched(pol='s', n_list=n_list_3d, d_list=d_full_3d, th_0=th_0, lam_vac=lams)
            res_p = coh_tmm_torch_batched(pol='p', n_list=n_list_3d, d_list=d_full_3d, th_0=th_0, lam_vac=lams)
        else:
            res_s = tmm.coh_tmm_torch(pol='s', n_list=n_list, d_list=d_full_3d, th_0=th_0, lam_vac=lams)
            res_p = tmm.coh_tmm_torch(pol='p', n_list=n_list, d_list=d_full_3d, th_0=th_0, lam_vac=lams)
```

*   **Línea 860 (`for epoch in range(num_epochs)`)**: Comienza el ciclo de entrenamiento o ajuste por gradiente.
*   **Línea 861 (`optimizer.zero_grad()`)**: Borra los gradientes del paso anterior. En PyTorch, los gradientes se acumulan por defecto, por lo que es necesario resetearlos a cero en cada iteración.
*   **Línea 863 (`d_fisico = reconstruct_d_fisico(d_opt)`)**: Llama a la función del bloque 9 para transformar logits a espesores físicos.
*   **Línea 864-865 (`d_full = torch.cat(...)`)**: Agrega las columnas infinitas de los extremos al principio y final de la matriz de espesores de las capas intermedias, y las expande a 3D copiando el valor para todas las longitudes de onda (forma final: `[semillas, capas, wavelengths]`).
*   **Línea 867-870 (`if is_parametric: ...`)**: Si se optimiza el índice, llama a la reconstrucción paramétrica cromática y ejecuta el cálculo de matrices de transferencia en su variante batcheada y paralela (`coh_tmm_torch_batched`) para obtener reflectancias s y p.
*   **Línea 871-873 (`else: ...`)**: Si los índices son estáticos, ejecuta el solver de TMM vectorial convencional sobre el tensor estático `n_list` expandido.

---

## 14. Cálculo de Coeficientes Elipsométricos y Pérdida MSE

Líneas dedicadas a resolver las ecuaciones ópticas elipsométricas y a guiar la retropropagación.

```python
        r_s = res_s['r']
        r_p = res_p['r']
        
        rho = torch.conj(r_p / r_s)
        psi_teo = torch.atan(torch.abs(rho))
        delta_teo = torch.angle(rho)
        
        Is_teo = torch.sin(2 * psi_teo) * torch.sin(delta_teo)
        Ic_teo = torch.sin(2 * psi_teo) * torch.cos(delta_teo)
        
        error_Is = (Is_teo - Is_exp) ** 2
        error_Ic = (Ic_teo - Ic_exp) ** 2
        
        loss_per_start = (error_Is + error_Ic).mean(dim=-1)
        loss = loss_per_start.sum()
        
        loss.backward()
        optimizer.step()
```

*   **Línea 875-876 (`r_s / r_p = res_s/p['r']`)**: Extrae los coeficientes de reflexión complejos calculados.
*   **Línea 878-880 (`rho = ...`)**: Calcula la relación de reflexión compleja $\rho = r_p / r_s$ (aplicando conjugado complejo debido a las convenciones de fase). Extrae los ángulos teóricos $\psi$ (arco tangente del módulo) y $\Delta$ (fase o ángulo complejo).
*   **Línea 882-883 (`Is_teo / Ic_teo = ...`)**: Aplica las identidades trigonométricas para proyectar $\psi$ y $\Delta$ en los coeficientes elipsométricos continuos $I_s$ e $I_c$ teóricos.
*   **Línea 885-886 (`error_Is / error_Ic = ...`)**: Calcula el error al cuadrado contra las mediciones experimentales para cada punto espectral.
*   **Línea 888 (`loss_per_start = ...mean(dim=-1)`)**: Calcula el Error Cuadrático Medio (MSE) promedio de todas las longitudes de onda, de manera individual para cada semilla.
*   **Línea 889 (`loss = loss_per_start.sum()`)**: Suma las pérdidas de todas las semillas en un único escalar. Esto permite propagar los gradientes de todas las optimizaciones simultáneamente en una sola llamada de Autograd.
*   **Línea 891 (`loss.backward()`)**: **Retropropagación automática**. PyTorch calcula de manera exacta las derivadas de la pérdida con respecto a todos los logits optimizables (espesores e índices).
*   **Línea 892 (`optimizer.step()`)**: **Paso de optimización**. El optimizador Adam actualiza los logits en base a las derivadas calculadas para minimizar el error.

---

## 15. Clampeo de Variables y Extracción del Mínimo Global

Líneas finales de procesamiento que previenen problemas de gradiente y extraen los parámetros del mejor ajuste absoluto.

```python
        with torch.no_grad():
            d_opt.data.clamp_(-5, 5)
            if is_parametric:
                p_opt.data.clamp_(-5, 5)
        
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            mejor_loss = loss_per_start.min().item()
            print(f"Epoch {epoch}/{num_epochs} | Mejor Loss (MSE): {mejor_loss:.6f}")
            
    # --- EXTRACCIÓN DEL MÍNIMO ---
    best_idx = loss_final.argmin()
    best_thicknesses = d_final_fisico[best_idx].cpu().detach().numpy()
    best_Is_curve = Is_teo_final[best_idx].cpu().detach().numpy()
    best_Ic_curve = Ic_teo_final[best_idx].cpu().detach().numpy()
```

*   **Línea 896-899 (`d_opt/p_opt.data.clamp_(-5, 5)`)**: Limita el valor absoluto de los logits a un rango de $[-5, 5]$ en modo sin gradientes. Esto evita la saturación de la función sigmoide y mantiene activos los gradientes para la siguiente iteración.
*   **Línea 901-903 (`if epoch % 50 == 0 ...`)**: Cada 50 épocas, extrae la pérdida de la mejor semilla en esa iteración y la imprime en consola para monitorear la convergencia del ajuste.
*   **Línea 933 (`best_idx = loss_final.argmin()`)**: Una vez completado el bucle de épocas, encuentra cuál de las semillas paralelas (ej: la número 142 de las 600) obtuvo la menor pérdida MSE absoluta.
*   **Línea 935-938 (`best_thicknesses = ...`)**: Extrae los espesores físicos, la curva de $I_s$ y la curva de $I_c$ teóricos correspondientes a esa semilla óptima, los mueve de vuelta a la CPU y los convierte en arreglos comunes de numpy para graficar o guardar.
