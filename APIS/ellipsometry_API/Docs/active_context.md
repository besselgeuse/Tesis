# Contexto Activo del Proyecto TMM-R

Este archivo registra el estado actual del desarrollo del proyecto de simulación elipsométrica, sus cambios recientes y la arquitectura vigente.

## 1. Cambios Recientes (Última Sesión)

- **Restricción de Índice de Refracción Máximo ($n_{max\_limits}$):**
  - Se implementó el parámetro opcional `n_max_limits` en `fit_ellipsometry_torch` y `autorange_fit_ellipsometry_torch` en [tmm_utils_Rodrigo.py](file:///d:/archivos/TESIS/TMM-R/segundo_cuatri/tmm_utils_Rodrigo.py) para acotar físicamente el valor de $n(\lambda)$ devuelto por modelos de dispersión (como Cauchy).
  - La verificación de validez se realiza al finalizar la optimización multi-start (post-épocas), filtrando las semillas que superen el valor máximo permitido en cualquier longitud de onda del rango y seleccionando la semilla válida con menor error ($\text{MSE}/\chi^2$).
- **Corrección de Contracción en Autorange (`adjust_val_range`):**
  - Se corrigió la función auxiliar `adjust_val_range` en `autorange_fit_ellipsometry_torch` para impedir contracciones indeseadas de la cota superior ($d_{max}$) cuando una variable o espesor colinda con su cota mínima absoluta (`0.0` nm).
- **Simulación TMM Directa y Validación de Ambigüedad de Fase (`Ajuste_AL2O3.py`):**
  - En [Ajuste_AL2O3.py](file:///d:/archivos/TESIS/TMM-R/segundo_cuatri/capas_AR_triple_o_mas/Ajuste_AL2O3.py) se agregaron celdas dedicadas para calcular y comparar la respuesta TMM de muestras gruesas (~970 nm) utilizando tanto la versión PyTorch (`coh_tmm_torch_batched`) como la versión NumPy (`tmm.coh_tmm`), comprobando equivalencia perfecta a nivel de precisión de máquina ($\sim 10^{-15}$).
  - Se analizó y resolvió la ambigüedad periódica de interferencia óptica ($m\lambda/2n\cos\theta$) que provocaba convergencia a la rama de 60-70 nm cuando los rangos de espesor no se ajustaban al régimen físico de ~900 nm.

- **Espesores de Capa Fijos e Independientes:** Se implementó soporte completo tanto en el frontend como en el backend para permitir fijar de forma independiente el espesor de cualquier capa intermedia finita.
  - **Interfaz de Espesores Mixtos:** En [index.html](file:///d:/archivos/TESIS/TMM-R/APIS/ellipsometry_API/static/index.html#L1286) se rediseñó la sección de espesores (`.layer-row-thickness`) agregando inputs de rango (`thick-min` y `thick-max`), un input de espesor fijo (`thick-fixed`) y un checkbox de alternancia `"Opt"`.
  - **Visibilidad Clásica:** Se restauró y mejoró la visibilidad del panel de espesor en [updateBoundsSection()](file:///d:/archivos/TESIS/TMM-R/APIS/ellipsometry_API/static/index.html#L1361). La barra de espesores se muestra en **cualquier capa** (sin importar su posición en la lista) si y solo si tiene un modelo de dispersión paramétrico seleccionado (Cauchy o Bruggeman). Si es estática, se oculta, manteniendo el comportamiento intuitivo y robusto original de la SPA.
  - **Mapeo de Datos:** En `getLayersData()` se lee el checkbox de optimización. Si se desmarca `"Opt"`, se asigna el valor fijo tanto al mínimo como al máximo (`d_min = d_max = valor_fijo`), el cual se propaga al backend de forma retrocompatible.
- **Resolución de Fallos de Transacción y Guardado en MySQL:**
  - **Limpieza de Metadata Locks:** Se diagnosticó que una transacción huérfana de hace más de 13,000 segundos retenía un lock de metadatos exclusivo sobre la tabla `stacks`, provocando que los procesos de `ALTER TABLE` y subsiguientes de escritura (`INSERT`/`SELECT`) se colgaran. Esto se resolvió ejecutando sentencias `KILL` sobre las conexiones inactivas y procesos encolados bloqueantes.
  - **Mapeo Robusto de Espesores Intermedios:** Se corrigió un bug en [main.py](file:///d:/archivos/TESIS/TMM-R/APIS/ellipsometry_API/main.py#L217) donde las capas intermedias estáticas causaban desalineamiento de índices en el almacenamiento. Ahora el espesor se extrae y guarda basándose directamente en el índice absoluto de la capa.
  - **Gestión de Excepciones db.rollback():** Se incorporó `db.rollback()` explícito en el bloque `except` del endpoint `/simular_stack` de `main.py` para asegurar la liberación inmediata de locks de sesión ante cualquier interrupción o excepción física.
- **Caso Borde - Cero Variables Optimizables:**
  - En [tmm_utils_Rodrigo.py](file:///d:/archivos/TESIS/TMM-R/APIS/ellipsometry_API/tmm_utils_Rodrigo.py#L818) se robusteció la inicialización del optimizador. Los tensores se filtran antes de ser registrados en `optim.Adam` para incluir únicamente aquellos que requieran optimización real (con elementos).
  - Si no hay variables optimizables (todas las capas e índices fijos), se crea un `DummyOptimizer` que implementa `zero_grad()` y `step()` como operaciones vacías (`pass`). En el bucle de optimización se condiciona la llamada a `loss.backward()` y `optimizer.step()` a que existan tensores optimizables, evitando la excepción clásica de PyTorch (`RuntimeError: element 0 of tensors does not require grad`) cuando no hay gradientes.
- **Control de STOP y Cancelación Activa:** Se implementó un mecanismo de cancelación controlada para simulaciones elipsométricas de larga duración. Se inyectó el callback `check_cancel_fn` en el motor de PyTorch (`tmm_utils_Rodrigo.py` y `fit_elipsometrico.py`) para abortar levantando `InterruptedError`. Se expuso el endpoint `POST /cancelar_simulacion/{sim_id}` en `main.py` y se incorporó un botón de STOP interactivo rojo en el frontend (`static/index.html`). Las peticiones canceladas retornan un status `"cancelled"` controlado (HTTP 200).
- **Soporte para Parámetros de Dispersión Fijos:** Se diseñó un flujo de optimización paramétrica mixto. En la interfaz web (`index.html`) se incorporó un checkbox `"Opt"` para cada parámetro de dispersión ($A, B, C$ para Cauchy y $f$ para Bruggeman) para conmutar entre rango y valor fijo. El backend en `tmm_utils_Rodrigo.py` detecta si un límite paramétrico es idéntico en su cota inferior y superior, excluyendo el parámetro del vector de logits optimizables de PyTorch y re-inyectándolo de forma constante y diferenciable en el bloque de reconstrucción de índices de refracción.
- **Chi-Cuadrado Mínimo Histórico Real:** Se implementó el registro del error mínimo histórico a través de todas las épocas de optimización para cada semilla. Esto corrige la inestabilidad de las últimas épocas del optimizador Adam y reporta el valor mínimo real absoluto. Este valor se persiste en la columna `chi2_min` del modelo de base de datos MySQL (con migración automática de DDL al arranque) y se expone en la sección de resultados e historial.
- **Soporte para Desviación Estándar y Chi-Cuadrado Reducido ($\chi^2_{red}$):** Se actualizó el optimizador para admitir opcionalmente archivos de desviación estándar de las mediciones experimentales. El sistema propaga analíticamente los errores $\sigma_\psi$ y $\sigma_\Delta$ a los parámetros de trabajo $I_s$ y $I_c$ en el plano del plano complejo, y calcula la pérdida mediante la métrica del Chi-Cuadrado Reducido ponderada por estos desvíos.
- **Resolución de Duplicados Espectrales y Filtrado de Footers:** Se implementó una lógica de filtrado dinámico para omitir las secciones finales de resumen (`# MINIMA:` y `# MAXIMA:`) de los archivos del elipsómetro, previniendo longitudes de onda no monótonas y valores `NaN` en `fit_elipsometrico.py` y `graficos.py`.
- **Modelos de Entrada Flexibles en la API:** Se extendió el modelo `SimParam` en `main.py` para admitir `data_path`, `std_path` y `skiprows` opcionales en el body de la petición, permitiendo un flujo dinámico y retrocompatible.
- **Funcionalidad Autorange Dinámico (Rango Adaptativo de Parámetros):** Se implementó una lógica de rango adaptativo (`autorange_fit_ellipsometry_torch`) en `tmm_utils_Rodrigo.py` que previene la saturación de espesores y constantes de dispersión durante la optimización. Detecta si un parámetro está cerca del límite del rango, lo expande un 25% hacia esa dirección y contrae el límite opuesto un 10%, ejecutando iterativamente el ajuste hasta lograr convergencia. Se corrigió un error de tipo `ValueError` debido al desempaquetado de listas anidadas en los límites de Bruggeman (`f_bounds`) y se unificó la numeración de capas del stack óptico en los logs del backend. Esta funcionalidad fue integrada en `fit_elipsometrico.py`, expuesta en la API Pydantic (`main.py`) y agregada como checkbox de optimización en la interfaz frontend (`static/index.html`).
- **Estabilización Numérica (Clamping de n y k):** Se implementó clamping estricto en `tmm_utils_Rodrigo.py` para impedir que la parte real del índice de refracción ($n \ge 1.0$) y la parte imaginaria ($k \ge 0.0$) tomen valores físicamente imposibles en los bloques de Cauchy y Cauchy Absorbente durante la optimización.
- **Restricción de Límites en Frontend:** Se acotaron los límites por defecto en `static/index.html` para los parámetros de Cauchy ($A \in [0.0, 4.0]$, $B,C \in [-4.0, 4.0]$) y la fracción de Bruggeman ($f \in [0.0, 0.5]$) para guiar la búsqueda del gradiente y prevenir divergencias.
- **Creación de Skills de Automatización:** Se implementó un plugin de skills del sistema (`C:\Users\mentu\AppData\Local\TMM\`) con herramientas personalizadas:
  - `tmm-debug-nan`: Checklist de depuración en caso de pérdida NaN.
  - `tmm-add-material`: Automatización para ingesta de nuevos materiales ópticos.
  - `tmm-analyze-fit`: Graficado de resultados de ajuste y dispersión $n(\lambda)$.
  - `tmm-export-results`: Generación de reportes listos para incluir en la tesis (LaTeX, CSV, Markdown).
  - `git-selective-commit`: Automatización de commits con exclusión selectiva de archivos.
  - `Updater`: Mantenedor automático de este archivo.

## 2. Estado Actual de Componentes

| Capa | Estado y Detalles |
|---|---|
| **Física (`tmm_core.py`)** | Estable. Contiene algoritmos vectorizados e implementaciones diferenciables con PyTorch (`complex128`). |
| **Numérica (`tmm_utils_Rodrigo.py`)** | Mapea logits a físicos con sigmoide, aplica clamping de seguridad ($n \ge 1.0$, $k \ge 0.0$), soporta la mezcla de parámetros fijos y optimizables de dispersión. Soporta de forma nativa e independiente espesores fijos y optimizables, y estabiliza el optimizador ante cero parámetros optimizables mediante `DummyOptimizer` y control de gradientes vacíos. |
| **Ajuste (`fit_elipsometrico.py`)** | Orquesta la carga de datos experimentales, inicialización de tensores y propagación del callback de cancelación (`check_cancel_fn`). |
| **API (`main.py`)** | FastAPI. Expone `/cancelar_simulacion/{sim_id}` (retorno controlado `"cancelled"`). Implementa migración automática DDL de `chi2_min` usando transacciones de SQLAlchemy y mitigando bloqueos indefinidos de arranque mediante `SET SESSION lock_wait_timeout = 3`. |
| **Persistencia (`models.py`)** | Actualizado. Agregada la columna `chi2_min` (FLOAT) a la tabla de `Stack`. |
| **Frontend (`static/index.html`)** | SPA. Incorpora el botón de STOP al lado de Ejecutar, checkbox "Opt" para conmutar entre rango y valor fijo tanto en espesores de capa intermedia como en parámetros individuales de dispersión Cauchy/Bruggeman. |

## 3. Próximos Pasos y Tareas Pendientes

- Validar el comportamiento de `f_air` en el modelo Bruggeman y habilitar opcionalmente su optimización si el usuario proporciona límites en la configuración.

