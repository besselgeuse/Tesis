# Guía Didáctica: Funcionamiento de `fit_ellipsometry_torch`

Esta guía explica detalladamente la lógica, la física y los trucos matemáticos implementados en la función `fit_ellipsometry_torch`. El objetivo es que seas capaz de comprender el diseño del código, replicar su funcionamiento o adaptarlo a otros problemas de ajuste físico de películas delgadas.

---

## 1. Fundamentos Físicos de la Elipsometría

La elipsometría es una técnica de caracterización óptica no destructiva que mide el cambio en el estado de polarización de la luz al reflejarse en una muestra.

### El Parámetro Fundamental $\rho$
Cuando la luz incide sobre una muestra con un ángulo $\theta_0$, los coeficientes de reflexión compleja para las polarizaciones transversal eléctrica ($s$) y transversal magnética ($p$) son $r_s$ y $r_p$, respectivamente. La elipsometría mide la relación entre estos coeficientes:

$$\rho = \frac{r_p}{r_s} = \tan(\psi) e^{i \Delta}$$

Donde:
*   $\psi$ (Psi) representa la relación de amplitudes entre las reflexiones de las ondas $p$ y $s$ ($\tan\psi = |r_p| / |r_s|$). Su rango físico es $[0, \pi/2]$ (o $[0^\circ, 90^\circ]$).
*   $\Delta$ (Delta) representa la diferencia de fase introducida por la reflexión entre ambas componentes ($\Delta = \delta_p - \delta_s$). Su rango físico es $[-\pi, \pi]$ (o $[-180^\circ, 180^\circ]$).

### Parámetros Elipsométricos Medidos: $I_s$ e $I_c$
En la práctica, muchos elipsómetros modernos (especialmente los de modulador de fase o analizador rotatorio) no miden $\psi$ y $\Delta$ directamente. En su lugar, miden tres intensidades normalizadas conocidas como los parámetros de Stokes o coeficientes elipsométricos. Dos de los más comunes son:

$$I_s = \sin(2\psi) \sin(\Delta)$$
$$I_c = \sin(2\psi) \cos(\Delta)$$

**¿Por qué el código ajusta $I_s$ e $I_c$ en lugar de $\psi$ y $\Delta$?**
1.  **Discontinuidades de Fase:** El ángulo $\Delta$ tiene una discontinuidad de salto de fase natural en $-\pi$ y $\pi$ (o $-180^\circ$ y $180^\circ$). Si el optimizador intenta ajustar $\Delta$ directamente cerca de estos bordes, los gradientes se vuelven inestables o erróneos debido al salto abrupto. Al usar $\sin(\Delta)$ y $\cos(\Delta)$, la fase se comporta como funciones continuas y suaves sin saltos bruscos.
2.  **Tratamiento de Singularidades:** Cuando $\psi \approx 0$ (poca reflectancia en la componente $p$), la fase $\Delta$ se vuelve indeterminada y extremadamente ruidosa. Multiplicar por $\sin(2\psi)$ atenúa naturalmente el impacto del ruido de $\Delta$ en estas zonas.

### ¿Por qué la incidencia normal no funciona?
En incidencia normal ($\theta_0 = 0^\circ$), las polarizaciones $s$ y $p$ son físicamente equivalentes y degeneradas. Por convención geométrica, $r_p = -r_s$. Esto implica que:

$$\rho = \frac{r_p}{r_s} = -1 \implies \tan(\psi) = 1 \implies \psi = 45^\circ$$
$$\Delta = 180^\circ \text{ (o } \pi \text{ rad)}$$

Bajo estas condiciones:
*   $I_s = \sin(90^\circ) \sin(180^\circ) = 0$
*   $I_c = \sin(90^\circ) \cos(180^\circ) = -1$

Dado que $I_s = 0$ e $I_c = -1$ para cualquier material y espesor a $0^\circ$, **es imposible realizar elipsometría a incidencia normal**. Por eso elipsómetros reales operan a ángulos oblicuos (típicamente entre $55^\circ$ y $75^\circ$), cerca del ángulo de Brewster del sustrato, donde la sensibilidad a los espesores y capas delgadas es máxima.

---

## 2. Optimización Paralela y Multi-Start con PyTorch

Ajustar espesores y propiedades de refracción a partir de curvas elipsométricas es un problema altamente **no lineal** y con abundantes **mínimos locales** (debido al comportamiento oscilatorio de la interferencia óptica).

### El Enfoque Tradicional vs. Enfoque PyTorch
*   **Enfoque tradicional:** Algoritmos como Levenberg-Marquardt (usado en `scipy.optimize.curve_fit`) o L-BFGS-B calculan derivadas numéricas aproximadas por diferencias finitas. Si se inician desde una única semilla, quedan atrapados casi de inmediato en el mínimo local más cercano. Correr muchas semillas de manera secuencial es sumamente lento.
*   **Enfoque PyTorch:** 
    1.  **Gradientes exactos (Autograd):** PyTorch calcula los gradientes analíticos exactos de la función de pérdida con respecto a todos los parámetros optimizables mediante diferenciación automática. Esto acelera drásticamente la convergencia de optimizadores basados en gradiente como **Adam**.
    2.  **Multi-start masivo en GPU:** En lugar de ejecutar una sola optimización, representamos los parámetros como tensores de dimensión `[num_starts, num_parametros]`. PyTorch procesa de forma vectorial y paralela todas las semillas (ej. 1000 ejecuciones simultáneas) aprovechando al máximo los núcleos paralelos de la GPU o la CPU.

---

## 3. El Truco del Sigmoide (Sigmoid Trick)

Los espesores y los parámetros ópticos de Cauchy tienen restricciones físicas estrictas (límites o *bounds*), por ejemplo, un espesor $d$ no puede ser menor a $0$ ni mayor a $150\text{ nm}$.

Optimizadores clásicos de PyTorch como `Adam` o `SGD` no soportan límites directamente; asumen que los parámetros pueden tomar cualquier valor real entre $-\infty$ y $\infty$. Si un espesor se volviera negativo en un paso de optimización, la simulación física colapsaría.

### Mapeo Matemático con Sigmoide
Para solucionar esto sin añadir pesadas restricciones matemáticas, aplicamos una transformación que mapea el espacio ilimitado del optimizador al espacio físico acotado de la simulación.

Sea un parámetro físico $p_{\text{phys}}$ acotado en el intervalo $[p_{\text{min}}, p_{\text{max}}]$. Definimos una variable interna llamada **logit** ($p_{\text{logit}}$) que puede tomar cualquier valor real en $(-\infty, \infty)$. La relación viene dada por la función sigmoide ($\sigma$):

$$p_{\text{phys}} = p_{\text{min}} + (p_{\text{max}} - p_{\text{min}}) \cdot \sigma(p_{\text{logit}})$$

Donde la función sigmoide se define como:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
  p_logit (Espacio del Optimizador)       p_phys (Espacio Físico de Simulación)
         (-inf, +inf)                        [p_min, p_max]
              │                                    │
              ▼                                    ▼
       ┌─────────────┐       ┌──────────────────────────────────────────────┐
       │   Optim.    │ ───>  │ p_phys = p_min + (p_max-p_min) * sigmoide(x) │
       └─────────────┘       └──────────────────────────────────────────────┘
```

*   Si $p_{\text{logit}} \to -\infty$, $\sigma(p_{\text{logit}}) \to 0 \implies p_{\text{phys}} \to p_{\text{min}}$.
*   Si $p_{\text{logit}} \to \infty$, $\sigma(p_{\text{logit}}) \to 1 \implies p_{\text{phys}} \to p_{\text{max}}$.
*   Si $p_{\text{logit}} = 0$, $\sigma(p_{\text{logit}}) = 0.5 \implies p_{\text{phys}}$ está exactamente en el centro del intervalo.

### Clampeo de Logits
Cuando el optimizador aleja mucho una variable de su óptimo, el logit puede crecer demasiado (ej. $+10$). En estos extremos lejanos de la sigmoide, su derivada $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ tiende exponencialmente a cero. Esto se conoce como **saturación de gradiente** (o *vanishing gradient*), y hace que el optimizador se "congele" y deje de actualizar esa variable.

Para evitarlo, el código realiza un clampeo preventivo en cada época:
```python
with torch.no_grad():
    d_opt.data.clamp_(-5, 5)
    if is_parametric:
        p_opt.data.clamp_(-5, 5)
```
Dado que $\sigma(-5) \approx 0.0067$ y $\sigma(5) \approx 0.9933$, limitar los logits a $[-5, 5]$ permite explorar el $98.6\%$ del rango físico disponible manteniendo siempre gradientes con magnitudes saludables para seguir optimizando.

---

## 4. Modelos de Dispersión Paramétricos

Las capas reales cambian su índice de refracción $n$ y coeficiente de extinción $k$ según la longitud de onda $\lambda$. En lugar de optimizar un valor de $n$ independiente para cada longitud de onda (lo cual crearía cientos de variables redundantes y sobreajuste), representamos el comportamiento cromático mediante ecuaciones físicas parametrizadas.

### A. Modelo de Cauchy Transparente
Es excelente para dieléctricos de banda ancha (como $\text{SiO}_2$ o $\text{Al}_2\text{O}_3$) en el espectro visible e infrarrojo cercano, donde la absorción es nula ($k=0$):

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$

Donde:
*   $\lambda$ se expresa típicamente en micrómetros ($\mu\text{m}$) o nanómetros escalados para evitar problemas numéricos. En nuestro código de PyTorch, las longitudes de onda están escaladas para mantener estabilidad numérica:
    ```python
    inv_lam2 = 1e4 / (lams_2d ** 2)  # Escala lambda en nm a micras de forma implícita
    inv_lam4 = 1e9 / (lams_2d ** 4)
    ```
*   **Parámetros a optimizar:** $A$ (índice base), $B$ (dispersión de primer orden), $C$ (dispersión de segundo orden).

### B. Modelo de Cauchy Absorbente
Útil para materiales con una absorción débil o moderada en el espectro ultravioleta/azul (como algunos óxidos metálicos como $\text{TiO}_2$). Modelamos la parte compleja del índice de refracción $\tilde{n} = n + i k$ con coeficientes independientes para la absorción:

$$n(\lambda) = A + \frac{B}{\lambda^2} + \frac{C}{\lambda^4}$$
$$k(\lambda) = D + \frac{E}{\lambda^2} + \frac{F}{\lambda^4}$$

*   **Parámetros a optimizar:** 6 parámetros en total ($A, B, C$ para $n$ y $D, E, F$ para $k$). El código asegura que $k$ no tome valores físicos negativos aplicando:
    ```python
    k_calc = torch.clamp(k_calc, min=0.0)
    ```

### C. Modelo de Bruggeman (Effective Medium Approximation - EMA)
Permite modelar una capa mezcla (por ejemplo, una capa porosa o rugosa) compuesta por dos fases: una matriz densa y aire/vacío ($n_{\text{air}} = 1.0$).

La relación matemática que define el índice complejo efectivo $n_{\text{eff}}$ de la mezcla viene dada por la ecuación cuadrática de Bruggeman:

$$f_{\text{air}} \frac{n_{\text{air}}^2 - n_{\text{eff}}^2}{n_{\text{air}}^2 + 2 n_{\text{eff}}^2} + (1 - f_{\text{air}}) \frac{n_{\text{base}}^2 - n_{\text{eff}}^2}{n_{\text{base}}^2 + 2 n_{\text{eff}}^2} = 0$$

Donde:
*   $n_{\text{base}}$ es el índice complejo del material denso subyacente.
*   $f_{\text{air}}$ es la fracción volumétrica de aire (porosidad).
*   **Configuración en el código:**
    *   **Fijo:** Podemos especificar una fracción constante (ej. `f_air = 0.42`).
    *   **Optimizable:** Definimos un rango de búsqueda (ej. `f_bounds = (0.3, 0.7)`). En este caso, se añade una variable extra al optimizador para encontrar la fracción volumétrica óptima de forma dinámica.

---

## 5. Función de Pérdida y Bucle de Optimización

El bucle optimiza de forma conjunta todos los espesores y coeficientes de dispersión calculando el error entre el modelo teórico y el experimental.

1.  **Reconstrucción:** En cada época, los logits de espesores ($d_{\text{opt}}$) y parámetros ópticos ($p_{\text{opt}}$) se transforman a sus dimensiones físicas correspondientes ($d_{\text{fisico}}$, $n_{\text{complex}}$) usando la sigmoide.
2.  **Simulación Óptica (coh_tmm_torch_batched):** Ejecuta la formulación de matriz de transferencia coherente de forma batcheada para todas las semillas y longitudes de onda en paralelo para obtener $r_s$ y $r_p$.
3.  **Simulación de Parámetros Teóricos:**
    $$\rho = \frac{r_p}{r_s} \implies \psi_{\text{teo}} = \arctan(|\rho|), \quad \Delta_{\text{teo}} = \text{angle}(\rho)$$
    $$I_{s,\text{teo}} = \sin(2\psi_{\text{teo}}) \sin(\Delta_{\text{teo}})$$
    $$I_{c,\text{teo}} = \sin(2\psi_{\text{teo}}) \cos(\Delta_{\text{teo}})$$
4.  **Cálculo de Pérdida (Loss MSE):**
    Calculamos el Error Cuadrático Medio por cada semilla $j$:
    
    $$\text{Loss}_j = \frac{1}{N} \sum_{\lambda} \left[ (I_{s,\text{teo}}^{(j)}(\lambda) - I_{s,\text{exp}}(\lambda))^2 + (I_{c,\text{teo}}^{(j)}(\lambda) - I_{c,\text{exp}}(\lambda))^2 \right]$$
    
    Para optimizar de forma eficiente con Autograd, sumamos las pérdidas de todas las semillas y ejecutamos `loss.backward()`.
5.  **Extracción del Mínimo Global:**
    Al finalizar las épocas, seleccionamos la semilla que logró el menor error final utilizando `loss_final.argmin()`. Sus parámetros óptimos son retornados como el ajuste definitivo.

---

## 6. Ejemplos de Configuración de la API

A continuación se muestran ejemplos prácticos de cómo configurar la API mejorada:

### Ejemplo 1: Capa de TiO2 rugosa/porosa con Bruggeman y Cauchy Absorbente
Queremos ajustar un stack sobre Silicio, donde la capa densa de $\text{TiO}_2$ es paramétrica (Cauchy con absorción y límites personalizados) y la capa superior es porosa (Bruggeman optimizando la fracción de aire):

```python
from fit_elipsometrico import ajuste_elipsometrico

# Nombre de la capa paramétrica puede ser cualquier etiqueta dummy
layer_names = ['air', 'T1_porosa', 'T1_densa', 'Si']

# Configuración avanzada de modelos
layer_models = [
    None,  # air
    {
        'model': 'bruggeman',
        'f_bounds': (0.1, 0.6)  # Optimizar fracción de aire entre 10% y 60%
    },
    {
        'model': 'cauchy_absorbent',
        # Definimos bounds personalizados para A, B, C, D, E, F:
        'bounds': [
            (2.0, 2.7),   # A (n base alto para TiO2)
            (0.0, 1.5),   # B
            (-0.5, 0.5),  # C
            (0.0, 0.1),   # D (absorción k base baja)
            (0.0, 0.5),   # E
            (-0.1, 0.1)   # F
        ]
    },
    None  # Sustrato estático Si
]

# Límites de espesores para capas intermedias (T1_porosa y T1_densa)
d_bounds = [(1.0, 40.0), (20.0, 120.0)]

best_thick, best_Is, best_Ic, best_params, wl = ajuste_elipsometrico(
    data_path='./Datos-28-5/TiO2_Si_Sputtering_sincinta.txt',
    skiprows=5,
    layer_names=layer_names,
    layer_models=layer_models,
    d_bounds=d_bounds,
    theta_0=69.5,
    num_starts=300,
    num_epochs=400,
    use_cuda=True,
    cache_path='./cache'  # Se guardará el hash de la simulación para carga instantánea posterior
)
```
