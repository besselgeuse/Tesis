# -*- coding: utf-8 -*-
"""
==============================================================================
GUÍA DIDÁCTICA: ¿Qué hace fit_ellipsometry_torch? Bloque por bloque.
==============================================================================

Este archivo explica en detalle cómo funciona la función fit_ellipsometry_torch
de tmm_utils_Rodrigo.py. Está pensado para que puedas entenderlo SIN tener que
leer el código fuente, y con ejemplos ejecutables donde sea posible.

Usá el modo de celdas (Ctrl+Enter en cada celda de VS Code / Jupyter).
"""

#%%
# =============================================================================
# RESUMEN CONCEPTUAL: ¿Qué problema resuelve esta función?
# =============================================================================
#
# Tenés medidas experimentales de un elipsómetro: para cada longitud de onda λ,
# el instrumento te da dos ángulos: ψ (psi) y Δ (delta).
# Esos ángulos codifican cuánto cambia la polarización de la luz al reflejarse
# en tu muestra (ej: aire / SiO2 / Si).
#
# Tu pregunta es: "¿Qué espesor tiene mi capa de SiO2, y cuál es su índice
# de refracción n(λ)?"
#
# La función busca los parámetros (espesores, A, B, C de Cauchy) que hacen que
# el modelo TMM reproduzca los ángulos experimentales lo mejor posible.
#
# ESQUEMA DEL FLUJO:
#
#  Parámetros         Modelo TMM           Pérdida
#   iniciales    →   (cálculo óptico)  →  (error vs exp)  →  Gradiente
#       ↑                                                         |
#       └─────────────── Optimizador Adam ←──────────────────────┘
#                        (ajusta parámetros)
#
# Esto se repite durante `num_epochs` iteraciones, con `num_starts` puntos
# de partida distintos en paralelo (para no quedar atrapado en un mínimo local).

#%%
# =============================================================================
# BLOQUE 0: Firma de la función y parámetros de entrada
# =============================================================================
#
# def fit_ellipsometry_torch(
#     n_list,         # Índices de refracción de cada capa, tensor [capas × λs]
#     d_bounds,       # Límites de espesores para cada capa finita, ej: [(5, 100), ...]
#     lams,           # Longitudes de onda en nm, tensor [num_wl]
#     Is_exp,         # Is experimental = sin(2ψ)·sin(Δ), tensor [num_wl]
#     Ic_exp,         # Ic experimental = sin(2ψ)·cos(Δ), tensor [num_wl]
#     c_list=None,    # Ignorado (compatibilidad con versiones anteriores)
#     th_0=0.0,       # Ángulo de incidencia en RADIANES (ej: 71° → np.radians(71))
#     num_starts=50,  # Cuántos puntos de partida aleatorios probar en paralelo
#     num_epochs=150, # Cuántas iteraciones de gradiente descendente
#     lr=2.0,         # Learning rate del optimizador Adam
#     use_cuda=False, # True → usar GPU (mucho más rápido con muchas semillas)
#     layer_models=None,  # ej: [None, 'cauchy', None] → ajusta Cauchy en capa 1
#     layer_names=None    # ej: ['air', 'SiO2_Daniel', 'Si'] → para el output
# )
#
# ¿Por qué Is e Ic en lugar de ψ y Δ directamente?
# Porque Is e Ic son continuas (no saltan de 360° a 0°), lo que hace que la
# optimización por gradiente funcione mucho mejor. La transformación es:
#
#   Is = sin(2ψ) · sin(Δ)
#   Ic = sin(2ψ) · cos(Δ)
#
# Y la inversa:
#   ψ = 0.5 · arcsin(√(Is² + Ic²))
#   Δ = arctan2(Is, Ic)

import numpy as np
import matplotlib.pyplot as plt

# Ejemplo de la transformación Is/Ic ↔ ψ/Δ:
psi_ejemplo = np.radians(30)   # 30 grados
delta_ejemplo = np.radians(120) # 120 grados

Is_ej = np.sin(2 * psi_ejemplo) * np.sin(delta_ejemplo)
Ic_ej = np.sin(2 * psi_ejemplo) * np.cos(delta_ejemplo)
print(f"Is = {Is_ej:.4f},  Ic = {Ic_ej:.4f}")

# Recuperar ψ y Δ:
psi_rec = 0.5 * np.degrees(np.arcsin(np.sqrt(Is_ej**2 + Ic_ej**2)))
delta_rec = np.degrees(np.arctan2(Is_ej, Ic_ej))
print(f"ψ recuperada = {psi_rec:.2f}°  (esperado: 30°)")
print(f"Δ recuperada = {delta_rec:.2f}°  (esperado: 120°)")

#%%
# =============================================================================
# BLOQUE 1: Elegir dispositivo y mover datos a GPU o CPU
# =============================================================================
#
# Las primeras líneas de la función detectan si hay GPU disponible y mueven
# todos los tensores al mismo dispositivo.
#
# >>> device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
# >>> lams = lams.to(device)
# >>> Is_exp = Is_exp.to(device)
# >>> Ic_exp = Ic_exp.to(device)
# >>> n_list = n_list.to(device)
#
# ¿Por qué importa esto?
# Todos los tensores que participan en una operación PyTorch DEBEN estar en el
# mismo dispositivo. Si uno está en CPU y otro en GPU, da error. Por eso, al
# principio se mueve todo al lugar correcto de una vez.
#
# EJEMPLO CONCEPTUAL:
# CPU: como calculadora de escritorio → correcta pero lenta con muchos datos
# GPU: como miles de calculadoras en paralelo → ideal para matrices grandes
#      Con 1000 semillas y 828 longitudes de onda = 828.000 operaciones/época
#      La GPU puede hacer todas estas operaciones SIMULTÁNEAMENTE.

#%%
# =============================================================================
# BLOQUE 2: Detectar cuáles capas tienen modelos paramétricos
# =============================================================================
#
# Cuando llamás a la función con layer_models=[None, 'cauchy', None], la función
# arma una lista interna (disp_layers) con la información de cada capa que tiene
# un modelo a ajustar.
#
# Para 'cauchy': n(λ) = A + B·(1e4/λ²) + C·(1e9/λ⁴)
#   → 3 parámetros a ajustar: A, B, C
#   → límites por defecto: A ∈ [1.0, 3.0], B ∈ [-1.0, 1.0], C ∈ [-1.0, 1.0]
#
# Para 'bruggeman': mezcla efectiva (EMA) de dos materiales (no tiene parámetros
#   propios, hereda el n del material adyacente calculado con Cauchy)
#
# El resultado disp_layers queda algo así:
#
# disp_layers = [
#   {
#     'layer_idx': 2,          # índice en n_list (capa 2 = SiO2_Daniel)
#     'model_type': 'cauchy',
#     'param_bounds': [(1.0, 3.0), (-1.0, 1.0), (-1.0, 1.0)],
#     'num_params': 3,
#     'start_idx': 0           # posición en el vector global de parámetros
#   }
# ]
#
# total_disp_params = 3  (A, B, C del Cauchy)

# Ejemplo numérico de la ecuación de Cauchy:
lam_nm = np.linspace(400, 900, 500)
A, B, C = 1.46, 0.003, 0.0    # valores típicos de SiO2
n_cauchy = A + B * 1e4 / lam_nm**2 + C * 1e9 / lam_nm**4

plt.figure(figsize=(8, 4))
plt.plot(lam_nm, n_cauchy, 'b-', linewidth=2)
plt.xlabel('λ [nm]')
plt.ylabel('n(λ)')
plt.title('Ecuación de Cauchy: n(λ) = A + B·(1e4/λ²) + C·(1e9/λ⁴)')
plt.grid(True, alpha=0.3)
plt.show()
print(f"n a 500nm = {A + B*1e4/500**2 + C*1e9/500**4:.4f}")
print(f"n a 800nm = {A + B*1e4/800**2 + C*1e9/800**4:.4f}")

#%%
# =============================================================================
# BLOQUE 3: La Transformación LOGIT — el truco clave de la optimización
# =============================================================================
#
# Los parámetros físicos tienen límites: el espesor no puede ser negativo ni
# exceder un máximo. Pero el optimizador Adam mueve los parámetros libremente
# en (-∞, +∞). ¿Cómo los limitamos?
#
# Respuesta: usamos la función LOGIT (la inversa del sigmoid) para transformar
# los parámetros a un espacio sin límites, y sigmoid para volver al espacio físico.
#
# Dado un parámetro físico p en [p_min, p_max]:
#
#   1. Normalizar a [0, 1]:   u = (p - p_min) / (p_max - p_min)
#   2. Aplicar logit:         logit = log(u / (1 - u))     ← va a (-∞, +∞)
#
# Y el camino inverso (adentro de la optimización):
#
#   1. Aplicar sigmoid:       u = 1 / (1 + exp(-logit))   ← vuelve a [0, 1]
#   2. Escalar al rango:      p = p_min + (p_max - p_min) * u
#
# Por eso en el código ves:
#   col = d_min + (d_max - d_min) * torch.sigmoid(d_opt_tensor[:, o_idx])

u = np.linspace(-6, 6, 300)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(u, 1 / (1 + np.exp(-u)), 'b-', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
plt.axvline(-5, color='red', linestyle=':', label='clamp = ±5')
plt.axvline(5, color='red', linestyle=':')
plt.xlabel('logit (parámetro interno)')
plt.ylabel('sigmoid(logit) = valor normalizado')
plt.title('sigmoid: (−∞,+∞) → [0,1]')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Ejemplo: espesor entre 5 y 100 nm
d_min, d_max = 5.0, 100.0
plt.plot(u, d_min + (d_max - d_min) / (1 + np.exp(-u)), 'g-', linewidth=2)
plt.axhline(d_min, color='gray', linestyle='--', alpha=0.5)
plt.axhline(d_max, color='gray', linestyle='--', alpha=0.5)
plt.axvline(-5, color='red', linestyle=':', label='clamp = ±5')
plt.axvline(5, color='red', linestyle=':')
plt.xlabel('logit (parámetro interno)')
plt.ylabel('Espesor [nm]')
plt.title(f'Espesor: (−∞,+∞) → [{d_min}, {d_max}] nm')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ¿Por qué el clamp en ±5?
# sigmoid(5) = 0.9933 → el optimizador puede alcanzar el 99.3% del rango
# sigmoid(15) ≈ 1.0000 → el gradiente es casi 0, el parámetro queda ATRAPADO
print("sigmoid(±5)  →", 1/(1+np.exp(-5)), "y", 1/(1+np.exp(5)))
print("sigmoid(±15) →", 1/(1+np.exp(-15)), "y", 1/(1+np.exp(15)))
print("d(sigmoid)/dx en x=5  →", (1/(1+np.exp(-5))) * (1 - 1/(1+np.exp(-5))))
print("d(sigmoid)/dx en x=15 →", (1/(1+np.exp(-15))) * (1 - 1/(1+np.exp(-15))))

#%%
# =============================================================================
# BLOQUE 4: Inicialización Multi-Start (el "por qué 1000 semillas")
# =============================================================================
#
# Un problema de optimización no-lineal puede tener MUCHOS mínimos locales.
# Si arrancás desde un solo punto, podés quedar atrapado en uno que no es el mejor.
#
# La estrategia multi-start:
# → Generar num_starts puntos de partida ALEATORIOS dentro del rango físico
# → Optimizar TODOS EN PARALELO en la GPU (gratis en términos de tiempo!)
# → Al final elegir el que dio el menor error
#
# >>> d_init_phys = torch.empty(num_starts, device=device).uniform_(d_min, d_max)
#
# Esto crea un vector de 1000 espesores iniciales aleatorios entre d_min y d_max.
# Luego se transforman a logits y se guardan en d_opt[semilla, capa].
#
# d_opt tiene forma [num_starts, num_opt_layers]  ← un logit por (semilla, capa)
# p_opt tiene forma [num_starts, total_disp_params] ← un logit por (semilla, param)
#
# VISUALIZACIÓN de la inicialización:
np.random.seed(42)
num_starts_ej = 500
d_min, d_max = 1.0, 80.0
d_inicio = np.random.uniform(d_min, d_max, num_starts_ej)

plt.figure(figsize=(8, 3))
plt.hist(d_inicio, bins=30, color='steelblue', edgecolor='white')
plt.xlabel('Espesor inicial [nm]')
plt.ylabel('Cantidad de semillas')
plt.title(f'{num_starts_ej} semillas de inicio distribuidas uniformemente\nentre {d_min} y {d_max} nm')
plt.grid(True, alpha=0.3)
plt.show()

#%%
# =============================================================================
# BLOQUE 5: Las funciones internas reconstruct_d_fisico y reconstruct_n_list_batched
# =============================================================================
#
# Estas dos funciones son el "puente" entre los logits del optimizador y los
# valores físicos que necesita el modelo TMM.
#
# --- reconstruct_d_fisico ---
# Recibe d_opt [num_starts, num_capas_opt] en espacio logit
# Devuelve d_fisico [num_starts, num_capas_finitas] en nanómetros
#
#   col = d_min + (d_max - d_min) * sigmoid(logit)
#
# Las capas con espesor FIJO simplemente ponen su valor constante.
# Al final, se adjuntan columnas de ∞ al inicio y al final para el superestrato
# y sustrato (que tienen "espesor infinito" por definición del TMM):
#   d_full = [∞, d_capa1, d_capa2, ..., ∞]
#
# --- reconstruct_n_list_batched ---
# Recibe p_opt [num_starts, total_disp_params] en espacio logit
# Devuelve n_list_3d [num_starts, num_layers, num_wl] compleja
#
# Pasos:
# 1. Copia capas estáticas: repite el tensor n[idx] para cada semilla
#    → expand() no copia memoria, es eficiente
# 2. Calcula capas Cauchy usando los logits actuales:
#    A, B, C = sigmoid(logit_A, B, C) escalados a sus rangos
#    n = A + B * inv_lam2 + C * inv_lam4  ← inv_lam2/4 precomputados
# 3. Calcula capas Bruggeman usando el n de la capa Cauchy adyacente:
#    n_brugg = n_eff_torch(n_cauchy, n_aire=1.0, f=0.5)
# 4. Apila todo: torch.stack(layer_tensors, dim=1)
#
# ¿Por qué torch.stack al final en vez de ir escribiendo en un tensor?
# Porque escribir en un tensor existente es una operación IN-PLACE que rompe
# el grafo de gradientes de PyTorch (da error en backward()).
# Al usar una lista de Python y apilar al final, cada tensor es "nuevo" y
# PyTorch puede trazar todos los gradientes correctamente.

#%%
# =============================================================================
# BLOQUE 6: El bucle central — una época de optimización
# =============================================================================
#
# El bucle principal hace esto por cada epoch:
#
# for epoch in range(num_epochs):
#     optimizer.zero_grad()           # 1. Borrar gradientes acumulados del paso anterior
#
#     d_fisico = reconstruct_d_fisico(d_opt)           # 2. Logits → espesores físicos
#     d_full_3d = [∞, d_fisico, ∞] expandido a [batch, capas, λs]
#
#     n_list_3d = reconstruct_n_list_batched(p_opt)    # 3. Logits → índices n(λ)
#
#     res_s = coh_tmm_torch_batched('s', n_list_3d, d_full_3d, th_0, lams)
#     res_p = coh_tmm_torch_batched('p', ...)          # 4. TMM para pol. s y p
#
#     r_s = res_s['r']   # Coeficiente de reflexión complejo para pol. s
#     r_p = res_p['r']   # Coeficiente de reflexión complejo para pol. p
#
#     rho = conj(r_p / r_s)          # 5. Relación elipsométrica ρ = tan(ψ)·e^(iΔ)
#     psi_teo = atan(|ρ|)            # 6. Extraer ψ y Δ del modelo
#     delta_teo = angle(ρ)
#
#     Is_teo = sin(2ψ)·sin(Δ)        # 7. Calcular Is e Ic teóricos
#     Ic_teo = sin(2ψ)·cos(Δ)
#
#     loss = mean((Is_teo - Is_exp)² + (Ic_teo - Ic_exp)²)  # 8. Error cuadrático
#
#     loss.backward()                # 9. PyTorch calcula ∂loss/∂logit para TODO
#     optimizer.step()               # 10. Adam actualiza los logits
#
#     d_opt.data.clamp_(-5, 5)       # 11. Evitar saturación del sigmoid
#     p_opt.data.clamp_(-5, 5)

#%%
# =============================================================================
# BLOQUE 7: La ecuación elipsométrica ρ = r_p / r_s
# =============================================================================
#
# El elipsómetro mide la relación entre los coeficientes de reflexión de las
# dos polarizaciones (p y s). Esta relación se llama ρ y contiene ψ y Δ:
#
#   ρ = r_p / r_s = tan(ψ) · e^(iΔ)
#
# De donde:
#   |ρ| = tan(ψ)  →  ψ = arctan(|ρ|)
#   arg(ρ) = Δ
#
# NOTA IMPORTANTE sobre la convención del conjugado:
# El código usa rho = conj(r_p / r_s), no rho = r_p / r_s directamente.
# Esto es porque la convención de la señal del TMM y la del elipsómetro pueden
# diferir en el signo de Δ. El conjugado cambia el signo de la parte imaginaria,
# lo que equivale a negar Δ. Esta convención debe ser consistente con cómo
# mediste Is e Ic en tu elipsómetro.
#
# Ejemplo de cómo r_p y r_s son complejos:
r_p_ej = 0.3 + 0.2j   # número complejo: amplitud + fase
r_s_ej = -0.4 + 0.1j
rho_ej = np.conj(r_p_ej / r_s_ej)
psi_ej = np.degrees(np.arctan(np.abs(rho_ej)))
delta_ej = np.degrees(np.angle(rho_ej))
print(f"r_p = {r_p_ej},  r_s = {r_s_ej}")
print(f"ρ = {rho_ej:.4f}")
print(f"ψ = {psi_ej:.2f}°,  Δ = {delta_ej:.2f}°")
Is_calc = np.sin(2*np.radians(psi_ej)) * np.sin(np.radians(delta_ej))
Ic_calc = np.sin(2*np.radians(psi_ej)) * np.cos(np.radians(delta_ej))
print(f"Is = {Is_calc:.4f},  Ic = {Ic_calc:.4f}")

#%%
# =============================================================================
# BLOQUE 8: El TMM batched — coh_tmm_torch_batched
# =============================================================================
#
# Esta es la función que resuelve la óptica real: dadas las capas y sus índices,
# calcula cómo se refleja la luz.
#
# El método TMM (Transfer Matrix Method) trabaja con matrices 2×2 que representan
# cada interfaz y cada capa. Para un stack de N capas:
#
#   M_total = M_frontal × M_capa1 × M_capa2 × ... × M_(N-1)
#
# Donde cada M_i = (1/t_i) × [[e^(-iδ), 0], [0, e^(iδ)]] × [[1, r_i], [r_i, 1]]
#
# Los coeficientes r_i y t_i son las ecuaciones de Fresnel en cada interfaz:
#
# Para polarización s:
#   r_s = (n_i·cos(θ_i) - n_f·cos(θ_f)) / (n_i·cos(θ_i) + n_f·cos(θ_f))
#   t_s = 2·n_i·cos(θ_i) / (n_i·cos(θ_i) + n_f·cos(θ_f))
#
# Para polarización p:
#   r_p = (n_f·cos(θ_i) - n_i·cos(θ_f)) / (n_f·cos(θ_i) + n_i·cos(θ_f))
#   t_p = 2·n_i·cos(θ_i) / (n_f·cos(θ_i) + n_i·cos(θ_f))
#
# El ángulo θ en cada capa se calcula con la Ley de Snell:
#   n_0·sin(θ_0) = n_i·sin(θ_i)  →  θ_i = arcsin(n_0·sin(θ_0) / n_i)
#
# La fase δ acumulada al atravesar la capa de espesor d:
#   δ = (2π/λ) · n · cos(θ) · d
#
# Al final, el coeficiente de reflexión total es:
#   r = M[1,0] / M[0,0]
#
# La versión "batched" hace todo esto para [num_starts × num_wl] a la vez
# usando operaciones matriciales de PyTorch sobre tensores 4D.
#
# Dimensiones de los tensores adentro de coh_tmm_torch_batched:
#   n_list:  [num_starts, num_layers, num_wl]   (indices de refracción)
#   d_list:  [num_starts, num_layers, num_wl]   (espesores broadcast en λ)
#   th_list: [num_starts, num_layers, num_wl]   (ángulos de Snell)
#   delta:   [num_starts, num_layers, num_wl]   (fases acumuladas)
#   Mtilde:  [num_starts, num_wl, 2, 2]         (matriz de transferencia total)
#   r:       [num_starts, num_wl]               (coeficiente de reflexión)

# Ejemplo a mano de la Ley de Snell a 71°:
n_aire = 1.0
n_SiO2 = 1.46
theta_0 = np.radians(71.0)
theta_SiO2 = np.degrees(np.arcsin(n_aire * np.sin(theta_0) / n_SiO2))
print(f"Ángulo en aire: 71°")
print(f"Ángulo en SiO2 (n=1.46): {theta_SiO2:.2f}°")

#%%
# =============================================================================
# BLOQUE 9: La función de pérdida (loss) y por qué es así
# =============================================================================
#
# loss_per_start = mean_λ( (Is_teo - Is_exp)² + (Ic_teo - Ic_exp)² )
#
# Esta es el error cuadrático medio (MSE) sumando las contribuciones de Is e Ic.
# No usa ψ y Δ directamente por dos razones:
#
# 1. CONTINUIDAD: Δ salta de 360° a 0° (discontinuidad), lo que confunde al
#    optimizador. Is e Ic son continuas (son seno y coseno del ángulo).
#
# 2. PONDERACIÓN FÍSICA: El factor sin(2ψ) en Is e Ic "pesa" automáticamente
#    menos las zonas donde ψ ≈ 0° o 90° (donde Δ es difícil de medir y ruidosa).
#
# loss = sum(loss_per_start)  ← suma sobre las 1000 semillas
# Esto es equivalente a optimizar cada semilla independientemente pero con un
# solo paso de backward(), lo que es muy eficiente en la GPU.
#
# Adam optimizer: no usa solo el gradiente actual sino un promedio móvil del
# gradiente y del cuadrado del gradiente. Esto lo hace más robusto que el
# gradiente descendente simple:
#   m = β1·m + (1-β1)·∂loss/∂logit   (momento de primer orden)
#   v = β2·v + (1-β2)·(∂loss/∂logit)²  (momento de segundo orden)
#   logit_nuevo = logit - lr · m / (√v + ε)

# Visualizar cómo varía Is e Ic con el espesor (intuición de la pérdida):
lam_demo = np.linspace(450, 900, 100)
espesores = [5, 20, 50, 80, 120]  # nm

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
colores = plt.cm.viridis(np.linspace(0, 1, len(espesores)))

for i, d in enumerate(espesores):
    # Cauchy simplificado SiO2
    n_sio2 = 1.46 + 0.003 * 1e4 / lam_demo**2
    # Fase simple en la capa (aproximación para visualización)
    delta_fase = 2 * np.pi * n_sio2 * d / lam_demo * np.cos(np.radians(30))
    # Is e Ic aproximados (sin TMM completo, solo para ilustrar)
    Is_approx = 0.5 * np.sin(2 * delta_fase)
    Ic_approx = 0.5 * np.cos(2 * delta_fase)
    ax[0].plot(lam_demo, Is_approx, color=colores[i], label=f'd={d}nm')
    ax[1].plot(lam_demo, Ic_approx, color=colores[i], label=f'd={d}nm')

ax[0].set_title('Is ≈ sin(2ψ)sin(Δ) varía mucho con el espesor (ilustración)')
ax[1].set_title('Ic ≈ sin(2ψ)cos(Δ) varía mucho con el espesor')
for a in ax:
    a.set_xlabel('λ [nm]')
    a.legend(fontsize=9)
    a.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
# =============================================================================
# BLOQUE 10: El clamp de logits — cómo se evita la saturación del sigmoid
# =============================================================================
#
# Después de optimizer.step(), se hace:
#   d_opt.data.clamp_(-5, 5)
#   p_opt.data.clamp_(-5, 5)
#
# Problema que resuelve:
# Adam con lr=1.5 puede cambiar un logit en ~1.5 por paso.
# Después de 10 pasos, el logit puede estar en ±15.
# sigmoid(15) ≈ 0.9999999, y sigmoid'(15) ≈ 3×10⁻⁷ (prácticamente 0).
# Con gradiente ≈ 0, el optimizador no puede mover el parámetro → ATRAPADO.
#
# Solución: clampear en ±5:
#   sigmoid(±5) ≈ 0.993/0.007  → acceso al 98.6% del rango físico
#   sigmoid'(±5) ≈ 0.0066      → gradiente vivo, 22000x mayor que en ±15
#
# El clamp se hace sobre .data (no sobre el tensor directamente) para no
# crear un nodo en el grafo de autograd que interfiera con el backward().

logits = np.linspace(-10, 10, 1000)
grad_sig = (1/(1+np.exp(-logits))) * (1 - 1/(1+np.exp(-logits)))

plt.figure(figsize=(9, 4))
plt.semilogy(logits, grad_sig, 'b-', linewidth=2)
plt.axvline(-5, color='red', linestyle='--', label='clamp ±5')
plt.axvline(5, color='red', linestyle='--')
plt.fill_between(logits, grad_sig, where=np.abs(logits) <= 5,
                 alpha=0.2, color='green', label='zona segura (gradiente vivo)')
plt.xlabel('logit')
plt.ylabel("d(sigmoid)/d(logit)  [escala log]")
plt.title('Gradiente del sigmoid: por qué clampear en ±5 es importante')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
# =============================================================================
# BLOQUE 11: Extracción del mejor resultado (después del entrenamiento)
# =============================================================================
#
# Después de las num_epochs épocas, se hace una última pasada SIN gradiantes
# (with torch.no_grad():) para evaluar todas las semillas con los parámetros
# finales y elegir la mejor:
#
# >>> best_idx = loss_final.argmin()
#
# Esto da el índice (número de semilla) que logró el menor error final.
# Luego se extrae:
#
# best_thicknesses = d_final_fisico[best_idx]  ← espesores en nm, numpy array
# best_Is_curve    = Is_teo_final[best_idx]    ← curva Is teórica, numpy array
# best_Ic_curve    = Ic_teo_final[best_idx]    ← curva Ic teórica, numpy array
# best_params      = dict con A, B, C de cada capa paramétrica
#
# El diccionario best_params contiene:
# {
#   'SiO2_Daniel': {
#     'model': 'cauchy',
#     'A': 1.462,
#     'B': 0.0031,
#     'C': -0.0002
#   }
# }
#
# Para recuperar el índice de refracción ajustado podés hacer:
# from tmm_utils_Rodrigo import cauchy_fn
# n_ajustado = cauchy_fn(A, B, C)(lams_np)

#%%
# =============================================================================
# BLOQUE 12: Modelo de Bruggeman — física del medio efectivo
# =============================================================================
#
# Cuando tenés una capa que es una mezcla de dos materiales (ej: SiO2 con poros
# de aire), el índice efectivo no es el promedio aritmético. El modelo de
# Bruggeman (EMA = Effective Medium Approximation) resuelve:
#
#   (1-f) · (ε₁ - ε_eff)/(ε₁ + 2·ε_eff)  +  f · (ε₂ - ε_eff)/(ε₂ + 2·ε_eff) = 0
#
# donde ε = n² (permitividad), f = fracción de relleno del material 2 (aquí aire).
# Esta ecuación cuadrática tiene solución analítica:
#
#   ω = (1-f)·(ε₂ - 2ε₁) + f·(ε₁ - 2ε₂)
#   n_eff = √( (√(ω² + 8ε₁ε₂) - ω) / 4 )
#
# En el código, la función n_eff_torch implementa esto con tensores PyTorch
# para que los gradientes fluyan correctamente hacia los parámetros A,B,C de
# Cauchy del SiO2 base.
#
# EJEMPLO: ¿Cómo varía n_eff con la fracción de aire?

fracciones = np.linspace(0, 1, 100)
n_SiO2_val = 1.46   # índice del SiO2
n_aire_val = 1.00   # índice del aire

e1 = n_SiO2_val**2
e2 = n_aire_val**2
n_promedio = fracciones * n_aire_val + (1-fracciones) * n_SiO2_val  # lineal (incorrecto)

omega = (1 - fracciones) * (e2 - 2*e1) + fracciones * (e1 - 2*e2)
n_brugg = np.sqrt((np.sqrt(omega**2 + 8*e1*e2) - omega) / 4)

plt.figure(figsize=(8, 4))
plt.plot(fracciones, n_brugg, 'b-', linewidth=2, label='Bruggeman (correcto)')
plt.plot(fracciones, n_promedio, 'r--', linewidth=2, label='Promedio lineal (incorrecto)')
plt.xlabel('Fracción de aire (f)')
plt.ylabel('n efectivo a λ = 550nm')
plt.title('Modelo de Bruggeman: mezcla SiO₂-Aire')
plt.axvline(0.5, color='green', linestyle=':', label='f=0.5 (50% de cada uno)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print(f"n efectivo con 50% de aire (Bruggeman): {n_brugg[50]:.4f}")
print(f"n efectivo con 50% de aire (promedio):  {n_promedio[50]:.4f}")

#%%
# =============================================================================
# RESUMEN FINAL: Diagrama de flujo completo
# =============================================================================
#
# ENTRADA:
#   - Datos experimentales: Is_exp[828], Ic_exp[828]  (828 λ entre 400-900nm)
#   - Stack físico: ['air', 'SiO2_Si_brugge', 'SiO2_Daniel', 'Si']
#   - d_bounds: [(0.1, 5), (1, 80)] nm para las dos capas finitas
#   - layer_models: [None, 'bruggeman', 'cauchy', None]
#
# INICIALIZACIÓN (una sola vez):
#   - Genera 1000 logits de espesor aleatorios → d_opt [1000, 2]
#   - Genera 1000×3 logits de Cauchy aleatorios → p_opt [1000, 3]
#   - Adam optimizer apunta a d_opt y p_opt
#   - Precomputa inv_lam2 e inv_lam4 (constantes para todas las épocas)
#
# CADA ÉPOCA (×150):
#   d_opt [1000,2] → sigmoid → d_fisico [1000,2] → d_full_3d [1000,4,828]
#   p_opt [1000,3] → sigmoid → A,B,C → n_Cauchy [1000,828] → Bruggeman → n_list_3d [1000,4,828]
#   TMM batched: (n_list_3d, d_full_3d) → r_s [1000,828], r_p [1000,828]
#   r_s, r_p → ρ = conj(r_p/r_s) → ψ, Δ → Is_teo, Ic_teo [1000,828]
#   loss = mean((Is_teo - Is_exp)² + (Ic_teo - Ic_exp)²) → escalar
#   backward() → gradientes en d_opt y p_opt
#   Adam.step() → actualiza d_opt y p_opt
#   clamp(-5,5) → evita saturación del sigmoid
#
# EXTRACCIÓN (una vez al final):
#   Evaluar las 1000 semillas con los logits finales
#   best_idx = argmin(loss_final)  ← índice de la mejor semilla
#   best_thicknesses = d_fisico[best_idx]  → [d_brugge, d_SiO2] en nm
#   best_params = {'SiO2_Daniel': {'A': ..., 'B': ..., 'C': ...}}
#   best_Is, best_Ic = Is_teo[best_idx], Ic_teo[best_idx]
#
# SALIDA:
#   best_thicknesses, best_Is, best_Ic, best_params

print("=" * 60)
print("Guía completa de fit_ellipsometry_torch leída correctamente.")
print("Ejecutá cada celda (Ctrl+Enter) para ver los ejemplos.")
print("=" * 60)
