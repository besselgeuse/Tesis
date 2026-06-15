# TMM-R: Guía de Arquitectura del Sistema

> **Propósito:** Documento de referencia para el agente de IA. Describe la arquitectura completa
> del proyecto de simulación elipsométrica basado en el Método de la Matriz de Transferencia (TMM),
> separado por capas funcionales.

---

## 1. Visión General

Este proyecto simula y ajusta parámetros elipsométricos de películas delgadas antirreflectantes
usadas en celdas solares. Usa dos librerías centrales:

- **TMM (Transfer Matrix Method):** Calcula coeficientes de Fresnel de stacks multicapa.
- **PyTorch Autograd:** Encuentra espesores y parámetros de dispersión óptimos minimizando
  el error cuadrático medio (MSE) entre datos simulados y experimentales.

El sistema tiene dos modos de uso:
1. **API web** (`APIS/ellipsometry_API/`): Servicio FastAPI + MySQL con frontend, autenticación
   y persistencia. Pensado para usuarios finales.
2. **Scripts offline** (`tmm-Rodrigo/`): Scripts Python para análisis directo sin servidor.
   Pensado para investigación y desarrollo.

---

## 2. Arquitectura por Capas

```
┌─────────────────────────────────────────────────────┐
│  FRONTEND  (static/index.html)                      │
│  SPA con formularios para configurar stack y ver    │
│  resultados. Se comunica con la API via fetch().    │
├─────────────────────────────────────────────────────┤
│  CAPA API  (main.py)                                │
│  FastAPI + OAuth2/JWT. Endpoints REST.              │
│  Modelos Pydantic: SimParam, Stack, Layer, LayerModel│
├─────────────────────────────────────────────────────┤
│  CAPA DE PERSISTENCIA  (database.py, models.py)     │
│  SQLAlchemy + MySQL. Tablas: usuarios, stacks.      │
├─────────────────────────────────────────────────────┤
│  CAPA DE AJUSTE  (fit_elipsometrico.py)             │
│  Orquesta: carga datos → construye n_list → llama   │
│  al optimizador → guarda caché → devuelve resultado.│
├─────────────────────────────────────────────────────┤
│  CAPA NUMÉRICA  (tmm_utils_Rodrigo.py)              │
│  Modelos de dispersión, optimización PyTorch,       │
│  funciones batched, sigmoide para acotamiento.      │
├─────────────────────────────────────────────────────┤
│  CAPA FÍSICA  (tmm_core.py)                         │
│  TMM coherente/incoherente. Fresnel. Snell.         │
│  Versiones numpy y torch.                           │
├─────────────────────────────────────────────────────┤
│  CAPA DE MATERIALES  (Materials_config.py,          │
│  catalogo_de_materiales.txt, indices/)              │
│  Archivos .nk/.nkv con datos ópticos tabulados.     │
├─────────────────────────────────────────────────────┤
│  INFRAESTRUCTURA  (docker-compose.yml, dockerfile,  │
│  requirements.txt, .env)                            │
└─────────────────────────────────────────────────────┘
```

---

## 3. Descripción Detallada de Cada Capa

### 3.1 Capa Física — `tmm_core.py`

**Origen:** Fork del paquete `tmm` de Steven Byrnes, vectorizado por Simón Saint André.

**Funciones clave:**
| Función | Descripción |
|---------|-------------|
| `coh_tmm(pol, n_list, d_list, th_0, lam_vac)` | TMM coherente (numpy). Retorna `R`, `T`, `r`, `t` |
| `inc_tmm(pol, n_list, d_list, c_list, th_0, lam_vac)` | TMM incoherente (numpy). Para capas gruesas |
| `coh_tmm_torch(pol, n_list, d_list, th_0, lam_vac)` | TMM coherente diferenciable (PyTorch) |
| `inc_tmm_torch(pol, n_list, d_list, c_list, th_0, lam_vac)` | TMM incoherente diferenciable (PyTorch) |
| `list_snell_torch(n_list, th_0)` | Ley de Snell vectorizada sobre todas las capas |

**Convenciones:**
- Ángulos en **radianes** (internamente).
- Longitudes de onda en **nanómetros** (nm).
- Índices de refracción complejos: `n + ik` (donde k es el coeficiente de extinción).
- `d_list[0]` y `d_list[-1]` siempre son `inf` (superestrato y sustrato semi-infinitos).
- Polarización: `'s'` o `'p'`.

---

### 3.2 Capa Numérica — `tmm_utils_Rodrigo.py`

**Es el archivo más grande y central del proyecto (~1225 líneas).** Contiene:

#### Modelos de dispersión (funciones fábrica)

| Función | Fórmula | Parámetros |
|---------|---------|------------|
| `cauchy_fn(A, B, C)` | `n(λ) = A + B×10⁴/λ² + C×10⁹/λ⁴` | A, B, C (reales) |
| `constant_fn(val)` | `n(λ) = val` (constante) | val |
| `brugg_fn(n_a, n_b, f_b)` | Bruggeman EMA | f_b: fracción de especie b |
| `n_eff(n1, n2, f)` | `n_eff = √((√(ω²+8ε₁ε₂) − ω) / 4)` | Implementación numérica |
| `n_eff_torch(n1, n2, f)` | Igual pero diferenciable (PyTorch) | Para Autograd |

#### Funciones de carga de datos ópticos

| Función | Formato de entrada |
|---------|-------------------|
| `load_interp(filename, ...)` | Archivos `.nk`, `.nkv` (3 columnas: λ, n, k) |
| `load_interp_in3(filename, ...)` | Archivos `.in3` (formato especial con secciones separadas de n y k) |
| `load_fn(filename, ...)` | Archivos genéricos de 2 columnas |

**Unidades de entrada configurables:** `'nm'` (default) o `'um'` (se multiplica por 1000).

#### Funciones de optimización

| Función | Propósito |
|---------|-----------|
| `calculate_RT_torch(...)` | Optimiza espesores para minimizar reflectancia (solar cells) |
| `fit_ellipsometry_torch(...)` | Optimiza espesores + parámetros de dispersión para elipsometría |
| `autorange_fit_ellipsometry_torch(...)` | Optimización en dos fases: coarse-to-fine |
| `coh_tmm_torch_batched(...)` | TMM batched (semillas × longitudes de onda) para paralelismo |

#### Mecanismo de acotamiento (sigmoide + logit)

Los parámetros físicos (espesores, Cauchy A/B/C) están acotados a rangos válidos usando
la transformación sigmoide:

```
valor_físico = valor_min + (valor_max - valor_min) × σ(logit)
```

Donde `logit` es la variable libre optimizada por Adam (`d_opt` para espesores, `p_opt` para
parámetros de dispersión). El mapeo inverso (inicialización) es:

```
logit = log(p / (1 - p)),  donde p = (valor - valor_min) / (valor_max - valor_min)
```

**Anti-saturación:** Los logits se clampean a [-5, 5] para evitar que `σ(logit)` sature
en 0 o 1, lo que anularía los gradientes.

#### Variables optimizables en el bucle de entrenamiento

| Variable | Shape | `requires_grad` | Qué representa |
|----------|-------|------------------|----------------|
| `d_opt` | `[num_starts, num_opt_layers]` | ✅ True | Logits de espesores |
| `p_opt` | `[num_starts, total_disp_params]` | ✅ True | Logits de parámetros Cauchy |

**Importante:** `d_opt` y `p_opt` son variables independientes. `∂p_opt/∂d_opt = 0`.

#### Flujo de datos en cada época

```
d_opt → sigmoid → d_fisico → [inf, d1, d2, ..., inf] → d_full_3d
p_opt → sigmoid → A, B, C → Cauchy → n_complex → n_list_3d
                                                          ↓
              coh_tmm_torch_batched(n_list_3d, d_full_3d) → r_s, r_p
                                                          ↓
              ρ = conj(r_p / r_s) → ψ, Δ → Is_teo, Ic_teo
                                                          ↓
              MSE(Is_teo - Is_exp) + MSE(Ic_teo - Ic_exp) → loss
                                                          ↓
              loss.backward() → gradientes → Adam.step()
```

---

### 3.3 Capa de Ajuste — `fit_elipsometrico.py`

**Función principal:** `ajuste_elipsometrico()`

Orquesta todo el proceso de ajuste:

1. Lee el archivo de datos experimentales (`.txt` con columnas: λ, ψ, Δ).
2. Convierte ψ, Δ a parámetros Is, Ic: `Is = sin(2ψ)·sin(Δ)`, `Ic = sin(2ψ)·cos(Δ)`.
3. Filtra longitudes de onda > 830 nm (ruido).
4. Carga los materiales estáticos del diccionario.
5. Construye `n_list_torch` (tensor complejo de índices de refracción).
6. Llama a `fit_ellipsometry_torch()` o `autorange_fit_ellipsometry_torch()`.
7. Opcionalmente guarda/carga caché con pickle.

**Parámetro clave:** `autorange=False` → Si True, usa la optimización coarse-to-fine.

---

### 3.4 Capa API — `main.py`

**Framework:** FastAPI con Uvicorn.

**Autenticación:** OAuth2 con tokens JWT firmados con `SECRET_KEY` (variable de entorno).

**Modelos Pydantic:**

| Modelo | Campos principales |
|--------|-------------------|
| `LayerModel` | `model_type` (cauchy/cauchy_absorbent/bruggeman), `bounds` |
| `Layer` | `name`, `model` (LayerModel), `d_min`, `d_max` |
| `Stack` | `name`, `layers` (List[Layer]) |
| `SimParam` | `num_starts`, `num_epochs`, `lr`, `use_cuda`, `th_0`, `autorange`, `stack` |

**Endpoints:**

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Redirect a `/static/index.html` |
| `POST` | `/register` | Registrar usuario nuevo |
| `POST` | `/token` | Login (obtener JWT) |
| `POST` | `/simular_stack` | Ejecutar simulación elipsométrica |
| `GET` | `/mostrar_resultados` | Listar stacks del usuario |
| `GET` | `/mostrar_resultados/{id}` | Obtener stack por ID |
| `GET` | `/descargar_resultados/{id}` | Descargar resultados en TXT |

---

### 3.5 Capa de Persistencia — `database.py`, `models.py`

**Base de datos:** MySQL 8.0 (via Docker o local).

**Tablas:**

| Tabla | Columnas clave |
|-------|---------------|
| `usuarios` | `id`, `email`, `hashed_password` (bcrypt) |
| `stacks` | `id`, `name`, `layers` (JSON), `best_Is`, `best_Ic`, `wl_exp`, `Is_exp`, `Ic_exp`, `user_id` (FK) |

**Seguridad:**
- `SECRET_KEY` se inyecta via `.env` (excluido de git).
- Contraseñas hasheadas con bcrypt.
- Cada usuario solo ve sus propios stacks (filtrado por `user_id`).

---

### 3.6 Capa de Materiales

**Archivos de datos ópticos** (`indices/`, `indices/nkdata/optical/`):
- Formato `.nk`: 3 columnas (λ, n, k), skiprows=1, unidades nm o μm.
- Formato `.nkv`: Similar a .nk.
- Formato `.in3`: Secciones separadas para n y k con headers de conteo de puntos.

**Catálogo** (`catalogo_de_materiales.txt`):
Diccionario Python evaluado con `eval()` que mapea nombres a lambdas de carga.

**Materiales disponibles:**
air, GaAs, InGaP, MgF2, vidrio, SiO2, Si, Rutilo (TiO2 bulk), T1_densa, T1_porosa,
ZnSe, AlAs, Al2O3, HfO2, SiNx, Aluminum.

---

### 3.7 Frontend — `static/index.html`

Single Page Application (SPA) HTML+JS puro (~73KB). Pestañas:
- **Stack óptico:** Configurar capas, modelos de dispersión y rangos de espesores.
- **Parámetros:** Configurar ángulo de incidencia, lr, num_starts, num_epochs, CUDA.
- **Historial:** Ver y descargar resultados previos.

---

### 3.8 Infraestructura

| Archivo | Propósito |
|---------|-----------|
| `docker-compose.yml` | MySQL 8.0 + API FastAPI. Healthcheck en MySQL. |
| `dockerfile` | Python 3.11, instala requirements, uvicorn en puerto 8000 |
| `requirements.txt` | numpy, scipy, torch, fastapi, sqlalchemy, pymysql, etc. |
| `.env` | `SECRET_KEY` (excluido de git) |
| `.gitignore` | Excluye venv, __pycache__, .env |

---

## 4. Problemas Conocidos y Trampas

### NaN en la loss
**Causa más frecuente:** Combinaciones de parámetros de Cauchy que producen índices de
refracción negativos o extremadamente pequeños, causando divisiones por cero en TMM.
**Mitigación:** Acotar rangos de búsqueda, usar `autorange`, clampear logits a [-5, 5].

### Ruido experimental > 830 nm
Los datos del elipsómetro por encima de 830 nm son ruidosos.
El filtrado se aplica en `fit_elipsometrico.py` (líneas 130-170).

### Learning rate
`lr=1.5` es agresivo. Funciona bien con rangos acotados pero puede diverger con rangos amplios.

### Bruggeman f_air
En `tmm-Rodrigo/tmm_utils_Rodrigo.py`, `f_air` está hardcodeada a 0.5 y no se optimiza.
En `APIS/ellipsometry_API/tmm_utils_Rodrigo.py`, `f_air` es optimizable si se pasa `f_bounds`.

---

## 5. Convenciones del Proyecto

- **Unidades de longitud de onda:** Siempre nm internamente.
- **Ángulos:** Grados en la interfaz de usuario, radianes internamente.
- **Índices de refracción:** Complejos `n + ik`. k ≥ 0 siempre.
- **Orden del stack:** `[superestrato (air), capa1, capa2, ..., sustrato (Si)]`.
- **d_bounds:** Solo para capas finitas (excluye superestrato y sustrato).
- **Variables de optimización:** En espacio logit (no acotado), mapeadas a físico via sigmoide.
