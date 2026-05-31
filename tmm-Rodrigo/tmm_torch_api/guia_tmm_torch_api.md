# 📘 Guía Didáctica: `tmm_torch_api`

> **Objetivo**: Entender en profundidad cada pieza del proyecto para que puedas replicarlo desde cero.

---

## 🗂️ Mapa del Proyecto

Antes de entrar al código, entendamos cómo se relacionan los archivos:

```
tmm-Rodrigo/
├── tmm_utils_Rodrigo.py       ← Lógica TMM original
├── tmm_core.py                ← Motor de cálculo óptico
│
└── tmm_torch_api/             ← Tu API (todo vive aquí)
    ├── main.py                ← Corazón de la API (FastAPI)
    ├── database.py            ← Conexión a MySQL
    ├── models.py              ← Tablas de la base de datos
    ├── tmm_torch_simulator.py ← Simulador con PyTorch
    ├── dockerfile             ← Receta para construir el contenedor
    ├── docker-compose.yml     ← Orquestador (API + MySQL)
    └── static/                ← Interfaz web HTML/JS
```

**Analogía**: Imagina que tu proyecto es un restaurante:
- `main.py` → El menú y el mesero (recibe pedidos, los procesa, devuelve resultados)
- `database.py` → La dirección de la despensa
- `models.py` → Los moldes/formularios para guardar ingredientes
- `tmm_torch_simulator.py` → El chef (ejecuta el cálculo real)
- `dockerfile` → La receta para construir la cocina
- `docker-compose.yml` → El plano del restaurante completo

---

# PARTE 1: `main.py` — El Corazón de la API

## 🧱 Bloque 1: Configuración de Rutas del Sistema (líneas 1–16)

```python
import sys
import os

API_DIR = os.path.dirname(os.path.abspath(__file__))
TMM_DIR = os.path.abspath(os.path.join(API_DIR, '..'))

if API_DIR not in sys.path:
    sys.path.append(API_DIR)
if TMM_DIR not in sys.path:
    sys.path.append(TMM_DIR)
```

### ¿Qué hace?
Python necesita saber **dónde buscar** los archivos cuando haces `import`. Por defecto solo busca en ciertas carpetas estándar.

| Variable | Valor en tu PC | Valor en Docker |
|---|---|---|
| `__file__` | `d:\...\tmm_torch_api\main.py` | `/app/main.py` |
| `API_DIR` | `d:\...\tmm_torch_api` | `/app` |
| `TMM_DIR` | `d:\...\tmm-Rodrigo` | `/app/..` → (ver nota) |

- `os.path.abspath(__file__)` → Convierte a ruta absoluta (sin `..` ni `.`)
- `os.path.dirname(...)` → Obtiene la carpeta que contiene el archivo
- `sys.path.append(...)` → Le dice a Python: "también busca módulos aquí"

> **¿Por qué el `if ... not in`?** Para no agregar la misma ruta dos veces si el módulo se reimporta. Es una buena práctica defensiva.

---

## 🧱 Bloque 2: Matplotlib en Modo Servidor (líneas 19–22)

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

### ¿Qué hace?
Matplotlib, por defecto, intenta **abrir una ventana gráfica** (GUI) para mostrar plots. En un servidor (Docker, Linux headless) **no hay pantalla**, por lo que fallará con un error.

- `matplotlib.use('Agg')` → Cambia el "backend" a `Agg` (Anti-Grain Geometry), que **solo renderiza en memoria** y permite guardar imágenes a archivo sin necesitar pantalla.
- **⚠️ Importante**: Esta línea debe ir **antes** de `import matplotlib.pyplot as plt`, de lo contrario ya se habrá inicializado el backend por defecto.

---

## 🧱 Bloque 3: Imports del Proyecto (líneas 25–33)

```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Annotated, List
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import tmm_torch_simulator as st
```

### Desglose línea por línea:

| Import | Rol |
|---|---|
| `FastAPI` | La clase principal que crea tu aplicación web |
| `HTTPException` | Para lanzar errores HTTP (ej. 404, 500) con un mensaje |
| `Depends` | Sistema de inyección de dependencias de FastAPI |
| `status` | Constantes de códigos HTTP (`status.HTTP_201_CREATED = 201`) |
| `StaticFiles` | Sirve archivos estáticos (HTML, CSS, JS) desde una carpeta |
| `RedirectResponse` | Respuesta que redirige al navegador a otra URL |
| `BaseModel` | Clase base de Pydantic para crear esquemas de validación |
| `Annotated`, `List` | Tipos de Python para anotaciones tipadas |
| `models` | Tu archivo `models.py` con las clases de la BD |
| `engine, SessionLocal` | Motor y fábrica de sesiones de SQLAlchemy |
| `Session` | Tipo de sesión de base de datos |
| `tmm_torch_simulator as st` | Tu simulador TMM con alias corto `st` |

---

## 🧱 Bloque 4: Crear la App y Montar Archivos Estáticos (líneas 36–40)

```python
app = FastAPI()

STATIC_DIR = os.path.join(API_DIR, 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

### ¿Qué hace?
- `FastAPI()` → Crea la instancia de tu aplicación. Es el objeto central al que le añades todos los endpoints.
- `app.mount("/static", ...)` → Le dice a FastAPI: "todo lo que llegue a la URL `/static/...` debe ser respondido con archivos de la carpeta `static/`".

**Ejemplo**: Si alguien visita `http://localhost:8000/static/index.html`, FastAPI buscará y devolverá `tmm_torch_api/static/index.html` directamente.

---

## 🧱 Bloque 5: Evento de Startup (líneas 43–46)

```python
@app.on_event("startup")
def startup_event():
    models.Base.metadata.create_all(bind=engine)
    print("[INFO] Database tables verified/created.")
```

### ¿Qué hace?
`@app.on_event("startup")` registra una función para que se ejecute **una sola vez cuando la API arranca**, antes de aceptar cualquier solicitud.

- `models.Base.metadata.create_all(bind=engine)` → Revisa todos los modelos (clases) que heredan de `Base` y, si sus tablas no existen en la base de datos, **las crea automáticamente**.

> **¿Por qué no lo hacemos al inicio del módulo?** Porque el motor de base de datos podría no estar listo todavía (especialmente en Docker donde MySQL tarda en arrancar).

---

## 🧱 Bloque 6: Schemas Pydantic — Validación de Datos (líneas 52–75)

Pydantic es la librería que usa FastAPI para **validar automáticamente** los datos que entran y salen de la API.

### `CapaSchema` (líneas 52–55)
```python
class CapaSchema(BaseModel):
    material: str
    coherencia: str
```
Representa **una sola capa** del stack óptico. Cuando alguien envía un JSON como:
```json
{"material": "SiO2", "coherencia": "c"}
```
Pydantic verifica que `material` y `coherencia` sean strings. Si falta alguno o el tipo es incorrecto, **FastAPI devuelve un error 422 automáticamente** sin que tengas que escribir ningún código de validación.

---

### `ThickSchema` (líneas 57–61)
```python
class ThickSchema(BaseModel):
    indice_capa: int
    min: float
    max: float
```
Define el **rango de optimización** de una capa (cuánto puede variar su espesor en nm). `indice_capa` es un `int` y `min`/`max` son `float`.

---

### `StackCreateSchema` (líneas 64–76) — El Schema Principal
```python
class StackCreateSchema(BaseModel):
    nombre: str
    capas: List[CapaSchema]
    thicks: List[ThickSchema]
    sustrato: str
    pol: str
    num_starts: int = 500
    num_epochs: int = 200
    lr: float = 1.5
    use_cuda: bool = True
    th_0: float = 0.0
```

Este es el schema que valida **todo el cuerpo del POST request**. Puntos clave:

| Campo | Tipo | Significado |
|---|---|---|
| `nombre` | `str` | Nombre identificador del experimento |
| `capas` | `List[CapaSchema]` | Lista de capas (cada una validada por `CapaSchema`) |
| `thicks` | `List[ThickSchema]` | Rangos de optimización por capa |
| `sustrato` | `str` | Material del sustrato (ej. "GaAs") |
| `pol` | `str` | Polarización de luz ("s" o "p") |
| `num_starts` | `int = 500` | Parámetro del optimizador (valor por defecto: 500) |
| `num_epochs` | `int = 200` | Épocas de entrenamiento (default: 200) |
| `lr` | `float = 1.5` | Learning rate del optimizador (default: 1.5) |
| `use_cuda` | `bool = True` | Usar GPU si está disponible (default: True) |
| `th_0` | `float = 0.0` | Ángulo de incidencia en grados (default: 0°) |

Los campos con `= valor` son **opcionales** en el JSON: si no los envías, usan el default.

---

## 🧱 Bloque 7: Dependency de Base de Datos (líneas 79–86)

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
```

### ¿Qué hace?
Este es el patrón **"dependency injection"** de FastAPI con el sistema de contexto de Python (`yield`).

**Analogía**: Imagina que `get_db` es como pedir prestado un libro de la biblioteca:
1. `db = SessionLocal()` → Abres la sesión (pides el libro)
2. `yield db` → Le entregas el libro a quien lo necesita (el endpoint)
3. `finally: db.close()` → El libro **siempre** se devuelve al final, haya error o no

**¿Qué es `yield` vs `return`?**
- `return` devuelve el valor y la función termina.
- `yield` convierte la función en un **generador**. FastAPI ejecuta el código hasta `yield`, entrega el valor al endpoint, y cuando el endpoint termina (con éxito o error), continúa ejecutando lo que está después de `yield` (en este caso el `finally`).

**¿Qué es `Annotated[Session, Depends(get_db)]`?**
- `Depends(get_db)` → Le dice a FastAPI: "para proveer este parámetro, ejecuta la función `get_db`"
- `Annotated[Session, ...]` → Combina el tipo (`Session`) con la anotación de dependencia

En cualquier endpoint donde pongas `db: db_dependency`, FastAPI automáticamente ejecutará `get_db()`, te dará la sesión activa, y la cerrará al terminar.

---

## 🧱 Bloque 8: Endpoint `GET /` — Redirección (líneas 91–93)

```python
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")
```

El endpoint más simple: cuando alguien visita la raíz (`http://localhost:8000/`), redirige automáticamente a la interfaz web en `/static/index.html`.

- `@app.get("/")` → El decorador que registra la función como handler del método GET en la ruta `/`
- `async def` → Función asíncrona (permite que FastAPI maneje múltiples peticiones concurrentemente)
- `RedirectResponse` → Responde con un código HTTP 307/308 que le dice al navegador: "ve a esta otra URL"

---

## 🧱 Bloque 9: Endpoint `POST /registrar_stack` — El Más Importante (líneas 95–164)

```python
@app.post("/registrar_stack", status_code=status.HTTP_201_CREATED)
async def registrar_stack(registro: StackCreateSchema, db: db_dependency):
```

Este endpoint es el corazón del sistema. Recibe un stack, lo simula y guarda los resultados.

### Firma de la función:
- `registro: StackCreateSchema` → FastAPI lee el body JSON y lo valida/convierte automáticamente
- `db: db_dependency` → FastAPI inyecta la sesión de base de datos

### Paso a paso interno:

#### Paso 1 — Preparar los thicks (líneas 103)
```python
thicks = [{"min": t.min, "max": t.max} for t in registro.thicks]
```
Transforma la lista de objetos `ThickSchema` en una lista de diccionarios simples que entiende el simulador. Es una **list comprehension**: crea una lista nueva iterando sobre `registro.thicks` y convirtiendo cada elemento.

#### Paso 2 — Ejecutar la simulación (líneas 106–117)
```python
best_thickness, best_reflectancia, best_jsc_am15, best_jsc_am0, lams = st.ejecutar_simulacion(
    stack=[capa.material for capa in registro.capas],
    thicks=thicks,
    c_list=[capa.coherencia for capa in registro.capas],
    ...
)
```
Llama al simulador TMM con PyTorch. El simulador devuelve **5 valores a la vez** (unpacking múltiple de Python):
- `best_thickness` → Array con los espesores óptimos encontrados
- `best_reflectancia` → Curva de reflectancia espectral óptima
- `best_jsc_am15` → Corriente de cortocircuito máxima bajo AM1.5
- `best_jsc_am0` → Corriente bajo AM0
- `lams` → Vector de longitudes de onda usadas

#### Paso 3 — Serializar valores (líneas 120–121)
```python
js_am0_val = float(best_jsc_am0)
js_am15_val = float(best_jsc_am15)
```
Convierte tensores de PyTorch a floats de Python nativos (necesario para guardar en BD y JSON).

#### Paso 4 — Construir diccionario de espesores (líneas 124–127)
```python
espesores_optimos = {}
for idx, val in enumerate(best_thickness):
    material_name = registro.capas[idx].material
    espesores_optimos[f"{idx}_{material_name}"] = float(val)
```
Crea un diccionario legible con clave `"0_SiO2": 45.3`, `"1_TiO2": 78.1"`, etc. `enumerate()` devuelve pares `(índice, valor)`.

#### Paso 5 — Guardar el Stack en BD (líneas 130–140)
```python
full_capas = ["air"] + [c.material for c in registro.capas] + [registro.sustrato]
full_coherencia = ["i"] + [c.coherencia for c in registro.capas] + ["i"]

stack_db = models.Stack(
    nombre=registro.nombre,
    capas_json=json.dumps(full_capas),
    ...
)
db.add(stack_db)
db.flush()  # ← obtiene el idstack SIN hacer commit todavía
```
- Se añade `"air"` al inicio y el sustrato al final (el simulador TMM siempre los necesita)
- `json.dumps(...)` convierte la lista Python a string JSON para guardar en la columna `Text`
- `db.flush()` escribe el objeto a la BD **sin confirmar la transacción**, para poder obtener el `idstack` generado automáticamente y usarlo en el siguiente paso

#### Paso 6 — Guardar Resultados (líneas 143–152)
```python
resultado_db = models.Results(
    ref_optima=json.dumps(best_reflectancia.tolist()),
    lams=json.dumps(lams.tolist()),
    ...
    idstack=stack_db.idstack
)
db.add(resultado_db)
db.commit()  # ← AHORA sí confirma ambos inserts
```
- `.tolist()` convierte arrays NumPy/PyTorch a listas Python (necesario para `json.dumps`)
- `db.commit()` confirma **ambos** inserts (Stack y Results) en una sola transacción

#### Manejo de Errores (líneas 162–164)
```python
except Exception as e:
    db.rollback()
    raise HTTPException(status_code=500, detail=f"Error en la simulación: {str(e)}")
```
Si **cualquier** cosa falla:
- `db.rollback()` → Cancela todos los cambios en la BD (no queda nada a medias)
- `raise HTTPException(...)` → Devuelve un error 500 al cliente con el mensaje del error

---

## 🧱 Bloque 10: `GET /mostrar_resultados` — Listar Todo (líneas 167–188)

```python
@app.get("/mostrar_resultados", status_code=status.HTTP_200_OK)
async def obtener_resultados(db: db_dependency):
    stacks = db.query(models.Stack).all()
    resultado = []
    for s in stacks:
        res = db.query(models.Results).filter(models.Results.idstack == s.idstack).first()
        resultado.append({...})
    return resultado
```

- `db.query(models.Stack).all()` → `SELECT * FROM stack` (todos los registros)
- `.filter(models.Results.idstack == s.idstack)` → `WHERE idstack = ?`
- `.first()` → Devuelve solo el primer resultado (equivalente a `LIMIT 1`)
- `json.loads(s.capas_json)` → Convierte el string JSON guardado de vuelta a lista Python
- `if s.capas_json else None` → Evita error si el campo está vacío (operador ternario)

---

## 🧱 Bloque 11: `GET /mostrar_resultados/{idstack}` — Un Solo Resultado (líneas 191–214)

```python
@app.get("/mostrar_resultados/{idstack}", status_code=status.HTTP_200_OK)
async def obtener_resultado_por_id(idstack: int, db: db_dependency):
    s = db.query(models.Stack).filter(models.Stack.idstack == idstack).first()
    if s is None:
        raise HTTPException(status_code=404, detail="Stack no encontrado")
```

- `{idstack}` en la ruta es un **path parameter**: FastAPI lo extrae de la URL y lo convierte a `int` automáticamente
- Si no se encuentra, se lanza un `404 Not Found` (convención REST: "el recurso no existe")
- Este endpoint incluye además `ref_optima` y `lams` (los arrays completos), que el endpoint de lista omite para no sobrecargar la respuesta

---

# PARTE 2: Archivos de Soporte

## `database.py` — La Conexión a MySQL

```python
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:Dinosaurio1724@localhost:3306/tmmm_torch_api")
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

### Anatomía de la URL de conexión:
```
mysql+pymysql://root:Dinosaurio1724@localhost:3306/tmmm_torch_api
│      │        │    │              │         │    │
│      │        │    │              │         │    └── Nombre de la base de datos
│      │        │    │              │         └────── Puerto MySQL (default: 3306)
│      │        │    │              └──────────────── Host (en Docker: "db")
│      │        │    └─────────────────────────────── Contraseña
│      │        └──────────────────────────────────── Usuario
│      └───────────────────────────────────────────── Driver Python
└──────────────────────────────────────────────────── Dialect SQLAlchemy
```

- `os.getenv("DATABASE_URL", "mysql+...")` → Lee la variable de entorno `DATABASE_URL`. Si no existe (ej. cuando corres local), usa el string hardcodeado. En Docker, la variable de entorno definida en `docker-compose.yml` sobreescribe el default.
- `echo=True` → SQLAlchemy imprime todas las queries SQL en consola (útil para debug)
- `autocommit=False` → Las transacciones NO se confirman automáticamente (debes llamar `db.commit()` explícitamente)
- `autoflush=False` → No envía cambios pendientes a la BD antes de cada query
- `declarative_base()` → Crea la clase base de la que heredan todos tus modelos

---

## `models.py` — Las Tablas de la Base de Datos

```python
class Stack(Base):
    __tablename__ = "stack"
    idstack = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    capas_json = Column(Text)
    thicks_json = Column(Text)
    coherencia_json = Column(Text)
    fecha = Column(DateTime, default=datetime.now)

    resultados = relationship("Results", back_populates="stack")
```

Cada clase que hereda de `Base` representa **una tabla**. Cada `Column` es una columna.

| Columna | Tipo SQL | Notas |
|---|---|---|
| `idstack` | `INTEGER PRIMARY KEY` | Auto-incremental, indexado |
| `nombre` | `VARCHAR(100)` | Máximo 100 caracteres |
| `capas_json` | `TEXT` | String largo (JSON serializado) |
| `fecha` | `DATETIME` | `default=datetime.now` se evalúa al insertar |

**`relationship("Results", back_populates="stack")`**: Define una relación ORM que te permite acceder a `stack.resultados` y obtener todos los `Results` asociados, sin escribir SQL de JOIN manualmente.

---

# PARTE 3: `dockerfile` — La Receta del Contenedor

```dockerfile
FROM python:3.12.2-slim
```
**Punto de partida**: Usa una imagen oficial de Python 3.12.2 en versión `slim` (sin herramientas innecesarias → imagen más pequeña).

---

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```
**Instalar dependencias del sistema operativo**:
- `libgomp1` → La librería OpenMP de GNU, necesaria para que PyTorch pueda usar **paralelismo de CPU** (multi-threading). Sin esto, PyTorch falla al importarse en algunos sistemas.
- `--no-install-recommends` → No instala paquetes recomendados opcionales (ahorra espacio)
- `rm -rf /var/lib/apt/lists/*` → Limpia la caché de apt para reducir el tamaño de la imagen

---

```dockerfile
WORKDIR /app
```
Define `/app` como el directorio de trabajo. Todos los comandos siguientes se ejecutan desde `/app`.

---

```dockerfile
COPY requirements.txt .
```
**Copia SOLO el `requirements.txt` primero** (antes que el resto del código). ¿Por qué? Por el sistema de **capas de Docker**:
- Docker cachea cada instrucción. Si copias el código completo y luego instalas dependencias, cualquier cambio en el código invalida el cache de pip (muy lento).
- Copiando solo `requirements.txt` primero, si no cambias dependencias, Docker reutiliza el cache de pip (muy rápido).

---

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```
**Instala PyTorch CPU por separado**: La versión CPU es mucho más pequeña que la versión con CUDA. Usamos el índice de paquetes oficial de PyTorch para la versión específica sin GPU.

---

```dockerfile
RUN grep -v "torch" requirements.txt > requirements_docker.txt && \
    pip install --no-cache-dir -r requirements_docker.txt
```
**Filtrar torch del requirements.txt**:
- `grep -v "torch"` → Muestra todas las líneas del archivo que **NO** contienen "torch"
- `> requirements_docker.txt` → Guarda esas líneas en un archivo nuevo
- Instala las dependencias restantes (FastAPI, SQLAlchemy, etc.) sin reinstalar torch

---

```dockerfile
COPY . .
```
**Copia todo el código** al contenedor (ahora que las dependencias ya están instaladas).

---

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
**Comando por defecto al arrancar el contenedor**:
- `uvicorn` → Servidor ASGI para FastAPI (ASGI = Asynchronous Server Gateway Interface)
- `main:app` → "En el módulo `main.py`, usa el objeto llamado `app`"
- `--host 0.0.0.0` → Acepta conexiones de **cualquier IP** (no solo localhost). Sin esto, el contenedor sería inaccesible desde fuera.
- `--port 8000` → Puerto donde escucha

---

# PARTE 4: `docker-compose.yml` — El Orquestador

Docker Compose permite definir y levantar **múltiples contenedores** que trabajan juntos.

```yaml
version: "3.8"
services:
  db:
    ...
  api-tmm:
    ...
volumes:
  mysql_data:
```

**Estructura**: `services` define cada contenedor. `volumes` define almacenamiento persistente.

---

## Servicio `db` — MySQL

```yaml
db:
  image: mysql:8.0
  container_name: db-tmm
  restart: always
  environment:
    MYSQL_ROOT_PASSWORD: Dinosaurio1724
    MYSQL_DATABASE: tmmm_torch_api
  ports:
    - "3306:3306"
  volumes:
    - mysql_data:/var/lib/mysql
  healthcheck:
    test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-uroot", "-pDinosaurio1724"]
    interval: 5s
    timeout: 5s
    retries: 10
    start_period: 30s
```

| Clave | Significado |
|---|---|
| `image: mysql:8.0` | Usa la imagen oficial de MySQL 8.0 (no construye desde Dockerfile) |
| `container_name: db-tmm` | Nombre del contenedor (visible en `docker ps`) |
| `restart: always` | Se reinicia automáticamente si falla o si reinicias Docker |
| `MYSQL_ROOT_PASSWORD` | Variable de entorno que MySQL usa para crear el usuario root |
| `MYSQL_DATABASE` | MySQL crea esta base de datos automáticamente al arrancar |
| `ports: "3306:3306"` | `HOST:CONTENEDOR` → Expone el puerto 3306 del contenedor al host |
| `volumes: mysql_data:/var/lib/mysql` | Persiste los datos de MySQL en un volumen (sobreviven si matas el contenedor) |

### `healthcheck` — La Clave para el Timing
```yaml
healthcheck:
  test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-uroot", "-pDinosaurio1724"]
  interval: 5s
  timeout: 5s
  retries: 10
  start_period: 30s
```
MySQL tarda ~15-30 segundos en arrancar. El healthcheck le pregunta a MySQL "¿estás listo?" cada 5 segundos, hasta 10 veces, esperando 30 segundos antes de empezar a preguntar.

- `test` → Comando que se ejecuta para verificar la salud
- `interval` → Cada cuánto se chequea
- `retries` → Cuántos fallos consecutivos para marcar como "unhealthy"
- `start_period` → Tiempo de gracia al inicio (no cuenta como fallo)

---

## Servicio `api-tmm` — Tu API

```yaml
api-tmm:
  build: .
  container_name: api-tmm
  restart: on-failure
  ports:
    - "8000:8000"
  volumes:
    - .:/app
  working_dir: /app
  environment:
    - DATABASE_URL=mysql+pymysql://root:Dinosaurio1724@db:3306/tmmm_torch_api
  depends_on:
    db:
      condition: service_healthy
  command: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

| Clave | Significado |
|---|---|
| `build: .` | Construye la imagen usando el `dockerfile` en el directorio actual (`.`) |
| `restart: on-failure` | Se reinicia solo si termina con error (no si lo detienes manualmente) |
| `volumes: .:/app` | **Monta** tu carpeta local en `/app` del contenedor → editas código y el contenedor lo ve en tiempo real |
| `environment: DATABASE_URL=...` | Inyecta variable de entorno. Nota: el host es `db` (nombre del servicio), no `localhost` |
| `depends_on: db: condition: service_healthy` | **No arranca hasta que `db` pase el healthcheck**. Esto evita el error "MySQL no está listo" |
| `command: uvicorn ... --reload` | Sobreescribe el `CMD` del Dockerfile. `--reload` recarga automáticamente cuando cambias código |

### 🔑 La Magia de la Red Interna de Docker
Cuando usas Docker Compose, todos los servicios se conectan en una **red virtual privada**. Los servicios se pueden comunicar entre sí usando el **nombre del servicio como hostname**. Por eso en `DATABASE_URL` ponemos `@db:3306` (el servicio se llama `db`), no `@localhost:3306`.

---

## Volumen `mysql_data`

```yaml
volumes:
  mysql_data:
```

Un **named volume**: Docker gestiona dónde se almacena físicamente en tu sistema. Los datos de MySQL sobreviven a `docker compose down`. Para borrarlos: `docker compose down -v`.

---

# 🧩 Flujo Completo de una Petición

```
Cliente (navegador/curl)
        │
        │ POST /registrar_stack (JSON body)
        ▼
┌─────────────────────┐
│    FastAPI (main.py) │
│  1. Valida JSON      │  ← Pydantic (StackCreateSchema)
│  2. Abre sesión BD   │  ← get_db() via Depends
│  3. Llama simulador  │  ← tmm_torch_simulator.ejecutar_simulacion()
│  4. Guarda stack     │  ← db.add(stack_db); db.flush()
│  5. Guarda resultados│  ← db.add(resultado_db); db.commit()
│  6. Retorna JSON     │
└─────────────────────┘
        │
        │ Respuesta: 201 Created + JSON con resultados
        ▼
    Cliente
```

---

# 📝 Conceptos Clave para Replicar

| Concepto | Cómo se usa en este proyecto |
|---|---|
| **FastAPI** | Framework web. `@app.get()`, `@app.post()` para endpoints |
| **Pydantic** | Validación de tipos. Clases que heredan de `BaseModel` |
| **SQLAlchemy ORM** | Clases Python = Tablas SQL. `db.query()`, `db.add()`, `db.commit()` |
| **Dependency Injection** | `Depends(get_db)` inyecta la sesión automáticamente |
| **Docker Layer Caching** | COPY requirements.txt antes que el código → instalar deps es rápido |
| **Docker Networking** | Servicios se comunican por nombre de servicio |
| **Healthcheck** | Evita race conditions entre servicios que tardan en arrancar |
| **Volumes** | Persistencia de datos y hot-reload de código |
| **Matplotlib Agg backend** | Usar matplotlib en servidor sin pantalla |
| **sys.path manipulation** | Hacer importables módulos fuera del package actual |
