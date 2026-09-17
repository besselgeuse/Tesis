# Guía Educativa: Autenticación con JWT y OAuth2 en FastAPI

Esta guía explica en detalle cómo funciona el sistema de registro, inicio de sesión (login) y protección de rutas que hemos implementado en el proyecto. El objetivo es que comprendas la lógica de cada línea de código para que puedas replicarlo y adaptarlo en tu proyecto final de tesis (`tmm_torch_api`).

---

## 1. Concepto Fundamental: ¿Por qué usamos JWT y OAuth2?

En el desarrollo web moderno, las APIs suelen ser **stateless** (sin estado). Esto significa que el servidor no recuerda al cliente entre una petición y otra. 

Para solucionar esto de manera segura, usamos dos tecnologías:
1. **OAuth2 (estándar de la industria):** Define el protocolo de comunicación. En nuestro caso, el flujo *Resource Owner Password Credentials*, donde el cliente envía usuario/contraseña a cambio de una credencial.
2. **JWT (JSON Web Token):** Es la credencial física en sí. Es un string compacto y seguro que contiene información del usuario firmada digitalmente por el servidor.

---

## 2. El Flujo Completo de la Autenticación

```
[ Registro ]
Usuario/Password  -->  [Servidor] (Genera Hash bcrypt)  -->  [Base de Datos] (Guarda Hash)

[ Inicio de Sesión / Login ]
Usuario/Password  -->  [Servidor] (Compara con Hash DB)  -->  Si coincide: Retorna Token JWT

[ Consumo Protegido ]
Cliente hace petición con Token en Headers  -->  [Servidor] (Verifica Firma y Expiración)  -->  Retorna Datos
```

---

## 3. Explicación del Código Paso a Paso

### A. Gestión Segura de Contraseñas (en `models.py`)

No guardamos contraseñas reales. Guardamos **hashes** generados con el algoritmo **bcrypt**. Usamos la librería `bcrypt` de Python directamente, la cual es más rápida y compatible con versiones modernas de Python (como la 3.12+), evitando los bugs y advertencias de desuso (*deprecation*) de librerías antiguas como `passlib`.

```python
import bcrypt

class User(Base):
    # ... definición de columnas ...

    @staticmethod
    def generar_hash(password: str) -> str:
        # Generar una sal (salt) aleatoria
        salt = bcrypt.gensalt()
        # Encriptar la contraseña (requiere codificar el texto a bytes)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        # Retornamos el hash decodificado como string para guardarlo en la base de datos
        return hashed.decode('utf-8')

    def verificar_password(self, password: str) -> bool:
        # Comparamos la contraseña recibida contra el hash guardado de forma segura
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))
        except Exception:
            return False
```
* **`bcrypt.gensalt()`**: Crea un valor aleatorio único llamado *salt* (sal) de forma automática. Esto asegura que si dos usuarios se registran con la misma contraseña, sus hashes resultantes sigan siendo completamente diferentes.
* **`generar_hash`**: Es un método estático. Toma la contraseña plana, genera la sal, calcula el hash usando `bcrypt.hashpw` y lo guarda como un string.
* **`verificar_password`**: Es un método de instancia. Utiliza `bcrypt.checkpw` para validar la contraseña de forma segura y evitar ataques de temporización (timing attacks).


---

### B. Generación de Tokens de Acceso (en `main.py`)

Un token JWT es un objeto JSON firmado. Consta de tres partes separadas por puntos: `Cabecera.Payload.Firma`.

```python
SECRET_KEY = "tu_clave_secreta_super_segura_para_el_laboratorio"
ALGORITHM = "HS256"

def crear_token_acceso(datos: dict, tiempo_vida_minutos: int = 60) -> str:
    convertir_datos = datos.copy()
    expiracion = datetime.utcnow() + timedelta(minutes=tiempo_vida_minutos)
    convertir_datos.update({"exp": expiracion})
    
    return jwt.encode(convertir_datos, SECRET_KEY, algorithm=ALGORITHM)
```
* **`SECRET_KEY`**: La clave privada que usa el servidor para firmar los tokens. **Nunca debe revelarse al público.**
* **`ALGORITHM`**: `HS256` es el algoritmo de encriptación simétrica para firmar la petición.
* **`expiracion`**: Añadimos la clave `"exp"` (expiration claim) al payload. El token dejará de ser válido automáticamente cuando transcurra el tiempo indicado (en este caso, 60 minutos).
* **`jwt.encode(...)`**: Une la cabecera, los datos (`payload`) y los firma usando la clave secreta. El resultado es el token JWT que el cliente guardará en su navegador.

---

### C. El Endpoint de Registro `/register`

Permite dar de alta a nuevos usuarios asegurando que no se repitan correos.

```python
@app.post("/register", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usercreate: UserCreate, db: db_dependency):
    # 1. Verificar si el email ya existe
    usuario_existente = db.query(models.User).filter(models.User.email == usercreate.email).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El email ya está registrado.")
    
    # 2. Si no existe, crear usuario nuevo con la contraseña hasheada
    nuevo_usuario = models.User(
        email=usercreate.email, 
        hashed_password=models.User.generar_hash(usercreate.password)
    )
    
    db.add(nuevo_usuario)
    db.commit()
    return {"message": "Usuario creado exitosamente"}
```

---

### D. El Endpoint de Inicio de Sesión `/token`

El cliente envía sus credenciales y, si son válidas, recibe su token de acceso.

```python
@app.post("/token")
def iniciar_sesion(db: db_dependency, form_data: OAuth2PasswordRequestForm = Depends()):
    # 1. Buscar al usuario por el correo (OAuth2 lo mapea como 'username')
    usuario = db.query(models.User).filter(models.User.email == form_data.username).first()
    
    # 2. Verificar existencia y comparar contraseña usando el hash
    if not usuario or not usuario.verificar_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 3. Generar token con la propiedad 'sub' (subject) conteniendo el email
    token_jwt = crear_token_acceso(datos={"sub": usuario.email})
    
    # 4. Devolver la respuesta obligatoria en el estándar de OAuth2
    return {"access_token": token_jwt, "token_type": "bearer"}
```
* **`OAuth2PasswordRequestForm`**: FastAPI expone este formulario por defecto. Espera que los datos se envíen como `form-data` con los campos específicos `username` (nuestro email) y `password`.
* **`token_type: "bearer"`**: El estándar exige retornar el tipo de token como "bearer" (portador).

---

### E. Protección de Rutas: Inyección de Dependencias

Para bloquear las rutas, primero creamos la función que valide el token en cada petición.

```python
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def obtener_usuario_actual(token: Annotated[str, Depends(oauth2_scheme)], db: db_dependency) -> models.User:
    excepcion_credenciales = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 1. Decodificar el token usando la clave secreta y el algoritmo
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise excepcion_credenciales
    except jwt.PyJWTError:
        # Si la firma es falsa, el token está corrupto o expirado, lanza error
        raise excepcion_credenciales
        
    # 2. Buscar al usuario que viene en el token
    usuario = db.query(models.User).filter(models.User.email == email).first()
    if usuario is None:
        raise excepcion_credenciales
    return usuario
```
* **`OAuth2PasswordBearer(tokenUrl="token")`**: Le dice a FastAPI que busque una cabecera `Authorization: Bearer <token>` en la petición. Si no existe, lanza un error `401 Unauthorized` automáticamente. El parámetro `tokenUrl="token"` indica a la documentación interactiva dónde debe ir a pedir el token.
* **`jwt.decode`**: Lee la información del token y **valida la firma criptográfica**. Si el token fue alterado, lanzará un error `PyJWTError` y bloqueará la petición.

Para aplicar esta seguridad a cualquier endpoint, simplemente inyectamos esta función usando `Depends()`:

```python
@app.post("/trajectory_data", status_code=status.HTTP_201_CREATED)
def create_trajectory_data(
    trajectory: trajectory_data_request, 
    db: db_dependency,
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)] # <-- RUTA PROTEGIDA
):
    # Si la ejecución llega aquí, significa que 'usuario_actual' ya es un usuario autenticado válido.
    # ...
```

---

## 4. Guía de Pruebas Manuales con Swagger UI

La forma más rápida e interactiva de probar que todo esto funciona es utilizando la documentación automática de FastAPI:

1. Levanta tu proyecto y ve a: [http://localhost:8000/docs](http://localhost:8000/docs).
2. Verás que los endpoints `/trajectory_data` tienen un icono de **candado cerrado** a la derecha. Esto indica que están protegidos.
3. Si intentas consumir el endpoint GET `/trajectory_data/{id}` directamente, verás que la API te responde con un error `401 Unauthorized`.
4. Haz clic en el endpoint `/register`, dale a *Try it out* y registra un usuario de prueba.
5. Sube a la esquina superior derecha y haz clic en el botón verde **Authorize**.
6. Introduce el email y contraseña que acabas de registrar en los campos `username` y `password` y haz clic en **Authorize**.
7. Verás que el candado ahora está abierto/activo. A partir de este momento, Swagger UI adjuntará automáticamente tu token JWT en cada petición que hagas.
8. Vuelve a probar los endpoints de trayectorias. Ahora verás que responden de manera exitosa (`200 OK` o `201 Created`).
