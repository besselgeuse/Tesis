from fastapi import FastAPI, status, HTTPException, Depends
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from database import engine, SessionLocal
from typing import Annotated, List, Tuple, Optional
from sqlalchemy.orm import Session
from fit_elipsometrico import ajuste_elipsometrico
import models
import json
import jwt
import io
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
import os
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "clave_secreta_genérica")
ALGORITHM = "HS256"

# inicializamos la app
app = FastAPI()

# Servir archivos estáticos
import os
API_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(API_DIR, 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

data_path='./Datos-09-6/TiO2_Si_SI_SP_S1.txt'
Data_R= None#'./Datos-04-6/1TIO2-2.txt'

#creo la tabla
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def crear_token_acceso(datos:dict,tiempo_vida_minutos:int=60) -> str:
    convertir_datos = datos.copy()
    expiracion = datetime.utcnow() + timedelta(minutes=tiempo_vida_minutos)
    convertir_datos.update({"exp":expiracion})
    return jwt.encode(convertir_datos,SECRET_KEY,algorithm=ALGORITHM)

def obtener_usuario_actual(token: Annotated[str, Depends(oauth2_scheme)], db: db_dependency) -> models.User:
    excepcion_credenciales = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="no se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise excepcion_credenciales
    except jwt.PyJWTError:
        raise excepcion_credenciales
    usuario = db.query(models.User).filter(models.User.email == email).first()
    if usuario is None:
        raise excepcion_credenciales
    return usuario

class UserCreate(BaseModel):
    email: str
    password: str

class LayerModel(BaseModel):
    model_type: str = Field(..., description="Tipo de modelo: 'cauchy', 'cauchy_absorbent', 'bruggeman'")
    # Para Cauchy: [[A_min, A_max], [B_min, B_max], [C_min, C_max]]
    # Para Bruggeman: [[f_min, f_max]]
    bounds: Optional[List[Tuple[float, float]]] = None
# 2. Definimos una Capa
class Layer(BaseModel):
    name: str = Field(..., description="Nombre del material (ej. 'air', 'SiO2', 'Si')")
    model: Optional[LayerModel] = Field(None, description="Modelo de dispersión. None si es una capa estática")
    
    # d_min y d_max son opcionales porque el superestrato y sustrato no los usan
    d_min: Optional[float] = Field(None, description="Límite inferior del espesor en nm")
    d_max: Optional[float] = Field(None, description="Límite superior del espesor en nm")
# 3. Definimos el Stack de capas ordenadas
class Stack(BaseModel):
    name: str
    layers: List[Layer]
# 4. Definimos los parámetros de la simulación/ajuste
class SimParam(BaseModel):
    num_starts: int = 100
    num_epochs: int = 200
    lr: float = 1.5
    use_cuda: bool = True
    th_0: float = 69.5
    stack: Stack

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.post('/register', status_code=status.HTTP_201_CREATED)
def registrar_usuario(usercreate: UserCreate, db: db_dependency):
    usuario_existente = db.query(models.User).filter(models.User.email == usercreate.email).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="email actualmente en uso")
    nuevo_usuario = models.User(
        email=usercreate.email,
        hashed_password=models.User.generar_hash(usercreate.password)
    )    
    db.add(nuevo_usuario)
    db.commit()
    return {"message": "Usuario registrado exitosamente"}

@app.post('/token')
def iniciar_sesion(db: db_dependency, form_data: OAuth2PasswordRequestForm = Depends()):
    usuario = db.query(models.User).filter(models.User.email == form_data.username).first()
    if usuario is None or not usuario.verificar_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    jwt_token = crear_token_acceso(datos={"sub": usuario.email})
    return {"access_token": jwt_token, "token_type": "Bearer"}


@app.post('/simular_stack', status_code=status.HTTP_201_CREATED)
async def simular_stack(
    registro: SimParam, 
    db: db_dependency, 
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    try:
        # 1. Extraer nombres y modelos en el formato esperado por el script físico
        layer_names = [layer.name for layer in registro.stack.layers]
        
        # Convertir Pydantic LayerModel a diccionarios de Python plano
        layer_models = []
        for layer in registro.stack.layers:
            if layer.model is None:
                layer_models.append(None)
            else:
                model_dict = {
                    "model": layer.model.model_type
                }
                if layer.model.model_type == 'bruggeman':
                    model_dict["f_bounds"] = layer.model.bounds
                else:
                    model_dict["bounds"] = layer.model.bounds
                layer_models.append(model_dict)
        
        d_bounds = [(layer.d_min, layer.d_max) for layer in registro.stack.layers[1:-1]]

        # 2. Ejecutar la simulación con Autograd
        best_thicknesses, best_Is, best_Ic, best_params, wl_exp, Is_exp, Ic_exp = ajuste_elipsometrico(
            data_path=data_path,
            Data_R=Data_R,
            skiprows=5,
            layer_names=layer_names,
            layer_models=layer_models,
            d_bounds=d_bounds,
            theta_0=registro.th_0,
            num_starts=registro.num_starts,
            num_epochs=registro.num_epochs,
            lr=registro.lr,
            use_cuda=registro.use_cuda,
        )

        # 3. Construir la lista estructurada de capas para guardar en la BD
        layer_db = []
        o_idx = 0
        for idx, name in enumerate(layer_names):
            if layer_models[idx] is None:
                # Capa estática (como aire o silicio)
                layer_db.append({
                    "name": name,
                    "model": None,
                    "thickness": None,
                    "params": {}
                })
            else:
                # Capa parametrizada y optimizada
                t_val = float(best_thicknesses[o_idx])
                
                # Extraer parámetros (A, B, C o f_air, linked_to) de forma segura y libre de colisiones
                sub_dict = best_params.get(idx) or best_params.get(name, {})
                params_cleaned = {}
                for k, v in sub_dict.items():
                    if k == 'model':
                        continue
                    try:
                        params_cleaned[k] = float(v)
                    except (ValueError, TypeError):
                        params_cleaned[k] = v
                
                layer_db.append({
                    "name": name,
                    "model": registro.stack.layers[idx].model.model_type,
                    "thickness": t_val,
                    "params": params_cleaned
                })
                o_idx += 1        
        # 4. Crear el modelo de base de datos
        stack_db = models.Stack(
            name=registro.stack.name,
            layers=layer_db,
            best_Is=best_Is.tolist(),
            best_Ic=best_Ic.tolist(),
            wl_exp=wl_exp.tolist(),  # Guardar la lista de longitudes de onda experimental
            Is_exp=Is_exp.tolist(),
            Ic_exp=Ic_exp.tolist(),
            user_id=usuario_actual.id
        )

        db.add(stack_db)
        db.flush()  # Generar id autoincremental
        db.commit()

        return {
            "message": "Stack registrado exitosamente",
            "stack": {
                "id": stack_db.id,
                "nombre": stack_db.name,
                "capas": stack_db.layers,
                "best_Is": stack_db.best_Is,
                "best_Ic": stack_db.best_Ic,
                "wl_exp": stack_db.wl_exp,
                "Is_exp": stack_db.Is_exp,
                "Ic_exp": stack_db.Ic_exp,
                "fecha": str(stack_db.created_at)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/mostrar_resultados", status_code=status.HTTP_200_OK)
async def obtener_resultados(
    db: db_dependency, 
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    """Retorna todos los stacks registrados del usuario autenticado."""
    stacks = db.query(models.Stack).filter(models.Stack.user_id == usuario_actual.id).all()
    resultado = []
    for s in stacks:
        resultado.append({
            "id": s.id,
            "nombre": s.name,
            "capas": s.layers,
            "best_Is": s.best_Is,
            "best_Ic": s.best_Ic,
            "wl_exp": s.wl_exp,
            "Is_exp": s.Is_exp,
            "Ic_exp": s.Ic_exp,
            "fecha": str(s.created_at),
        })
    return resultado

@app.get("/mostrar_resultados/{id}", status_code=status.HTTP_200_OK)
async def obtener_resultados_id(
    id: int, 
    db: db_dependency, 
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    """Retorna un stack registrado del usuario autenticado por ID."""
    stack = db.query(models.Stack).filter(models.Stack.id == id, models.Stack.user_id == usuario_actual.id).first()
    if not stack:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stack no encontrado o no tienes permiso para verlo")
        
    return {
        "id": stack.id,
        "nombre": stack.name,
        "capas": stack.layers,
        "best_Is": stack.best_Is,
        "best_Ic": stack.best_Ic,
        "wl_exp": stack.wl_exp,
        "Is_exp": stack.Is_exp,
        "Ic_exp": stack.Ic_exp,
        "fecha": str(stack.created_at),
    }

@app.get('/descargar_resultados/{id}', status_code=status.HTTP_200_OK)
async def descargar_resultados(
    id: int, 
    db: db_dependency, 
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    stack = db.query(models.Stack).filter(models.Stack.id == id, models.Stack.user_id == usuario_actual.id).first()
    if not stack:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stack no encontrado")
    
    stream = io.StringIO()
    stream.write(f"Resultados del Stack: {stack.name}\n")
    stream.write(f"Fecha: {stack.created_at}\n")
    stream.write(f"ID: {stack.id}\n\n")
    stream.write("información del stack\n")
    stream.write("=====================\n\n")
    
    for layer in stack.layers:
        stream.write(f"Material: {layer['name']}\n")
        stream.write(f"Modelo de Dispersión: {layer.get('model', 'N/A')}\n")
        if layer.get('thickness') is not None:
            stream.write(f"Espesor: {layer['thickness']:.2f} nm\n")
        if layer.get('params'):
            stream.write(f"Parámetros del Modelo: {layer['params']}\n")
        stream.write("\n")
        
    stream.write("=====================\n\n")
    stream.write("wl,best_Ic,best_Is,Ic_exp,Is_exp\n")
    if stack.Ic_exp and stack.Is_exp:
        for wl, best_ic, best_is, ic_exp, is_exp in zip(stack.wl_exp, stack.best_Ic, stack.best_Is, stack.Ic_exp, stack.Is_exp):
            stream.write(f"{wl},{best_ic},{best_is},{ic_exp},{is_exp}\n")
    else:
        for wl, best_ic, best_is in zip(stack.wl_exp, stack.best_Ic, stack.best_Is):
            stream.write(f"{wl},{best_ic},{best_is},,\n")

    file_content = stream.getvalue()
    stream.close()
    
    
    return StreamingResponse(
        io.BytesIO(file_content.encode('utf-8')),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=resultados_stack_{stack.id}.txt"}
    )