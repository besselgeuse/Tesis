from fastapi import FastAPI, status, HTTPException, Depends
from pydantic import BaseModel
from database import engine, sessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
from functions import Tiro_parabolico as tp
import models
import json
import jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer


SECRET_KEY = "tu_clave_secreta_super_segura_para_el_laboratorio"
ALGORITHM = "HS256"


# inicializamos la app
app = FastAPI()

#creo la tabla
models.Base.metadata.create_all(bind=engine)

class trajectory_data_request(BaseModel):
    theta: float
    v_inicial: float

class UserCreate(BaseModel):
    email: str
    password: str


def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def crear_token_acceso(datos: dict, tiempo_vida_minutos: int = 60) -> str:
    convertir_datos = datos.copy()
    expiracion = datetime.utcnow() + timedelta(minutes=tiempo_vida_minutos)
    convertir_datos.update({"exp": expiracion})
    
    # Codificar y firmar el token JWT
    return jwt.encode(convertir_datos, SECRET_KEY, algorithm=ALGORITHM)

def obtener_usuario_actual(token: Annotated[str, Depends(oauth2_scheme)], db: db_dependency) -> models.User:
    excepcion_credenciales = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
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


@app.post("/register", status_code=status.HTTP_201_CREATED)
def registrar_usuario(usercreate: UserCreate, db: db_dependency):
    usuario_existente = db.query(models.User).filter(models.User.email == usercreate.email).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El email ya está registrado.")
    
    nuevo_usuario = models.User(email=usercreate.email, hashed_password=models.User.generar_hash(usercreate.password))
    db.add(nuevo_usuario)
    db.commit()
    return {"message": "Usuario creado exitosamente"}

@app.post("/token")
def iniciar_sesion(db: db_dependency, form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2PasswordRequestForm mapea el campo 'username' al email ingresado
    usuario = db.query(models.User).filter(models.User.email == form_data.username).first()
    
    if not usuario or not usuario.verificar_password(form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generar el token empaquetando el email del usuario
    token_jwt = crear_token_acceso(datos={"sub": usuario.email})
    
    # Es obligatorio para el estándar OAuth2 retornar el token_type como 'bearer'
    return {"access_token": token_jwt, "token_type": "bearer"}


@app.post("/trajectory_data", status_code=status.HTTP_201_CREATED)
def create_trajectory_data(
    trajectory: trajectory_data_request, 
    db: db_dependency,
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    try:
        th, vi = trajectory.theta, trajectory.v_inicial
        x,y,t = tp(th, vi)
        new_trajectory = models.trajectory_data(theta=json.dumps(th), v_inicial=json.dumps(vi), t=json.dumps(t.tolist()), x=json.dumps(x.tolist()), y=json.dumps(y.tolist()))
        db.add(new_trajectory)
        db.commit()
        db.refresh(new_trajectory)
        return {
            "id": new_trajectory.id,
            "theta": new_trajectory.theta,
            "v_inicial": new_trajectory.v_inicial,
            "x": new_trajectory.x,
            "y": new_trajectory.y,
            "t": new_trajectory.t
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
@app.get("/trajectory_data/{id}", status_code=status.HTTP_200_OK)
def get_trajectory_data(
    id: int, 
    db: db_dependency,
    usuario_actual: Annotated[models.User, Depends(obtener_usuario_actual)]
):
    t = db.query(models.trajectory_data).filter(models.trajectory_data.id == id).first()
    if t is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trajectory not found")
    return {
        "id" : t.id,
        "theta" : t.theta,
        "v_inicial" : t.v_inicial,
        "x" : t.x,
        "y" : t.y,
        "t" : t.t
    }