from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Annotated
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

class ingresoBase(BaseModel):
    documentoingreso: str
    nombrepersona: str
    apellidopersona: str

class ingresoBase2(BaseModel):
    idregistro: int
    documentoingreso: str
    nombrepersona: str
    apellidopersona: str

class egresoBase(BaseModel):
    documentosalida: str
    nombresalida: str
    apellidosalida: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency=Annotated[Session, Depends(get_db)]


@app.post("/registro",status_code = status.HTTP_201_CREATED)
async def crear_registro(registro:ingresoBase,db:db_dependency):
    registro_db = models.Ingreso(**registro.dict())
    db.add(registro_db)
    db.commit()
    #db.refresh(registro_db)
    return "el registro se realizo exitosamente"

@app.get("/lista_registros/", status_code = status.HTTP_200_OK)
async def consultar_registros(db:db_dependency):
    registros = db.query(models.Ingreso).all()
    return registros

@app.get("/consultar_registro/{documento}", status_code = status.HTTP_200_OK)
async def consultar_registro_dni(documento:str,db:db_dependency):
    registro = db.query(models.Ingreso).filter(models.Ingreso.documentoingreso == documento).first()
    if registro is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registro no encontrado")
    return registro

@app.delete("/borrar_registro/{documento}",status_code = status.HTTP_200_OK)
async def borrar_registro(documento:str,db:db_dependency):
    registro_borrar = db.query(models.Ingreso).filter(models.Ingreso.documentoingreso == documento).first()
    if registro_borrar is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registro no encontrado")
    db.delete(registro_borrar)
    db.commit()
    return "Registro borrado exitosamente"

@app.post("/actualizar_registro/",status_code=status.HTTP_200_OK)
async def actualizar_registro(registro:ingresoBase2,db:db_dependency):
    registro_actualizar = db.query(models.Ingreso).filter(models.Ingreso.idregistro == registro.idregistro).first()
    if registro_actualizar is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registro no encontrado")
    registro_actualizar.documentoingreso = registro.documentoingreso
    registro_actualizar.nombrepersona = registro.nombrepersona
    registro_actualizar.apellidopersona = registro.apellidopersona
    db.commit()
    return "Registro actualizado exitosamente"

@app.post("/egreso",status_code = status.HTTP_201_CREATED)
async def crear_egreso(egreso:egresoBase,db:db_dependency):
    egreso_db = models.Egreso(**egreso.dict())
    db.add(egreso_db)
    db.commit()
    #db.refresh(egreso_db)
    return "el egreso se realizo exitosamente"