import sys
import os
import json
import numpy as np

# ── Configuración de rutas ──────────────────────────────────────────────
# API_DIR = carpeta donde vive este archivo (tmm_torch_api/)
# TMM_DIR = carpeta padre (tmm-Rodrigo/) donde están tmm_utils_Rodrigo.py y tmm_core.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
# TMM_DIR = os.path.abspath(os.path.join(API_DIR, '..'))

# Agregar API_DIR y TMM_DIR al sys.path para importaciones
if API_DIR not in sys.path:
    sys.path.append(API_DIR)
# if TMM_DIR not in sys.path:
#     sys.path.append(TMM_DIR)


# Configurar matplotlib para modo servidor (sin ventana)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Imports del proyecto ────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Annotated, List
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
import tmm_torch_simulator as st

# ── App FastAPI ─────────────────────────────────────────────────────────
app = FastAPI()

# ── Servir archivos estáticos (interfaz gráfica) ───────────────────────
STATIC_DIR = os.path.join(API_DIR, 'static')
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Ensure tables are created at startup
@app.on_event("startup")
def startup_event():
    models.Base.metadata.create_all(bind=engine)
    print("[INFO] Database tables verified/created.")


# ── Schemas Pydantic (validación de entrada) ────────────────────────────
from typing import Union, Annotated, List

class CapaSchema(BaseModel):
    """Una capa del stack: espesor en nm (o 'inf'), nombre del material, coherencia ('c' o 'i')"""
    material: str
    coherencia: str

class ThickSchema(BaseModel):
    """Rango de optimización para una capa: índice de la capa en el stack, min, max y paso en nm"""
    indice_capa: int
    min: float
    max: float


class StackCreateSchema(BaseModel):
    """Esquema completo para registrar un stack y ejecutar la simulación"""
    nombre: str
    capas: List[CapaSchema]
    thicks: List[ThickSchema]
    sustrato:str
    pol:str
    num_starts:int = 500
    num_epochs:int = 200
    lr:float = 1.5
    use_cuda:bool = True
    th_0:float = 0.0


# ── Dependency de base de datos ─────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/registrar_stack", status_code=status.HTTP_201_CREATED)
async def registrar_stack(registro: StackCreateSchema, db: db_dependency):
    """
    Recibe un stack, ejecuta la simulación TMM y guarda tanto el stack
    como los resultados optimizados en la base de datos.
    """
    try:
        # 1. Construir thicks como lista de dicts para el simulador
        thicks = [{"min": t.min, "max": t.max} for t in registro.thicks]

        # 2. Ejecutar simulación TMM (PyTorch)
        best_thickness, best_reflectancia, best_jsc_am15, best_jsc_am0, lams = st.ejecutar_simulacion(
            stack=[capa.material for capa in registro.capas], 
            thicks=thicks, 
            c_list=[capa.coherencia for capa in registro.capas], 
            sustrato=registro.sustrato,
            pol=registro.pol,
            num_starts=registro.num_starts,
            num_epochs=registro.num_epochs,
            lr=registro.lr,
            use_cuda=registro.use_cuda,
            th_0=registro.th_0
        )

        # 3. Serializar valores numéricos
        js_am0_val = float(best_jsc_am0)
        js_am15_val = float(best_jsc_am15)

        # 4. Construir mapeo de espesores óptimos
        espesores_optimos = {}
        for idx, val in enumerate(best_thickness):
            material_name = registro.capas[idx].material
            espesores_optimos[f"{idx}_{material_name}"] = float(val)

        # 5. Guardar stack en la base de datos (incluyendo air y sustrato en el json como indica el modelo)
        full_capas = ["air"] + [c.material for c in registro.capas] + [registro.sustrato]
        full_coherencia = ["i"] + [c.coherencia for c in registro.capas] + ["i"]

        stack_db = models.Stack(
            nombre=registro.nombre,
            capas_json=json.dumps(full_capas),
            coherencia_json=json.dumps(full_coherencia),
            thicks_json=json.dumps(thicks),
        )
        db.add(stack_db)
        db.flush()  # para obtener el idstack generado

        # 6. Guardar resultados optimizados en la base de datos
        resultado_db = models.Results(
            ref_optima=json.dumps(best_reflectancia.tolist()),
            lams=json.dumps(lams.tolist()),
            js_am0=f"{js_am0_val:.4f}",
            js_am15=f"{js_am15_val:.4f}",
            espesores_optimos=json.dumps(espesores_optimos),
            idstack=stack_db.idstack
        )
        db.add(resultado_db)
        db.commit()

        return {
            "mensaje": "Simulación completada y guardada exitosamente",
            "idstack": stack_db.idstack,
            "js_am0_max": js_am0_val,
            "js_am15_max": js_am15_val,
            "espesores_optimos": espesores_optimos
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error en la simulación: {str(e)}")


@app.get("/mostrar_resultados", status_code=status.HTTP_200_OK)
async def obtener_resultados(db: db_dependency):
    """Retorna todos los stacks registrados con un resumen de sus resultados."""
    stacks = db.query(models.Stack).all()
    resultado = []
    for s in stacks:
        res = db.query(models.Results).filter(models.Results.idstack == s.idstack).first()
        resultado.append({
            "idstack": s.idstack,
            "nombre": s.nombre,
            "capas": json.loads(s.capas_json) if s.capas_json else None,
            "coherencias": json.loads(s.coherencia_json) if s.coherencia_json else None,
            "thicks": json.loads(s.thicks_json) if s.thicks_json else None,
            "fecha": str(s.fecha),
            "resultados": {
                "idresults": res.idresults,
                "js_am0": res.js_am0,
                "js_am15": res.js_am15,
                "espesores_optimos": json.loads(res.espesores_optimos) if res.espesores_optimos else None,
            } if res else None
        })
    return resultado


@app.get("/mostrar_resultados/{idstack}", status_code=status.HTTP_200_OK)
async def obtener_resultado_por_id(idstack: int, db: db_dependency):
    """Retorna un stack específico con sus resultados completos."""
    s = db.query(models.Stack).filter(models.Stack.idstack == idstack).first()
    if s is None:
        raise HTTPException(status_code=404, detail="Stack no encontrado")

    res = db.query(models.Results).filter(models.Results.idstack == s.idstack).first()
    return {
        "idstack": s.idstack,
        "nombre": s.nombre,
        "capas": json.loads(s.capas_json),
        "coherencias": json.loads(s.coherencia_json),
        "thicks": json.loads(s.thicks_json),
        "fecha": str(s.fecha),
        "resultados": {
            "idresults": res.idresults,
            "js_am0": res.js_am0,
            "js_am15": res.js_am15,
            "espesores_optimos": json.loads(res.espesores_optimos),
            "ref_optima": json.loads(res.ref_optima),
            "lams": json.loads(res.lams),
        } if res else None
    }