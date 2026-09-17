# Archivo: test_api.py
from fastapi.testclient import TestClient
from main import app
import json

# Creamos un cliente falso
client = TestClient(app)

def test_crear_trayectoria():
    # Simulamos un usuario haciendo un POST
    respuesta = client.post("/trajectory_data", json={"theta": 45.0, "v_inicial": 10.0})
    
    # Comprobamos que el código HTTP sea 201 (Creado)
    assert respuesta.status_code == 201
    
    # Comprobamos que la API nos devuelva el JSON que esperamos
    datos = respuesta.json()
    assert "id" in datos
    assert float(json.loads(datos["theta"])) == 45.0
    
def test_datos_invalidos():
    # Qué pasa si mando texto en vez de números?
    respuesta = client.post("/trajectory_data", json={"theta": "cuarenta", "v_inicial": 10.0})
    assert respuesta.status_code == 422 # Error de validación de Pydantic
