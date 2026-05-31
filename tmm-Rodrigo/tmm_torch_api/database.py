from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Cadena de conexión a MySQL
# Formato: mysql+pymysql://usuario:contraseña@localhost:3306/nombre_base_de_datos
# He puesto 'tmmm_torch_api' según el log de tu conexión, cámbialo a 'tmm_torch_api' si fue un error de tipeo al crearla.
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:Dinosaurio1724@localhost:3306/tmmm_torch_api")

# Crear motor MySQL (se quitó check_same_thread porque es exclusivo de SQLite)
engine = create_engine(DATABASE_URL, echo=True)

# Crear la sesión y la base declarativa
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

