
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


# Configuración de la base de datos
engine = create_engine(
    "mysql+pymysql://root:Dinosaurio1724@localhost:3306/ejemplo1"
)
#creo la sesion
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base = declarative_base()
