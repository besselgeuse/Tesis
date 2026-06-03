from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

Database_URL = os.getenv("DATABASE_URL","mysql+pymysql://root:Dinosaurio1724@localhost:3306/prueba_api")

engine = create_engine(Database_URL, echo=True)
sessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()