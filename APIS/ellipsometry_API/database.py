from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:Dinosaurio1724@localhost:3306/ellipsometry_fit")

engine = create_engine(DATABASE_URL, echo=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()