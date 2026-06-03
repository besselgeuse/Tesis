from sqlalchemy import String,Integer,Column
from database import Base


class Ingreso(Base):
    __tablename__="ingresos"
    idregistro=Column(Integer,primary_key=True,index=True)
    documentoingreso = Column(String(11))
    nombrepersona = Column(String(50))
    apellidopersona = Column(String(50))

class Egreso(Base):
    __tablename__ = "egresos"
    ideregreso = Column(Integer, primary_key=True, index=True)
    documentosalida = Column(String(11))
    nombresalida = Column(String(50))
    apellidosalida = Column(String(50))
    
