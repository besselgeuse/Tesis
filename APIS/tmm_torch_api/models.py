from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime



class Stack(Base):
    __tablename__ = "stack"
    idstack = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(100))
    capas_json = Column(Text)       # JSON: ["air", "T1_porosa", "T1_densa", "InGaP", "GaAs"...]
    thicks_json = Column(Text)      # JSON: [{"min": 10, "max": 120, "paso": 1}, ...]
    coherencia_json = Column(Text)  # JSON: ["i", "c", "c", "c", "i"] (siempre debe coincidir en longitud con capas_json)
    fecha = Column(DateTime, default=datetime.now)

    resultados = relationship("Results", back_populates="stack")




class Results(Base):
    __tablename__ = "results"
    idresults = Column(Integer, primary_key=True, index=True)
    ref_optima = Column(Text)        # Reflectancia óptima serializada como JSON (1D array)
    lams = Column(Text)              # Longitudes de onda como JSON
    js_am0 = Column(String(50))      # Max Jsc AM0 (valor escalar)
    js_am15 = Column(String(50))     # Max Jsc AM1.5 (valor escalar)
    espesores_optimos = Column(Text)  # JSON con espesores óptimos encontrados
    idstack = Column(Integer, ForeignKey("stack.idstack"))

    stack = relationship("Stack", back_populates="resultados")
