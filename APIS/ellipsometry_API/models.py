from sqlalchemy import Column, Integer, Text, DateTime, String, JSON, ForeignKey
from database import Base
from datetime import datetime
import bcrypt


class Stack(Base):
    __tablename__ = "stacks"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    
    # Guarda una lista de diccionarios, ej: 
    # [{"name": "air"}, {"name": "TiO2", "model": "cauchy", "thickness": 50, "A": 1.5, "B": 0.01, "C": 0.01}, ...]
    layers = Column(JSON, nullable=False)
    best_Is = Column(JSON)
    best_Ic = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relación con el usuario dueño del stack
    user_id = Column(Integer, ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False)

class User(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    @staticmethod
    def generar_hash(password: str) -> str:
        # Generar hash usando la librería bcrypt directamente
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')

    def verificar_password(self, password: str) -> bool:
        # Verificar la contraseña contra el hash
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))
        except Exception:
            return False
