# pyrefly: ignore [missing-import]
from sqlalchemy import Column, Integer, Text, DateTime, String
from database import Base
from datetime import datetime
import bcrypt

class trajectory_data(Base):
    __tablename__ = "trajectory_data"
    id = Column(Integer, primary_key=True, index=True)
    t = Column(Text)
    x = Column(Text)
    y = Column(Text)
    theta = Column(Text)
    v_inicial = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

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