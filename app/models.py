from sqlalchemy import Column, Integer, String, Float
from .database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String)
    amount = Column(Float)
    category = Column(String, nullable=True)
    explanation = Column(String, nullable=True)
