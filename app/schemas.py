from pydantic import BaseModel
from typing import Optional

class TransactionBase(BaseModel):
    description: str
    amount: float

class TransactionCreate(TransactionBase):
    pass

class Transaction(TransactionBase):
    id: int
    category: Optional[str] = None
    explanation: Optional[str] = None

    class Config:
        from_attributes = True
