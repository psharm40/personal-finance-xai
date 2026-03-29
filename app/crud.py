from sqlalchemy.orm import Session
from sqlalchemy import func
from . import models, schemas

def create_user_transaction(db: Session, transaction: schemas.TransactionCreate, category: str, explanation: str):
    db_transaction = models.Transaction(
        description=transaction.description,
        amount=transaction.amount,
        category=category,
        explanation=explanation
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction

def get_category_totals(db: Session):
    return db.query(
        models.Transaction.category, 
        func.sum(models.Transaction.amount).label("total")
    ).group_by(models.Transaction.category).all()
