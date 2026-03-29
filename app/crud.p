from sqlalchemy.orm import Session
from . import models, schemas

def create_user_transaction(db: Session, transaction: schemas.TransactionCreate, category: str, explanation: str):
    db_transaction = models.Transaction(
        description=transaction.description,
        amount=transaction.amount,
        category=category,      # Added this
        explanation=explanation # Added this
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    return db_transaction
EOF


