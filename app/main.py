import pickle
import numpy as np
import shap
import os
from fastapi import FastAPI, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from . import crud, models, schemas
from .database import engine, get_db
from app.ml.train import update_model

models.Base.metadata.create_all(bind=engine)
app = FastAPI()

with open('app/ml/model.pkl', 'rb') as f:
    ml_model = pickle.load(f)
with open('app/ml/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
explainer = shap.TreeExplainer(ml_model)

@app.post("/transactions/", response_model=schemas.Transaction)
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    test_vec = vectorizer.transform([transaction.description]).toarray()
    prediction = ml_model.predict(test_vec)[0]
    explanation_text = f"Categorized as {prediction}"
    return crud.create_user_transaction(db=db, transaction=transaction, category=prediction, explanation=explanation_text)

@app.get("/transactions/")
def read_transactions(db: Session = Depends(get_db)):
    return db.query(models.Transaction).all()

@app.post("/transactions/{transaction_id}/correct")
def correct_transaction(transaction_id: int, new_category: str = Query(...), db: Session = Depends(get_db)):
    transaction = db.query(models.Transaction).filter(models.Transaction.id == transaction_id).first()
    if not transaction: return {"error": "Not found"}
    transaction.category = new_category
    db.commit()
    update_model(transaction.description, new_category)
    return {"message": "AI learned!"}

# Serve from the correct directory
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/") 
async def read_index():
    return FileResponse(os.path.join(static_path, 'index.html'))
