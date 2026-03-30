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

# Load the AI Brain
with open('app/ml/model.pkl', 'rb') as f:
    ml_model = pickle.load(f)
with open('app/ml/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize the XAI Explainer
explainer = shap.TreeExplainer(ml_model)

@app.post("/transactions/", response_model=schemas.Transaction)
def create_transaction(transaction: schemas.TransactionCreate, db: Session = Depends(get_db)):
    # 1. Transform text to math
    test_vec = vectorizer.transform([transaction.description]).toarray()
    
    # 2. Get the prediction
    prediction = ml_model.predict(test_vec)[0]

    # 3. Calculate SHAP values (The Math Minor part)
    shap_output = explainer.shap_values(test_vec)
    class_idx = list(ml_model.classes_).index(prediction)

    # Handle SHAP output format
    if isinstance(shap_output, list):
        shap_vals = shap_output[class_idx][0]
    else:
        shap_vals = shap_output[0, :, class_idx]

    # 4. Find the most influential word
    feature_names = vectorizer.get_feature_names_out()
    top_word_idx = np.argmax(shap_vals)
    top_word = feature_names[top_word_idx]

    # 5. Build the smart explanation
    explanation_text = f"Categorized as {prediction} because of the word '{top_word}'"

    return crud.create_user_transaction(
        db=db,
        transaction=transaction,
        category=prediction,
        explanation=explanation_text
    )

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/") 
async def read_index():
    return FileResponse(os.path.join(static_dir, 'index.html'))
