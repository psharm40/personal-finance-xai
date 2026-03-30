import pickle
import numpy as np
import shap
import os
from fastapi import FastAPI, Depends
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
    
    shap_output = explainer.shap_values(test_vec)
    class_idx = list(ml_model.classes_).index(prediction)
    
    if isinstance(shap_output, list):
        shap_vals = shap_output[class_idx][0]
    else:
        shap_vals = shap_output[0, :, class_idx]
    
    feature_names = vectorizer.get_feature_names_out()
    top_word_idx = np.argmax(shap_vals)
    top_word = feature_names[top_word_idx]
    
    explanation_text = f"Categorized as {prediction} because of the word '{top_word}'"

    return crud.create_user_transaction(
        db=db, 
        transaction=transaction, 
        category=prediction, 
        explanation=explanation_text
    )

@app.get("/analytics/")
def get_analytics(db: Session = Depends(get_db)):
    totals = crud.get_category_totals(db)
    return {category: total for category, total in totals}

def correct_transaction(transaction_id: int, new_category: str, db: Session = Depends(get_db)): 
	transaction = db.query(models.Transaction).filter(models.Transaction.id == 
transaction_id).first()
	if not transaction: 
		return {"error": "Transaction not found"}

	transaction.category = new_category 
	db.commit()

	update_model(transaction.description, new_category)

	return {"message": f"AI learned that {transaction.description} is {new_category}!"}

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
