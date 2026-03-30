import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    # Load the training data
    df = pd.read_csv("app/ml/transactions.csv")
    
    # Vectorize the descriptions (Math Minor Flex: TF-IDF)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['description'])
    y = df['category']
    
    # Train the Random Forest
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    
    # Save the 'Brain' files
    with open('app/ml/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('app/ml/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

def update_model(new_description, new_category):
    # 1. Load existing data
    df = pd.read_csv("app/ml/transactions.csv")
    
    # 2. Add the new "learned" row (The Self-Learning part)
    new_row = pd.DataFrame([[new_description, new_category]], columns=['description', 'category'])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # 3. Save it back to the source
    df.to_csv("app/ml/transactions.csv", index=False)
    
    # 4. Retrain so the app gets smarter immediately
    train_model()

if __name__ == "__main__":
    train_model()
