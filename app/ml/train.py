import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("app/ml/transactions.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['description'])
y = df['category']

model = RandomForestClassifier(n_estimators=50)
model.fit(X, y)

with open('app/ml/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('app/ml/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
