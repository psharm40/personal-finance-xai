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

def update_model(new_description, new_category): 
	df = pd.read_csv("app/ml/transactions.csv"_

	# this adds the new "learned row" to the transactions.csv file
	new_row = pd.DataFrame([[new_description, new_category]], columns =['description', 
'category']) 
	df = pd.concat([df, new_row], ignore_index=True)

	# save our update csv file 
	df.to_csv("app/ml/transactions.csv", index=False) 
	
	vectorizer = TfidVectorizer() 
	X = vectorizer.fit_transform(df['description']
	y = df['category'] 
	model = RandomForestClassifier(n_estimator=50)
	model.fit(X, y) 

	with open('app/ml/model.pkl', 'wb') as f:
		pickle.dump(modedl, f)
	with open('app/ml/vectorizer.pkl', 'wb') as f:
		pickle.dump(vectorizer, f)


	
