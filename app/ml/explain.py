import pickle
import pandas as pd
import shap
import numpy as np

# 1. Load the "Brain" (Model) and "Dictionary" (Vectorizer)
with open('app/ml/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('app/ml/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 2. Prepare a "Test" transaction
test_description = ["Starbucks Morning Coffee"]

# 3. Turn the words into numbers (Vectorization)
test_vec = vectorizer.transform(test_description).toarray()

# 4. Get the Prediction
prediction = model.predict(test_vec)[0]
probs = model.predict_proba(test_vec)

print(f"--- Analysis for: '{test_description[0]}' ---")
print(f"AI Prediction: {prediction}")

# 5. THE XAI PART: Calculate SHAP values
# This explains which words 'pushed' the model toward this choice
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_vec)

# Get the list of all words the AI knows
feature_names = vectorizer.get_feature_names_out()

# Find the index of the predicted category (e.g., is 'Dining' index 0 or 1?)
class_idx = list(model.classes_).index(prediction)

# Look at the impact of each word for THAT specific category
current_shap_values = shap_values[class_idx][0]

print("\nWhy did the AI choose this?")
for i, val in enumerate(current_shap_values):
    if val > 0: # If the word had a positive impact on this choice
        print(f"-> Word '{feature_names[i]}' added {val:.2f} to the confidence.")
