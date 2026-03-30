This is a full-stack financial management tool that uses explainable AI to categorize transactions. This system provides human-readable transparency into how the AI reaches its conclusions.

Live Demo: https://personal-finance-xai.onrender.com/

Math behind the AI: 
- TF-IDF Vectorization: Converts transaction text into weighted numerical vectors, priortizing unique names over generic words. For example, if every sentence had "the", then "the" wouldn't be so valuable. But a word like "Starbucks" holds more values because it only appears in a few labels.
- Random Forest Classifier: A learning method that uses 50+ decision trees to "vote" on the most likely spending category.
- SHAP (Explainability): Integrate Shapley values to attribute exactly which words influenced the AI's classification.

Tech Stack 
- Backend: FastAPI (Python 3.9) - REST API for high-performance data handling.
- Database: SQLAIchemy + SQLite - relational mapping for transaction history and AI metadata.
- DevOps: Render (CI/CD) - Automated deployment pipeline linked directly to GitHub for live demo of the project.
- Frontend: JavaScript, CSS3, HTML5 - For the project dashboard (which is pretty simple as of right now)

Key Features: 
- HSelf-Learning: Users can correct the AI misclassification and the corrections made by user get updates in the database so that hte history is correct. 
