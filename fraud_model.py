import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Sample dataset (you can expand this)
data = {
    "message": [
        "You have won $1000 lottery, click here to claim",   # Fraud
        "Update your bank details immediately or account blocked", # Fraud
        "Meeting at 10 AM tomorrow, please join",            # Safe
        "Your package will be delivered today",              # Safe
        "Congratulations! You won a free vacation",          # Fraud
        "Project deadline is extended to next week"          # Safe
    ],
    "label": ["fraud", "fraud", "safe", "safe", "fraud", "safe"]
}

df = pd.DataFrame(data)

# Train model
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

print("Model trained. Accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "fraud_detector.pkl")