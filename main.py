import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

print("Churn model training started...")

# Sample dataset (temporary demo data)
data = pd.DataFrame({
    "recency": [10, 20, 5, 30, 2, 15, 40, 3],
    "frequency": [5, 2, 10, 1, 12, 3, 1, 15],
    "monetary": [200, 100, 500, 50, 700, 120, 30, 800],
    "churn": [0, 1, 0, 1, 0, 1, 1, 0]
})

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", accuracy)

joblib.dump(model, "churn_model.pkl")

print("Model saved successfully.")