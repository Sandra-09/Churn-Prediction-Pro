import pandas as pd
from src.data.ingest import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
from src.features.build_features import build_features


ARTIFACT_PATH = "artifacts/model.pkl"

def train_model(data: pd.DataFrame):
    
    data = build_features(data)
    
    df = load_data("data/raw/telco_churn.csv")
    df=df.copy()
    df["Churn"] = LabelEncoder().fit_transform(df["Churn"])
    

    X = data
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(eval_metric="logloss")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    print("Model Accuracy:", accuracy)

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, ARTIFACT_PATH)

    print(f"Model saved to {ARTIFACT_PATH}")

    return model