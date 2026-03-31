import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib


def train_model(data: pd.DataFrame):

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

    return model