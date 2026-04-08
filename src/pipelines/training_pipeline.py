import mlflow
import mlflow.sklearn

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.data.ingest import load_data


def run_training_pipeline():

    print("Training pipeline started...")

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run():

        df = load_data("data/raw/telco_churn.csv")

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna()

        X = df.drop(columns=["customerID", "Churn"])
        y = df["Churn"].map({"No": 0, "Yes": 1})
        
        categorical_features = X.select_dtypes(include=["object"]).columns
        numeric_features = X.select_dtypes(exclude=["object"]).columns
        
        preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),])
     

        model = XGBClassifier()

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)

        print("Model Accuracy:", acc)

        # log metric
        mlflow.log_metric("accuracy", acc)

        # log model
        mlflow.sklearn.log_model(pipeline, "model")

        joblib.dump(pipeline, "artifacts/model.pkl")

    print("Training pipeline completed.")