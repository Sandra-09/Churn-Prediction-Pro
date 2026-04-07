import joblib
import pandas as pd

from src.features.build_features import build_features

MODEL_PATH = "artifacts/model.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict(data: dict):

    model = load_model()

    df = pd.DataFrame([data])

    df = build_features(df)

    prediction = model.predict(df)

    return prediction