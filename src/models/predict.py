import joblib
import pandas as pd

MODEL_PATH = "artifacts/model.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict(customer_data):

    model = load_model()

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)

    return prediction[0]