from fastapi import FastAPI
from src.models.predict import predict

app = FastAPI(title="Churn Prediction API")


@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}


@app.post("/predict")
def predict_churn(customer: dict):

    prediction = predict(customer)

    if prediction == 1:
        result = "Customer will churn"
    else:
        result = "Customer will stay"

    return {"prediction": result}