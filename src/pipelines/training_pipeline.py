import pandas as pd
from src.models.train import train_model


def run_training_pipeline():

    print("Training pipeline started...")

    # Temporary demo dataset
    data = pd.DataFrame({
        "recency": [10, 20, 5, 30, 2, 15, 40, 3],
        "frequency": [5, 2, 10, 1, 12, 3, 1, 15],
        "monetary": [200, 100, 500, 50, 700, 120, 30, 800],
        "churn": [0, 1, 0, 1, 0, 1, 1, 0]
    })

    model = train_model(data)

    print("Training pipeline completed.")

    return model