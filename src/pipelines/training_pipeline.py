from src.data.ingest import load_data
from src.models.train import train_model


def run_training_pipeline():

    print("Training pipeline started...")

    df = load_data("data/raw/telco_churn.csv")

    model = train_model(df)

    print("Training pipeline completed.")

    return model