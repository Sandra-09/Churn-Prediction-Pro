import pandas as pd


def load_data(path: "str"):

    print("Loading dataset...")

    df = pd.read_csv("data\\raw\\Telco-Customer-Churn.csv")

    print("Dataset shape:", df.shape)

    return df