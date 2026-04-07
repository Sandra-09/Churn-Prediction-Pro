import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Telco churn data into model-ready features
    """
    df = df.copy()
    df=df.drop(["Churn"], axis=1, errors="ignore")
    
    cat_df=df.select_dtypes(include=["object", "category"])
    num_df=df.select_dtypes(include=["int64", "float64"])
    
    trans=ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_df.columns),
            ("scaler", StandardScaler(), num_df.columns)
        ]
    )
    df=trans.fit_transform(df)
    
    return df