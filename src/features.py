"""
features.py
Feature engineering + sklearn preprocessing pipeline.
The pipeline object must be saved alongside the model at training time
and reloaded at inference time — same transformations, always.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import CAT_FEATURES, NUM_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-driven features from raw columns.
    EDA finding: EMI_Ratio threshold at 2.0x is the key nonlinear signal.
    EDA finding: log transforms needed for right-skewed income and loan amount.
    """
    df = df.copy()

    # Total household income
    df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Loan-to-annual-income ratio
    # LoanAmount is in ₹000s, Total_Income is monthly — multiply by 12 for annual
    df["Loan_Income_Ratio"] = (df["LoanAmount"] * 1000) / (
    df["Total_Income"].replace(0, np.nan) * 12
)
    # Log transforms to handle right skew (EDA finding)
    df["Log_LoanAmount"]  = np.log1p(df["LoanAmount"])
    df["Log_TotalIncome"] = np.log1p(df["Total_Income"])

    # Drop raw columns that are now represented by engineered features
    df.drop(columns=["ApplicantIncome", "CoapplicantIncome",
                     "LoanAmount", "Total_Income"], inplace=True)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Returns an unfitted ColumnTransformer.
    Numeric features: StandardScaler
    Categorical features: OneHotEncoder
    Must be fit on training data only — never on the full dataset.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ), CAT_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    # Quick sanity check — run directly to verify output shape
    df_raw = pd.read_csv("data/processed/train.csv")

    print("Before engineering:", df_raw.shape)
    df_eng = engineer_features(df_raw.drop("Loan_Status", axis=1))
    print("After engineering: ", df_eng.shape)
    print("Columns:", list(df_eng.columns))
    print("Missing values:", df_eng.isnull().sum().sum())
    print("Loan_Income_Ratio sample:\n", df_eng["Loan_Income_Ratio"].describe().round(3))

    # Test the preprocessor fits without error
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(df_eng)
    print(f"\nTransformed shape: {X_transformed.shape}")
    print("Preprocessor working correctly.")