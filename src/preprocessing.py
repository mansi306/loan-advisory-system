"""
preprocessing.py
Loads raw data, cleans it, and saves train/test splits to data/processed/.
Run directly: python src/preprocessing.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Allow imports from project root
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_PATH, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and handle missing values."""
    df = pd.read_csv(path)

    # Drop ID column — carries no predictive signal
    df.drop(columns=["Loan_ID"], inplace=True)

    # Median impute numeric columns
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Mode impute categorical columns (exclude target)
    cat_cols = df.select_dtypes(include="str").columns.drop("Loan_Status")
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Encode target: Y → 1, N → 0
    df["Loan_Status"] = (df["Loan_Status"] == "Y").astype(int)

    return df


def split_and_save(df: pd.DataFrame) -> None:
    """Stratified train/test split, saved to data/processed/."""
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,           # preserves 69/31 class ratio in both splits
        random_state=RANDOM_STATE
    )

    # Save splits
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(
        PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(
        PROCESSED_DIR / "test.csv", index=False)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train approval rate: {y_train.mean():.3f}")
    print(f"Test approval rate:  {y_test.mean():.3f}")
    print(f"Saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    df = load_and_clean(RAW_DATA_PATH)

    print("After cleaning:")
    print(f"  Shape: {df.shape}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Class balance: {df['Loan_Status'].value_counts(normalize=True).to_dict()}")

    split_and_save(df)