"""
test_features.py
Basic tests for feature engineering functions.
Run: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.features import engineer_features, build_preprocessor


# ── Sample data for tests ──────────────────────────────
SAMPLE = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 2000,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}


def make_df(overrides=None):
    data = SAMPLE.copy()
    if overrides:
        data.update(overrides)
    return pd.DataFrame([data])


# ── Tests ──────────────────────────────────────────────

def test_loan_income_ratio_calculated():
    """Loan_Income_Ratio must equal (LoanAmount*1000) / (TotalIncome*12)."""
    df = engineer_features(make_df())
    expected = (150 * 1000) / ((5000 + 2000) * 12)
    assert abs(df["Loan_Income_Ratio"].iloc[0] - expected) < 1e-6


def test_log_transforms_non_negative():
    """Log transforms must never produce negative values."""
    df = engineer_features(make_df())
    assert (df["Log_LoanAmount"] >= 0).all()
    assert (df["Log_TotalIncome"] >= 0).all()


def test_no_missing_values_after_engineering():
    """Engineer features must produce zero missing values."""
    df = engineer_features(make_df())
    assert df.isnull().sum().sum() == 0


def test_zero_income_no_inf():
    """Zero total income must not produce inf in Loan_Income_Ratio."""
    df = engineer_features(make_df({
        "ApplicantIncome": 0,
        "CoapplicantIncome": 0
    }))
    assert not np.isinf(df["Loan_Income_Ratio"].iloc[0])


def test_preprocessor_output_shape():
    """Preprocessor must return 2D array with expected number of columns."""
    df = engineer_features(make_df())
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    assert X.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] > 5  # at least numeric + some OHE columns