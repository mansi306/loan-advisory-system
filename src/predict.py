"""
predict.py
Loads saved pipeline, runs inference, generates SHAP explanations.
Called by the Streamlit app — never contains training logic.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_DIR, THRESHOLD
from src.features import engineer_features


# ── Load artifacts once at module import ───────────────
pipeline    = joblib.load(MODEL_DIR / "best_pipeline.joblib")
model       = pipeline.named_steps["clf"]
preprocessor = pipeline.named_steps["pre"]
explainer   = shap.TreeExplainer(model)


def predict_with_explanation(applicant: dict) -> dict:
    """
    Generate approval prediction + SHAP-based explanation.

    Args:
        applicant: raw applicant features as dict

    Returns:
        {
          probability: float,
          approved: bool,
          factors: [(feature_name, shap_value), ...] top 6 by magnitude
        }
    """
    df = engineer_features(pd.DataFrame([applicant]))
    X  = preprocessor.transform(df)

    prob = float(model.predict_proba(X)[0][1])

    # shap_values returns list [class_0, class_1] for tree classifiers
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        shap_vals = np.array(sv[1][0])
    else:
        shap_vals = np.array(sv[0])
    shap_vals = shap_vals.flatten()

    feat_names = preprocessor.get_feature_names_out()

    # Sort by absolute magnitude — biggest impact first
    factors = sorted(
        zip(feat_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:6]

    return {
        "probability": round(prob, 3),
        "approved":    prob >= THRESHOLD,
        "factors":     [(name, round(float(val), 4)) for name, val in factors]
    }


if __name__ == "__main__":
    # Test with a sample rejected applicant
    sample_rejected = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 3000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 0,       # no credit history — expect rejection
        "Property_Area": "Urban"
    }

    sample_approved = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 8000,
        "CoapplicantIncome": 3000,
        "LoanAmount": 100,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,       # has credit history — expect approval
        "Property_Area": "Urban"
    }

    print("=== REJECTED APPLICANT ===")
    r1 = predict_with_explanation(sample_rejected)
    print(f"Probability: {r1['probability']:.1%}")
    print(f"Decision:    {'APPROVED' if r1['approved'] else 'REJECTED'}")
    print("Top factors:")
    for name, val in r1["factors"]:
        direction = "hurts" if val < 0 else "helps"
        print(f"  {name:<35} {val:+.4f}  ({direction})")

    print("\n=== APPROVED APPLICANT ===")
    r2 = predict_with_explanation(sample_approved)
    print(f"Probability: {r2['probability']:.1%}")
    print(f"Decision:    {'APPROVED' if r2['approved'] else 'REJECTED'}")
    print("Top factors:")
    for name, val in r2["factors"]:
        direction = "hurts" if val < 0 else "helps"
        print(f"  {name:<35} {val:+.4f}  ({direction})")