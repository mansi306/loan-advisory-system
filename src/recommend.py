"""
recommend.py
Rule-based recommendation engine.
Takes SHAP factors and applicant data, returns ranked improvement actions.
Rule-based by design — recommendations must be auditable, not probabilistic.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


# ── Recommendation rules keyed by feature name ─────────
RULES = {
    "Credit_History": {
        "action": "Establish a credit history",
        "detail": (
            "No credit record is the single most impactful factor. "
            "A small loan or credit card repaid on time over 6-12 months "
            "can significantly improve approval chances."
        ),
        "impact": "High"
    },
    "Loan_Income_Ratio": {
        "action": "Reduce loan-to-income ratio",
        "detail_fn": lambda applicant: (
            f"Your loan-to-annual-income ratio is "
            f"{(applicant['LoanAmount'] * 1000) / ((applicant['ApplicantIncome'] + applicant['CoapplicantIncome']) * 12):.2f}x. "
            f"Try reducing loan amount or adding a co-applicant income."
        ),
        "impact": "High"
    },
    "Log_TotalIncome": {
        "action": "Increase total household income",
        "detail": (
            "Adding a co-applicant with steady income improves "
            "total income and directly strengthens your application."
        ),
        "impact": "Medium"
    },
    "Log_LoanAmount": {
        "action": "Consider a smaller loan amount",
        "detail": (
            "Reducing the requested loan amount improves your "
            "approval probability and lowers repayment risk."
        ),
        "impact": "Medium"
    },
    "Loan_Amount_Term": {
        "action": "Extend repayment term to 360 months",
        "detail": (
            "A longer repayment term reduces monthly EMI burden, "
            "making the loan more manageable relative to your income."
        ),
        "impact": "Medium"
    },
}

# Clean display names for frontend
DISPLAY_NAMES = {
    "num__Credit_History":      "Credit_History",
    "num__Loan_Income_Ratio":   "Loan_Income_Ratio",
    "num__Log_TotalIncome":     "Log_TotalIncome",
    "num__Log_LoanAmount":      "Log_LoanAmount",
    "num__Loan_Amount_Term":    "Loan_Amount_Term",
    "cat__Education_Graduate":  "Education",
    "cat__Married_Yes":         "Married",
    "cat__Gender_Male":         "Gender",
    "cat__Property_Area_Urban": "Property_Area",
}


def generate_recommendations(applicant: dict,
                              shap_factors: list) -> list[dict]:
    """
    Generate ranked improvement recommendations for rejected applicants.

    Args:
        applicant:    raw applicant dict
        shap_factors: list of (feature_name, shap_value) from predict.py

    Returns:
        List of recommendation dicts, ordered by impact (worst blocker first)
    """
    recs = []

    # Special case: no credit history is always flagged regardless of SHAP
    if applicant.get("Credit_History", 1) == 0:
        recs.append({
            "action":    RULES["Credit_History"]["action"],
            "detail":    RULES["Credit_History"]["detail"],
            "impact":    "High",
            "shap_value": -1.0,
            "feature":   "Credit_History"
        })

    for feature_name, shap_val in shap_factors:
        if shap_val >= 0:
            continue  # only address negative factors (blockers)

        # Map encoded name to base feature name
        base_name = DISPLAY_NAMES.get(feature_name, "")

        # Skip Credit_History — already handled above
        if base_name == "Credit_History":
            continue

        if base_name not in RULES:
            continue

        rule = RULES[base_name]

        # Use dynamic detail if available, else static
        if "detail_fn" in rule:
            try:
                detail = rule["detail_fn"](applicant)
            except Exception:
                detail = "Consider reducing your loan amount relative to income."
        else:
            detail = rule["detail"]

        recs.append({
            "action":    rule["action"],
            "detail":    detail,
            "impact":    rule["impact"],
            "shap_value": shap_val,
            "feature":   base_name
        })

    # Return top 3, already ordered by impact
    return recs[:3]


if __name__ == "__main__":
    from src.predict import predict_with_explanation

    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 3000,
        "CoapplicantIncome": 0,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 0,
        "Property_Area": "Urban"
    }

    result = predict_with_explanation(sample)
    recs   = generate_recommendations(sample, result["factors"])

    print(f"Probability: {result['probability']:.1%}")
    print(f"Decision:    {'APPROVED' if result['approved'] else 'REJECTED'}")
    print(f"\nRecommendations ({len(recs)} found):")

    if not recs:
        print("  No improvement actions needed.")
    else:
        for i, rec in enumerate(recs, 1):
            print(f"\n  {i}. [{rec['impact']} impact] {rec['action']}")
            print(f"     {rec['detail']}")