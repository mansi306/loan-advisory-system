# 🏦 Loan Advisory System

AI-powered loan approval prediction with explainable decisions and personalized improvement guidance.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.51-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

---

## The Problem

Every year, millions of loan applications are rejected by banks and financial institutions — often without a clear explanation. Applicants know they were rejected, but not why, and not what to do next.

This isn't just a personal frustration. A credit officer reviewing hundreds of applications has neither the time nor the obligation to explain each rejection in actionable terms. First-time borrowers, self-employed individuals, and applicants from semi-urban and rural areas are left guessing.

> The question isn't just "will this loan be approved?" — it's "what would need to change for it to be approved?"

## What This System Does Differently

This project builds a full decision-support system, not just a prediction model.

| Feature | Description |
|---|---|
| Approval prediction | Random Forest model, AUC 0.749, 5-fold cross-validation |
| SHAP explainability | Per-prediction explanation, not just global feature importance |
| Recommendations | Ranked, specific improvement actions for rejected applicants |
| Fairness awareness | Geography and gender gaps analysed and documented |

---

## Key EDA Findings

1. **Credit history is a gate, not a factor** — applicants with credit history are approved at 78% vs 8% without. A 70-point gap that dwarfs every other variable.
2. **Income alone is a weak predictor** — a low-income applicant with credit history is approved more often than a high-income applicant without one.
3. **Loan-to-income ratio has a threshold effect** — approval drops sharply above 2.0x annual income, motivating the `Loan_Income_Ratio` engineered feature.
4. **Geography proxies income** — the rural approval gap shrinks significantly when controlling for income bracket.
5. **Class imbalance (69/31)** — motivated use of StratifiedKFold and AUC as primary metric over accuracy.

---

## Model Results

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.727 | 0.796 | 0.783 | 0.976 | 0.868 |
| Decision Tree | 0.638 | 0.688 | 0.773 | 0.775 | 0.772 |
| **Random Forest** | **0.749** | **0.766** | **0.780** | **0.920** | **0.843** |
| XGBoost | 0.714 | 0.737 | 0.784 | 0.855 | 0.817 |

**Selected: Random Forest** — highest AUC, best recall, compatible with SHAP TreeExplainer.

Decision Tree excluded due to clear overfitting signal (lowest AUC at 0.638).

---

## Project Structure
loan-advisory-system/
├── data/
│   ├── raw/              # original CSV (not tracked in git)
│   └── processed/        # train/test splits (not tracked in git)
├── notebooks/
│   └── 01_EDA.ipynb      # exploratory analysis with business insights
├── src/
│   ├── preprocessing.py  # cleaning and stratified splitting
│   ├── features.py       # feature engineering + sklearn pipeline
│   ├── train.py          # model comparison and selection
│   ├── predict.py        # SHAP-based inference
│   └── recommend.py      # rule-based recommendation engine
├── models/               # saved joblib artifacts (not tracked in git)
├── tests/
│   └── test_features.py  # 5 unit tests, all passing
├── app/
│   └── streamlit_app.py  # UI layer only, no model logic
├── config.py             # all constants and paths
└── requirements.txt

---

## How to Run Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/mansi306/loan-advisory-system.git
cd loan-advisory-system
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Add dataset
# Download loan_data.csv from Kaggle and place in data/raw/

# 3. Run pipeline
python src/preprocessing.py
python src/train.py

# 4. Launch app
streamlit run app/streamlit_app.py
```

---

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Streamlit · Plotly · Fairlearn
