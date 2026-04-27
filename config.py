from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
RAW_DATA_PATH   = BASE_DIR / "data/raw/loan_data.csv"
PROCESSED_DIR   = BASE_DIR / "data/processed"
MODEL_DIR       = BASE_DIR / "models"

# ── Model settings ─────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5
THRESHOLD       = 0.5

# ── Features ───────────────────────────────────────────
CAT_FEATURES = [
    "Gender", "Married", "Education",
    "Self_Employed", "Property_Area", "Dependents"
]
NUM_FEATURES = [
    "Log_TotalIncome", "Log_LoanAmount",
    "Loan_Income_Ratio", "Loan_Amount_Term", "Credit_History"
]