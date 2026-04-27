"""
train.py
Trains and compares 4 models using StratifiedKFold cross-validation.
Saves the best pipeline (preprocessor + model) to models/.
Run directly: python src/train.py
"""

import pandas as pd
import joblib
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).parent.parent))
from config import (PROCESSED_DIR, MODEL_DIR, RANDOM_STATE, CV_FOLDS)
from src.features import engineer_features, build_preprocessor


CANDIDATES = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
}


def load_training_data():
    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    X = engineer_features(df.drop("Loan_Status", axis=1))
    y = df["Loan_Status"]
    return X, y


def compare_models(X, y) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    results = {}

    print(f"\nComparing {len(CANDIDATES)} models — {CV_FOLDS}-fold CV\n")
    print(f"{'Model':<25} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 57)

    for name, model in CANDIDATES.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline([("pre", preprocessor), ("clf", model)])
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"]
        )
        results[name] = {
            "auc":       scores["test_roc_auc"].mean(),
            "accuracy":  scores["test_accuracy"].mean(),
            "precision": scores["test_precision"].mean(),
            "recall":    scores["test_recall"].mean(),
            "f1":        scores["test_f1"].mean(),
            "model":     model
        }
        r = results[name]
        print(f"{name:<25} {r['auc']:.3f}  {r['accuracy']:.3f}"
              f"  {r['precision']:.3f}  {r['recall']:.3f}  {r['f1']:.3f}")

    return results


def save_best_model(results: dict, X, y) -> None:
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]

    print(f"\nBest model: {best_name} (AUC {results[best_name]['auc']:.3f})")

    preprocessor = build_preprocessor()
    final_pipeline = Pipeline([("pre", preprocessor), ("clf", best_model)])
    final_pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_DIR / "best_pipeline.joblib")

    comparison = pd.DataFrame(results).T.drop(columns=["model"])
    comparison.to_csv(MODEL_DIR / "model_comparison.csv")

    print(f"Pipeline saved to {MODEL_DIR / 'best_pipeline.joblib'}")
    print("Model comparison saved to models/model_comparison.csv")


if __name__ == "__main__":
    X, y = load_training_data()
    print(f"Training data: {X.shape}, Class balance: {y.mean():.3f}")
    results = compare_models(X, y)
    save_best_model(results, X, y)