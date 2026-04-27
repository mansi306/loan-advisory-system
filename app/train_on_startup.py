import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def ensure_model_exists():
    from config import MODEL_DIR, PROCESSED_DIR, RAW_DATA_PATH
    
    model_path = MODEL_DIR / "best_pipeline.joblib"
    if model_path.exists():
        print("Model already exists, skipping training.")
        return

    print("Training model for first time...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    from src.preprocessing import load_and_clean, split_and_save
    df = load_and_clean(RAW_DATA_PATH)
    split_and_save(df)

    from src.train import load_training_data, compare_models, save_best_model
    X, y = load_training_data()
    results = compare_models(X, y)
    save_best_model(results, X, y)
    print("Model trained and saved successfully.")