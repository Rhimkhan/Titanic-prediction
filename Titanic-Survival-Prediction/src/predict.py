"""
predict.py
==========
Titanic Survival Prediction — Generate Predictions & Submission
---------------------------------------------------------------
Loads:
  • Best model from outputs/models/best_model.pkl
  • Scaler    from outputs/models/scaler.pkl
  • Test data from data/processed/X_test.csv
  • PassengerIds from data/processed/test_ids.csv

Outputs:
  • outputs/submission.csv  <- ready for Kaggle upload

Run from project root:
    python src/predict.py
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT_DIR / "data" / "processed"
MODEL_DIR  = ROOT_DIR / "outputs" / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
VIZ_DIR    = OUTPUT_DIR / "visualizations"

# ─────────────────────────────────────────────
# 1. Load Artifacts
# ─────────────────────────────────────────────
def load_artifacts():
    model_path  = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    for p in [model_path, scaler_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"\n[ERROR] '{p}' not found.\n"
                "Please run: python src/model_training.py first."
            )

    with open(model_path,  "rb") as f: model  = pickle.load(f)
    with open(scaler_path, "rb") as f: scaler = pickle.load(f)

    print(f"[INFO] Loaded model  : {type(model).__name__}")
    print(f"[INFO] Loaded scaler : {type(scaler).__name__}")
    return model, scaler

# ─────────────────────────────────────────────
# 2. Load Test Data
# ─────────────────────────────────────────────
def load_test_data():
    X_test   = pd.read_csv(PROC_DIR / "X_test.csv")
    test_ids = pd.read_csv(PROC_DIR / "test_ids.csv").squeeze()
    print(f"[INFO] X_test: {X_test.shape}  |  PassengerIds: {len(test_ids)}")
    return X_test, test_ids

# ─────────────────────────────────────────────
# 3. Predict
# ─────────────────────────────────────────────
def predict(model, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    y_pred   = model.predict(X_scaled)
    y_prob   = model.predict_proba(X_scaled)[:, 1]
    return y_pred, y_prob

# ─────────────────────────────────────────────
# 4. Generate Submission CSV
# ─────────────────────────────────────────────
def save_submission(test_ids, y_pred):
    submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived"   : y_pred.astype(int)
    })
    path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(path, index=False)
    print(f"\n[DONE] Submission saved -> {path}")
    print(f"       Shape: {submission.shape}")
    print(f"       Predicted survivors: {y_pred.sum()} / {len(y_pred)}")
    return submission

# ─────────────────────────────────────────────
# 5. Prediction Summary Plots
# ─────────────────────────────────────────────
def plot_prediction_summary(y_pred, y_prob, test_ids):
    sns.set_theme(style="darkgrid", palette="muted")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Prediction Summary", fontsize=14, fontweight="bold")

    # Pie chart — survival breakdown
    labels = ["Did Not Survive", "Survived"]
    counts = [(y_pred == 0).sum(), (y_pred == 1).sum()]
    colors = ["#DD8452", "#4C72B0"]
    axes[0].pie(counts, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Survival Distribution")

    # Probability histogram
    axes[1].hist(y_prob, bins=25, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold = 0.5")
    axes[1].set_title("Predicted Survival Probability")
    axes[1].set_xlabel("P(Survived)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Confidence: high vs low confidence predictions
    high_conf = ((y_prob >= 0.8) | (y_prob <= 0.2)).sum()
    low_conf  = len(y_prob) - high_conf
    axes[2].bar(["High Confidence\n(prob ≤0.2 or ≥0.8)", "Low Confidence\n(0.2–0.8)"],
                [high_conf, low_conf],
                color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=1.5)
    axes[2].set_title("Prediction Confidence")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    path = VIZ_DIR / "prediction_summary.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  TITANIC — GENERATING PREDICTIONS")
    print("="*60)

    # Load model & scaler
    print("\n[STEP 1] Loading model artifacts...")
    model, scaler = load_artifacts()

    # Load test data
    print("\n[STEP 2] Loading test data...")
    X_test, test_ids = load_test_data()

    # Handle any residual nulls (safety net)
    null_count = X_test.isnull().sum().sum()
    if null_count > 0:
        print(f"[WARN] {null_count} null(s) in X_test — filling with column medians")
        X_test = X_test.fillna(X_test.median())

    # Predict
    print("\n[STEP 3] Running predictions...")
    y_pred, y_prob = predict(model, scaler, X_test)
    print(f"  Predictions: {y_pred.shape[0]} passengers")
    print(f"  Survival rate: {y_pred.mean()*100:.1f}%")

    # Save submission
    print("\n[STEP 4] Saving submission...")
    submission = save_submission(test_ids, y_pred)

    # Plots
    print("\n[STEP 5] Generating prediction visualisations...")
    plot_prediction_summary(y_pred, y_prob, test_ids)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print(f"  Upload outputs/submission.csv to Kaggle.")
    print("  Kaggle URL: https://www.kaggle.com/competitions/titanic")
    print("="*60 + "\n")

    # Print a sample of the submission
    print("--- Sample Submission (first 10 rows) ---")
    print(submission.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
