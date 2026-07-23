"""
model_training.py
=================
Titanic Survival Prediction — Model Training & Evaluation
----------------------------------------------------------
Models:
  • Logistic Regression  (baseline)
  • Random Forest        (ensemble)
  • XGBoost              (gradient boosting)

Workflow:
  1. Load processed data from data/processed/
  2. Scale features
  3. Hyperparameter tune each model with GridSearchCV (5-fold CV)
  4. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
  5. Save best models to outputs/models/
  6. Generate: confusion matrices, ROC curves, feature importance

Run from project root:
    python src/model_training.py
"""

import warnings
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from pathlib import Path
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
VIZ_DIR   = ROOT_DIR / "outputs" / "visualizations"

for d in [MODEL_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load Processed Data
# ─────────────────────────────────────────────
def load_processed():
    X = pd.read_csv(PROC_DIR / "X_train.csv")
    y = pd.read_csv(PROC_DIR / "y_train.csv").squeeze()
    print(f"[INFO] Loaded X: {X.shape}  y: {y.shape}")
    print(f"[INFO] Class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}")
    return X, y

# ─────────────────────────────────────────────
# 2. Scale Features
# ─────────────────────────────────────────────
def scale_features(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns)
    # Save scaler for prediction step
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[INFO] Scaler saved.")
    return X_scaled, scaler

# ─────────────────────────────────────────────
# 3. Model Definitions & Parameter Grids
# ─────────────────────────────────────────────
def get_models_and_params():
    models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(
                random_state=42, max_iter=1000, solver="lbfgs"
            ),
            "params": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l2"],
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth"   : [None, 4, 6, 8],
                "min_samples_split": [2, 5],
                "min_samples_leaf" : [1, 2],
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=42, eval_metric="logloss",
                verbosity=0, use_label_encoder=False
            ),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth"   : [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample"   : [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
        }
    }
    return models

# ─────────────────────────────────────────────
# 4. GridSearchCV Tuning
# ─────────────────────────────────────────────
def tune_model(name, config, X, y, cv=5):
    print(f"\n[TUNE] {name} — GridSearchCV (cv={cv}) ...")
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator  = config["estimator"],
        param_grid = config["params"],
        cv         = cv_strategy,
        scoring    = "accuracy",
        n_jobs     = -1,
        verbose    = 0
    )
    gs.fit(X, y)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV acc : {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_

# ─────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────
def evaluate_model(name, model, X, y):
    """
    Cross-validated metrics using 5-fold stratified CV.
    Returns a dict of metric -> mean value.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {}
    for metric_name, scorer in [
        ("accuracy",  "accuracy"),
        ("precision", "precision"),
        ("recall",    "recall"),
        ("f1",        "f1"),
        ("roc_auc",   "roc_auc"),
    ]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        metrics[metric_name] = scores.mean()

    print(f"\n[EVAL] {name}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    return metrics

# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
COLORS = ["#4C72B0", "#DD8452", "#55A868"]

def plot_confusion_matrices(models_dict, X, y):
    """Plot confusion matrices for all models side-by-side."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model), color in zip(axes, models_dict.items(), COLORS):
        # Use a single fold for the confusion matrix display
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(cv.split(X, y))
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = m.predict(X.iloc[val_idx])

        cm = confusion_matrix(y.iloc[val_idx], y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Survived", "Survived"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy: {accuracy_score(y.iloc[val_idx], y_pred):.3f}",
                     fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices (Validation Fold)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = VIZ_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

def plot_roc_curves(models_dict, X, y):
    fig, ax = plt.subplots(figsize=(9, 7))
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(cv.split(X, y))

    for (name, model), color in zip(models_dict.items(), COLORS):
        m = clone(model)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = m.predict_proba(X.iloc[val_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y.iloc[val_idx], y_prob)
        auc = roc_auc_score(y.iloc[val_idx], y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    path = VIZ_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

def plot_feature_importance(models_dict, feature_names):
    """Feature importances for RF and XGBoost."""
    tree_models = {k: v for k, v in models_dict.items()
                   if hasattr(v, "feature_importances_")}
    if not tree_models:
        return

    n = len(tree_models)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"{name} — Feature Importance", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    path = VIZ_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

def plot_metrics_comparison(results: dict):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(results.keys())
    values = {m: [results[model][m] for model in model_names] for m in metrics}

    x = np.arange(len(model_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, vals) in enumerate(values.items()):
        ax.bar(x + i * width, vals, width, label=metric.capitalize(),
               color=sns.color_palette("muted")[i])

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = VIZ_DIR / "metrics_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

# ─────────────────────────────────────────────
# 7. Save Models & Results
# ─────────────────────────────────────────────
def save_artifacts(models_dict, results, best_params):
    # Save individual models
    for name, model in models_dict.items():
        path = MODEL_DIR / f"{name.lower()}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[SAVE] Model -> {path}")

    # Save best model (by ROC-AUC)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models_dict[best_name]
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[BEST] '{best_name}' selected as best model (ROC-AUC: {results[best_name]['roc_auc']:.4f})")
    print(f"[SAVE] Best model -> {MODEL_DIR / 'best_model.pkl'}")

    # Save results summary
    summary = {
        "best_model": best_name,
        "results": results,
        "best_params": best_params,
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Results -> {MODEL_DIR / 'training_results.json'}")
    return best_name

# ─────────────────────────────────────────────
# 8. Main Pipeline
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  TITANIC — MODEL TRAINING & HYPERPARAMETER TUNING")
    print("="*60)

    # Load data
    X, y = load_processed()

    # Scale
    print("\n[STEP 1] Scaling features...")
    X_scaled, _ = scale_features(X)

    # Models config
    models_config = get_models_and_params()
    best_models   = {}
    all_params    = {}
    results       = {}

    # Tune each model
    print("\n[STEP 2] Hyperparameter tuning (this may take a few minutes)...")
    for name, config in models_config.items():
        model, params, _ = tune_model(name, config, X_scaled, y)
        best_models[name] = model
        all_params[name]  = params

    # Evaluate
    print("\n[STEP 3] Cross-validated evaluation...")
    for name, model in best_models.items():
        results[name] = evaluate_model(name, model, X_scaled, y)

    # Re-fit on full training data (so models are ready for prediction)
    print("\n[STEP 4] Fitting final models on full training set...")
    for name, model in best_models.items():
        model.fit(X_scaled, y)
        print(f"  {name} fitted OK")

    # Visualisations
    print("\n[STEP 5] Generating visualisations...")
    plot_confusion_matrices(best_models, X_scaled, y)
    plot_roc_curves(best_models, X_scaled, y)
    plot_feature_importance(best_models, X.columns.tolist())
    plot_metrics_comparison(results)

    # Save artifacts
    print("\n[STEP 6] Saving models & results...")
    best_name = save_artifacts(best_models, results, all_params)

    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE — Best Model: {best_name}")
    print("  Run next: python src/predict.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
