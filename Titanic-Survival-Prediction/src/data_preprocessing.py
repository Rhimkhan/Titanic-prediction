"""
data_preprocessing.py
=====================
Titanic Survival Prediction - Data Preprocessing & Feature Engineering
----------------------------------------------------------------------
Steps:
  1. Load train.csv and test.csv
  2. Handle missing values (Age, Embarked, Fare, Cabin)
  3. Feature engineering (Title, FamilySize, IsAlone, AgeGroup, FareGroup)
  4. Encode categorical variables
  5. Save processed datasets to data/processed/

Run from project root:
    python src/data_preprocessing.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
PROC_DIR   = DATA_DIR / "processed"
VIZ_DIR    = ROOT_DIR / "outputs" / "visualizations"

for d in [PROC_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data():
    train_path = DATA_DIR / "train.csv"
    test_path  = DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] '{train_path}' not found.\n"
            "Please download the Titanic dataset from:\n"
            "  https://www.kaggle.com/competitions/titanic/data\n"
            "and place train.csv and test.csv inside the 'data/' folder."
        )

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"[INFO] Loaded train: {train.shape} | test: {test.shape}")
    return train, test

# ─────────────────────────────────────────────
# 2. Exploratory Summaries
# ─────────────────────────────────────────────
def print_summary(df, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"  {name}  ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"{'='*50}")
    print(df.dtypes.to_string())
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None")
    else:
        for col, cnt in missing.items():
            print(f"  {col:20s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────
TITLE_MAP = {
    "Mr"      : "Mr",
    "Miss"    : "Miss",
    "Mrs"     : "Mrs",
    "Master"  : "Master",
    "Dr"      : "Rare",  "Rev"     : "Rare",
    "Col"     : "Rare",  "Major"   : "Rare",
    "Mlle"    : "Miss",  "Countess": "Rare",
    "Ms"      : "Miss",  "Lady"    : "Rare",
    "Jonkheer": "Rare",  "Don"     : "Rare",
    "Dona"    : "Rare",  "Capt"    : "Rare",
    "Sir"     : "Rare",
}

def extract_title(name: str) -> str:
    match = re.search(r',\s*([^.]+)\.', name)
    title = match.group(1).strip() if match else "Unknown"
    return TITLE_MAP.get(title, "Rare")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Title
    df["Title"] = df["Name"].apply(extract_title)

    # FamilySize & IsAlone
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # AgeGroup (bins after imputing Age)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        right=True
    )

    # FareGroup (quartile bins after imputing Fare)
    df["FareGroup"] = pd.qcut(
        df["Fare"].fillna(df["Fare"].median()),
        q=4,
        labels=["Low", "Mid", "High", "VeryHigh"],
        duplicates="drop"
    )

    return df

# ─────────────────────────────────────────────
# 4. Handle Missing Values
# ─────────────────────────────────────────────
def impute_age(df: pd.DataFrame, medians: dict = None) -> tuple:
    """
    Impute Age using median age grouped by (Title, Pclass).
    Returns imputed df and the median lookup dict.
    """
    if medians is None:
        medians = df.groupby(["Title", "Pclass"])["Age"].median().to_dict()

    def _fill(row):
        if pd.isnull(row["Age"]):
            key = (row["Title"], row["Pclass"])
            return medians.get(key, df["Age"].median())
        return row["Age"]

    df["Age"] = df.apply(_fill, axis=1)
    return df, medians

def handle_missing(train: pd.DataFrame, test: pd.DataFrame):
    """Full missing-value handling for both splits."""
    # ── Embarked: 2 missing in train ──
    mode_embarked = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(mode_embarked)
    test["Embarked"]  = test["Embarked"].fillna(mode_embarked)

    # ── Fare: 1 missing in test ──
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    test["Fare"]  = test["Fare"].fillna(train["Fare"].median())

    # ── Cabin: ~77 % missing — encode presence only ──
    train["HasCabin"] = train["Cabin"].notna().astype(int)
    test["HasCabin"]  = test["Cabin"].notna().astype(int)

    # ── Age: impute from (Title, Pclass) medians ──
    # Must extract Title before imputation
    train["Title"] = train["Name"].apply(extract_title)
    test["Title"]  = test["Name"].apply(extract_title)

    train, age_medians = impute_age(train)
    test,  _           = impute_age(test, age_medians)

    return train, test

# ─────────────────────────────────────────────
# 5. Encode Categorical Features
# ─────────────────────────────────────────────
TITLE_ENC   = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
AGE_GRP_ENC = {"Child": 0, "Teen": 1, "YoungAdult": 2, "Adult": 3, "Senior": 4}
FARE_GRP_ENC= {"Low": 0, "Mid": 1, "High": 2, "VeryHigh": 3}
SEX_ENC     = {"male": 0, "female": 1}
EMB_ENC     = {"S": 0, "C": 1, "Q": 2}

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "FamilySize", "IsAlone",
    "AgeGroup", "FareGroup", "HasCabin"
]
TARGET = "Survived"

def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sex"]      = df["Sex"].map(SEX_ENC)
    df["Embarked"] = df["Embarked"].map(EMB_ENC)
    df["Title"]    = df["Title"].map(TITLE_ENC).fillna(4)
    df["AgeGroup"] = df["AgeGroup"].map(AGE_GRP_ENC)
    df["FareGroup"]= df["FareGroup"].map(FARE_GRP_ENC)
    return df

# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
def plot_eda(train_raw: pd.DataFrame):
    sns.set_theme(style="darkgrid", palette="muted")

    # Survival rate by key features
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Titanic — EDA Overview", fontsize=16, fontweight="bold")

    cols = ["Pclass", "Sex", "Embarked", "FamilySize", "Title", "HasCabin"]
    for ax, col in zip(axes.flat, cols):
        if col in train_raw.columns:
            survival = train_raw.groupby(col)["Survived"].mean().sort_values()
            survival.plot(kind="bar", ax=ax, color=sns.color_palette("muted"), edgecolor="white")
            ax.set_title(f"Survival Rate by {col}")
            ax.set_ylabel("Survival Rate")
            ax.set_xlabel(col)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = VIZ_DIR / "eda_survival_by_feature.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

    # Age distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    for survived, grp in train_raw.groupby("Survived"):
        label = "Survived" if survived else "Did Not Survive"
        ax.hist(grp["Age"].dropna(), bins=30, alpha=0.6, label=label)
    ax.set_title("Age Distribution by Survival")
    ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.legend()
    path = VIZ_DIR / "eda_age_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

    # Correlation heatmap (numeric only)
    num_cols = train_raw.select_dtypes(include=np.number).columns.tolist()
    corr = train_raw[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    path = VIZ_DIR / "eda_correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[VIZ] Saved -> {path}")

# ─────────────────────────────────────────────
# 7. Main Pipeline
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  TITANIC — DATA PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)

    # Load
    train, test = load_data()
    print_summary(train, "Train (raw)")
    print_summary(test,  "Test  (raw)")

    # Handle missing values (also creates Title column)
    print("\n[STEP 1] Handling missing values...")
    train, test = handle_missing(train, test)

    # Feature engineering
    print("[STEP 2] Engineering features...")
    train = engineer_features(train)
    test  = engineer_features(test)

    # EDA plots (using post-imputation data but before encoding)
    print("[STEP 3] Generating EDA visualisations...")
    plot_eda(train)

    # Encode
    print("[STEP 4] Encoding categorical features...")
    train = encode(train)
    test  = encode(test)

    # Select features
    X_train = train[FEATURES].copy()
    y_train = train[TARGET].copy()
    X_test  = test[FEATURES].copy()
    test_ids = test["PassengerId"]

    # Sanity check
    print(f"\n[INFO] X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"[INFO] X_test : {X_test.shape}")
    null_train = X_train.isnull().sum().sum()
    null_test  = X_test.isnull().sum().sum()
    print(f"[INFO] Remaining nulls — train: {null_train} | test: {null_test}")

    # Save processed data
    X_train.to_csv(PROC_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROC_DIR / "y_train.csv", index=False)
    X_test.to_csv(PROC_DIR / "X_test.csv",  index=False)
    test_ids.to_csv(PROC_DIR / "test_ids.csv", index=False)

    print(f"\n[DONE] Processed data saved to: {PROC_DIR}")
    print("       Run next: python src/model_training.py\n")

if __name__ == "__main__":
    main()
