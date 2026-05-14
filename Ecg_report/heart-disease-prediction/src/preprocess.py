"""
src/preprocess.py
-----------------
Data loading, cleaning, and feature engineering for the heart-disease pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
TARGET_COL = "target"

FEATURE_INFO = {
    "age":      {"label": "Age (years)",               "type": "numeric"},
    "sex":      {"label": "Sex (1=Male, 0=Female)",    "type": "binary"},
    "cp":       {"label": "Chest Pain Type (0-3)",     "type": "categorical"},
    "trestbps": {"label": "Resting BP (mm Hg)",        "type": "numeric"},
    "chol":     {"label": "Serum Cholesterol (mg/dl)", "type": "numeric"},
    "fbs":      {"label": "Fasting BS > 120 mg/dl",   "type": "binary"},
    "restecg":  {"label": "Resting ECG Results (0-2)", "type": "categorical"},
    "thalach":  {"label": "Max Heart Rate Achieved",   "type": "numeric"},
    "exang":    {"label": "Exercise Induced Angina",   "type": "binary"},
    "oldpeak":  {"label": "ST Depression (oldpeak)",   "type": "numeric"},
    "slope":    {"label": "Slope of Peak ST (0-2)",    "type": "categorical"},
    "ca":       {"label": "Major Vessels (0-4)",       "type": "categorical"},
    "thal":     {"label": "Thalassemia Type (0-3)",    "type": "categorical"},
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV and perform sanity checks."""
    df = pd.read_csv(filepath)
    expected = set(FEATURE_COLS + [TARGET_COL])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and remove outliers."""
    df = df.copy()

    # Impute numeric missing values with median
    for col in FEATURE_COLS:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Remove extreme outliers using IQR for numeric columns
    numeric_cols = [c for c, v in FEATURE_INFO.items() if v["type"] == "numeric"]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df = df[(df[col] >= Q1) & (df[col] <= Q3)]

    return df.reset_index(drop=True)


def get_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Return scaled train/test splits + fitted scaler."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler
