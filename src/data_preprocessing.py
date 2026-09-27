"""
data_preprocessing.py
----------------------
Loads the raw e-commerce dataset and applies cleaning, feature engineering,
and encoding steps needed before modeling.

Usage:
    from src.data_preprocessing import load_and_clean_data, build_features

    df = load_and_clean_data("data/raw/E_Commerce_Dataset.xlsx")
    X, y, feature_names = build_features(df)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Churn"
ID_COL = "CustomerID"
CATEGORICAL_AS_OBJECT = ["Churn", "CityTier"]


def load_and_clean_data(path: str | Path, sheet_name: str = "E Comm") -> pd.DataFrame:
    """Load the raw Excel sheet and apply cleaning steps.

    Cleaning steps:
      1. Drop the CustomerID column (identifier, not predictive).
      2. Cast Churn/CityTier to categorical dtype for correct downstream handling.
      3. Impute missing numeric values with each column's own median
         (NOTE: the original exploratory notebook had a bug here -- it called
         ``df.fillna(df[i].median())`` which fills the *entire* dataframe with a
         single column's median. Fixed below to fill only the affected column).
      4. Cap outliers in numeric columns using the IQR method (winsorizing).
      5. Engineer an `Avg_cashbk_per_order` feature.

    Returns a cleaned DataFrame ready for feature building.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    for col in CATEGORICAL_AS_OBJECT:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # Impute missing numeric values with that column's own median (bug-fixed).
    for col in df.columns:
        if df[col].isnull().sum() > 0 and is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    # Winsorize numeric columns to reduce the influence of extreme outliers.
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            lower, upper = _iqr_bounds(df[col])
            df[col] = np.clip(df[col], lower, upper)

    # Feature engineering: average cashback earned per order.
    df["Avg_cashbk_per_order"] = df["CashbackAmount"] / df["OrderCount"].replace(0, np.nan)
    df["Avg_cashbk_per_order"] = df["Avg_cashbk_per_order"].fillna(0)

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def build_features(df: pd.DataFrame):
    """One-hot encode categoricals and scale numeric features.

    Returns (X, y, feature_names, scaler).
    """
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != TARGET_COL]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

    y = df_encoded[TARGET_COL]
    X = df_encoded.drop(columns=[TARGET_COL])

    return X, y, X.columns.tolist(), scaler
