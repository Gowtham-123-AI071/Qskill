"""
Validation utilities for raw and processed data.
Simple checks that raise informative exceptions when things are wrong.
"""
from typing import List
import pandas as pd


class ValidationError(Exception):
    pass


def assert_columns_present(df: pd.DataFrame, expected: List[str]):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing expected columns: {missing}")


def assert_no_nulls(df: pd.DataFrame):
    total_nulls = int(df.isnull().sum().sum())
    if total_nulls > 0:
        raise ValidationError(f"Dataset contains {total_nulls} null values")


def assert_numeric_columns_non_negative(df: pd.DataFrame, numeric_cols: List[str]):
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValidationError(f"Column '{col}' contains negative values")


def validate_raw_iris(df: pd.DataFrame):
    expected_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"]
    # also accept alternative names commonly used
    alt_cols = {
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)",
    }
    # Basic checks
    if df.shape[1] < 5:
        raise ValidationError("Raw iris should have at least 5 columns (4 features + target)")

    # No nulls
    assert_no_nulls(df)
