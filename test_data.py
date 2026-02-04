"""
Tests for data creation and processing.
These are simple checks to ensure the pipeline produces expected artifacts.
Run with: pytest -q
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.make_dataset import save_raw
from src.features.build_features import build_processed, PROCESSED_PATH


def test_raw_created(tmp_path):
    # Create raw dataset (idempotent)
    p = save_raw(force=True)  # ensure raw exists
    assert p.exists()
    df = pd.read_csv(p)
    assert df.shape[0] == 150  # Iris dataset has 150 rows
    assert "target" in df.columns


def test_processed_has_no_nulls():
    df = build_processed(save=True)
    assert df.isnull().sum().sum() == 0
    # expected columns
    expected = {"sepal_length", "sepal_width", "petal_length", "petal_width", "target"}
    assert expected.issubset(set(df.columns))
