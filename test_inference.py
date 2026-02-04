"""
Tests for model training and inference.
- Trains a quick model (if not present) and asserts a prediction shape.
- Keep tests lightweight by training only when necessary.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.predict import get_model, predict_from_dict
from src.models.train import train_and_save_model
from src.features.build_features import build_processed
from src.models.predict import MODEL_PATH

def test_train_if_missing():
    # Ensure processed data exists
    build_processed(save=True)
    # Train if model not exists
    if not Path(MODEL_PATH).exists():
        train_and_save_model()
    assert Path(MODEL_PATH).exists()

def test_predict_single():
    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    res = predict_from_dict(sample)
    assert "prediction" in res and "label" in res and "probabilities" in res
    assert isinstance(res["probabilities"], list)
    assert len(res["probabilities"]) == 3  # 3 classes
