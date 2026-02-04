"""
Utilities for computing and saving evaluation metrics.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> Dict[str, Any]:
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    # full text classification report
    report_text = classification_report(y_true, y_pred, zero_division=0)
    report["classification_report"] = report_text
    return report


def save_report(report: Dict[str, Any], filename: str = "metrics.json"):
    out_path = REPORT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path
