"""
Train script for the Iris classification model.
Produces a serialized sklearn pipeline at models/iris_pipeline.joblib
and a small metadata JSON describing the model and training metrics.
"""
import logging
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.models.metrics import classification_metrics, save_report
from src.features.build_features import build_processed

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "iris_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

SEED = 42
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    df = build_processed(save=True)
    return df


def train_and_save_model(random_seed: int = SEED) -> None:
    df = load_data()
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].values
    y = df["target"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_seed
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=random_seed))
    ])

    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 3, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    logger.info("Starting GridSearchCV with param_grid=%s", param_grid)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    logger.info("Best params: %s", gs.best_params_)
    y_pred = best.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    logger.info("Test accuracy: %.4f", accuracy)

    # Save pipeline
    joblib.dump(best, MODEL_PATH)
    logger.info("Saved model pipeline to %s", MODEL_PATH)

    # Save metadata and metrics
    metrics = classification_metrics(y_test, y_pred)
    metadata = {
        "model_path": str(MODEL_PATH),
        "best_params": gs.best_params_,
        "test_accuracy": accuracy,
        "metrics": metrics
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", METADATA_PATH)

    # Save a readable metrics file as well
    save_report(metrics, filename="metrics.json")


if __name__ == "__main__":
    train_and_save_model()
