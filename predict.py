"""
Prediction utility that loads the trained pipeline and performs predictions.
Exposes helpful wrappers for programmatic usage and for API server.
"""
from pathlib import Path
import joblib
import numpy as np
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "iris_pipeline.joblib"

# Default label mapping for iris dataset if needed
DEFAULT_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}


class ModelNotLoadedError(RuntimeError):
    pass


class IrisModel:
    def __init__(self, model_path: Path = MODEL_PATH, label_map: Dict[int, str] = DEFAULT_LABELS):
        self.model_path = Path(model_path)
        self._model = None
        self.label_map = label_map
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")
        self._model = joblib.load(self.model_path)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded")
        return self._model.predict(X)

    def predict_single(self, features: List[float]) -> Dict[str, Any]:
        import numpy as _np
        arr = _np.array(features, dtype=float).reshape(1, -1)
        pred = int(self.predict(arr)[0])
        proba = self.predict_proba(arr)[0].tolist()
        return {
            "prediction": pred,
            "label": self.label_map.get(pred, str(pred)),
            "probabilities": proba
        }


# Module-level loader for convenience
_model_instance: IrisModel = None


def get_model() -> IrisModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = IrisModel()
    return _model_instance


def predict_from_dict(data: Dict[str, float]) -> Dict[str, Any]:
    # Accepts dict with keys: sepal_length, sepal_width, petal_length, petal_width
    keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    features = [float(data[k]) for k in keys]
    model = get_model()
    return model.predict_single(features)


if __name__ == "__main__":
    # quick CLI test
    sample = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    print(predict_from_dict(sample))
