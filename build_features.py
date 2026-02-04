"""
Simple feature pipeline: read raw CSV and produce cleaned processed CSV.
For iris dataset this is minimal, but we include:
 - consistent column names (snake_case)
 - ensure target is integer (0,1,2)
 - save processed CSV to data/processed/iris_processed.csv
"""
from pathlib import Path
import pandas as pd
import logging
from src.utils.validation import validate_raw_iris

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "iris.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "iris_processed.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # map common sklearn names to snake_case
    mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
        "species": "target"
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    return df


def build_processed(save: bool = True) -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}. Run make_dataset.py first.")

    df = pd.read_csv(RAW_PATH)
    validate_raw_iris(df)
    df = normalize_column_names(df)

    # Ensure target is numeric (some versions of load_iris may have 'target' numeric already)
    if df["target"].dtype == object:
        # Try mapping names to numbers if present
        unique = df["target"].unique().tolist()
        logger.info("Mapping target categories %s to numeric labels", unique)
        mapping = {v: i for i, v in enumerate(sorted(unique))}
        df["target"] = df["target"].map(mapping)

    # Persist processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if save:
        df.to_csv(PROCESSED_PATH, index=False)
        logger.info("Saved processed dataset to %s", PROCESSED_PATH)
    return df


if __name__ == "__main__":
    build_processed()
