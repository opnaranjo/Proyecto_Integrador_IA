"""Fixtures compartidas por las pruebas."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Dataset sintético con la misma estructura que el real (rápido y determinista)."""
    rng = np.random.default_rng(0)
    n = 600
    fatigue = rng.uniform(0, 10, n)
    resource = rng.integers(1, 11, n).astype(float)
    designation = rng.integers(0, 6, n).astype(float)
    burn = np.clip(
        0.07 * fatigue + 0.02 * resource + 0.01 * designation + rng.normal(0, 0.03, n) - 0.05, 0, 1
    )
    df = pd.DataFrame(
        {
            "Employee ID": [f"id{i}" for i in range(n)],
            "Date of Joining": "2008-01-01",
            "Gender": rng.choice(["Female", "Male"], n),
            "Company Type": rng.choice(["Service", "Product"], n),
            "WFH Setup Available": rng.choice(["Yes", "No"], n),
            "Designation": designation,
            "Resource Allocation": resource,
            "Mental Fatigue Score": fatigue,
            "Burn Rate": burn,
        }
    )
    df.loc[rng.random(n) < 0.08, "Mental Fatigue Score"] = np.nan
    df.loc[rng.random(n) < 0.05, "Resource Allocation"] = np.nan
    df.loc[rng.random(n) < 0.04, "Burn Rate"] = np.nan
    return df


@pytest.fixture(scope="session")
def trained_predictor(synthetic_df):
    """BurnoutPredictor entrenado con datos sintéticos."""
    from src.data_processing import clean_data, split_data
    from src.model import BurnoutPredictor

    df = clean_data(synthetic_df)
    X_train, X_test, y_train, y_test = split_data(df, 0.2, 42)
    model = BurnoutPredictor.fit(X_train, y_train, {"n_estimators": 80, "max_depth": 3}, cv_folds=3)
    return model, X_test, y_test
