"""Pruebas del módulo de procesamiento de datos."""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    FEATURES,
    TARGET,
    DataValidationError,
    build_preprocessor,
    clean_data,
    split_data,
    validate_input,
)


def test_clean_data_drops_rows_without_target(synthetic_df):
    """Las filas sin Burn Rate se eliminan y no se imputa el objetivo."""
    out = clean_data(synthetic_df)
    assert out[TARGET].notna().all()
    assert len(out) == synthetic_df[TARGET].notna().sum()
    assert "Employee ID" not in out.columns and "Date of Joining" not in out.columns


def test_clean_data_keeps_feature_missing_values(synthetic_df):
    """Los faltantes de las variables se conservan para imputarlos dentro del pipeline."""
    out = clean_data(synthetic_df)
    assert out["Mental Fatigue Score"].isna().any()


def test_split_is_stratified(synthetic_df):
    """Train y test conservan proporciones similares de riesgo alto."""
    X_tr, X_te, y_tr, y_te = split_data(clean_data(synthetic_df), 0.2, 42)
    assert abs((y_tr >= 0.6).mean() - (y_te >= 0.6).mean()) < 0.03
    assert len(X_te) == pytest.approx(0.2 * (len(X_tr) + len(X_te)), abs=2)


def test_preprocessor_adds_missing_indicators_and_drops_gender(synthetic_df):
    """El preprocesador crea indicadores de faltante y no usa Gender."""
    df = clean_data(synthetic_df)
    prep = build_preprocessor().fit(df[FEATURES + ["Gender"]])
    cols = list(prep.get_feature_names_out())
    assert "missingindicator_Mental Fatigue Score" in cols
    assert not any("Gender" in c for c in cols)
    Xt = prep.transform(df[FEATURES])
    assert not Xt.isna().any().any()


def test_validate_input_accepts_valid_and_normalizes_case():
    row = pd.DataFrame(
        [
            {
                "Designation": 2,
                "Resource Allocation": 5,
                "Mental Fatigue Score": None,
                "Company Type": " service ",
                "WFH Setup Available": "yes",
            }
        ]
    )
    out = validate_input(row)
    assert out.loc[0, "Company Type"] == "Service"
    assert out.loc[0, "WFH Setup Available"] == "Yes"
    assert np.isnan(out.loc[0, "Mental Fatigue Score"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("Mental Fatigue Score", 11),
        ("Resource Allocation", 0),
        ("Designation", "alto"),
        ("Company Type", "Startup"),
    ],
)
def test_validate_input_rejects_invalid_values(field, value):
    row = {
        "Designation": 2,
        "Resource Allocation": 5,
        "Mental Fatigue Score": 5,
        "Company Type": "Service",
        "WFH Setup Available": "Yes",
    }
    row[field] = value
    with pytest.raises(DataValidationError):
        validate_input(pd.DataFrame([row]))


def test_validate_input_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="Faltan columnas"):
        validate_input(pd.DataFrame([{"Designation": 1}]))
