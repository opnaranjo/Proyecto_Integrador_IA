"""Pruebas de la lógica y la interfaz de la aplicación Streamlit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import EXAMPLES, aggregate_segments, build_input, predict_batch, risk_level

from src.data_processing import DataValidationError

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.pkl"


@pytest.mark.parametrize("value,label", [(0.1, "Bajo"), (0.45, "Moderado"), (0.6, "Alto"), (1.0, "Alto")])
def test_risk_level(value, label):
    assert risk_level(value)[0] == label


def test_build_input_handles_missing_fatigue():
    row = build_input(2, 5.0, None, "Service", "Yes")
    assert np.isnan(row.loc[0, "Mental Fatigue Score"])


def test_build_input_rejects_out_of_range():
    with pytest.raises(DataValidationError):
        build_input(9, 5.0, 5.0, "Service", "Yes")


def test_aggregate_segments_hides_small_groups():
    data = pd.DataFrame(
        {
            "pred": np.r_[np.full(12, 0.7), np.full(3, 0.2)],
            "alert": np.r_[np.ones(12, bool), np.zeros(3, bool)],
            "Company Type": ["Service"] * 12 + ["Product"] * 3,
        }
    )
    agg = aggregate_segments(data, "Company Type", min_size=10).set_index("Company Type")
    assert agg.loc["Service", "visible"] and agg.loc["Service", "pct_riesgo_alto"] == 100
    assert not agg.loc["Product", "visible"] and np.isnan(agg.loc["Product", "burn_rate_medio"])


def test_predict_batch_with_example_file(trained_predictor):
    model, _, _ = trained_predictor
    raw = pd.read_csv(ROOT / "app" / "assets" / "ejemplo_lote.csv")
    out = predict_batch(model, raw)
    assert len(out) == len(raw) and out["pred"].between(0, 1).all()


def test_examples_are_valid_inputs():
    for ex in EXAMPLES.values():
        build_input(
            ex["Designation"],
            ex["Resource Allocation"],
            ex["Mental Fatigue Score"],
            ex["Company Type"],
            ex["WFH Setup Available"],
        )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Requiere models/best_model.pkl (python -m src.train)")
def test_streamlit_app_runs_example_without_errors():
    """Ejecuta la app completa, carga un ejemplo y verifica que no haya excepciones."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=120).run()
    assert not at.exception
    next(b for b in at.button if b.label == "Riesgo alto").click().run()
    assert not at.exception
    assert any("Riesgo Alto" in m.value for m in at.markdown)


def test_load_predictor_retrains_when_model_file_is_unreadable(tmp_path, monkeypatch):
    """Si el .pkl no se puede leer (p. ej., otra versión de librerías), la app reentrena en memoria."""
    import helpers

    bad = tmp_path / "roto.pkl"
    bad.write_text("no es un modelo")
    cfg = helpers.load_config()
    cfg["paths"]["best_model"] = str(bad)
    cfg["model"]["params"] = {"n_estimators": 50, "max_depth": 3}
    cfg["model"]["cv_folds"] = 3
    monkeypatch.setattr(helpers, "load_config", lambda: cfg)
    model, origin = helpers.load_predictor()
    assert origin == "reentrenado"
    assert 0 < model.alert_cutoff <= 0.6
    assert model.metadata["test_metrics"]["r2"] > 0.8
