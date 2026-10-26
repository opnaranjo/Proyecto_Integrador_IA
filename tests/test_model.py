"""Pruebas del modelo, los intervalos y las métricas."""

import numpy as np
import pytest

from src.evaluate import alert_metrics, group_metrics, regression_metrics
from src.model import build_baselines


def test_predictions_in_valid_range(trained_predictor):
    model, X_test, _ = trained_predictor
    pred = model.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert (pred >= 0).all() and (pred <= 1).all()


def test_model_beats_mean_baseline(trained_predictor, synthetic_df):
    """El modelo debe ser claramente mejor que predecir la media."""
    model, X_test, y_test = trained_predictor
    rmse_model = regression_metrics(y_test, model.predict(X_test))["rmse"]
    assert rmse_model < 0.5 * float(np.std(y_test))


def test_intervals_contain_point_prediction(trained_predictor):
    model, X_test, _ = trained_predictor
    res = model.predict_interval(X_test)
    assert (res["lower"] <= res["pred"] + 1e-9).all()
    assert (res["upper"] >= res["pred"] - 1e-9).all()
    assert res["p_alert"].between(0, 1).all()


def test_intervals_wider_when_fatigue_missing(trained_predictor):
    """Sin el dato de fatiga, la incertidumbre debe ser mayor."""
    model, X_test, _ = trained_predictor
    res = model.predict_interval(X_test)
    width = res["upper"] - res["lower"]
    assert width[res["fatigue_imputed"]].mean() > width[~res["fatigue_imputed"]].mean()


def test_explanations_sum_to_prediction(trained_predictor):
    """Las contribuciones SHAP más la base reconstruyen la predicción del modelo."""
    model, X_test, _ = trained_predictor
    sample = X_test.head(5)
    contrib = model.explain(sample)
    raw = model.pipeline.predict(sample[contrib.columns.drop("base")])
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), raw, atol=1e-4)


def test_feature_importance_sums_to_one(trained_predictor):
    model, _, _ = trained_predictor
    imp = model.feature_importance()
    assert imp.sum() == pytest.approx(1.0)
    assert imp.index[0] == "Mental Fatigue Score"


def test_alert_metrics_perfect_prediction():
    y = np.array([0.1, 0.7, 0.65, 0.3])
    m = alert_metrics(y, y, 0.6)
    assert m["sensibilidad"] == 1.0 and m["precision"] == 1.0 and m["auc_roc"] == 1.0


def test_group_metrics_suppresses_small_groups():
    y = np.linspace(0, 1, 30)
    groups = np.array(["A"] * 25 + ["B"] * 5)
    table = group_metrics(y, y, groups, 0.6, min_size=10)
    assert list(table.index) == ["A"]


def test_baselines_are_buildable():
    names = set(build_baselines())
    assert {"Solo fatiga (lineal)", "Ridge", "SVR-RBF"} <= names


def test_select_alert_cutoff_reaches_target_recall():
    """El corte elegido alcanza la sensibilidad pedida y no supera el umbral de riesgo."""
    from src.model import select_alert_cutoff

    rng = np.random.default_rng(1)
    y = rng.uniform(0, 1, 2000)
    score = 0.5 + 0.8 * (y - 0.5) + rng.normal(0, 0.05, 2000)  # predicción contraída al centro
    cut = select_alert_cutoff(y, score, threshold=0.6, min_recall=0.85)
    assert cut <= 0.6
    assert (score[y >= 0.6] >= cut).mean() >= 0.85


def test_alert_uses_calibrated_cutoff(trained_predictor):
    model, X_test, _ = trained_predictor
    res = model.predict_interval(X_test)
    assert (res["alert"] == (res["pred"] >= model.alert_cutoff)).all()
