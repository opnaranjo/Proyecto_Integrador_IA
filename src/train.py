"""Entrenamiento del modelo final y de las versiones de referencia.

Uso:
    python -m src.train                  # entrena con config.yaml
    python -m src.train --skip-eval      # solo entrena y guarda los modelos
"""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from typing import Any

import joblib
import matplotlib
import sklearn
import xgboost

from src.data_processing import FEATURES, clean_data, load_raw_data, save_processed, split_data
from src.evaluate import full_metrics, run_evaluation
from src.model import BurnoutPredictor, build_baselines, build_xgb_pipeline
from src.utils import get_logger, load_config, resolve_path, set_seed

logger = get_logger(__name__)


def train(cfg: dict[str, Any] | None = None, evaluate: bool = True) -> BurnoutPredictor:
    """Entrena y guarda ``model_v1.pkl`` (XGBoost por defecto) y ``best_model.pkl``.

    Args:
        cfg: configuración; si es ``None`` se lee ``config.yaml``.
        evaluate: si se genera la evaluación completa en ``results/``.

    Returns:
        El ``BurnoutPredictor`` final entrenado.
    """
    cfg = cfg or load_config()
    seed = cfg["project"]["seed"]
    thr = cfg["alert"]["threshold"]
    set_seed(seed)

    df = clean_data(load_raw_data())
    save_processed(df)
    X_train, X_test, y_train, y_test = split_data(df, cfg["data"]["test_size"], seed)
    logger.info("Partición estratificada: %d train / %d test", len(X_train), len(X_test))

    # Versión 1: XGBoost con hiperparámetros por defecto y sin indicador de faltantes.
    v1 = build_xgb_pipeline(None, seed, add_missing_indicator=False).fit(X_train[FEATURES], y_train)
    joblib.dump(v1, resolve_path(cfg["paths"]["model_v1"]))
    logger.info("model_v1.pkl guardado")

    # Modelo final: hiperparámetros optimizados + indicador de faltantes + intervalos.
    predictor = BurnoutPredictor.fit(
        X_train,
        y_train,
        cfg["model"]["params"],
        seed,
        thr,
        cfg["model"]["interval_coverage"],
        cfg["model"]["cv_folds"],
        cfg["alert"].get("min_recall", 0.85),
    )
    logger.info("Corte de alerta calibrado con validación cruzada: %.3f", predictor.alert_cutoff)

    test_metrics = full_metrics(y_test, predictor.predict(X_test), thr, predictor.alert_cutoff)
    predictor.metadata = {
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "features": FEATURES,
        "params": cfg["model"]["params"],
        "threshold": thr,
        "alert_cutoff": predictor.alert_cutoff,
        "interval_coverage": cfg["model"]["interval_coverage"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_metrics": test_metrics,
        "train_medians": X_train[["Resource Allocation", "Mental Fatigue Score"]].median().to_dict(),
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    if evaluate:
        baselines = {n: m.fit(X_train, y_train) for n, m in build_baselines(seed).items()}
        baselines["XGBoost v1 (por defecto)"] = v1
        summary = run_evaluation(predictor, baselines, X_train, X_test, y_train, y_test, cfg)
        predictor.metadata["comparison"] = summary["comparacion"]
        predictor.metadata["interval_coverage_test"] = summary["cobertura_intervalos"]
        predictor.metadata["feature_importance"] = summary["importancia_variables"]
        predictor.metadata["fairness"] = summary["equidad"]

    joblib.dump(predictor, resolve_path(cfg["paths"]["best_model"]))
    logger.info(
        "best_model.pkl guardado | RMSE test %.4f | R2 %.4f", test_metrics["rmse"], test_metrics["r2"]
    )
    return predictor


def main() -> None:
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena el modelo de riesgo de burnout")
    parser.add_argument("--skip-eval", action="store_true", help="No generar la evaluación completa")
    parser.add_argument("--config", default=None, help="Ruta alternativa a config.yaml")
    args = parser.parse_args()
    try:
        train(load_config(args.config), evaluate=not args.skip_eval)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
