"""Lógica de la aplicación separada de la interfaz para facilitar las pruebas."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import FEATURES, validate_input  # noqa: E402
from src.utils import get_logger, load_config, resolve_path  # noqa: E402

logger = get_logger("app")

DESIGNATION_LABELS = {
    0: "0 · Nivel inicial",
    1: "1 · Junior",
    2: "2 · Intermedio",
    3: "3 · Senior",
    4: "4 · Líder / Gerencia media",
    5: "5 · Dirección",
}

RISK_LEVELS = [
    (0.0, 0.40, "Bajo", "#2e7d32"),
    (0.40, 0.60, "Moderado", "#e0a100"),
    (0.60, 1.01, "Alto", "#b3261e"),
]


def set_alert_cutoff(cutoff: float) -> None:
    """Ajusta el límite Moderado/Alto al corte de alerta calibrado del modelo.

    Se modifica la lista en su lugar para que todos los módulos que la importaron
    vean los mismos límites.
    """
    RISK_LEVELS[1] = (0.40, cutoff, "Moderado", "#e0a100")
    RISK_LEVELS[2] = (cutoff, 1.01, "Alto", "#b3261e")


EXAMPLES = {
    "Riesgo bajo": {
        "Designation": 1,
        "Resource Allocation": 3.0,
        "Mental Fatigue Score": 3.5,
        "Company Type": "Product",
        "WFH Setup Available": "Yes",
    },
    "Riesgo moderado": {
        "Designation": 2,
        "Resource Allocation": 5.0,
        "Mental Fatigue Score": 6.2,
        "Company Type": "Service",
        "WFH Setup Available": "Yes",
    },
    "Riesgo alto": {
        "Designation": 4,
        "Resource Allocation": 8.0,
        "Mental Fatigue Score": 8.6,
        "Company Type": "Service",
        "WFH Setup Available": "No",
    },
    "Sin dato de fatiga": {
        "Designation": 3,
        "Resource Allocation": 7.0,
        "Mental Fatigue Score": None,
        "Company Type": "Product",
        "WFH Setup Available": "No",
    },
}

FEATURE_LABELS = {
    "Designation": "Nivel jerárquico",
    "Resource Allocation": "Carga de trabajo asignada",
    "Mental Fatigue Score": "Fatiga mental",
    "Company Type": "Tipo de empresa",
    "WFH Setup Available": "Teletrabajo disponible",
}


def risk_level(value: float) -> tuple[str, str]:
    """Devuelve la etiqueta de nivel de riesgo y su color para un Burn Rate."""
    for lo, hi, label, color in RISK_LEVELS:
        if lo <= value < hi:
            return label, color
    return RISK_LEVELS[-1][2], RISK_LEVELS[-1][3]


def build_input(
    designation: int, resource: float | None, fatigue: float | None, company: str, wfh: str
) -> pd.DataFrame:
    """Construye y valida el DataFrame de una sola persona.

    Raises:
        DataValidationError: si algún valor está fuera de dominio.
    """
    row = pd.DataFrame(
        [
            {
                "Designation": designation,
                "Resource Allocation": np.nan if resource is None else resource,
                "Mental Fatigue Score": np.nan if fatigue is None else fatigue,
                "Company Type": company,
                "WFH Setup Available": wfh,
            }
        ]
    )
    return validate_input(row)


def retrain_predictor(cfg: dict):
    """Reentrena el modelo final solo con las dependencias de la aplicación.

    No usa ``src.train`` (que requiere librerías de gráficos y de equidad) ni
    escribe archivos: el modelo queda en memoria. Las métricas para la pestaña
    "Acerca del modelo" se leen de ``results/metrics/final_metrics.json`` si
    existe; si no, se calculan sobre el conjunto de prueba.
    """
    import json

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score

    from src.data_processing import clean_data, load_raw_data, split_data
    from src.model import BurnoutPredictor

    seed = cfg["project"]["seed"]
    thr = cfg["alert"]["threshold"]
    X_tr, X_te, y_tr, y_te = split_data(clean_data(load_raw_data()), cfg["data"]["test_size"], seed)
    model = BurnoutPredictor.fit(
        X_tr,
        y_tr,
        cfg["model"]["params"],
        seed,
        thr,
        cfg["model"]["interval_coverage"],
        cfg["model"]["cv_folds"],
        cfg["alert"].get("min_recall", 0.85),
    )
    pred = model.predict(X_te)
    model.metadata = {
        "version": "1.0.0 (reentrenado al iniciar)",
        "threshold": thr,
        "alert_cutoff": model.alert_cutoff,
        "test_metrics": {
            "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
            "mae": float(mean_absolute_error(y_te, pred)),
            "r2": float(r2_score(y_te, pred)),
            "sensibilidad": float(recall_score(y_te >= thr, pred >= model.alert_cutoff)),
        },
    }
    metrics_file = resolve_path(cfg["paths"]["metrics_dir"]) / "final_metrics.json"
    if metrics_file.exists():
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))
        model.metadata.update(
            {
                "comparison": saved.get("comparacion"),
                "interval_coverage_test": saved.get("cobertura_intervalos"),
                "feature_importance": saved.get("importancia_variables"),
                "fairness": saved.get("equidad"),
            }
        )
    return model


def load_predictor():
    """Carga ``models/best_model.pkl``; si falla, reentrena con los datos del repositorio.

    El reentrenamiento de respaldo cubre el caso en que la versión de las
    librerías del servidor no pueda leer el archivo serializado.

    Returns:
        Tupla ``(modelo, origen)`` con origen ``"archivo"`` o ``"reentrenado"``.
    """
    import joblib

    cfg = load_config()
    path = resolve_path(cfg["paths"]["best_model"])
    try:
        model = joblib.load(path)
        logger.info("Modelo cargado desde %s", path)
        return model, "archivo"
    except Exception as exc:  # noqa: BLE001 - cualquier fallo activa el respaldo
        logger.warning("No se pudo cargar %s (%s). Reentrenando en memoria...", path, exc)
        return retrain_predictor(cfg), "reentrenado"


def aggregate_segments(data: pd.DataFrame, by: str, min_size: int = 10) -> pd.DataFrame:
    """Resumen agregado por segmento con supresión de grupos pequeños.

    Args:
        data: predicciones con columnas ``pred`` y ``alert`` más la columna ``by``.
        by: variable de segmentación.
        min_size: tamaño mínimo para mostrar un segmento (k-anonimato).

    Returns:
        Tabla con n, Burn Rate medio, % en riesgo alto y una columna ``visible``.
    """
    g = (
        data.groupby(by, observed=True)
        .agg(
            personas=("pred", "size"),
            burn_rate_medio=("pred", "mean"),
            pct_riesgo_alto=("alert", "mean"),
        )
        .reset_index()
    )
    g["pct_riesgo_alto"] = g["pct_riesgo_alto"] * 100
    g["visible"] = g["personas"] >= min_size
    g.loc[~g["visible"], ["burn_rate_medio", "pct_riesgo_alto"]] = np.nan
    return g


def predict_batch(model, raw: pd.DataFrame) -> pd.DataFrame:
    """Valida un lote y devuelve las predicciones con su intervalo."""
    data = validate_input(raw)
    res = model.predict_interval(data)
    return pd.concat([data[FEATURES].reset_index(drop=True), res.reset_index(drop=True)], axis=1)


def what_if_curve(model, base: pd.DataFrame, feature: str, grid: np.ndarray) -> pd.DataFrame:
    """Predicción al variar una sola variable manteniendo las demás fijas."""
    rows = pd.concat([base] * len(grid), ignore_index=True)
    rows[feature] = grid
    return pd.DataFrame({feature: grid, "pred": model.predict(rows)})
