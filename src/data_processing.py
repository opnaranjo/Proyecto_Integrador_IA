"""Carga, limpieza, validación y preprocesamiento de los datos de burnout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

TARGET = "Burn Rate"
NUMERIC_FEATURES = ["Designation", "Resource Allocation", "Mental Fatigue Score"]
CATEGORICAL_FEATURES = ["Company Type", "WFH Setup Available"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Dominio válido de cada variable, usado para validar entradas de la aplicación.
VALID_RANGES = {
    "Designation": (0.0, 5.0),
    "Resource Allocation": (1.0, 10.0),
    "Mental Fatigue Score": (0.0, 10.0),
}
VALID_CATEGORIES = {
    "Company Type": ["Product", "Service"],
    "WFH Setup Available": ["No", "Yes"],
}


class DataValidationError(ValueError):
    """Error lanzado cuando los datos de entrada no cumplen el esquema esperado."""


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Lee el CSV original del dataset.

    Args:
        path: ruta al CSV; si es ``None`` se toma de ``config.yaml``.

    Returns:
        DataFrame con los datos crudos.

    Raises:
        FileNotFoundError: si el archivo no existe.
    """
    csv_path = resolve_path(path or load_config()["paths"]["raw_data"])
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}. Descarga train.csv según data/README.md.")
    df = pd.read_csv(csv_path)
    logger.info("Datos cargados: %d filas, %d columnas", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza acordada para el entrenamiento.

    Decisiones (justificadas en docs/analisis_datos.md):
      * Se eliminan las filas sin ``Burn Rate``: imputar la variable objetivo
        fabricaría etiquetas.
      * Se descartan ``Employee ID`` (identificador) y ``Date of Joining``
        (todas las fechas son de 2008 y su correlación con el objetivo es 0.001).
      * Los faltantes de ``Resource Allocation`` y ``Mental Fatigue Score`` se
        conservan como NaN; se imputan dentro del pipeline, ajustado solo con train.
      * ``Gender`` se conserva únicamente para auditar equidad.

    Args:
        df: DataFrame crudo.

    Returns:
        DataFrame limpio con las variables del modelo, ``Gender`` y el objetivo.
    """
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Faltan columnas requeridas: {missing_cols}")
    out = df.dropna(subset=[TARGET]).copy()
    keep = FEATURES + (["Gender"] if "Gender" in out.columns else []) + [TARGET]
    out = out[keep].reset_index(drop=True)
    logger.info("Filas con objetivo observado: %d (se descartaron %d)", len(out), len(df) - len(out))
    return out


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Divide en entrenamiento y prueba con estratificación por nivel de riesgo.

    La estratificación usa el umbral de alerta (0.6) para que ambas particiones
    tengan la misma proporción de casos de riesgo alto.

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``; X incluye ``Gender`` si existe.
    """
    strata = (df[TARGET] >= 0.6).astype(int)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strata)


def build_preprocessor(
    add_missing_indicator: bool = True, extra_categorical: list[str] | None = None
) -> ColumnTransformer:
    """Construye el preprocesador de scikit-learn.

    * Numéricas: imputación por mediana; con ``add_missing_indicator`` se añade
      una columna binaria que marca si el valor original faltaba. Esto permite al
      modelo distinguir un dato imputado de uno observado (mitigación de la
      auditoría de equidad).
    * Categóricas: one-hot con la primera categoría como referencia.

    Las columnas no listadas (por ejemplo ``Gender``) se descartan, salvo que se
    pasen en ``extra_categorical`` (solo se usa en el experimento de ablación).
    """
    numeric = SimpleImputer(strategy="median", add_indicator=add_missing_indicator)
    categorical = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    prep = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES + list(extra_categorical or [])),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Salida como DataFrame para conservar nombres de columnas hasta el modelo.
    return prep.set_output(transform="pandas")


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza datos de entrada provenientes de la aplicación o de un CSV.

    Reglas:
      * Deben existir todas las columnas de ``FEATURES``.
      * Numéricas convertibles a número y dentro de ``VALID_RANGES``;
        ``Resource Allocation`` y ``Mental Fatigue Score`` admiten vacío (NaN).
      * Categóricas dentro de ``VALID_CATEGORIES`` (se toleran mayúsculas/espacios).

    Args:
        df: datos a validar.

    Returns:
        Copia del DataFrame con tipos normalizados.

    Raises:
        DataValidationError: con un mensaje legible que enumera los problemas.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("No se recibieron datos para evaluar.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas: {', '.join(missing)}")

    out = df.copy()
    errors: list[str] = []
    for col, (lo, hi) in VALID_RANGES.items():
        values = pd.to_numeric(out[col], errors="coerce")
        bad_type = values.isna() & out[col].notna() & (out[col].astype(str).str.strip() != "")
        if bad_type.any():
            errors.append(f"'{col}' tiene {int(bad_type.sum())} valor(es) no numérico(s)")
        if col == "Designation" and values.isna().any():
            errors.append("'Designation' es obligatoria")
        out_of_range = values.notna() & ((values < lo) | (values > hi))
        if out_of_range.any():
            errors.append(
                f"'{col}' debe estar entre {lo:g} y {hi:g} ({int(out_of_range.sum())} fuera de rango)"
            )
        out[col] = values.astype(float)

    for col, allowed in VALID_CATEGORIES.items():
        normalized = out[col].astype(str).str.strip().str.capitalize()
        bad = ~normalized.isin(allowed)
        if bad.any():
            errors.append(f"'{col}' solo admite {allowed} ({int(bad.sum())} valor(es) inválido(s))")
        out[col] = normalized

    if errors:
        raise DataValidationError("; ".join(errors))
    return out


def save_processed(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Guarda el dataset limpio en ``data/processed``."""
    out = resolve_path(path or load_config()["paths"]["processed_data"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Dataset procesado guardado en %s", out)
    return out
