"""Definición de modelos: XGBoost principal, baselines e intervalos conformales."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.data_processing import FEATURES, build_preprocessor

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
}


def build_xgb_pipeline(
    params: dict[str, Any] | None = None,
    seed: int = 42,
    add_missing_indicator: bool = True,
    extra_categorical: list[str] | None = None,
) -> Pipeline:
    """Pipeline preprocesamiento + XGBRegressor.

    Args:
        params: hiperparámetros de XGBoost; si es ``None`` se usan los por defecto
            de la librería (útil como punto de partida "antes de optimizar").
        seed: semilla.
        add_missing_indicator: si se agregan indicadores de dato faltante.
        extra_categorical: categóricas adicionales (solo para ablación, p. ej. ``Gender``).
    """
    kwargs = dict(params or {})
    kwargs.update(
        {"random_state": seed, "n_jobs": -1, "tree_method": "hist", "objective": "reg:squarederror"}
    )
    return Pipeline(
        [
            ("prep", build_preprocessor(add_missing_indicator, extra_categorical)),
            ("model", xgb.XGBRegressor(**kwargs)),
        ]
    )


def build_baselines(seed: int = 42) -> dict[str, Pipeline]:
    """Modelos de referencia contra los que se compara el modelo final.

    * ``Media``: predice siempre el promedio de entrenamiento.
    * ``Solo fatiga (lineal)``: regresión lineal con una sola variable
      (Mental Fatigue Score imputada por mediana); baseline trivial exigido.
    * ``Ridge``: regresión lineal regularizada con todas las variables.
    * ``SVR-RBF``: réplica del enfoque de Teodósio et al. (2025).
    """
    fatigue_only = ColumnTransformer(
        [("fat", SimpleImputer(strategy="median"), ["Mental Fatigue Score"])], remainder="drop"
    )
    from sklearn.dummy import DummyRegressor

    return {
        "Media (sin modelo)": Pipeline([("prep", fatigue_only), ("model", DummyRegressor())]),
        "Solo fatiga (lineal)": Pipeline([("prep", fatigue_only), ("model", LinearRegression())]),
        "Ridge": Pipeline(
            [("prep", build_preprocessor()), ("sc", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "SVR-RBF": Pipeline(
            [
                ("prep", build_preprocessor()),
                ("sc", StandardScaler()),
                ("model", SVR(C=1.0, epsilon=0.02, gamma="scale")),
            ]
        ),
    }


def select_alert_cutoff(
    y_true, y_score, threshold: float = 0.6, min_recall: float = 0.85, grid: np.ndarray | None = None
) -> float:
    """Elige el corte de decisión de la alerta con predicciones fuera de muestra.

    Un caso es de riesgo alto si su Burn Rate real es mayor o igual a ``threshold``.
    Como el modelo contrae las predicciones hacia el centro, usar el mismo valor
    como corte deja escapar casos reales. Se elige el corte más alto (el que
    produce menos falsas alarmas) que aún alcanza la sensibilidad ``min_recall``.

    Args:
        y_true: Burn Rate real (solo entrenamiento).
        y_score: predicciones fuera de muestra (validación cruzada).
        threshold: definición de riesgo alto.
        min_recall: sensibilidad mínima exigida (KPI del proyecto).
        grid: cortes candidatos; por defecto de 0.40 a ``threshold`` cada 0.005.

    Returns:
        Corte de decisión; si ninguno alcanza la sensibilidad, el más bajo de la grilla.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    real = y_true >= threshold
    grid = np.round(np.arange(0.40, threshold + 1e-9, 0.005), 3) if grid is None else grid
    ok = [c for c in grid if (y_score[real] >= c).mean() >= min_recall]
    return float(max(ok)) if ok else float(min(grid))


@dataclass
class BurnoutPredictor:
    """Modelo final con intervalos de predicción conformales por segmento.

    Los intervalos se calibran con residuos fuera de muestra (validación cruzada
    sobre train), separados según si ``Mental Fatigue Score`` fue observado o
    imputado. Así el intervalo es más ancho, de forma honesta, cuando falta el
    dato más informativo.

    Attributes:
        pipeline: pipeline de scikit-learn ya entrenado.
        threshold: definición de riesgo alto (Burn Rate real >= threshold).
        alert_cutoff: corte de decisión sobre la predicción para emitir la alerta,
            calibrado con validación cruzada para alcanzar la sensibilidad objetivo.
        coverage: cobertura nominal del intervalo (por ejemplo 0.90).
        residuals: residuos OOF por segmento (``observed``/``imputed``).
        metadata: métricas y datos de versión para mostrar en la app.
    """

    pipeline: Pipeline
    threshold: float = 0.6
    alert_cutoff: float = 0.6
    coverage: float = 0.90
    residuals: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        threshold: float = 0.6,
        coverage: float = 0.90,
        cv_folds: int = 5,
        min_recall: float = 0.85,
    ) -> "BurnoutPredictor":
        """Entrena el pipeline y calibra los intervalos.

        Args:
            X: variables de entrada (se ignoran columnas no usadas como ``Gender``).
            y: Burn Rate.
            params: hiperparámetros de XGBoost.
            seed: semilla.
            threshold: umbral de alerta.
            coverage: cobertura del intervalo.
            cv_folds: pliegues para obtener residuos fuera de muestra.
            min_recall: sensibilidad objetivo para calibrar el corte de la alerta.
        """
        base = build_xgb_pipeline(params, seed)
        oof = cross_val_predict(
            clone(base), X[FEATURES], y, cv=KFold(cv_folds, shuffle=True, random_state=seed)
        )
        resid = np.asarray(y) - oof
        imputed = X["Mental Fatigue Score"].isna().to_numpy()
        residuals = {"observed": resid[~imputed], "imputed": resid[imputed]}
        cutoff = select_alert_cutoff(y, oof, threshold, min_recall)
        pipeline = base.fit(X[FEATURES], y)
        return cls(
            pipeline=pipeline,
            threshold=threshold,
            alert_cutoff=cutoff,
            coverage=coverage,
            residuals=residuals,
        )

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicción puntual de Burn Rate, recortada al rango [0, 1]."""
        return np.clip(self.pipeline.predict(X[FEATURES]), 0.0, 1.0)

    def _segment(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(X["Mental Fatigue Score"].isna(), "imputed", "observed")

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicción con intervalo conformal y probabilidad de superar el umbral.

        Returns:
            DataFrame con columnas ``pred``, ``lower``, ``upper``,
            ``p_alert`` (proporción de residuos de calibración que llevarían al
            valor real por encima de ``threshold``), ``alert`` (predicción mayor
            o igual a ``alert_cutoff``) y ``fatigue_imputed``.
        """
        pred = self.predict(X)
        seg = self._segment(X)
        alpha = 1.0 - self.coverage
        lower, upper, p_alert = [], [], []
        for p, s in zip(pred, seg):
            r = self.residuals.get(s)
            if r is None or len(r) == 0:
                r = np.concatenate(list(self.residuals.values()))
            lo_q, hi_q = np.quantile(r, [alpha / 2, 1 - alpha / 2])
            lower.append(max(0.0, p + lo_q))
            upper.append(min(1.0, p + hi_q))
            p_alert.append(float(np.mean(p + r >= self.threshold)))
        return pd.DataFrame(
            {
                "pred": pred,
                "lower": lower,
                "upper": upper,
                "p_alert": p_alert,
                "alert": pred >= self.alert_cutoff,
                "fatigue_imputed": seg == "imputed",
            },
            index=X.index,
        )

    # ---------------------------------------------------------- explanation
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuciones locales (valores SHAP exactos de TreeSHAP vía XGBoost).

        Las columnas transformadas se agrupan de vuelta a la variable original,
        de modo que la suma de contribuciones + ``base`` es la predicción sin recortar.
        """
        prep = self.pipeline.named_steps["prep"]
        booster = self.pipeline.named_steps["model"].get_booster()
        Xt = prep.transform(X[FEATURES])
        names = list(Xt.columns)
        contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)
        df = pd.DataFrame(contrib[:, :-1], columns=names, index=X.index)
        grouped = pd.DataFrame(index=X.index)
        for feat in FEATURES:
            cols = [
                c for c in names if c == feat or c.startswith(feat + "_") or c == f"missingindicator_{feat}"
            ]
            grouped[feat] = df[cols].sum(axis=1)
        grouped["base"] = contrib[:, -1]
        return grouped

    def feature_importance(self) -> pd.Series:
        """Importancia global por ganancia, agregada por variable original."""
        prep = self.pipeline.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        gain = self.pipeline.named_steps["model"].get_booster().get_score(importance_type="total_gain")
        raw = pd.Series({n: gain.get(n, 0.0) for n in names})
        out = {}
        for feat in FEATURES:
            cols = [
                c for c in names if c == feat or c.startswith(feat + "_") or c == f"missingindicator_{feat}"
            ]
            out[feat] = raw[cols].sum()
        s = pd.Series(out)
        return (s / s.sum()).sort_values(ascending=False)
