"""Evaluación del modelo: métricas, baselines, equidad, errores y pruebas de estrés.

Uso:
    python -m src.evaluate          # evalúa models/best_model.pkl sobre el test
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.data_processing import FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ métricas
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE y R² de una predicción continua."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def alert_metrics(y_true, y_pred, threshold: float = 0.6, cutoff: float | None = None) -> dict[str, float]:
    """Métricas de la alerta binaria.

    Un caso es de riesgo alto si su Burn Rate real es ≥ ``threshold``; el modelo
    emite la alerta si su predicción es ≥ ``cutoff`` (por defecto igual a
    ``threshold``). La sensibilidad es la métrica prioritaria: un caso de riesgo
    alto no detectado es el error más costoso para un programa de bienestar.
    """
    cutoff = threshold if cutoff is None else cutoff
    t = np.asarray(y_true) >= threshold
    p = np.asarray(y_pred) >= cutoff
    out = {
        "sensibilidad": float(recall_score(t, p, zero_division=0)),
        "especificidad": float(recall_score(~t, ~p, zero_division=0)),
        "precision": float(precision_score(t, p, zero_division=0)),
        "f1": float(f1_score(t, p, zero_division=0)),
        "tasa_alerta_real": float(t.mean()),
        "tasa_alerta_predicha": float(p.mean()),
    }
    out["auc_roc"] = float(roc_auc_score(t, y_pred)) if 0 < t.mean() < 1 else float("nan")
    return out


def full_metrics(y_true, y_pred, threshold: float = 0.6, cutoff: float | None = None) -> dict[str, float]:
    """Une métricas de regresión y de alerta."""
    return {**regression_metrics(y_true, y_pred), **alert_metrics(y_true, y_pred, threshold, cutoff)}


def compare_models(
    models: dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.6
) -> pd.DataFrame:
    """Tabla comparativa de modelos ya entrenados sobre el mismo test."""
    rows = []
    for name, m in models.items():
        pred = np.clip(m.predict(X_test), 0, 1)
        rows.append({"modelo": name, **full_metrics(y_test, pred, threshold)})
    return pd.DataFrame(rows).set_index("modelo")


# ------------------------------------------------------------------ equidad
def group_metrics(
    y_true,
    y_pred,
    groups: pd.Series,
    threshold: float = 0.6,
    min_size: int = 10,
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Métricas por subgrupo (para auditoría de equidad y análisis de errores).

    Los grupos con menos de ``min_size`` casos se omiten (protección de privacidad).
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    rows = []
    for g, sub in df.groupby("g", observed=True):
        if len(sub) < min_size:
            continue
        m = regression_metrics(sub["y"], sub["p"])
        t, a = sub["y"] >= threshold, sub["p"] >= (threshold if cutoff is None else cutoff)
        m.update(
            {
                "n": int(len(sub)),
                "tasa_alerta_real": float(t.mean()),
                "tasa_alerta_predicha": float(a.mean()),
                "sensibilidad": float(recall_score(t, a, zero_division=0)) if t.any() else float("nan"),
                "falsos_positivos": float(((~t) & a).sum() / max((~t).sum(), 1)),
            }
        )
        rows.append({"grupo": g, **m})
    return pd.DataFrame(rows).set_index("grupo")


def fairness_gaps(table: pd.DataFrame) -> dict[str, float]:
    """Brechas máximas entre grupos (max - min) de las métricas clave."""
    cols = ["mae", "rmse", "sensibilidad", "tasa_alerta_predicha", "falsos_positivos"]
    return {f"brecha_{c}": float(table[c].max() - table[c].min()) for c in cols if c in table}


def demographic_parity_difference(y_pred, groups, cutoff: float = 0.6) -> float:
    """DPD de la alerta con Fairlearn (diferencia de tasas de selección)."""
    from fairlearn.metrics import demographic_parity_difference as dpd

    a = (np.asarray(y_pred) >= cutoff).astype(int)
    return float(dpd(a, a, sensitive_features=np.asarray(groups)))


def equalized_odds_difference(
    y_true, y_pred, groups, threshold: float = 0.6, cutoff: float | None = None
) -> float:
    """Diferencia de odds igualadas con Fairlearn sobre la alerta binaria."""
    from fairlearn.metrics import equalized_odds_difference as eod

    t = (np.asarray(y_true) >= threshold).astype(int)
    a = (np.asarray(y_pred) >= (threshold if cutoff is None else cutoff)).astype(int)
    return float(eod(t, a, sensitive_features=np.asarray(groups)))


# ------------------------------------------------------------ intervalos
def interval_coverage(y_true, intervals: pd.DataFrame) -> pd.DataFrame:
    """Cobertura empírica y ancho medio del intervalo, por segmento de fatiga."""
    df = intervals.copy()
    df["y"] = np.asarray(y_true)
    df["cubre"] = (df["y"] >= df["lower"]) & (df["y"] <= df["upper"])
    df["ancho"] = df["upper"] - df["lower"]
    df["segmento"] = np.where(df["fatigue_imputed"], "Fatiga imputada", "Fatiga observada")
    out = df.groupby("segmento").agg(
        n=("y", "size"), cobertura=("cubre", "mean"), ancho_medio=("ancho", "mean")
    )
    out.loc["Total"] = [len(df), df["cubre"].mean(), df["ancho"].mean()]
    return out


# ----------------------------------------------------------- estrés (S5)
def _perturbed(
    predict: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    fn: Callable,
    reps: int,
    seed: int,
    threshold: float,
    cutoff: float | None = None,
) -> dict[str, float]:
    res = []
    for i in range(reps):
        X2, y2 = fn(X, y, np.random.default_rng(seed + i))
        res.append(full_metrics(y2, np.clip(predict(X2), 0, 1), threshold, cutoff))
    return pd.DataFrame(res).mean().to_dict()


def stress_tests(
    predict: Callable,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    threshold: float = 0.6,
    reps: int = 10,
    seed: int = 42,
    cutoff: float | None = None,
) -> pd.DataFrame:
    """Simulación de escenarios críticos aplicada al modelo de burnout.

    Escenarios:
      * Ruido en las entradas numéricas (σ × desviación estándar de train).
      * Pérdida aleatoria de Fatiga y Recursos (se dejan como faltantes).
      * La variable clave (Fatiga) deja de llegar por completo.
      * Cambio de prevalencia: se remuestrea el test para que la tasa de
        riesgo alto sea 10 %, 30 % o 50 %.
      * Error de captura en Designation (±1 nivel en el 20 % de casos).
    """
    num = ["Resource Allocation", "Mental Fatigue Score"]
    std = X_train[num].std()
    rows = [
        {
            "escenario": "Sin perturbar",
            "nivel": "-",
            **full_metrics(y_test, np.clip(predict(X_test), 0, 1), threshold, cutoff),
        }
    ]

    def noise(s):
        def f(X, y, rng):
            X2 = X.copy()
            for c in num:
                X2[c] = (X2[c] + rng.normal(0, 1, len(X2)) * std[c] * s).clip(0, 10)
            return X2, y

        return f

    for s in [0.1, 0.25, 0.5, 1.0]:
        rows.append(
            {
                "escenario": "Ruido en entradas",
                "nivel": f"σ={s}",
                **_perturbed(predict, X_test, y_test, noise(s), reps, seed, threshold, cutoff),
            }
        )

    def missing(p):
        def f(X, y, rng):
            X2 = X.copy()
            for c in num:
                X2.loc[rng.random(len(X2)) < p, c] = np.nan
            return X2, y

        return f

    for p in [0.1, 0.3, 0.5]:
        rows.append(
            {
                "escenario": "Datos faltantes",
                "nivel": f"{int(p * 100)}%",
                **_perturbed(predict, X_test, y_test, missing(p), reps, seed, threshold, cutoff),
            }
        )

    def drop_fatigue(X, y, rng):
        X2 = X.copy()
        X2["Mental Fatigue Score"] = np.nan
        return X2, y

    rows.append(
        {
            "escenario": "Fatiga no disponible",
            "nivel": "100%",
            **_perturbed(predict, X_test, y_test, drop_fatigue, 1, seed, threshold, cutoff),
        }
    )

    def prevalence(target):
        def f(X, y, rng):
            pos = np.flatnonzero(y.values >= threshold)
            neg = np.flatnonzero(y.values < threshold)
            n = len(y)
            idx = np.concatenate(
                [
                    rng.choice(pos, int(n * target), replace=True),
                    rng.choice(neg, n - int(n * target), replace=True),
                ]
            )
            return X.iloc[idx], y.iloc[idx]

        return f

    for t in [0.10, 0.30, 0.50]:
        rows.append(
            {
                "escenario": "Cambio de prevalencia",
                "nivel": f"{int(t * 100)}% riesgo alto",
                **_perturbed(predict, X_test, y_test, prevalence(t), reps, seed, threshold, cutoff),
            }
        )

    def designation_error(X, y, rng):
        X2 = X.copy()
        m = rng.random(len(X2)) < 0.2
        X2.loc[m, "Designation"] = (X2.loc[m, "Designation"] + rng.choice([-1, 1], m.sum())).clip(0, 5)
        return X2, y

    rows.append(
        {
            "escenario": "Error en Designation",
            "nivel": "20% ±1",
            **_perturbed(predict, X_test, y_test, designation_error, reps, seed, threshold, cutoff),
        }
    )
    return pd.DataFrame(rows)


def distribution_shift(
    build: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    column: str,
    train_value: str,
    seed: int = 42,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Entrena con un subgrupo y evalúa en el subgrupo complementario.

    Simula desplegar el modelo en un tipo de empresa distinto del de entrenamiento.
    """
    import warnings

    from sklearn.model_selection import train_test_split

    # El grupo distinto trae categorías no vistas; el aviso de one-hot es esperado.
    warnings.filterwarnings("ignore", message="Found unknown categories")
    ref = X[column] == train_value
    Xa, ya, Xb, yb = X[ref], y[ref], X[~ref], y[~ref]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, ya, test_size=0.3, random_state=seed)
    m = build().fit(Xa_tr[FEATURES], ya_tr)
    return pd.DataFrame(
        [
            {
                "evaluado_en": f"{column} = {train_value} (mismo grupo)",
                **full_metrics(ya_te, np.clip(m.predict(Xa_te[FEATURES]), 0, 1), threshold),
            },
            {
                "evaluado_en": f"{column} ≠ {train_value} (grupo distinto)",
                **full_metrics(yb, np.clip(m.predict(Xb[FEATURES]), 0, 1), threshold),
            },
        ]
    ).set_index("evaluado_en")


# ------------------------------------------------------------------ figuras
def plot_pred_vs_real(y_true, y_pred, path, threshold: float = 0.6) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25, color="#333")
    ax.plot([0, 1], [0, 1], color="#c0392b", lw=1)
    ax.axhline(threshold, color="#888", ls="--", lw=0.8)
    ax.axvline(threshold, color="#888", ls="--", lw=0.8)
    ax.set_xlabel("Burn Rate real")
    ax.set_ylabel("Burn Rate predicho")
    ax.set_title("Predicción vs. valor real (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, imputed, path) -> None:
    r = np.asarray(y_true) - np.asarray(y_pred)
    imputed = np.asarray(imputed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].hist(r[~imputed], bins=60, color="#444", alpha=0.8, density=True, label="Fatiga observada")
    axes[0].hist(r[imputed], bins=40, color="#c0392b", alpha=0.5, density=True, label="Fatiga imputada")
    axes[0].set_xlabel("Residuo (real − predicho)")
    axes[0].set_title("Distribución de residuos")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(y_pred, r, s=4, alpha=0.25, color="#333")
    axes[1].axhline(0, color="#c0392b", lw=1)
    axes[1].set_xlabel("Burn Rate predicho")
    axes[1].set_ylabel("Residuo")
    axes[1].set_title("Residuos vs. predicción")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    t = table.sort_values("rmse", ascending=False)
    colors = ["#c0392b" if "XGBoost final" in i else "#888" for i in t.index]
    axes[0].barh(t.index, t["rmse"], color=colors)
    axes[0].set_xlabel("RMSE (menor es mejor)")
    for i, v in enumerate(t["rmse"]):
        axes[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)
    axes[1].barh(t.index, t["sensibilidad"], color=colors)
    axes[1].set_xlabel("Sensibilidad de la alerta ≥ 0.6")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_yticklabels([])
    for i, v in enumerate(t["sensibilidad"]):
        axes[1].text(v, i, f" {v:.2f}", va="center", fontsize=8)
    fig.suptitle("Comparación con baselines en el conjunto de prueba", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stress(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    lab = table["escenario"] + " " + table["nivel"].where(table["nivel"] != "-", "")
    axes[0].barh(lab, table["rmse"], color="#555")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("RMSE")
    axes[0].axvline(table.loc[0, "rmse"], color="#c0392b", ls="--", lw=1)
    axes[1].barh(lab, table["sensibilidad"], color="#555")
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].axvline(table.loc[0, "sensibilidad"], color="#c0392b", ls="--", lw=1)
    axes[1].set_xlabel("Sensibilidad de la alerta")
    fig.suptitle("Pruebas de estrés del modelo final (línea roja = sin perturbar)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_metrics(table: pd.DataFrame, title: str, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col, lab in zip(
        axes,
        ["mae", "sensibilidad", "tasa_alerta_predicha"],
        ["MAE", "Sensibilidad", "Tasa de alerta predicha"],
    ):
        ax.bar(table.index.astype(str), table[col], color="#555")
        ax.set_title(lab, fontsize=10)
        for i, v in enumerate(table[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ orquestación
def run_evaluation(
    predictor,
    baselines: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Genera todas las métricas, tablas y figuras de evaluación en ``results/``.

    Args:
        predictor: ``BurnoutPredictor`` entrenado.
        baselines: modelos de referencia ya entrenados (nombre -> estimador).
        X_train, X_test, y_train, y_test: particiones (X incluye ``Gender``).
        cfg: configuración cargada de ``config.yaml``.

    Returns:
        Resumen con las métricas principales (se guarda también en JSON).
    """
    from src.model import build_xgb_pipeline
    from src.utils import resolve_path, save_json

    thr = cfg["alert"]["threshold"]
    seed = cfg["project"]["seed"]
    k = cfg["privacy"]["min_group_size"]
    fig = resolve_path(cfg["paths"]["figures_dir"])
    fig.mkdir(parents=True, exist_ok=True)
    met = resolve_path(cfg["paths"]["metrics_dir"])
    met.mkdir(parents=True, exist_ok=True)

    cut = getattr(predictor, "alert_cutoff", thr)
    pred = predictor.predict(X_test)
    final = full_metrics(y_test, pred, thr, cut)
    final_same = full_metrics(y_test, pred, thr)
    logger.info(
        "Test | corte %.3f | RMSE %.4f | MAE %.4f | R2 %.4f | Sensibilidad %.3f",
        cut,
        final["rmse"],
        final["mae"],
        final["r2"],
        final["sensibilidad"],
    )

    # 1. Comparación con baselines
    models = dict(baselines)
    models["XGBoost final (optimizado)"] = predictor
    comp = compare_models(models, X_test, y_test, thr)
    # Misma predicción del modelo final con el corte de alerta calibrado en train.
    comp.loc[f"XGBoost final + corte calibrado ({cut:.3f})"] = final
    comp.round(4).to_csv(met / "model_comparison.csv")
    plot_model_comparison(comp, fig / "model_comparison.png")

    # 2. Ablación: ¿aporta algo el género?
    with_gender = build_xgb_pipeline(cfg["model"]["params"], seed, extra_categorical=["Gender"])
    with_gender.fit(X_train[FEATURES + ["Gender"]], y_train)
    pg = np.clip(with_gender.predict(X_test[FEATURES + ["Gender"]]), 0, 1)
    abl = pd.DataFrame(
        [
            {"variante": "Con Gender", **full_metrics(y_test, pg, thr)},
            {"variante": "Sin Gender (final)", **final_same},
        ]
    ).set_index("variante")
    abl.round(4).to_csv(met / "ablation_gender.csv")

    # 3. Equidad y análisis de errores por subgrupo
    imputed = X_test["Mental Fatigue Score"].isna()
    segments = {
        "gender": X_test["Gender"],
        "fatigue": np.where(imputed, "Fatiga imputada", "Fatiga observada"),
        "designation": X_test["Designation"].astype(int).astype(str),
        "company": X_test["Company Type"],
        "wfh": X_test["WFH Setup Available"].map({"Yes": "Con teletrabajo", "No": "Sin teletrabajo"}),
    }
    fairness = {}
    for name, g in segments.items():
        t = group_metrics(y_test, pred, pd.Series(g, index=X_test.index), thr, k, cutoff=cut)
        t.round(4).to_csv(met / f"group_metrics_{name}.csv")
        fairness[name] = {
            **fairness_gaps(t),
            "dpd_alerta": demographic_parity_difference(pred, g, cut),
            "eod_alerta": equalized_odds_difference(y_test, pred, g, thr, cut),
        }
    plot_group_metrics(
        group_metrics(y_test, pred, X_test["Gender"], thr, k, cutoff=cut),
        "Métricas por género (variable no usada por el modelo)",
        fig / "fairness_gender.png",
    )
    plot_group_metrics(
        group_metrics(y_test, pred, pd.Series(segments["fatigue"], index=X_test.index), thr, k, cutoff=cut),
        "Métricas según disponibilidad de Mental Fatigue Score",
        fig / "fairness_fatigue.png",
    )
    plot_group_metrics(
        group_metrics(
            y_test, pred, pd.Series(segments["designation"], index=X_test.index), thr, k, cutoff=cut
        ),
        "Métricas por nivel de Designation",
        fig / "fairness_designation.png",
    )

    bins = pd.cut(y_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0001], include_lowest=True, right=False)
    err = group_metrics(y_test, pred, bins.astype(str), thr, k, cutoff=cut)
    err.round(4).to_csv(met / "error_by_target_range.csv")

    # 4. Intervalos de predicción
    intervals = predictor.predict_interval(X_test)
    cov = interval_coverage(y_test, intervals)
    cov.round(4).to_csv(met / "interval_coverage.csv")

    # 5. Pruebas de estrés y cambio de distribución
    stress = stress_tests(predictor.predict, X_test, y_test, X_train, thr, 10, seed, cutoff=cut)
    stress.round(4).to_csv(met / "stress_tests.csv", index=False)
    plot_stress(stress, fig / "stress_tests.png")
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    shift = distribution_shift(
        lambda: build_xgb_pipeline(cfg["model"]["params"], seed),
        X_all,
        y_all,
        "Company Type",
        "Service",
        seed,
        thr,
    )
    shift.round(4).to_csv(met / "distribution_shift.csv")

    # 6. Figuras de diagnóstico
    plot_pred_vs_real(y_test, pred, fig / "pred_vs_real.png", thr)
    plot_residuals(y_test, pred, imputed.to_numpy(), fig / "residuals.png")
    imp = predictor.feature_importance()
    imp.to_csv(met / "feature_importance.csv", header=["importancia"])

    summary = {
        "alert_cutoff": cut,
        "test": final,
        "test_corte_igual_umbral": final_same,
        "comparacion": comp.round(4).to_dict(orient="index"),
        "ablacion_genero": abl.round(4).to_dict(orient="index"),
        "equidad": fairness,
        "cobertura_intervalos": cov.round(4).to_dict(orient="index"),
        "importancia_variables": imp.round(4).to_dict(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    save_json(summary, met / "final_metrics.json")
    write_report(
        summary, comp, stress, cov, resolve_path(cfg["paths"]["reports_dir"]) / "reporte_evaluacion.md"
    )
    return summary


def _md_table(df: pd.DataFrame, cols: list[str], index_name: str) -> str:
    """Convierte un DataFrame en tabla Markdown con 3 decimales."""
    head = f"| {index_name} | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [
        f"| {i} | " + " | ".join(f"{df.loc[i, c]:.3f}" if pd.notna(df.loc[i, c]) else "" for c in cols) + " |"
        for i in df.index
    ]
    return "\n".join([head, sep, *rows])


def write_report(
    summary: dict[str, Any], comp: pd.DataFrame, stress: pd.DataFrame, cov: pd.DataFrame, path
) -> None:
    """Genera un reporte Markdown con los resultados de la evaluación final."""
    from datetime import datetime

    t = summary["test"]
    stress_idx = stress.assign(caso=stress["escenario"] + " " + stress["nivel"].replace("-", "")).set_index(
        "caso"
    )
    lines = [
        "# Reporte de evaluación del modelo final",
        "",
        f"Generado automáticamente por `src/evaluate.py` el {datetime.now():%Y-%m-%d %H:%M}.",
        f"Conjunto de prueba: {summary['n_test']} registros. Corte de alerta: {summary['alert_cutoff']:.3f}.",
        "",
        "## Métricas principales",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in t.items()],
        "",
        "## Comparación con modelos de referencia",
        "",
        _md_table(comp, ["rmse", "mae", "r2", "sensibilidad", "precision", "f1", "auc_roc"], "Modelo"),
        "",
        "## Cobertura de los intervalos de predicción (90 % nominal)",
        "",
        _md_table(cov, ["n", "cobertura", "ancho_medio"], "Segmento"),
        "",
        "## Pruebas de estrés",
        "",
        _md_table(stress_idx, ["rmse", "sensibilidad", "precision", "f1"], "Escenario"),
        "",
        "## Equidad (brechas entre grupos)",
        "",
        "| Segmentación | Brecha MAE | Brecha sensibilidad | DPD alerta | EOD alerta |",
        "|---|---|---|---|---|",
        *[
            f"| {g} | {v['brecha_mae']:.3f} | {v['brecha_sensibilidad']:.3f} | "
            f"{v['dpd_alerta']:.3f} | {v['eod_alerta']:.3f} |"
            for g, v in summary["equidad"].items()
        ],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Reporte guardado en %s", path)


def main() -> dict[str, Any]:
    """Evalúa el modelo guardado en ``models/best_model.pkl`` sobre el test."""
    import joblib

    from src.data_processing import clean_data, load_raw_data, split_data
    from src.model import build_baselines
    from src.utils import load_config, resolve_path

    cfg = load_config()
    seed = cfg["project"]["seed"]
    predictor = joblib.load(resolve_path(cfg["paths"]["best_model"]))
    df = clean_data(load_raw_data())
    X_train, X_test, y_train, y_test = split_data(df, cfg["data"]["test_size"], seed)
    baselines = {n: m.fit(X_train, y_train) for n, m in build_baselines(seed).items()}
    v1 = joblib.load(resolve_path(cfg["paths"]["model_v1"]))
    baselines["XGBoost v1 (por defecto)"] = v1
    return run_evaluation(predictor, baselines, X_train, X_test, y_train, y_test, cfg)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
