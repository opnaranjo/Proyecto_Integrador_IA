"""Optimización y análisis de sensibilidad de hiperparámetros (metodología del Workshop S5).

Flujo:
  1. Búsqueda bayesiana con Optuna (TPE) minimizando el RMSE de validación
     cruzada, calculado solo sobre el conjunto de entrenamiento.
  2. Importancia de hiperparámetros (fANOVA de Optuna).
  3. Efecto individual: gráficos *slice* y dependencia parcial sobre un modelo
     sustituto (Random Forest que predice el RMSE a partir de los hiperparámetros).
  4. Interacciones: dependencia parcial 2D del par más importante.

Uso:
    python -m src.optimize --trials 60
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import KFold, cross_val_score

from src.data_processing import FEATURES, clean_data, load_raw_data, split_data
from src.model import build_xgb_pipeline
from src.utils import get_logger, load_config, resolve_path, save_json

logger = get_logger(__name__)

# Configuración ajustada a mano en el diagnóstico de sobreajuste, previa a Optuna.
PRIOR_MANUAL_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.5,
}

SEARCH_SPACE_DOC = {
    "n_estimators": "100 a 1000 (entero)",
    "learning_rate": "0.01 a 0.30 (log)",
    "max_depth": "2 a 10 (entero)",
    "min_child_weight": "1 a 20 (log)",
    "subsample": "0.5 a 1.0",
    "colsample_bytree": "0.5 a 1.0",
    "reg_lambda": "0.001 a 10 (log)",
    "reg_alpha": "0.001 a 10 (log)",
}


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Define el espacio de búsqueda de hiperparámetros de XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def cv_rmse(
    params: dict[str, Any] | None, X: pd.DataFrame, y: pd.Series, folds: int = 5, seed: int = 42
) -> float:
    """RMSE promedio de validación cruzada para un conjunto de hiperparámetros."""
    pipe = build_xgb_pipeline(params, seed)
    scores = cross_val_score(
        pipe,
        X[FEATURES],
        y,
        cv=KFold(folds, shuffle=True, random_state=seed),
        scoring="neg_root_mean_squared_error",
    )
    return float(-scores.mean())


def run_study(
    X: pd.DataFrame, y: pd.Series, n_trials: int = 60, folds: int = 5, seed: int = 42
) -> optuna.Study:
    """Ejecuta la búsqueda bayesiana y devuelve el estudio de Optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", study_name="xgb_burnout", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda t: cv_rmse(suggest_params(t), X, y, folds, seed), n_trials=n_trials, show_progress_bar=False
    )
    logger.info("Mejor RMSE CV: %.5f con %s", study.best_value, study.best_params)
    return study


def param_importances(study: optuna.Study, seed: int = 42) -> pd.Series:
    """Importancia de hiperparámetros con fANOVA (proporción de varianza explicada)."""
    imp = optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
    )
    return pd.Series(imp).sort_values(ascending=False)


def surrogate_model(trials: pd.DataFrame, seed: int = 42) -> tuple[RandomForestRegressor, list[str]]:
    """Ajusta un Random Forest que aproxima RMSE = f(hiperparámetros)."""
    cols = [c for c in trials.columns if c.startswith("params_")]
    sur = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed)
    sur.fit(trials[cols].astype(float), trials["value"])
    return sur, cols


# ------------------------------------------------------------------ figuras
def plot_history(trials: pd.DataFrame, baseline_rmse: float, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.scatter(trials["number"], trials["value"], s=18, color="#888", label="Prueba")
    ax.plot(trials["number"], trials["value"].cummin(), color="#111", lw=2, label="Mejor hasta el momento")
    ax.axhline(
        baseline_rmse, color="#c0392b", ls="--", lw=1.2, label=f"XGBoost por defecto ({baseline_rmse:.4f})"
    )
    ax.set_xlabel("Número de prueba")
    ax.set_ylabel("RMSE (CV 5 pliegues)")
    ax.set_title("Historial de optimización con Optuna (TPE)")
    ax.set_ylim(top=min(trials["value"].max(), baseline_rmse * 1.25))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importances(imp: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    s = imp.sort_values()
    ax.barh(s.index, s.values, color="#444")
    for i, v in enumerate(s.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Importancia fANOVA (fracción de varianza del RMSE)")
    ax.set_title("Importancia de hiperparámetros")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_slices(trials: pd.DataFrame, order: list[str], path) -> None:
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow), sharey=True)
    for ax, p in zip(np.ravel(axes), order):
        ax.scatter(
            trials[f"params_{p}"],
            trials["value"],
            c=trials["number"],
            cmap="Greys",
            s=16,
            edgecolor="#333",
            lw=0.3,
        )
        ax.set_title(p, fontsize=10)
        if p in ("learning_rate", "min_child_weight", "reg_lambda", "reg_alpha"):
            ax.set_xscale("log")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    np.ravel(axes)[0].set_ylabel("RMSE CV")
    fig.suptitle(
        "Gráficos slice: RMSE de cada prueba según el valor del hiperparámetro (color = orden de la prueba)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pdp(sur, trials: pd.DataFrame, cols: list[str], order: list[str], path) -> None:
    feats = [f"params_{p}" for p in order]
    n = len(feats)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow))
    X = trials[cols].astype(float)
    PartialDependenceDisplay.from_estimator(sur, X, feats, ax=np.ravel(axes)[:n], line_kw={"color": "#111"})
    for ax in np.ravel(axes)[:n]:
        ax.set_xlabel(ax.get_xlabel().replace("params_", ""))
        ax.set_ylabel("RMSE parcial")
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle("Dependencia parcial del RMSE (modelo sustituto Random Forest)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_interaction(sur, trials: pd.DataFrame, cols: list[str], pair: tuple[str, str], path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.6))
    X = trials[cols].astype(float)
    disp = PartialDependenceDisplay.from_estimator(
        sur, X, [(f"params_{pair[0]}", f"params_{pair[1]}")], ax=ax, contour_kw={"cmap": "Greys"}
    )
    a = disp.axes_[0, 0]
    a.set_xlabel(pair[0])
    a.set_ylabel(pair[1])
    a.scatter(trials[f"params_{pair[0]}"], trials[f"params_{pair[1]}"], s=8, color="#c0392b", alpha=0.6)
    a.set_title(f"Interacción {pair[0]} × {pair[1]} (RMSE parcial)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def interaction_strength(
    sur, trials: pd.DataFrame, cols: list[str], pair: tuple[str, str], grid: int = 15
) -> float:
    """Estadístico H de Friedman aproximado para un par de hiperparámetros.

    Compara la dependencia parcial conjunta con la suma de las individuales.
    Valores cercanos a 0 indican efectos aditivos (sin interacción).
    """
    X = trials[cols].astype(float).reset_index(drop=True)
    a, b = f"params_{pair[0]}", f"params_{pair[1]}"
    ga = np.quantile(X[a], np.linspace(0.05, 0.95, grid))
    gb = np.quantile(X[b], np.linspace(0.05, 0.95, grid))

    def pd_value(assign: dict[str, float]) -> float:
        Z = X.copy()
        for k, v in assign.items():
            Z[k] = v
        return float(sur.predict(Z).mean())

    pa = np.array([pd_value({a: v}) for v in ga])
    pb = np.array([pd_value({b: v}) for v in gb])
    pab = np.array([[pd_value({a: va, b: vb}) for vb in gb] for va in ga])
    pa_c, pb_c = pa - pa.mean(), pb - pb.mean()
    pab_c = pab - pab.mean()
    num = ((pab_c - pa_c[:, None] - pb_c[None, :]) ** 2).sum()
    den = (pab_c**2).sum()
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def main(n_trials: int | None = None) -> dict[str, Any]:
    """Ejecuta la optimización completa y guarda tablas y figuras en ``results/``."""
    cfg = load_config()
    seed = cfg["project"]["seed"]
    n_trials = n_trials or cfg["optimization"]["n_trials"]
    folds = cfg["optimization"]["cv_folds"]
    fig_dir = resolve_path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir = resolve_path(cfg["paths"]["metrics_dir"])
    met_dir.mkdir(parents=True, exist_ok=True)

    df = clean_data(load_raw_data())
    X_train, _, y_train, _ = split_data(df, cfg["data"]["test_size"], seed)

    default_rmse = cv_rmse(None, X_train, y_train, folds, seed)
    manual_rmse = cv_rmse(PRIOR_MANUAL_PARAMS, X_train, y_train, folds, seed)
    logger.info(
        "RMSE CV XGBoost por defecto: %.5f | configuración manual previa: %.5f", default_rmse, manual_rmse
    )

    study = run_study(X_train, y_train, n_trials, folds, seed)
    trials = study.trials_dataframe()
    trials.to_csv(met_dir / "optuna_trials.csv", index=False)

    imp = param_importances(study, seed)
    imp.to_csv(met_dir / "hyperparam_importance.csv", header=["importance"])
    order = list(imp.index)
    sur, cols = surrogate_model(trials, seed)
    pair = (order[0], order[1])
    h_top = interaction_strength(sur, trials, cols, pair)
    pair2 = ("max_depth", "n_estimators")
    h_depth = interaction_strength(sur, trials, cols, pair2)
    pair3 = ("learning_rate", "n_estimators")
    h_lr = interaction_strength(sur, trials, cols, pair3)

    plot_history(trials, default_rmse, fig_dir / "opt_history.png")
    plot_importances(imp, fig_dir / "opt_param_importance.png")
    plot_slices(trials, order, fig_dir / "opt_slices.png")
    plot_pdp(sur, trials, cols, order, fig_dir / "opt_pdp.png")
    plot_interaction(sur, trials, cols, pair, fig_dir / "opt_interaction_top.png")
    plot_interaction(sur, trials, cols, pair2, fig_dir / "opt_interaction_depth_trees.png")
    plot_interaction(sur, trials, cols, pair3, fig_dir / "opt_interaction_lr_trees.png")

    q = trials["value"].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).to_dict()
    summary = {
        "n_trials": n_trials,
        "cv_folds": folds,
        "search_space": SEARCH_SPACE_DOC,
        "cv_rmse_default_xgb": default_rmse,
        "cv_rmse_manual_config": manual_rmse,
        "cv_rmse_best": study.best_value,
        "best_params": study.best_params,
        "rmse_quantiles_across_trials": {str(k): v for k, v in q.items()},
        "param_importance": imp.to_dict(),
        "interaction_H": {
            f"{pair[0]} x {pair[1]}": h_top,
            f"{pair2[0]} x {pair2[1]}": h_depth,
            f"{pair3[0]} x {pair3[1]}": h_lr,
        },
    }
    save_json(summary, met_dir / "optimization_summary.json")
    save_json(study.best_params, cfg["paths"]["best_params"])
    return summary


if __name__ == "__main__":
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=None, help="Número de pruebas de Optuna")
    args = parser.parse_args()
    main(args.trials)
