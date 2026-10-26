"""Aplicación Streamlit: Monitor de Riesgo de Burnout Organizacional.

Ejecutar localmente desde la raíz del repositorio:
    streamlit run app/app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from helpers import (
    DESIGNATION_LABELS,
    EXAMPLES,
    FEATURE_LABELS,
    RISK_LEVELS,
    aggregate_segments,
    build_input,
    load_predictor,
    predict_batch,
    risk_level,
    set_alert_cutoff,
    what_if_curve,
)

from src.data_processing import FEATURES, DataValidationError
from src.utils import load_config

ASSETS = Path(__file__).parent / "assets"
CFG = load_config()
MIN_GROUP = CFG["privacy"]["min_group_size"]
THRESHOLD = CFG["alert"]["threshold"]
ACCENT = "#7a1f2b"

st.set_page_config(
    page_title="Monitor de Riesgo de Burnout",
    page_icon="🧭",
    layout="wide",
    menu_items={"About": "Proyecto Integrador · Maestría en Inteligencia Artificial · UEES"},
)

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.kpi {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: .8rem 1rem; height: 100%;}
.kpi .lab {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .75;}
.kpi .val {font-size: 1.7rem; font-weight: 700; line-height: 1.2;}
.kpi .sub {font-size: .8rem; opacity: .75;}
.pill {display:inline-block; padding:.15rem .7rem; border-radius:999px; color:#fff; font-weight:600;}
</style>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ estado
@st.cache_resource(show_spinner="Cargando el modelo...")
def get_model():
    """Carga el modelo una sola vez por servidor."""
    return load_predictor()


def init_state() -> None:
    defaults = {"history": [], "last": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_example(name: str) -> None:
    """Callback de los botones de ejemplo: carga valores en el formulario."""
    ex = EXAMPLES[name]
    st.session_state["in_designation"] = int(ex["Designation"])
    st.session_state["in_resource"] = float(ex["Resource Allocation"] or 5.0)
    st.session_state["in_resource_na"] = ex["Resource Allocation"] is None
    st.session_state["in_fatigue"] = float(ex["Mental Fatigue Score"] or 5.0)
    st.session_state["in_fatigue_na"] = ex["Mental Fatigue Score"] is None
    st.session_state["in_company"] = ex["Company Type"]
    st.session_state["in_wfh"] = ex["WFH Setup Available"]
    st.session_state["autorun"] = True


def alert_cutoff() -> float:
    """Corte de alerta vigente (límite inferior del nivel Alto)."""
    return RISK_LEVELS[2][0]


def fmt_prob(p: float) -> str:
    """Formatea una probabilidad evitando mostrar certezas absolutas (0 % o 100 %)."""
    if p > 0.99:
        return "> 99 %"
    if p < 0.01:
        return "< 1 %"
    return f"{p * 100:.0f} %"


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="lab">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


# --------------------------------------------------------------- gráficos
def gauge(value: float, lower: float, upper: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".2f", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.2, 0.4, 0.6, 0.8, 1]},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [{"range": [lo, min(hi, 1)], "color": c} for lo, hi, _, c in RISK_LEVELS],
                "threshold": {"line": {"color": "#111", "width": 5}, "thickness": 0.9, "value": value},
            },
        )
    )
    fig.add_annotation(
        x=0.5, y=-0.08, showarrow=False, font={"size": 13}, text=f"Intervalo 90 %: {lower:.2f} a {upper:.2f}"
    )
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=10, b=30))
    return fig


def waterfall(contrib: pd.Series, base: float) -> go.Figure:
    c = contrib.reindex(FEATURES)
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute"] + ["relative"] * len(c) + ["total"],
            y=["Promedio de la población"] + labels + ["Predicción"],
            x=[base] + list(c.values) + [0],
            text=[f"{base:.2f}"] + [f"{v:+.3f}" for v in c.values] + [f"{base + c.sum():.2f}"],
            textposition="outside",
            increasing={"marker": {"color": "#b3261e"}},
            decreasing={"marker": {"color": "#2e7d32"}},
            totals={"marker": {"color": ACCENT}},
            connector={"line": {"color": "#999", "width": 1}},
        )
    )
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Burn Rate",
        xaxis_range=[min(0, base + c[c < 0].sum()) - 0.02, max(base + c[c > 0].sum(), base) + 0.18],
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return fig


def what_if_figure(curve: pd.DataFrame, current: float | None, pred: float) -> go.Figure:
    fig = go.Figure()
    for lo, hi, _, col in RISK_LEVELS:
        fig.add_hrect(y0=lo, y1=min(hi, 1), fillcolor=col, opacity=0.10, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=curve["Mental Fatigue Score"],
            y=curve["pred"],
            mode="lines",
            line={"color": ACCENT, "width": 3},
            name="Predicción",
        )
    )
    if current is not None:
        fig.add_trace(
            go.Scatter(
                x=[current],
                y=[pred],
                mode="markers",
                name="Caso actual",
                marker={"size": 13, "color": "#111"},
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_range=[0, 1],
        xaxis_title="Fatiga mental (0 a 10)",
        yaxis_title="Burn Rate predicho",
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


# ------------------------------------------------------------ pestañas
def tab_individual(model) -> None:
    st.subheader("Autoevaluación y simulación de escenarios")
    st.caption(
        "Ingresa los datos de una situación laboral para estimar el nivel de riesgo de burnout. "
        "Los datos no se guardan en ningún servidor; el historial solo vive en esta sesión del navegador."
    )

    st.write("**Prueba con un ejemplo:**")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=load_example, args=(name,), width="stretch", key=f"ex_{name}")

    for k, v in {
        "in_designation": 2,
        "in_resource": 5.0,
        "in_resource_na": False,
        "in_fatigue": 6.2,
        "in_fatigue_na": False,
        "in_company": "Service",
        "in_wfh": "Yes",
    }.items():
        st.session_state.setdefault(k, v)

    with st.form("formulario"):
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                "Nivel jerárquico (Designation)",
                options=list(DESIGNATION_LABELS),
                format_func=DESIGNATION_LABELS.get,
                key="in_designation",
                help="0 = nivel inicial, 5 = dirección.",
            )
            st.slider(
                "Carga de trabajo asignada (Resource Allocation)",
                1.0,
                10.0,
                step=1.0,
                key="in_resource",
                help="Recursos u horas asignadas, en escala de 1 a 10.",
            )
            st.checkbox("No dispongo de este dato", key="in_resource_na")
            st.radio(
                "Tipo de empresa",
                ["Service", "Product"],
                horizontal=True,
                key="in_company",
                format_func={"Service": "Servicios", "Product": "Producto"}.get,
            )
        with c2:
            st.slider(
                "Fatiga mental (Mental Fatigue Score)",
                0.0,
                10.0,
                step=0.1,
                key="in_fatigue",
                help="Autopercepción de fatiga mental de 0 (ninguna) a 10 (máxima).",
            )
            st.checkbox(
                "No dispongo de este dato ",
                key="in_fatigue_na",
                help="El modelo puede estimar sin este dato, pero con mucha más incertidumbre.",
            )
            st.radio(
                "¿Tiene teletrabajo disponible?",
                ["Yes", "No"],
                horizontal=True,
                key="in_wfh",
                format_func={"Yes": "Sí", "No": "No"}.get,
            )
        submitted = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

    if submitted or st.session_state.pop("autorun", False):
        try:
            with st.spinner("Calculando..."):
                row = build_input(
                    st.session_state.in_designation,
                    None if st.session_state.in_resource_na else st.session_state.in_resource,
                    None if st.session_state.in_fatigue_na else st.session_state.in_fatigue,
                    st.session_state.in_company,
                    st.session_state.in_wfh,
                )
                res = model.predict_interval(row).iloc[0]
                contrib = model.explain(row).iloc[0]
            st.session_state.last = {"row": row, "res": res, "contrib": contrib}
            level, _ = risk_level(res["pred"])
            st.session_state.history.append(
                {
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    **{
                        FEATURE_LABELS[f]: (
                            "Sin dato"
                            if pd.isna(v := row[f].iloc[0])
                            else f"{v:g}"
                            if isinstance(v, float)
                            else str(v)
                        )
                        for f in FEATURES
                    },
                    "Burn Rate": round(float(res["pred"]), 3),
                    "Intervalo 90 %": f"{res['lower']:.2f} a {res['upper']:.2f}",
                    "Prob. riesgo alto": round(float(res["p_alert"]), 3),
                    "Nivel": level,
                }
            )
        except DataValidationError as exc:
            st.error(f"Revisa los datos ingresados: {exc}")
            return
        except Exception:  # noqa: BLE001
            st.error("Ocurrió un problema al calcular la predicción. Intenta de nuevo o recarga la página.")
            return

    last = st.session_state.last
    if last is None:
        st.info("Completa el formulario y pulsa **Calcular riesgo**, o elige un ejemplo.")
        return

    res, row, contrib = last["res"], last["row"], last["contrib"]
    level, color = risk_level(res["pred"])
    st.divider()
    st.markdown(
        f"### Resultado: <span class='pill' style='background:{color}'>Riesgo {level}</span>",
        unsafe_allow_html=True,
    )
    if res["fatigue_imputed"]:
        st.warning(
            "No se ingresó la fatiga mental, la variable más informativa. El modelo usó la mediana "
            "poblacional y el intervalo es casi el doble de ancho: interpreta el resultado con cautela."
        )

    g1, g2 = st.columns([1.1, 1])
    with g1:
        st.plotly_chart(gauge(res["pred"], res["lower"], res["upper"]), width="stretch")
    with g2:
        k1, k2 = st.columns(2)
        k1.markdown(kpi("Burn Rate estimado", f"{res['pred']:.2f}", "escala 0 a 1"), unsafe_allow_html=True)
        k2.markdown(
            kpi("Prob. de riesgo alto", fmt_prob(res["p_alert"]), f"Burn Rate ≥ {THRESHOLD:.1f}"),
            unsafe_allow_html=True,
        )
        st.write("")
        k3, k4 = st.columns(2)
        k3.markdown(
            kpi(
                "Intervalo 90 %", f"{res['lower']:.2f} – {res['upper']:.2f}", "rango probable del valor real"
            ),
            unsafe_allow_html=True,
        )
        k4.markdown(
            kpi(
                "Confianza del dato",
                "Reducida" if res["fatigue_imputed"] else "Normal",
                "según datos disponibles",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "La probabilidad se estima con la distribución de errores del modelo en datos "
            "no vistos (predicción conformal)."
        )

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**¿Qué explica este resultado?**")
        st.plotly_chart(waterfall(contrib[FEATURES], float(contrib["base"])), width="stretch")
        st.caption("Contribución de cada variable (valores SHAP). Rojo sube el riesgo, verde lo reduce.")
    with e2:
        st.markdown("**¿Qué pasaría si cambia la fatiga mental?**")
        base = row.copy()
        curve = what_if_curve(model, base, "Mental Fatigue Score", np.linspace(0, 10, 51))
        current = None if res["fatigue_imputed"] else float(row["Mental Fatigue Score"].iloc[0])
        st.plotly_chart(what_if_figure(curve, current, res["pred"]), width="stretch")
        st.caption(
            "Manteniendo fijas las demás variables. Es una asociación del modelo, no una relación causal."
        )

    with st.expander("Sugerencias generales de bienestar", expanded=level != "Bajo"):
        st.markdown(
            "- Conversa sobre la carga de trabajo y las prioridades con tu líder o el área de bienestar.\n"
            "- Planifica pausas activas y desconexión fuera del horario laboral.\n"
            "- Si el cansancio es persistente, busca apoyo profesional de salud ocupacional o psicológica.\n\n"
            "*Esta herramienta no es un diagnóstico médico ni psicológico.*"
        )

    if st.session_state.history:
        st.divider()
        st.markdown("#### Historial y comparación de esta sesión")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, width="stretch", hide_index=True)
        if len(hist) > 1:
            fig = go.Figure(
                go.Bar(
                    x=[f"#{i + 1} {h}" for i, h in enumerate(hist["hora"])],
                    y=hist["Burn Rate"],
                    marker_color=[risk_level(v)[1] for v in hist["Burn Rate"]],
                    text=hist["Burn Rate"],
                    textposition="outside",
                )
            )
            fig.add_hline(y=alert_cutoff(), line_dash="dash", annotation_text="Corte de alerta")
            fig.update_layout(
                height=280, yaxis_range=[0, 1.1], margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Burn Rate"
            )
            st.plotly_chart(fig, width="stretch")
        b1, b2 = st.columns(2)
        b1.download_button(
            "Descargar historial (CSV)",
            hist.to_csv(index=False).encode("utf-8-sig"),
            "historial_predicciones.csv",
            "text/csv",
            width="stretch",
        )
        if b2.button("Borrar historial", width="stretch"):
            st.session_state.history = []
            st.rerun()


def tab_batch(model) -> None:
    st.subheader("Análisis organizacional agregado")
    st.caption(
        f"Sube un CSV con varias personas para obtener **solo resultados agregados**. Por diseño, la "
        f"aplicación no muestra predicciones individuales en esta vista y oculta cualquier segmento "
        f"con menos de {MIN_GROUP} personas."
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        up = st.file_uploader("Archivo CSV", type=["csv"], help="Columnas: " + ", ".join(FEATURES))
    with c2:
        st.download_button(
            "Descargar plantilla CSV",
            (ASSETS / "plantilla.csv").read_bytes(),
            "plantilla_burnout.csv",
            "text/csv",
            width="stretch",
        )
        use_example = st.button("Usar datos de ejemplo (400 registros)", width="stretch")

    if use_example:
        st.session_state.batch_source = "ejemplo"
    if up is not None:
        st.session_state.batch_source = "archivo"
    src = st.session_state.get("batch_source")
    if src is None:
        st.info("Sube un archivo o pulsa **Usar datos de ejemplo**.")
        return

    try:
        raw = (
            pd.read_csv(up)
            if (src == "archivo" and up is not None)
            else pd.read_csv(ASSETS / "ejemplo_lote.csv")
        )
        if len(raw) > 50_000:
            st.error("El archivo supera las 50 000 filas permitidas.")
            return
        with st.spinner(f"Evaluando {len(raw)} registros..."):
            data = predict_batch(model, raw)
    except DataValidationError as exc:
        st.error(
            f"El archivo no tiene el formato esperado: {exc}. Descarga la plantilla para ver el formato."
        )
        return
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
        st.error("No se pudo leer el archivo. Verifica que sea un CSV separado por comas y en UTF-8.")
        return

    if len(data) < MIN_GROUP:
        st.warning(f"Se necesitan al menos {MIN_GROUP} registros para mostrar resultados agregados.")
        return

    n_imp = int(data["fatigue_imputed"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Personas evaluadas", f"{len(data):,}"), unsafe_allow_html=True)
    k2.markdown(kpi("Burn Rate medio", f"{data['pred'].mean():.2f}"), unsafe_allow_html=True)
    k3.markdown(
        kpi(
            "Con alerta de riesgo alto",
            f"{data['alert'].mean() * 100:.1f} %",
            f"predicción ≥ {alert_cutoff():.2f}",
        ),
        unsafe_allow_html=True,
    )
    k4.markdown(
        kpi("Sin dato de fatiga", f"{n_imp / len(data) * 100:.1f} %", "resultados menos precisos"),
        unsafe_allow_html=True,
    )

    h1, h2 = st.columns(2)
    with h1:
        fig = go.Figure(go.Histogram(x=data["pred"], nbinsx=30, marker_color=ACCENT))
        fig.add_vline(x=alert_cutoff(), line_dash="dash", annotation_text="Corte de alerta")
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Distribución del Burn Rate estimado",
            xaxis_title="Burn Rate",
            yaxis_title="Personas",
        )
        st.plotly_chart(fig, width="stretch")
    with h2:
        counts = (
            data["pred"]
            .apply(lambda v: risk_level(v)[0])
            .value_counts()
            .reindex(["Bajo", "Moderado", "Alto"])
            .fillna(0)
        )
        fig = go.Figure(
            go.Bar(
                x=counts.index,
                y=counts.values,
                marker_color=[c for *_, c in RISK_LEVELS],
                text=[f"{v / len(data) * 100:.0f} %" for v in counts.values],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Personas por nivel de riesgo",
            yaxis_title="Personas",
        )
        st.plotly_chart(fig, width="stretch")

    seg_opts = {
        "Nivel jerárquico": "Designation",
        "Tipo de empresa": "Company Type",
        "Teletrabajo": "WFH Setup Available",
    }
    seg_label = st.selectbox("Ver resultados por segmento", list(seg_opts))
    agg = aggregate_segments(data, seg_opts[seg_label], MIN_GROUP)
    vis = agg[agg["visible"]]
    xlab = (
        vis[seg_opts[seg_label]].astype(int).map(DESIGNATION_LABELS)
        if seg_opts[seg_label] == "Designation"
        else vis[seg_opts[seg_label]].astype(str)
    )
    fig = go.Figure(
        go.Bar(
            x=xlab,
            y=vis["pct_riesgo_alto"],
            marker_color=ACCENT,
            text=[f"{v:.0f} %" for v in vis["pct_riesgo_alto"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% en riesgo alto", xaxis_title=seg_label
    )
    st.plotly_chart(fig, width="stretch")
    show = agg.rename(
        columns={
            seg_opts[seg_label]: seg_label,
            "personas": "Personas",
            "burn_rate_medio": "Burn Rate medio",
            "pct_riesgo_alto": "% riesgo alto",
        }
    )
    show["Personas"] = [
        f"{n:,}" if v else f"< {MIN_GROUP}" for n, v in zip(show["Personas"], show["visible"])
    ]
    show = show.drop(columns="visible")
    if seg_opts[seg_label] == "Designation":
        show[seg_label] = show[seg_label].astype(int).map(DESIGNATION_LABELS)
    hidden = f"Oculto (< {MIN_GROUP} personas)"
    show["Burn Rate medio"] = show["Burn Rate medio"].map(lambda v: hidden if pd.isna(v) else f"{v:.2f}")
    show["% riesgo alto"] = show["% riesgo alto"].map(lambda v: hidden if pd.isna(v) else f"{v:.1f} %")
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button(
        "Descargar resumen agregado (CSV)",
        show.to_csv(index=False).encode("utf-8-sig"),
        "resumen_agregado_burnout.csv",
        "text/csv",
    )


def tab_about(model) -> None:
    md = model.metadata
    tm = md.get("test_metrics", {})
    st.subheader("Acerca del modelo")
    st.markdown(
        "El modelo estima el **Burn Rate** (índice de agotamiento laboral de 0 a 1) a partir de cinco variables "
        "laborales. Es un **XGBoost** de regresión con hiperparámetros optimizados con Optuna, un indicador de "
        "dato faltante para la fatiga mental e **intervalos de predicción conformales** calibrados por separado "
        "según si la fatiga fue observada o imputada. Se entrenó con el dataset público *Are Your Employees "
        "Burning Out?* (HackerEarth, 21 626 registros con objetivo)."
    )
    k = st.columns(5)
    for col, (lab, key, fmt) in zip(
        k,
        [
            ("RMSE", "rmse", "{:.3f}"),
            ("MAE", "mae", "{:.3f}"),
            ("R²", "r2", "{:.3f}"),
            ("Sensibilidad alerta", "sensibilidad", "{:.2f}"),
            ("AUC alerta", "auc_roc", "{:.3f}"),
        ],
    ):
        col.markdown(
            kpi(lab, fmt.format(tm.get(key, float("nan"))), "conjunto de prueba"), unsafe_allow_html=True
        )

    comp = md.get("comparison")
    if comp:
        st.markdown("#### Comparación con modelos de referencia")
        dfc = pd.DataFrame(comp).T[["rmse", "mae", "r2", "sensibilidad", "precision", "auc_roc"]]
        dfc.columns = ["RMSE", "MAE", "R²", "Sensibilidad", "Precisión", "AUC"]
        st.dataframe(dfc.style.format("{:.3f}").highlight_min("RMSE", color="#e8d6d9"), width="stretch")
        st.caption(
            "El baseline exigido de una sola variable (solo fatiga) tiene RMSE 0.087; el modelo final lo "
            "reduce en un tercio y detecta más casos de riesgo alto."
        )
    imp = md.get("feature_importance")
    c1, c2 = st.columns(2)
    if imp:
        with c1:
            s = pd.Series(imp).sort_values()
            fig = go.Figure(
                go.Bar(
                    x=s.values,
                    y=[FEATURE_LABELS.get(i, i) for i in s.index],
                    orientation="h",
                    marker_color=ACCENT,
                    text=[f"{v * 100:.1f} %" for v in s.values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=40, t=30, b=10),
                title="Importancia global (ganancia)",
                xaxis_range=[0, 1],
                xaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, width="stretch")
    cov = md.get("interval_coverage_test")
    if cov:
        with c2:
            st.markdown("**Calidad de los intervalos (90 % nominal)**")
            dcov = pd.DataFrame(cov).T.rename(
                columns={"n": "Casos", "cobertura": "Cobertura real", "ancho_medio": "Ancho medio"}
            )
            st.dataframe(
                dcov.style.format({"Casos": "{:.0f}", "Cobertura real": "{:.1%}", "Ancho medio": "{:.2f}"}),
                width="stretch",
            )
    st.markdown("#### Limitaciones y advertencias")
    st.markdown(
        "- El dataset proviene de una competencia de HackerEarth; su origen exacto y el significado preciso de "
        "Burn Rate no están documentados públicamente. Los resultados son de **validación simulada**, no clínica.\n"
        "- La fatiga mental explica cerca del 80 % de la predicción: si ese dato falta o está mal medido, el error se duplica.\n"
        "- No incluye área, antigüedad, salario ni factores personales; no capta causas del burnout.\n"
        "- Entrenado con datos de 2008 de una población específica: puede no generalizar a otras organizaciones o países.\n"
        f"- Versión {md.get('version', '1.0.0')} · entrenado el {md.get('trained_at', 's/f')} · "
        f"riesgo alto definido como Burn Rate ≥ {THRESHOLD:.1f}; la alerta se emite desde una predicción de "
        f"{alert_cutoff():.2f}, corte calibrado con validación cruzada para detectar al menos el 85 % de los casos."
    )


def tab_ethics(model) -> None:
    st.subheader("Ética y uso responsable")
    st.error(
        "**Uso prohibido:** esta herramienta **no** debe usarse para contratar, despedir, evaluar desempeño, "
        "ascender ni sancionar a ninguna persona."
    )
    st.markdown(
        "**Para qué sí sirve:** que una persona reflexione sobre su propia situación, y que un equipo de bienestar "
        "o talento humano observe tendencias **agregadas** para priorizar programas preventivos."
    )
    fair = model.metadata.get("fairness", {})
    g = fair.get("gender", {})
    f = fair.get("fatigue", {})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Equidad por género")
        st.markdown(
            "El género **no se usa** como variable del modelo; se conserva solo para auditar. "
            f"La diferencia de error (MAE) entre mujeres y hombres es {g.get('brecha_mae', 0):.3f} y la diferencia "
            f"de sensibilidad es {g.get('brecha_sensibilidad', 0) * 100:.1f} puntos. La diferencia en tasa de alertas "
            f"({g.get('dpd_alerta', 0) * 100:.0f} puntos) refleja una diferencia real en los datos y no un error del modelo."
        )
    with c2:
        st.markdown("#### Grupo más vulnerable: datos incompletos")
        st.markdown(
            "Las personas sin dato de fatiga mental reciben predicciones con casi el doble de error. "
            "Mitigación: indicador de dato faltante en el modelo, intervalos más anchos y aviso visible en la app. "
            f"Brecha de sensibilidad remanente: {f.get('brecha_sensibilidad', 0) * 100:.1f} puntos."
        )
    st.markdown("#### Privacidad")
    st.markdown(
        f"- La app no guarda datos: todo se procesa en memoria y se descarta al cerrar la sesión.\n"
        f"- La vista organizacional solo muestra agregados y oculta segmentos con menos de {MIN_GROUP} personas.\n"
        "- Los datos de salud mental son datos sensibles según la Ley Orgánica de Protección de Datos Personales "
        "del Ecuador (LOPDP): un uso real requiere consentimiento explícito, finalidad limitada y evaluación de impacto."
    )
    st.markdown("#### Responsabilidad")
    st.markdown(
        "Las decisiones siempre las toma una persona. Cualquier alerta debe derivar en una conversación "
        "voluntaria y confidencial con profesionales de bienestar, nunca en una acción disciplinaria."
    )


def tab_help() -> None:
    st.subheader("Cómo usar la aplicación")
    st.markdown(
        "1. En **Evaluación individual**, elige un ejemplo o completa el formulario y pulsa *Calcular riesgo*.\n"
        "2. Lee el nivel de riesgo, el intervalo y la probabilidad de riesgo alto. Revisa qué variables pesaron más.\n"
        "3. Usa la curva *¿Qué pasaría si...?* para explorar escenarios.\n"
        "4. Compara varias predicciones en el historial y descárgalo en CSV.\n"
        "5. En **Análisis organizacional**, sube un CSV con la plantilla para ver resultados agregados.\n\n"
        "El tema claro u oscuro se cambia en el menú ⋮ (arriba a la derecha) → *Settings*."
    )
    st.markdown("#### Preguntas frecuentes")
    with st.expander("¿Qué significa Burn Rate?"):
        st.write(
            "Es un índice de 0 a 1 del dataset original que representa el nivel de agotamiento laboral; "
            "valores de 0.6 o más se consideran riesgo alto en este proyecto."
        )
    with st.expander("¿Por qué el resultado cambia poco al modificar el tipo de empresa o el teletrabajo?"):
        st.write(
            "Porque en los datos esas variables casi no aportan información una vez conocidas la fatiga y la carga."
        )
    with st.expander("¿Qué hago si no conozco mi puntaje de fatiga?"):
        st.write(
            "Marca 'No dispongo de este dato'. El modelo estimará igualmente, con un intervalo más amplio."
        )
    with st.expander("Mi CSV da error"):
        st.write(
            "Descarga la plantilla, respeta los nombres de columna exactos, usa 'Service'/'Product' y 'Yes'/'No', "
            "y guarda el archivo como CSV UTF-8 separado por comas."
        )


# ----------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.title("🧭 Monitor de Riesgo de Burnout Organizacional")
    st.markdown(
        "Estimación del riesgo de agotamiento laboral con aprendizaje automático, diseñada para "
        "**prevención y bienestar**, no para evaluar personas."
    )
    try:
        model, origin = get_model()
    except Exception:  # noqa: BLE001
        st.error("No fue posible cargar el modelo. Intenta recargar la página en unos minutos.")
        st.stop()
    set_alert_cutoff(float(getattr(model, "alert_cutoff", THRESHOLD)))
    if origin == "reentrenado":
        st.toast("El modelo se reentrenó al iniciar por compatibilidad de versiones.")

    with st.sidebar:
        st.markdown("### Proyecto Integrador")
        st.markdown("Maestría en Inteligencia Artificial · UEES\n\n**Autor:** Oscar Paúl Naranjo Castro")
        st.divider()
        st.markdown("**Niveles de riesgo**")
        for lo, hi, lab, col in RISK_LEVELS:
            st.markdown(
                f"<span class='pill' style='background:{col}'>{lab}</span> &nbsp;{lo:.2f} a {min(hi, 1):.2f}",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("Herramienta académica de apoyo. No sustituye la evaluación de profesionales de salud.")

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Evaluación individual",
            "Análisis organizacional",
            "Acerca del modelo",
            "Ética y uso responsable",
            "Ayuda",
        ]
    )
    with t1:
        tab_individual(model)
    with t2:
        tab_batch(model)
    with t3:
        tab_about(model)
    with t4:
        tab_ethics(model)
    with t5:
        tab_help()


main()
