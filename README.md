# Proyecto Integrador IA — Predicción de Riesgo de Burnout Organizacional

**Maestría en Inteligencia Artificial — UEES**
**Autor:** Oscar Paúl Naranjo Castro
**Docente:** Gladys María Villegas Rugel

## Descripción

Sistema que predice el nivel de riesgo de burnout organizacional a partir de variables de clima laboral (carga de trabajo, horas extra, satisfacción, antigüedad), mediante un modelo de regresión complementado con una capa de clasificación de riesgo (bajo/medio/alto), visualizado en un dashboard interactivo desarrollado en Streamlit.

> **Nota de transparencia:** la variable objetivo "Burn Rate" es una métrica sintética generada para un reto de ciencia de datos (HackerEarth), no un diagnóstico clínico validado (ej. Maslach Burnout Inventory). Este proyecto la utiliza explícitamente como una **proxy operacional** del riesgo de burnout. Ver el Documento de Alcance del Proyecto para el detalle completo de esta limitación.

## Dataset

- **Principal:** ["Are Your Employees Burning Out?"](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) (Kaggle/HackerEarth), 22,750 registros.
- **Plan B:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), en caso de inaccesibilidad del dataset principal.

## Métricas Objetivo

| KPI | Mínimo Aceptable | Objetivo (Excelencia) |
|---|---|---|
| RMSE | ≤ 0.15 | ≤ 0.10 |
| R² | ≥ 0.60 | ≥ 0.75 |
| Recall (clase "alto riesgo") | ≥ 75% | ≥ 85% |
| Precision (clase "alto riesgo") | ≥ 60% | ≥ 70% |
| Latencia del dashboard | < 5,000 ms | < 2,000 ms |

## Estructura del Repositorio

```
Proyecto_Integrador_IA/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── sprint1_eda.ipynb
│   ├── sprint2_modelado_base.ipynb
│   ├── sprint3_optimizacion.ipynb
│   └── sprint4_testing_final.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── dashboard/
│       └── app.py
└── docs/
    └── Presentacion_Proyecto.pdf
```

## Cómo Ejecutar

1. Clonar el repositorio: `git clone https://github.com/opnaranjo/Proyecto_Integrador_IA.git`
2. Abrir los notebooks en Google Colab, o instalar dependencias localmente: `pip install -r requirements.txt`
3. Descargar el dataset desde Kaggle (requiere cuenta gratuita) y colocarlo en `data/`
4. Ejecutar los notebooks en orden (Sprint 1 → Sprint 4)
5. Para el dashboard: `streamlit run src/dashboard/app.py`

## Cronograma

Ver el Cronograma con Metodología Ágil en la Presentación del Proyecto (`docs/Presentacion_Proyecto.pdf`) para el detalle de los 4 sprints, user stories y criterios de aceptación.

## Licencia y Uso Académico

Proyecto desarrollado con fines académicos para el Proyecto Integrador de la Maestría en Inteligencia Artificial (UEES). El uso de asistencia de IA generativa está declarado conforme a los lineamientos del programa.
