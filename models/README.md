# Modelos entrenados

| Archivo | Contenido | RMSE test | R² test | Sensibilidad alerta |
|---|---|---|---|---|
| `best_model.pkl` | **Modelo de producción.** Objeto `BurnoutPredictor` (ver `src/model.py`): pipeline de preprocesamiento + XGBoost optimizado con Optuna, indicador de fatiga faltante, residuos de calibración para intervalos conformales y metadatos (métricas, versiones, fecha). | 0.0582 | 0.914 | 0.790 |
| `model_v1.pkl` | Versión inicial: pipeline con XGBoost de hiperparámetros por defecto y sin indicador de faltantes. Se conserva para comparar antes y después de la optimización. | 0.0601 | 0.909 | 0.790 |
| `best_params.json` | Mejores hiperparámetros encontrados por Optuna (60 pruebas, CV de 5 pliegues). | | | |

## Cómo regenerarlos

```bash
python -m src.optimize --trials 60   # opcional: rehace la búsqueda de hiperparámetros
python -m src.train                  # entrena, evalúa y guarda ambos modelos
```

## Cómo usarlos

```python
import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")   # ejecutar desde la raíz del repositorio
x = pd.DataFrame([{"Designation": 3, "Resource Allocation": 6, "Mental Fatigue Score": 7.2,
                   "Company Type": "Service", "WFH Setup Available": "No"}])
model.predict_interval(x)   # pred, lower, upper, p_alert, alert, fatigue_imputed
model.explain(x)            # contribuciones SHAP por variable
```

## Compatibilidad

Los archivos `.pkl` se generaron con Python 3.13, scikit-learn 1.9.1 y XGBoost 3.4.1. Si otra versión no puede leerlos, la aplicación reentrena el modelo automáticamente al iniciar (unos 10 segundos) a partir de `data/raw/train.csv` y `config.yaml`.
