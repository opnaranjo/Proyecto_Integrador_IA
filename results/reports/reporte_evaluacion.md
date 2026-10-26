# Reporte de evaluación del modelo final

Generado automáticamente por `src/evaluate.py` el 2026-10-26 13:49.
Conjunto de prueba: 4326 registros. Corte de alerta: 0.580.

## Métricas principales

| Métrica | Valor |
|---|---|
| rmse | 0.0582 |
| mae | 0.0462 |
| r2 | 0.9140 |
| sensibilidad | 0.8642 |
| especificidad | 0.9229 |
| precision | 0.7782 |
| f1 | 0.8189 |
| tasa_alerta_real | 0.2383 |
| tasa_alerta_predicha | 0.2647 |
| auc_roc | 0.9700 |

## Comparación con modelos de referencia

| Modelo | rmse | mae | r2 | sensibilidad | precision | f1 | auc_roc |
|---|---|---|---|---|---|---|---|
| Media (sin modelo) | 0.199 | 0.159 | -0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| Solo fatiga (lineal) | 0.087 | 0.063 | 0.808 | 0.688 | 0.850 | 0.760 | 0.940 |
| Ridge | 0.070 | 0.053 | 0.875 | 0.740 | 0.825 | 0.780 | 0.961 |
| SVR-RBF | 0.059 | 0.046 | 0.912 | 0.794 | 0.837 | 0.815 | 0.969 |
| XGBoost v1 (por defecto) | 0.060 | 0.047 | 0.908 | 0.790 | 0.826 | 0.808 | 0.967 |
| XGBoost final (optimizado) | 0.058 | 0.046 | 0.914 | 0.790 | 0.838 | 0.814 | 0.970 |
| XGBoost final + corte calibrado (0.580) | 0.058 | 0.046 | 0.914 | 0.864 | 0.778 | 0.819 | 0.970 |

## Cobertura de los intervalos de predicción (90 % nominal)

| Segmento | n | cobertura | ancho_medio |
|---|---|---|---|
| Fatiga imputada | 411.000 | 0.922 | 0.324 |
| Fatiga observada | 3915.000 | 0.910 | 0.171 |
| Total | 4326.000 | 0.911 | 0.186 |

## Pruebas de estrés

| Escenario | rmse | sensibilidad | precision | f1 |
|---|---|---|---|---|
| Sin perturbar  | 0.058 | 0.864 | 0.778 | 0.819 |
| Ruido en entradas σ=0.1 | 0.065 | 0.809 | 0.818 | 0.813 |
| Ruido en entradas σ=0.25 | 0.074 | 0.791 | 0.795 | 0.793 |
| Ruido en entradas σ=0.5 | 0.101 | 0.754 | 0.707 | 0.730 |
| Ruido en entradas σ=1.0 | 0.168 | 0.693 | 0.547 | 0.611 |
| Datos faltantes 10% | 0.065 | 0.846 | 0.773 | 0.808 |
| Datos faltantes 30% | 0.079 | 0.794 | 0.763 | 0.778 |
| Datos faltantes 50% | 0.094 | 0.729 | 0.749 | 0.739 |
| Fatiga no disponible 100% | 0.101 | 0.733 | 0.696 | 0.714 |
| Cambio de prevalencia 10% riesgo alto | 0.057 | 0.864 | 0.551 | 0.673 |
| Cambio de prevalencia 30% riesgo alto | 0.059 | 0.862 | 0.824 | 0.843 |
| Cambio de prevalencia 50% riesgo alto | 0.061 | 0.866 | 0.917 | 0.891 |
| Error en Designation 20% ±1 | 0.060 | 0.864 | 0.774 | 0.817 |

## Equidad (brechas entre grupos)

| Segmentación | Brecha MAE | Brecha sensibilidad | DPD alerta | EOD alerta |
|---|---|---|---|---|
| gender | 0.001 | 0.051 | 0.111 | 0.051 |
| fatigue | 0.037 | 0.082 | 0.019 | 0.082 |
| designation | 0.014 | 0.298 | 1.000 | 1.000 |
| company | 0.001 | 0.010 | 0.019 | 0.010 |
| wfh | 0.001 | 0.026 | 0.244 | 0.102 |
