# Datos del proyecto

## Fuente

| Campo | Detalle |
|---|---|
| Nombre | *Are Your Employees Burning Out?* |
| Origen | HackerEarth Machine Learning Challenge: [Predict the employee burn out rate](https://www.hackerearth.com/problem/machine-learning/predict-the-employee-burn-out-rate-7-6340b4e3) |
| Distribución pública | Kaggle: [blurredmachine/are-your-employees-burning-out](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) |
| Archivo usado | `train.csv` (único archivo con la variable objetivo) |
| Licencia | La publicación no declara una licencia abierta explícita. Se usa con fines exclusivamente académicos y no comerciales, citando la fuente. Antes de cualquier uso comercial debe verificarse la licencia con los titulares. |

## Estructura

```
data/
├── raw/train.csv                   # copia sin modificar del archivo original
└── processed/burnout_clean.csv     # generado por src/data_processing.py
```

`data/processed/burnout_clean.csv` se regenera con `python -m src.train`.

## Diccionario de variables

| Variable | Tipo | Descripción | Faltantes | Uso en el modelo |
|---|---|---|---|---|
| Employee ID | Texto | Identificador anónimo | 0 % | Descartada |
| Date of Joining | Fecha | Fecha de ingreso (solo año 2008) | 0 % | Descartada (correlación 0.001 con el objetivo) |
| Gender | Categórica | Female / Male | 0 % | **Excluida del modelo**, solo auditoría de equidad |
| Company Type | Categórica | Service / Product | 0 % | Entrada (one-hot) |
| WFH Setup Available | Categórica | Teletrabajo disponible: Yes / No | 0 % | Entrada (one-hot) |
| Designation | Ordinal 0 a 5 | Nivel jerárquico (0 = inicial, 5 = dirección) | 0 % | Entrada |
| Resource Allocation | Numérica 1 a 10 | Recursos u horas de trabajo asignadas | 6.1 % | Entrada (mediana + indicador) |
| Mental Fatigue Score | Numérica 0 a 10 | Nivel de fatiga mental reportado | 9.3 % | Entrada (mediana + indicador) |
| Burn Rate | Numérica 0 a 1 | Índice de agotamiento laboral (objetivo) | 4.9 % | Objetivo; filas sin valor se eliminan |

## Tamaño

* 22 750 registros y 9 columnas en `train.csv`.
* 21 626 registros con objetivo observado, divididos en 17 300 de entrenamiento y 4 326 de prueba (80/20, estratificado por alerta, semilla 42).

## Limitaciones conocidas

* No hay documentación pública sobre cómo se midieron *Mental Fatigue Score* ni *Burn Rate*; podrían ser datos simulados o derivados.
* No incluye área o departamento, antigüedad real, edad, salario ni país.
* Representa una población de 2008; no se garantiza que generalice a organizaciones actuales ni al contexto ecuatoriano.
