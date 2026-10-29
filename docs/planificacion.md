# Planificación del proyecto

| Campo | Detalle |
|---|---|
| Proyecto | Predicción de Riesgo de Burnout Organizacional con Dashboard Interactivo |
| Programa | Maestría en Inteligencia Artificial, Universidad de Especialidades Espíritu Santo (UEES) |
| Asignatura | Proyecto Integrador en Inteligencia Artificial |
| Autor | Oscar Paúl Naranjo Castro (trabajo individual) |
| Docente | Gladys María Villegas Rugel |
| Entregable | Proyecto final (repositorio, aplicación desplegada y documentación) |

> **Nota sobre el contexto simulado.** El proyecto es académico y no cuenta con una organización patrocinadora. Los roles de "analista de talento humano", "comité de ética" o "gerente de área" que aparecen en este documento son construcciones hipotéticas que modelan cómo se gestionaría el sistema en una empresa real.

## 1. Definición del problema

El burnout (síndrome de agotamiento laboral) se reconoce como un fenómeno ocupacional que resulta del estrés crónico en el trabajo no gestionado con éxito (OMS, CIE-11, código QD85). Sus efectos se traducen en ausentismo, rotación, errores y deterioro de la salud de las personas. En la práctica, las organizaciones suelen detectarlo tarde, cuando el desgaste ya es visible.

El proyecto aborda una pregunta concreta: **¿es posible estimar, con datos laborales básicos, qué segmentos de una organización presentan mayor riesgo de agotamiento, de forma suficientemente precisa y transparente como para orientar programas preventivos de bienestar?**

## 2. Objetivos

**Objetivo general.** Estimar el índice de agotamiento laboral (*Burn Rate*) a partir de variables de clima laboral mediante un modelo de regresión con una capa de alerta de riesgo alto, y ponerlo a disposición de usuarios no técnicos en una aplicación web interactiva, desplegada en la nube y diseñada con salvaguardas éticas.

**Objetivos específicos y estado final.**

| # | Objetivo específico | Meta | Resultado | Estado |
|---|---|---|---|---|
| 1 | Analizar el dataset y verificar su completitud | > 95 % de celdas completas | 97.7 % | Cumplido |
| 2 | Entrenar y validar un modelo de regresión | RMSE ≤ 0.10, R² ≥ 0.75 | RMSE 0.058, R² 0.914 (test) | Cumplido |
| 3 | Implementar la alerta de riesgo alto | Sensibilidad ≥ 85 %, precisión ≥ 70 % | 86.4 % y 77.8 % (test) | Cumplido |
| 4 | Superar un baseline trivial de una sola variable | Mejora demostrable | RMSE 0.087 → 0.058 (33 % menos) | Cumplido |
| 5 | Desarrollar la aplicación con panel de interpretabilidad | Respuesta < 2 000 ms | ≈ 30 ms de cómputo por predicción con explicación (CPU local) | Cumplido |
| 6 | Auditar equidad y documentar limitaciones éticas | Auditoría por subgrupo | Ver `consideraciones_eticas.md` | Cumplido |

## 3. Justificación de la relevancia

* La Organización Mundial de la Salud incluye el burnout en la CIE-11 como fenómeno ocupacional, lo que lo sitúa en la agenda de salud laboral.
* Gallup (*State of the Global Workplace*) y la Society for Human Resource Management (SHRM) reportan de forma recurrente altos niveles de estrés y agotamiento laboral y su vínculo con la rotación. Estas cifras son externas y se citan como contexto: el proyecto no estima un retorno económico propio porque no dispone de una organización real ni de una línea base.
* La literatura revisada muestra un precedente directo (Teodósio et al., 2025, SVM con interfaz para uso organizacional, R² = 0.84) sobre el mismo dataset. Este proyecto lo extiende con una alerta orientada a la decisión, intervalos de incertidumbre, explicabilidad local y una auditoría de equidad.

## 4. Alcance

**Incluye**

* Pipeline reproducible de preprocesamiento, entrenamiento, optimización y evaluación.
* Modelo XGBoost de regresión con alerta calibrada e intervalos de predicción conformales.
* Aplicación Streamlit con evaluación individual (autoevaluación y simulación), análisis organizacional agregado por CSV, explicaciones SHAP e información del modelo.
* Auditoría de equidad, pruebas de estrés y documentación ética.
* Despliegue en un servidor gratuito en la nube con URL pública.

**No incluye**

* Integración con sistemas de nómina o talento humano (HRIS) ni datos en tiempo real.
* Diagnóstico clínico: *Burn Rate* es una variable proxy de un reto de ciencia de datos, no un instrumento validado como el Maslach Burnout Inventory.
* Autenticación de usuarios, almacenamiento de datos personales o análisis de texto libre.
* Validación con trabajadores reales o en el contexto ecuatoriano (propuesta como trabajo futuro).

## 5. Cronograma: planificado frente a real

Metodología ágil adaptada a un proyecto individual (el autor asume los roles de Product Owner, Scrum Master y equipo de desarrollo).

| Sprint | Planificado | Real | Desvío y causa |
|---|---|---|---|
| 1. Investigación y diseño | Dataset verificado, EDA inicial, arquitectura | Completado; presentación entregada el 20/09/2026 | Sin desvío. La retroalimentación docente añadió dos requisitos: comparar con un baseline de una variable y reconocer la ausencia de variable de área |
| 2. Desarrollo base | Preprocesamiento, modelo base, clasificación de riesgo | Completado: ficha técnica, comparación de algoritmos y EDA | Se descartó imputar el objetivo (planteado al inicio) para no fabricar etiquetas |
| 3. Optimización y validación | Ajuste de hiperparámetros, validación cruzada, dashboard | Diagnóstico de sobreajuste, auditoría de equidad (12/10/2026) y optimización con Optuna | El dashboard se rediseñó para mostrar solo agregados en la vista organizacional, a raíz de la auditoría ética |
| 4. Finalización | Pruebas integrales, documentación, presentación | Repositorio final, pruebas automatizadas, despliegue y videos | Se añadió el despliegue en la nube como requisito obligatorio; la fecha límite de entrega fue el 30/10/2026 |

## 6. Recursos

| Tipo | Recurso | Costo |
|---|---|---|
| Datos | *Are Your Employees Burning Out?* (HackerEarth, distribuido en Kaggle) | 0 USD |
| Hardware | Computador personal y Google Colab (CPU); el entrenamiento completo toma menos de 1 minuto y la optimización unos 2 minutos | 0 USD |
| Software | Python 3.13, scikit-learn, XGBoost, Optuna, Fairlearn, Streamlit, Plotly (todo de código abierto) | 0 USD |
| Despliegue | Streamlit Community Cloud (alternativa: Hugging Face Spaces con Docker) | 0 USD |
| Humano | Autor, con una dedicación estimada de 60 horas | No aplica |

## 7. Riesgos y mitigación

| Riesgo | Probabilidad | Impacto | Mitigación aplicada | Estado |
|---|---|---|---|---|
| El dataset no representa burnout clínico | Alta | Alto | Declarar *Burn Rate* como proxy en toda la documentación y en la app | Mitigado (riesgo residual aceptado) |
| Una sola variable (fatiga) explica casi todo y el modelo no aporta valor | Alta | Medio | Comparación explícita con baseline de una variable; métricas de alerta | Mitigado: el modelo reduce el error un 33 % |
| Fuga de información en el preprocesamiento | Media | Alto | Todo el preprocesamiento dentro de un `Pipeline`; test usado una sola vez | Mitigado |
| Uso del sistema para decisiones sobre personas | Media | Muy alto | Vista organizacional solo agregada, supresión de grupos < 10, advertencias, prohibición documentada | Mitigado parcialmente (depende de la organización usuaria) |
| Sesgo por género u otros grupos | Media | Alto | Exclusión de género del modelo y auditoría por subgrupo | Monitoreado |
| Incompatibilidad de versiones al desplegar | Media | Medio | Versiones fijadas y reentrenamiento automático si el modelo no carga | Mitigado |
| Falta de tiempo al cierre | Media | Medio | Priorización MoSCoW y automatización de la evaluación | Mitigado |

## Referencias

* World Health Organization. *Burn-out an "occupational phenomenon": International Classification of Diseases* (2019).
* Gallup. *State of the Global Workplace* (informe anual).
* B. W. G. Teodósio, M. J. O. T. Lira, P. H. M. Araújo y L. R. C. Farias, "Support vector machine-based burnout risk prediction with an interactive interface for organizational use", arXiv:2510.25509, 2025.
