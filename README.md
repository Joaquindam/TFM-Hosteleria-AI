# TFM-Hosteleria-AI
Pipeline de analítica predictiva sobre los datos operativos de un restaurante de Pozuelo de Alarcón (Madrid). Arquitectura medallion (Bronze → Silver → Gold) sobre Parquet, con cinco modelos predictivos independientes y un agente de IA que generaliza la ingesta a otros restaurantes.

## Estructura del Repositorio
```
├── notebooks/
│   ├── 01_ingesta_bronze.ipynb
│   ├── 01_pdf_process.ipynb
│   ├── 02_depuracion_silver.ipynb
│   ├── 02b_depuracion_silver_agente.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_prediccion_ventas.ipynb
│   ├── 04_prediccion_comandas.ipynb
│   ├── 05_prediccion_facturacion_diaria_definitivo.ipynb
│   ├── 06_pred_noshows.ipynb
│   ├── 07_prediccion_ticket_medio_definitivo.ipynb
│   ├── 09_prediccion_facturacion_largo_plazo.ipynb
│   └── 10_prediccion_ticket_medio_largo_plazo.ipynb
├── tfm_io.py              # lectores de las fuentes crudas → Bronze
├── tfm_depuracion.py      # utilidades de depuración reutilizables
├── tfm_agente.py          # agente ReAct de adaptación a otros restaurantes
├── ml/
│   ├── features.py        # construcción reproducible de las tablas de features
│   └── revenue_predictor.py
├── data/                  # bronze / silver / gold
└── results/               # métricas, figuras y modelos serializados
```

## Sobre la numeración

La numeración refleja fases del pipeline, no un orden estricto de ficheros. Conviene leerla así:

- Dos notebooks 01 — son las dos vías de ingesta a Bronze, independientes entre sí: 01_ingesta_bronze para las fuentes tabulares del TPV y las fuentes externas, 01_pdf_process para la extracción de facturas en PDF.
- 02 y 02b — 02 depura los datos del restaurante original; 02b es la versión genérica, escrita para depurar la salida del agente sobre un restaurante cualquiera. Comparten las funciones de tfm_depuracion.py.
- Dos notebooks 04 — son dos enfoques distintos sobre el mismo problema de producto, deliberadamente separados: 04_prediccion_ventas modela la venta semanal de artículos individuales del catálogo (panel artículo × periodo); 04_prediccion_comandas modela la demanda agregada de platos con pérdida de Poisson y análisis de cesta. Sus conclusiones se contrastan en la memoria.

## Notebooks
 
| Notebook | Contenido |
|---|---|
| `01_ingesta_bronze` | Carga las fuentes crudas del TPV (tickets, ventas, reservas, propinas, catálogos) y las externas (meteorología, festivos, eventos) mediante los lectores de `tfm_io.py`. Vuelca 12 snapshots a Bronze sin aplicar ninguna decisión de negocio. |
| `01_pdf_process` | Extrae las facturas en PDF con un parser por coordenadas (`TicketPDFAgent`) que reconstruye las líneas a partir de las posiciones del texto. Produce `facturas_raw.parquet` con cabecera y detalle separados. |
| `02_depuracion_silver` | Audita y depura las 13 entidades de Bronze: tipos, normalización de estados de reserva y marcado de anomalías como `np.nan` en lugar de imputarlas. Cierra con la validación cruzada de cobertura entre tickets, meteorología y festivos. |
| `02b_depuracion_silver_agente` | Versión genérica del `02`, aplicable a la salida del agente sobre un restaurante arbitrario. Carga sólo las entidades que el agente haya producido y salta las secciones cuyos datos faltan en lugar de fallar. |
| `03_EDA` | Análisis exploratorio de facturación, reservas, producto, propinas, meteorología y festivos, con matriz de correlaciones y aviso explícito de *data leakage* en las variables de reservas. Construye y verifica la tabla maestra diaria de Gold (242 × 34), que alimenta a todos los modelos diarios. |
| `04_prediccion_ventas` | Modelo panel de venta semanal para los 82 artículos que superan el filtro de calidad de serie (≥ 90 % de periodos activos, 75,4 % de la facturación), con lags y EWMA agrupados por artículo. Compara los baselines de persistencia y EWMA frente a Ridge y HistGradientBoosting, con evaluación por artículo además de agregada. |
| `04_prediccion_comandas` | Modelo de demanda de platos por periodo con HistGradientBoosting de pérdida Poisson y Random Forest, validado por bloques temporales de ventana expansiva frente a los baselines de última semana y media móvil. Incluye forecast del siguiente periodo, estrategia de respaldo para artículos de cola y análisis de cesta. |
| `05_prediccion_facturacion_diaria_definitivo` | Predice la facturación del día siguiente con horizonte configurable, incorporando las reservas confirmadas con antelación como feature sin fuga y excluyendo toda variable que sólo se conozca al cerrar el día. Ajusta cinco modelos sobre `TimeSeriesSplit` y los evalúa una única vez en el tramo temporal final. |
| `06_pred_noshows` | Clasificación de no-show a nivel de reserva individual; único modelo que no parte de Gold, ya que aprovecha el histórico completo de reservas desde 2023. Compara regresión logística y HistGradientBoosting frente a tres baselines ingenuos, con análisis del segmento de alta confianza de cada uno. |
| `07_prediccion_ticket_medio_definitivo` | Mismo esquema que el `05` aplicado al ticket medio, con nueve modelos y batería completa de métricas de error. Añade la estimación del ruido de composición del indicador: el error que ningún modelo puede eliminar. |
| `09_prediccion_facturacion_largo_plazo` | Variante del `05` sin ninguna variable de memoria reciente, para fechas demasiado lejanas como para disponer de historial real. Usa `ml/features.py` para ver exactamente las mismas columnas menos las de historial, de modo que la diferencia de error mide el valor de esa memoria. |
| `10_prediccion_ticket_medio_largo_plazo` | Misma variante aplicada al ticket medio sobre las features del `07`. Sirve de contraste directo con el modelo de corto plazo para cuantificar el coste de predecir sin historial. |

## Módulos de apoyo
| Módulo |	Contenido|
|---|---|
| tfm_io.py	| Lectores específicos de cada fuente cruda hacia el esquema Bronze|
| tfm_depuracion.py	| Funciones de depuración reutilizables: tratamiento de atípicos, análisis de categóricas, patrones de valores perdidos |
| tfm_agente.py	| Agente ReAct (SDK de Anthropic) que identifica la entidad de un fichero desconocido y propone un mapeo de columnas al contrato Bronze, con salida estructurada forzada y confirmación humana obligatoria antes de escribir |
| ml/features.py	| Construcción reproducible de las tablas de features de facturación y ticket medio, compartida entre los notebooks de corto y largo plazo |
| ml/revenue_predictor.py	| Selección del modelo adecuado (corto o largo plazo) según el horizonte solicitado |


## Orden de ejecución

```
01_ingesta_bronze  →  01_pdf_process  →  02_depuracion_silver  →  03_EDA
                                                                   │
        ┌──────────────────────┬─────────────┬────────────────────┴──────┐
        ▼                      ▼             ▼                           ▼
04_prediccion_ventas   04_prediccion    05  →  09                 07  →  10
                        _comandas
                                              06_pred_noshows
                                    (independiente: parte de Silver)
```
 
Del `03` en adelante los notebooks consumen los Parquet de Silver y Gold: no es necesario reejecutar la ingesta si esos ficheros ya están presentes. El `02b` sólo hace falta cuando se trabaja con datos producidos por el agente.