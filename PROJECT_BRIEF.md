# PROJECT BRIEF — Restaurant Intelligence (TFM-Hosteleria-AI)

> Documento de referencia completo. En los prompts de cada fase se cita solo
> la sección concreta que toca (p.ej. "sección 9 y 10 para el MCP"). No hace
> falta releerlo entero cada vez.
>
> Generado a partir del estado real del repo (notebooks, modelos `.joblib`,
> datos bronze/silver/gold) más la arquitectura objetivo descrita en las
> conversaciones previas. Es un punto de partida razonable, no un documento
> cerrado: ajusta cualquier sección que no encaje con tu criterio antes de
> lanzar la fase correspondiente — sobre todo las secciones 3, 11, 21-23 y
> 30-34, que son más de producto/negocio que de código y dependen de
> decisiones tuyas sobre el alcance del TFM.

## 1. Visión general

Restaurant Intelligence es la capa de producto que envuelve el trabajo de
modelado ya hecho en este TFM (predicción de facturación diaria y de ticket
medio) para convertirlo en un asistente consultable por un gerente de
restaurante: un chat en lenguaje natural que combina esas predicciones con
conocimiento documental (reseñas, políticas, carta) y con datos operativos
(reservas, meteorología, eventos) para responder preguntas y proponer
escenarios de decisión.

## 2. Objetivos y alcance

- Exponer los modelos ML existentes (no reentrenarlos) a través de una API.
- Dar una interfaz visual (Streamlit) para explorar previsiones sin código.
- Añadir una capa de razonamiento (Agent) que decide qué fuente de información
  usar para cada pregunta, sin que el LLM sustituya al modelo ML.
- Incorporar un módulo de "AI Visibility": cómo aparece el restaurante cuando
  un usuario le pregunta a un LLM por recomendaciones en su categoría/zona.
- Fuera de alcance: automatizar reservas, modificar precios o menús en
  sistemas externos, cualquier acción que escriba en sistemas de terceros.

## 3. Usuarios y casos de uso

Usuario principal: gerente/propietario del restaurante, sin conocimientos
técnicos. Casos de uso tipo:
- "¿Cuál es nuestra previsión para mañana?"
- "¿Por qué la previsión de este sábado es más baja que el sábado pasado?"
- "¿Qué dicen las reseñas últimas sobre el servicio?"
- "Si lloviera este finde, ¿cómo cambiaría la previsión?"
- "¿Aparecemos cuando alguien le pregunta a ChatGPT por restaurantes de la zona?"

## 4. Modelos ML existentes (inputs/outputs)

Confirmado por auditoría del repo (ver `PROJECT_AUDIT.md` para el detalle
exacto de columnas y firma real):

- `results/models/mejor_modelo_facturacion_diaria.joblib` — predicción de
  facturación diaria. Entrenado y evaluado en
  `notebooks/05_prediccion_facturacion_diaria_definitivo.ipynb`, con métricas
  en `results/forecasting_final/`.
- `results/models/mejor_modelo_ticket_medio.joblib` — predicción de ticket
  medio. Entrenado en `notebooks/07_prediccion_ticket_medio_definitivo.ipynb`,
  con métricas en `results/ticket_medio_final/`.
- `results/models/mejor_modelo_facturacion_diaria_largo_plazo.joblib` y
  `..._ticket_medio_largo_plazo.joblib` — **modelos "de largo plazo"**,
  entrenados en `notebooks/09_prediccion_facturacion_largo_plazo.ipynb` y
  `10_prediccion_ticket_medio_largo_plazo.ipynb`. Mismos datos y mismo split
  que los modelos de corto plazo, pero **sin ninguna variable de memoria**
  (`facturacion_Xd_antes`, medias móviles, etc.) — solo calendario, festivos,
  eventos, meteorología y reservas ya confirmadas. Se usan cuando se predice
  una fecha demasiado lejana para tener un "hace 7 días" real (más de
  `MAX_FUTURE_GAP_DAYS` desde el último dato conocido). Resultado honesto:
  MAE test 494€ (vs 457€ del modelo corto, facturación) y 14.87€ (vs 13.85€,
  ticket medio) — solo ~7-8% peor, porque las variables de memoria pesaban
  menos de lo esperado frente al calendario (ver la explicabilidad de la
  Fase 2). `ml/revenue_predictor.py` y `ml/ticket_predictor.py` eligen
  automáticamente qué modelo usar según si hay historial reciente real;
  el resultado siempre indica `used_recent_history` y, si es `false`, un
  aviso explícito. Decisión tomada tras descartar una alternativa peor
  (rellenar las columnas de memoria del modelo corto con la media del
  histórico): eso solo funciona si el modelo tolera huecos (el de
  facturación/ticket sí, vía su imputer; el de ventas por artículo, más
  abajo, no) y en cualquier caso es una aproximación peor que un modelo
  entrenado para el caso real.
- `results/models/modelo_prediccion_ventas_articulo.joblib` — predicción de
  ventas (`units_por_dia`) de un artículo de carta concreto, a nivel
  **artículo × periodo semanal** (no diario), para 82 de los 251 artículos
  (los que tienen venta suficientemente regular). Entrenado en
  `notebooks/04_prediccion_ventas.ipynb` (Ridge). El notebook original no
  guardaba el modelo — se añadió la celda de guardado y se re-ejecutó el
  notebook completo en la Fase 3 (autorizado explícitamente, sin cambiar
  metodología). Integrado en `ml/features_sales.py` + `ml/sales_predictor.py`.
- `results/models/modelo_prediccion_noshows.joblib` — probabilidad de
  no-show de una reserva individual. Entrenado en
  `notebooks/06_pred_noshows.ipynb` (HistGradientBoostingClassifier
  ajustado). Señal débil según el propio notebook (PR-AUC ~0.04, apenas por
  encima de baselines ingenuos) — cualquier uso de este modelo debe dejarlo
  claro, nunca presentarlo como una certeza. **No es un Pipeline de
  sklearn**: el preprocesado (mapeo de `shift`/`zone`, one-hot de `origin`)
  se hizo a mano en el notebook y se replica en `ml/noshow_predictor.py`.
  Excluye el canal `walk in` (0% no-show observado, fuera del universo de
  entrenamiento). El formulario de reservas de la app (sección 5.1) no
  recoge todavía `origin`/`zone`/`antelación` — hoy solo se puede consultar
  este modelo introduciendo esos datos a mano en la página Forecast.

Antes de escribir `ml/`, la Fase 2 debe confirmar contra el notebook real:
formato exacto de entrada (¿DataFrame con qué columnas?, ¿fecha como índice
o columna?), variables de calendario/meteo/eventos/reservas usadas como
features, y forma de la salida (valor único, intervalo, o desglose).
**No asumir la interfaz — leerla del notebook y adaptarse a ella.**

## 5. Capa de datos (bronze/silver/gold)

- `data/bronze/` — datos crudos (incluye reseñas de Google en bruto).
- `data/silver/` — datos depurados (`notebooks/02_depuracion_silver.ipynb`).
- `data/gold/` — datos listos para modelado/consumo.

Cualquier endpoint o tool que necesite datos históricos debe leer de `gold`
cuando exista, no recalcular desde `bronze`.

### 5.1 Reservas — entrada manual desde la propia app (decisión confirmada)

Se descarta la sincronización con un sistema externo (sección 5.1 anterior).
El restaurante no tiene API de reservas ni un export fiable, así que las
reservas se introducen a mano dentro de la propia aplicación. Diseño:

1. Almacén: `data/gold/reservas.db` (SQLite) — una tabla simple
   `reservas(id, fecha, hora, comensales, nombre, estado, notas,
   creado_en, actualizado_en)`. SQLite en vez de parquet porque necesita
   soportar escrituras frecuentes desde la UI, no solo lectura por lotes.
2. Backend (sección 16): endpoints `POST /reservations` (crear),
   `PUT /reservations/{id}` (editar/cambiar estado), además del ya previsto
   `GET /reservations/{date}` (leer). Toda escritura pasa por el backend,
   nunca Streamlit escribiendo directo al fichero — así el backend puede
   validar (fecha válida, comensales > 0, etc.) en un solo sitio.
3. Frontend (sección 17): página "Reservas" con un formulario para añadir
   una reserva y una tabla editable con las reservas del día/semana
   seleccionada (alta, edición de estado — confirmada/cancelada/no-show,
   borrado).
4. `get_reservation_status` (sección 10) deja de ser mock permanente: lee de
   `data/gold/reservas.db`. Ya no hace falta `last_synced_at` (es tiempo
   real de verdad, no una sincronización periódica) — sí conviene devolver
   si hay 0 reservas registradas para esa fecha, para distinguir "no hay
   reservas" de "no se ha introducido nada todavía".
5. Fuera de alcance por ahora: importar reservas históricas masivas desde
   CSV, o volver a plantear una integración con un sistema de reservas
   real — si en el futuro el restaurante adopta uno, se sustituye este
   almacén sin tocar el resto del sistema (el resto solo depende de la
   interfaz de `get_reservation_status`, no de cómo se guardan los datos).

### 5.2 Persistencia y acumulación de datos (regla dura, ver CLAUDE.md)

Todo dato introducido a mano desde la app —reservas ahora, cualquier otro
dato manual que se añada en fases futuras— se trata como histórico
acumulativo, no como estado editable sin memoria:

- **Nunca DELETE físico.** Cancelar/borrar una reserva desde la UI marca la
  fila como `estado = 'cancelada'` (borrado lógico), no elimina el registro.
- **Editar crea historial, no sobrescribe sin rastro**: guardar
  `actualizado_en` en cada cambio; si en el futuro hace falta trazabilidad
  fina, se puede pasar a un modelo de "eventos" (una fila por cada cambio)
  en vez de una fila mutable — no necesario para el alcance actual del TFM,
  pero el esquema de `reservas.db` (sección 5.1) no debe impedirlo después.
- **Backups**: `data/gold/reservas.db` se trata como dato de usuario real,
  no como caché — debe estar en `.gitignore` igual que el resto de `data/`
  (confirmar en la Fase 1) pero conviene que la propia app o un script
  sencillo haga copia periódica (p.ej. copiar el fichero con fecha al
  arrancar el backend) para no perder reservas por un fallo de disco.
- **Objetivo a medio plazo**: este histórico de reservas reales (frente a
  las variables estimadas/mock usadas al entrenar los modelos actuales) es
  el candidato natural para una futura actualización de
  `mejor_modelo_facturacion_diaria.joblib` con una feature de "reservas
  confirmadas" real. Esto queda anotado aquí como motivación, pero
  **reentrenar sigue fuera de alcance de las fases actuales** — se anota
  para que quien continúe el proyecto sepa que el dato ya se está
  acumulando con ese fin.

## 6. Capa ml/ (adapter sobre modelos)

Módulo `ml/` con responsabilidad única: cargar los `.joblib` y exponer una
interfaz estable para el resto de la aplicación, independientemente de cómo
esté implementado el modelo por debajo.

- `ml/revenue_predictor.py` — `predict_revenue(date, input_data) -> dict`
- `ml/ticket_predictor.py` — `predict_avg_ticket(date, input_data) -> dict`
- `ml/explainability.py` — importancia de features / drivers de una predicción
  concreta (basarse en `importancia_permutacion.csv` de `results/*_final/` si
  no es viable recalcular SHAP en caliente).
- `ml/scenarios.py` — recalcula la predicción variando una o varias features
  de entrada (ver sección 19, "model-based scenario").

Si la interfaz real de los notebooks no coincide con esta propuesta, adaptar
`ml/` a la real y documentar la diferencia en un comentario al inicio del
archivo. Si hace falta transformar datos de forma no trivial para llegar al
formato que espera el modelo, ese código va en `adapters/`, no se reescribe
el notebook ni el pipeline original.

## 7. Base de conocimiento documental (fuentes para RAG)

- Reseñas de Google (`data/bronze/google_reviews_la_roca_*`,
  `results/reviews_overview/`).
- Documentos de reseñas en `data/bronze/resenia/` (.docx).
- Cualquier documento de carta, políticas del restaurante o FAQ que la autora
  aporte — si no existe, crear un archivo de ejemplo dentro de `knowledge/`
  con la primera línea `> DEMO — contenido de ejemplo, sustituir por el real`.

`knowledge/` no debe contener nunca datos inventados presentados como reales.

## 8. Diseño del sistema RAG

Implementado en la Fase 3:

- `rag/ingest.py` — carga y trocea (chunking) los documentos de `knowledge/`
  (por encabezado `##`) y las 965 reseñas reales de
  `data/silver/snapshots/resenas_silver.parquet` (una por reseña, con su
  texto y los aspectos positivos/negativos ya extraídos — no se reprocesa).
- `rag/embeddings.py` — **decisión tomada**: TF-IDF (`scikit-learn`), no
  embeddings neuronales ni vector store externo. Con ~976 chunks (400 platos
  + 3 documentos + 965 reseñas) un vector store real (Chroma/FAISS) es más
  infraestructura de la que hace falta; TF-IDF ya estaba disponible sin
  dependencias pesadas ni API key. La interfaz de `retriever.py` no depende
  de esto — se puede sustituir por embeddings reales sin tocar el resto del
  sistema si el corpus crece o hace falta mejor matching semántico.
- `rag/knowledge_base.py` — construye y cachea en memoria (un proceso, un
  `lru_cache`) el corpus vectorizado; se reconstruye en menos de un segundo,
  no hace falta persistir un índice en disco a este tamaño.
- `rag/retriever.py` — `retrieve_context(query, k=4) -> list[chunk]`,
  probado con queries reales (ver PROJECT_AUDIT/notas de la Fase 3).

El RAG **nunca** debe responder preguntas de cifras o predicciones — esas
siempre van por MCP/tools (ver sección 9).

## 9. Filosofía del MCP Server

El MCP Server (`restaurant-intelligence-mcp`) expone como "tools" todo dato
estructurado o calculado: predicciones, históricos, reservas, meteorología,
eventos, análisis de reseñas agregado, AI Visibility. Regla dura: si un dato
puede calcularse determinísticamente (modelo ML, consulta a `gold/`, API de
meteo), pasa por una tool MCP. El LLM nunca debe "recordar" o inventar estas
cifras a partir de contexto de RAG o de su propio conocimiento.

## 10. Catálogo de herramientas MCP

- `get_revenue_forecast(date)` — usa `ml/revenue_predictor.py`.
- `get_ticket_forecast(date)` — usa `ml/ticket_predictor.py`.
- `get_historical_revenue(date_from, date_to)` — lee de `data/gold/`.
- `get_reservation_status(date)` — lee de `data/gold/reservas.db`
  (ver sección 5.1, entrada manual desde la app). Solo cae a **MOCK** si
  esa tabla todavía no existe cuando se implemente esta tool.
- `get_weather_forecast(date)` — si `date` ya está en `data/gold` devuelve
  la meteorología real registrada (`source: "historical_data"`); si no, cae a
  **MOCK** (no hay API de meteorología en tiempo real integrada todavía).
- `get_events(date)` — lee
  `data/bronze/eventos_relevantes_pozuelo_la_roca_2025_2026.csv` (78 eventos
  reales curados por la autora, cobertura real 2025-10-10 a 2026-10-01).
  Devuelve `source: "historical_data"` si `date` cae dentro de ese rango
  (aunque ese día no haya ningún evento), **MOCK** solo fuera de rango.
- `get_review_analysis(date_from, date_to)` — **implementado con datos
  reales** en la Fase 3: agregados sobre
  `data/silver/snapshots/resenas_silver.parquet` (965 reseñas) y
  `resenas_absa_silver.parquet` (866 filas de sentimiento por aspecto/plato
  ya calculado) — distribución de sentimiento y top aspectos positivos/
  negativos. No es un placeholder.
- `get_ai_visibility(query_set)` — resultados de la batería de prompts
  (sección 11 y 37); MOCK/placeholder hasta que la Fase 5 tenga resultados reales.
- `get_competitor_analysis(competitor_ids=None)` — comparativa frente a
  competidores locales conocidos (sección 22 y 37): mismas métricas de
  AI Visibility calculadas también para ellos, más cualquier comparativa de
  reseñas agregadas que ya exista. MOCK/placeholder hasta tener lista de
  competidores confirmada por la autora (sección 22) y resultados reales.
- `run_revenue_scenario(date, overrides)` / `run_ticket_scenario(date, overrides)`
  — usan `ml/scenarios.py` (ya implementado en la Fase 2), nunca presentar
  el resultado como causal (ver sección 19).

Cada tool debe devolver, además del dato, un campo `source` (`"model"`,
`"historical_data"`, `"mock"`, etc.) para que el Agent pueda citar la fuente.

## 11. AI Visibility — definición y alcance

"AI Visibility" mide cómo aparece (o no aparece) el restaurante cuando se le
hace a un LLM una batería fija de prompts tipo "mejores restaurantes de
[categoría] en [zona]". **No es un ranking universal ni una métrica SEO
estándar, y nunca se presenta como "ranking de ChatGPT" ni como promesa de
posición**: es simplemente el resultado, agregado y repetible, de esa
batería de prompts contra uno o varios modelos. Cualquier UI o informe sobre
esto debe dejar claro ese matiz para no sobre-interpretar el resultado. Las
métricas concretas y la batería de prompts se detallan en la sección 37.

## 12. Arquitectura general del sistema

```
Streamlit (frontend)
      │  HTTP
      ▼
FastAPI (backend)  ──────────────┐
      │                          │
      ▼                          ▼
  ml/ (adapter)             AI Agent ── decide → RAG (knowledge/)
      │                          │        └────→ MCP tools (sección 10)
      ▼                          ▼
 modelos .joblib          respuesta en lenguaje de gerente + fuentes citadas
 (results/models/)
```

Los notebooks y el pipeline bronze/silver/gold siguen siendo la fuente de
verdad de datos y modelos; nada en `backend/`, `ml/`, `rag/` o `mcp/` los
sustituye, solo los envuelve.

## 13. AI Agent — orquestación y decisión

El Agent recibe la pregunta del usuario y decide, antes de responder:
1. ¿Necesita un dato estructurado/calculado? → llama a la(s) tool(s) MCP
   necesarias (sección 10), no todas por defecto.
2. ¿Necesita conocimiento documental (reseñas, políticas, contexto)? →
   `retrieve_context` del RAG (sección 8).
3. ¿Necesita ambos? → combina resultados, nunca deja que el texto del RAG
   "complete" o "corrija" una cifra que ya vino de una tool.

## 14. Agent — política RAG vs MCP (nunca mezclar)

Regla dura, repetida aquí porque es la que más falla en la práctica: si la
pregunta tiene una respuesta numérica calculable (previsión, histórico,
meteorología), esa cifra **solo** puede venir de una tool MCP. El RAG solo
aporta explicación cualitativa o contexto (p.ej. qué dicen las reseñas, qué
política aplica). Si el Agent no tiene una tool para un dato numérico, debe
decir que no lo tiene — nunca debe generar un número "razonable" desde su
propio conocimiento.

## 15. Agent — estilo de respuesta y trazabilidad de fuentes

- Lenguaje de gerente, no de ingeniero: evitar jerga técnica ("SHAP value",
  "R²"), traducir a impacto de negocio ("la lluvia está reduciendo la
  previsión en torno a un X%").
- Cada respuesta debe cerrar indicando qué fuentes usó (modelo ML,
  histórico, reservas, meteorología, RAG, AI Visibility), para que el
  gerente pueda distinguir un dato real de uno estimado o mock.

## 16. Backend FastAPI — estructura y endpoints

```
backend/
  main.py
  routers/
    forecast.py      # /forecast/{date}
    reservations.py  # GET /reservations/{date}, POST /reservations,
                       # PUT /reservations/{id}  (ver sección 5.1)
    weather.py        # /weather/{date}
    events.py          # /events/{date}
    agent.py           # /agent/chat
  schemas.py
```

Endpoints mínimos (Fase 2): `/health`, `/forecast/{date}`,
`GET /reservations/{date}`, `/weather/{date}`, `/events/{date}`.
Endpoints de escritura de reservas (`POST`/`PUT /reservations`) se añaden en
la misma fase que la página "Reservas" del frontend (sección 17), no antes.
Endpoint de la Fase 4: `POST /agent/chat`.

## 17. Frontend Streamlit — páginas y estructura

Navegación principal (7 páginas — las 6 del producto final más "Reservas",
que es infraestructura de datos real ya construida en la Fase 2 y no se
retira aunque no apareciera en la lista de 6 páginas pedida más tarde):

```
app/
  app.py
  config.py
  pages/
    1_Overview.py             # estado del restaurante en 10-30s (sección 12.1)
    2_Forecast.py              # selector de fecha + previsión + drivers
    3_Reservas.py               # alta/edición manual de reservas (sección 5.1)
    4_Customer_Intelligence.py   # reseñas: sentimiento/temas/evolución (sección 36)
    5_AI_Visibility.py            # métricas de visibilidad en IAs (sección 37)
    6_Scenarios.py                  # simulador what-if (sección 19; ml/scenarios.py ya existe)
    7_AI_Advisor.py                  # chat con el Agent (Fase 4)
```

Toda página que muestre datos mock debe indicarlo visualmente (badge o texto
"DEMO DATA"), no solo en el código. Principios de diseño de estas páginas:
ver sección 39.

### 17.1 Overview — qué debe mostrar

Debe poder leerse en 10-30 segundos. KPIs mínimos: facturación
reciente/actual, facturación prevista, ticket medio, reservas del día,
un nivel de riesgo (alto/medio/bajo — heurística simple, no un modelo nuevo:
p.ej. basado en cuánto se aleja la previsión del rango histórico habitual y
en si hay condiciones adversas conocidas como lluvia), meteorología, eventos,
principales drivers (reutilizando `ml/explainability.py`), alertas y una
recomendación breve en una frase. La recomendación no es texto libre
inventado por la UI: la genera el Agent (Fase 4) a partir de los mismos
datos — hasta entonces, esta página puede mostrar solo los datos, sin
sección de "recomendación", en vez de fabricar una.

## 18. Política de datos mock / demo

Se usa mock **solo** cuando el audit (`PROJECT_AUDIT.md`) confirma que no
existe capacidad real para ese dato (p.ej. no hay integración con sistema de
reservas). Todo mock debe:
- Marcarse en el código con `# MOCK - sin dato real`.
- Marcarse en la UI de forma visible.
- Devolver `source: "mock"` si sale por una tool MCP (sección 10).

## 19. Model-based scenarios (what-if)

`run_revenue_scenario` permite recalcular la previsión cambiando una o
varias variables de entrada (p.ej. "si lloviera"). Esto es una simulación
del modelo entrenado, **no un análisis causal**: el modelo puede no haber
visto esa combinación de variables en el histórico. Cualquier texto generado
sobre un escenario debe usar la expresión "model-based scenario" y evitar
lenguaje causal ("esto hará que...", "esto provocará...").

## 20. Seguridad y gestión de secretos

- Ninguna clave (API de meteo, LLM, etc.) en código, notebooks o git.
- `.env` (ya en `.gitignore`) + `.env.example` con las claves necesarias sin
  valores reales.
- Si en algún momento se detecta una clave hardcodeada en el repo, no
  reproducirla en ningún output ni commit — avisar y localizar el archivo.

## 21. AI Visibility — implementación técnica

```
ai_visibility/
  prompts.py       # batería fija de prompts de prueba
  runner.py         # ejecuta la batería contra uno o más LLMs
  metrics.py         # frecuencia de mención, posición, sentimiento de la mención
  competitors.py      # mismos prompts para competidores locales conocidos
  perception.py        # cruce con el análisis de reseñas (sección 23)
```

Mientras no haya ejecuciones reales, la página de Streamlit debe mostrar
estructura vacía o de ejemplo, marcada como "DEMO DATA", nunca resultados
inventados presentados como reales.

## 22. Competidores y benchmarking (AI Visibility)

La lista de competidores locales para comparar en `competitors.py` la debe
aportar la autora (no inventar nombres de negocios reales sin confirmación).
Si no se dispone de la lista, dejar la función preparada con una lista vacía
y un comentario indicando que debe rellenarse.

## 23. Percepción / sentiment (reseñas)

Se apoya en `results/reviews_overview/` y en el análisis ya hecho en
`notebooks/08_overview_resenas_google.ipynb` / `08_resenias.ipynb`. No
reemplaza ese análisis; `get_review_analysis` (sección 10) debe reutilizar
sus resultados agregados, no reprocesar las reseñas desde cero salvo que se
pida explícitamente.

## 24. Agent — system prompt y reglas de comportamiento

El system prompt del Agent debe incluir, como mínimo:
- Su rol (asesor de gestión para el restaurante, no un chatbot genérico).
- La regla de la sección 14 (RAG vs MCP, nunca mezclar).
- La instrucción de citar fuentes (sección 15).
- La prohibición explícita de inventar cifras, fechas o resultados no
  devueltos por una tool o por el RAG.
- El uso obligatorio de "model-based scenario" para cualquier simulación.

## 25. Agent — ejemplos de comportamiento esperado

- "¿Cuál es nuestra previsión para mañana?" → llama a
  `get_revenue_forecast(mañana)`, responde con la cifra y su fuente.
- "¿Por qué?" (turno siguiente) → usa la explicabilidad de la predicción ya
  obtenida (`ml/explainability.py` vía la tool), no vuelve a preguntar la
  fecha si ya está en contexto de la conversación.
- "¿Cómo estaremos este sábado y qué deberíamos hacer?" → combina
  `get_revenue_forecast`, `get_weather_forecast`, `get_events` y
  `get_reservation_status` para ese sábado, y opcionalmente RAG si la
  pregunta roza recomendaciones operativas (p.ej. turnos de personal), sin
  inventar una recomendación que no se derive de los datos anteriores.
  Estructura de respuesta obligatoria para este tipo de pregunta compuesta:
  **PREVISIÓN → EXPLICACIÓN → RIESGOS → RECOMENDACIÓN** (cuatro bloques
  cortos, no un párrafo único).
- "¿Qué dicen nuestros clientes?" → `get_review_analysis`, nunca RAG en
  crudo sobre reseñas individuales si lo que se pide es un agregado.
- "¿Cómo nos ve la IA?" → `get_ai_visibility`, con el matiz de la sección 11
  (no es un ranking).
- "¿Cómo estamos frente a la competencia?" → `get_competitor_analysis`.
- "¿Qué pasa si conseguimos 20 reservas más?" → `run_revenue_scenario` con
  el override correspondiente sobre `reservas_anticipadas`/
  `comensales_anticipados`, presentado explícitamente como
  "model-based scenario" (sección 19), nunca como promesa.

## 26. Testing — estrategia

Tests mínimos por fase, no cobertura exhaustiva:
- `tests/test_forecast.py` — `ml/` devuelve un output con la forma esperada
  para un input real de ejemplo.
- `tests/test_api.py` — cada endpoint responde 200 con un input válido.
- `tests/test_rag.py` — `retrieve_context` devuelve al menos un chunk para
  una query conocida sobre el contenido de `knowledge/`.
- `tests/test_mcp.py` — cada tool devuelve el schema esperado, incluido el
  campo `source`.
- `tests/test_agent.py` — las 3 preguntas de la sección 25 no lanzan
  excepción y citan al menos una fuente.

## 27. Documentación — README y estructura

`README.md` debe cubrir: arquitectura (referencia a la sección 12),
instalación, variables de entorno necesarias (sección 35), cómo levantar
backend y frontend, y estructura de carpetas del proyecto final.

## 28. Entorno de desarrollo y ejecución

```bash
uvicorn backend.main:app --reload
streamlit run app/app.py
```

Ambos procesos en paralelo durante desarrollo; el frontend consume el
backend por HTTP en `localhost`.

## 29. Checklist de calidad por fase

Antes de dar una fase por cerrada: el código nuevo ejecuta sin error, se ha
probado con al menos un ejemplo real (no solo import sin fallos), no ha
tocado notebooks/datos/modelos existentes salvo lectura, y cualquier mock
está marcado según la sección 18.

## 30. Roadmap post-TFM (fuera de alcance actual)

La **lectura** de reservas vía CSV/Excel sí está en alcance (sección 5.1).
Lo que queda fuera: cualquier **escritura** en un sistema de reservas
(crear/modificar/cancelar reservas), API de reservas oficial si en el futuro
el restaurante contrata una, despliegue en servidor, autenticación
multiusuario, panel de administración de `knowledge/` desde la UI. No
abordar en las fases actuales salvo que se pida explícitamente.

## 31. Limitaciones conocidas

- Los modelos `.joblib` están entrenados sobre el histórico disponible en
  `data/gold/`; cualquier escenario fuera de ese rango de variables es
  extrapolación, no predicción fiable (ver sección 19).
- AI Visibility depende de qué LLM se consulte y cuándo; no es estable en el
  tiempo ni comparable directamente con herramientas SEO tradicionales.

## 32. Glosario

- **Model-based scenario**: simulación del modelo ML cambiando inputs,
  sin implicar causalidad.
- **AI Visibility**: resultado de una batería fija de prompts contra uno o
  más LLMs, no un ranking oficial.
- **Fuente (`source`)**: campo que indica de dónde viene un dato devuelto
  por una tool (`model`, `historical_data`, `mock`, etc.).

## 33. Convenciones de nombres y estilo de código

Seguir el estilo ya presente en `src/` (snake_case, funciones con
docstring corta en español). Módulos nuevos (`ml/`, `rag/`, `mcp/`,
`ai_visibility/`, `adapters/`) siguen la misma convención.

## 34. Métricas de éxito del proyecto

Para el TFM: que el backend sirva previsiones reales consumidas por el
frontend, que el Agent responda las 3 preguntas de ejemplo (sección 25)
citando fuentes correctamente, y que quede documentado qué partes son
reales y cuáles son demo/mock.

## 35. Variables de entorno (.env)

```
OPENAI_API_KEY=
MODEL_NAME=
WEATHER_API_KEY=       # opcional, si no está, get_weather_forecast es MOCK
```

`.env.example` debe listar estas mismas claves sin valores. `.env` ya está
en `.gitignore` (confirmado en la Fase 1).

## 36. Customer Intelligence (análisis de reseñas)

Página dedicada a convertir las reseñas ya existentes (sección 7 y 23) en
información accionable para el gerente, no solo gráficos:

- **Sentimiento**: distribución positivo/neutro/negativo, reutilizando el
  análisis ya hecho en `notebooks/08_overview_resenas_google.ipynb` /
  `08_resenias.ipynb` y `results/reviews_overview/` — no reprocesar desde
  cero (ver sección 23).
- **Temas recurrentes**: qué se menciona más (servicio, comida, precio,
  ambiente, terraza...) — si el notebook 08 ya extrae esto, reutilizarlo;
  si no, es una extensión razonable de ese análisis, no un modelo nuevo.
- **Evolución**: tendencia del sentimiento/volumen de reseñas en el tiempo.
- **Fortalezas / debilidades / oportunidades**: no basta con mostrar el dato
  crudo — la página (o el Agent, cuando exista) debe traducir "80% de
  menciones negativas de 'espera'" en una fortaleza/debilidad explícita.
  Esto es interpretación textual sobre datos agregados reales, nunca una
  cifra inventada.
- `get_review_analysis` (sección 10) es la tool que expone estos agregados
  al Agent; esta página los expone directamente al gerente en la UI.

## 37. AI Visibility — métricas y batería de prompts (detalle)

Amplía las secciones 11, 21 y 22 con las métricas y la batería de prompts
concretas pedidas para el producto final:

**Batería fija de prompts** (`ai_visibility/prompts.py`), agrupada por
intención — cubrir al menos estas categorías, adaptando el texto exacto de
cada prompt a la zona/categoría real del restaurante:
- ubicación (p.ej. "restaurantes en Pozuelo de Alarcón"),
- ocasión (parejas, familias, grupos),
- producto (carne, terraza),
- precio (relación calidad-precio, gama),
- perfil de cliente (comidas de empresa).

**Métricas por prompt y agregadas** (`ai_visibility/metrics.py`):
- **Mention Rate**: % de ejecuciones de la batería en las que el restaurante
  aparece mencionado.
- **Recommendation Rate**: % de ejecuciones en las que aparece como
  recomendación explícita (no solo mencionado de pasada).
- **Posición aproximada**: si la respuesta del LLM lista varias opciones,
  en qué posición aparece (aproximada porque el LLM no devuelve un ranking
  estable ni reproducible al 100%).
- **Competidores**: qué otros negocios aparecen en las mismas respuestas
  (alimenta `get_competitor_analysis`, sección 10, y `competitors.py`,
  sección 21 — lista de competidores a confirmar por la autora, sección 22).
- **Atributos asociados**: con qué cualidades se menciona el restaurante
  (p.ej. "ambiente", "carne", "caro") cuando aparece.
- **Fuentes**: qué modelo(s) se consultaron y cuándo, para que el resultado
  sea auditable y repetible.

**Cómo se presenta**: nunca como "ranking de ChatGPT" ni prometiendo
posiciones — siempre como "resultado de esta batería de N prompts contra
estos modelos, en esta fecha" (ver sección 11).

## 38. Explicabilidad — permutación vs. SHAP (decisión pendiente)

La Fase 2 implementa `ml/explainability.py` usando la importancia por
permutación ya calculada y guardada en el entrenamiento
(`results/forecasting_final|ticket_medio_final/importancia_permutacion.csv`)
más el valor real de cada variable para el día consultado. Es una
explicación **global** (qué variables importan en general) aplicada a un
día concreto, no una explicación **por instancia** de verdad.

SHAP daría explicabilidad por instancia real (cuánto contribuyó cada
variable a *esta* predicción concreta), pero **no está calculado en ningún
notebook existente** (05 ni 07) — no es "ya disponible", habría que:
- añadir la librería `shap` como dependencia nueva,
- calcularlo en caliente en `ml/explainability.py` sobre el modelo ya
  cargado (viable para el mejor modelo de facturación, `RandomForestRegressor`,
  vía `TreeExplainer`, que es rápido; el de ticket medio es `Ridge`, con
  explicabilidad lineal trivial vía coeficientes, sin necesitar SHAP).

Recomendación: dejar la solución actual (permutación + valor del día) para
la Fase 2/3, y evaluar el cambio a SHAP como mejora incremental en la Fase 4
o 5 si el nivel de detalle no es suficiente para el Agent o para la demo del
TFM — no bloquea nada mientras tanto.

## 39. Principios de UX / diseño del producto

La aplicación debe leerse como un producto, no como un notebook convertido
en web:
- Profesional, moderno, limpio; poco ruido visual, gráficos solo donde
  aportan (no "un gráfico por variable").
- Responsive: usable en móvil y en escritorio.
- Lenguaje de negocio (ver sección 15), nunca jerga de ML en la UI.
- Cada pantalla debe responder, explícita o implícitamente, a cuatro
  preguntas: **¿Qué significa? ¿Por qué ocurre? ¿Qué riesgo existe?
  ¿Qué puedo hacer?** Una pantalla que solo muestra un número sin ninguna
  de estas respuestas está incompleta para este producto.
- Ninguna pantalla puede rellenar la parte de "qué puedo hacer" con una
  recomendación inventada — si esa recomendación no puede derivarse de un
  dato real ya mostrado (o del Agent, sección 13), la pantalla no debe
  fingir que la tiene.
