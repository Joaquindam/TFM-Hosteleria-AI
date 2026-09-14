# Fases

- [~] Fase 1 — Auditoría del repo. No se genero PROJECT_AUDIT.md formal; se
      hizo una exploracion dirigida del repo (notebooks 05/07, esquemas de
      gold/silver) para poder escribir CLAUDE.md, PROJECT_BRIEF.md y la Fase 2
      con datos reales en vez de plantilla generica. Pendiente si en algun
      momento hace falta el documento formal.
- [x] Fase 2 — ml/ (adapter sobre los dos modelos .joblib reales) + FastAPI
      backend + Streamlit basico (Overview, Forecast, Reservas). Probado de
      extremo a extremo: backend levantado, Streamlit levantado, paginas
      verificadas en navegador, ciclo completo de reservas (crear/editar/
      cancelar con borrado logico) probado con curl.
- [x] Fase 3 — RAG + MCP Server. Probado de extremo a extremo: RAG (TF-IDF,
      976 chunks: knowledge/ + 965 reseñas reales) probado con varias
      queries; servidor MCP real (`mcp` SDK, 12 tools) probado con
      `list_tools` y `call_tool` sobre 2 tools distintas. Varias tools
      resultaron ser reales en vez de mock al auditar los datos ya
      procesados del repo: `get_review_analysis` (965 reseñas + ABSA),
      `get_events` (78 eventos reales 2025-10 a 2026-10) y
      `get_weather_forecast` (real para fechas ya en el histórico).
      Solo siguen en MOCK `get_ai_visibility` y `get_competitor_analysis`
      (pendientes de la Fase 5 y de la lista de competidores de la autora).
- [ ] Fase 4 — AI Agent ("AI Advisor") + integracion Agent + RAG + MCP en
      /agent/chat. Debe seguir el patron PREVISION -> EXPLICACION -> RIESGOS
      -> RECOMENDACION para preguntas compuestas (ver seccion 25).
- [ ] Fase 5 — Paginas Customer Intelligence, AI Visibility (metricas
      concretas) y Scenarios (el motor ml/scenarios.py ya existe desde la
      Fase 2; falta el endpoint /scenario y la pagina).
- [ ] Fase 6 — Tests + README + pulido de UX (ver seccion 39) + .env.example.

- [x] Extra (post-Fase 3) — Integración de dos modelos nuevos del equipo
      (`04_prediccion_ventas.ipynb`, `06_pred_noshows.ipynb`) tras `git pull`:
      `ml/features_sales.py` + `ml/sales_predictor.py`,
      `ml/noshow_predictor.py`, routers `backend/routers/sales.py` y
      `backend/routers/noshow.py`, y dos secciones nuevas en
      `app/pages/2_Forecast.py`. Probado de extremo a extremo (backend +
      Streamlit en navegador). Ver PROJECT_BRIEF.md sección 4 para el
      detalle de cada modelo y sus limitaciones.

- [x] Extra — Modelos de largo plazo para facturación y ticket medio
      (`notebooks/09` y `10`), sin variables de memoria, para poder predecir
      fechas lejanas (p.ej. noviembre) sin inventar historial reciente.
      Resultado honesto: ~7-8% peor que el modelo corto, no un fallback
      degradado. `ml/revenue_predictor.py`/`ml/ticket_predictor.py` eligen
      el modelo automáticamente. Probado de extremo a extremo vía API.
      Pendiente si se quiere: el mismo tratamiento para ventas por
      artículo (de momento sigue con la aproximación de la media histórica
      del artículo, más débil — ver sección 4 del brief).

## Decisiones tomadas fuera del plan original (ver PROJECT_BRIEF.md para el detalle)

- La pagina "Reservas" (entrada manual, seccion 5.1) se mantiene como pagina
  propia ademas de las 6 que pide la especificacion ampliada (Overview,
  Forecast, Customer Intelligence, AI Visibility, Scenarios, AI Advisor):
  es funcionalidad real ya construida y es la fuente de datos de reservas
  que alimenta Overview/Forecast/MCP. No se ha eliminado.
- Explicabilidad: la Fase 2 usa importancia por permutacion ya calculada en
  el entrenamiento (no SHAP). SHAP no esta disponible todavia en el repo
  (los notebooks 05/07 no lo calculan) -- ver seccion 38 para la decision
  pendiente de si merece la pena anadirlo en una fase posterior.
