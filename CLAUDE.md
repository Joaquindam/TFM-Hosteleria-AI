# Reglas del proyecto TFM-Hosteleria-AI

## Contexto
Este repo ya contiene el trabajo del TFM: pipeline de datos (bronze/silver/gold),
notebooks de EDA y modelado, y dos modelos entrenados y serializados:
- `results/models/mejor_modelo_facturacion_diaria.joblib`
- `results/models/mejor_modelo_ticket_medio.joblib`
Helpers reutilizables en `src/` (tfm_io.py, tfm_depuracion.py, tfm_plots.py, tfm_theme.py).
NO se parte de cero. El objetivo de las siguientes fases es envolver ese trabajo
en un producto: Streamlit (frontend) + FastAPI (backend) + RAG + MCP + AI Agent.

## Reglas fijas (nunca romper)
- NUNCA reentrenar ni sustituir los modelos .joblib existentes salvo que se pida explícitamente.
- NUNCA borrar ni modificar notebooks, datasets (data/bronze|silver|gold) o modelos existentes.
- NUNCA guardar API keys en código/notebooks/git. Usar .env (ya está en .gitignore) + .env.example.
- NUNCA inventar datos, métricas o resultados. Si no hay datos reales, usar mock
  y marcarlo claramente como "DEMO DATA" en la UI y con comentario "# MOCK - sin dato real" en el código.
- RAG = conocimiento documental (reseñas, cartas, políticas). MCP/tools = datos
  estructurados (predicciones, reservas, meteo). NUNCA mezclar: el LLM no debe
  "inventar" una predicción que el modelo ML ya calcula.
- Los escenarios (what-if) se llaman "model-based scenario", nunca causalidad.
- AI Visibility no es un ranking universal, son resultados de una batería de prompts.
- Si hay incompatibilidad entre código existente (src/, notebooks) y arquitectura
  nueva: crear un adapter en adapters/, no reescribir el código original.
- Cambios mínimos y explícitos sobre código ya existente.
- Antes de tocar notebooks, comprobar `git status`: puede haber cambios sin commitear
  (trabajo en curso de la autora) que no hay que pisar ni descartar.
- Cualquier dato introducido a mano desde la app (reservas, y cualquier otro
  dato manual que se añada más adelante) es acumulativo: nunca se sobrescribe
  ni se borra en duro. Edición = nueva versión/estado, borrado = borrado lógico
  (marcar como cancelado/inactivo), nunca DELETE físico. El objetivo es que ese
  histórico crezca con el tiempo para poder alimentar futuras actualizaciones
  de los modelos — aunque el reentrenamiento en sí sigue sin hacerse salvo que
  se pida explícitamente (ver regla de "nunca reentrenar" más arriba).

## Estado del proyecto
Ver [PHASES.md](PHASES.md) para saber en qué fase estamos y qué está hecho.
Ver PROJECT_AUDIT.md (se genera en la Fase 1) para el inventario detallado del repo.

## Modo de trabajo
Trabajar solo en la fase indicada en el mensaje. No adelantar fases siguientes.
Después de cada fase: implementar → ejecutar → probar → corregir → parar y reportar.
