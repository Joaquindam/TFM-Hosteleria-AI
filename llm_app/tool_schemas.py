"""Definicion de las tools del LLM app en el formato que espera la Messages
API de Anthropic (`name`/`description`/`input_schema`), mas el dispatch a la
implementacion real. Reutiliza tal cual las funciones de
`mcp_server/tools.py` -- ese modulo ya es la "logica de negocio" del MCP
server (ver PROJECT_BRIEF.md secciones 9-10); aqui no se reimplementa nada,
solo se describe para que un LLM sepa cuando y como llamarlas.

Si en el futuro se quiere que un cliente MCP externo (Claude Desktop, etc.)
use exactamente estas mismas tools, ya existe `mcp_server/server.py` sirviendo
las mismas funciones por stdio -- este modulo es la version "in-process" para
el LLM app propio de esta aplicacion, sin la sobrecarga del protocolo MCP al
ser el mismo codebase.
"""
from __future__ import annotations

from mcp_server import tools

_WEATHER_SCHEMA = {
    "type": "object",
    "description": (
        "Las 7 variables meteorologicas. O se aportan las 7, o ninguna "
        "(nunca un subconjunto) -- si faltan, la funcion falla."
    ),
    "properties": {
        "temperature_max": {"type": "number"},
        "temperature_min": {"type": "number"},
        "temperature_mean": {"type": "number"},
        "precipitation_mm": {"type": "number"},
        "precipitation_hours": {"type": "number"},
        "wind_speed_max": {"type": "number"},
        "sunshine_duration_h": {"type": "number"},
    },
}

_EVENT_SCHEMA = {
    "type": "object",
    "description": "Variables de evento local a forzar en un escenario (todas opcionales).",
    "properties": {
        "tiene_evento": {"type": "integer", "enum": [0, 1]},
        "intensidad_evento": {"type": "integer"},
        "impacto_evento": {"type": "string"},
        "direccion_evento": {"type": "string"},
        "categoria_evento": {"type": "string"},
        "cat_evento": {"type": "string"},
    },
}

_SALES_EXOGENOUS_SCHEMA = {
    "type": "object",
    "description": (
        "Agregados del periodo semanal completo. O se aportan las 9, o "
        "ninguna -- si faltan, la funcion falla."
    ),
    "properties": {
        "n_dias_festivo": {"type": "integer"},
        "intensidad_evento_total": {"type": "integer"},
        "temperature_mean_periodo": {"type": "number"},
        "precipitation_mm_total": {"type": "number"},
        "wind_speed_max_periodo": {"type": "number"},
        "sunshine_duration_h_total": {"type": "number"},
        "comensales_completada_periodo": {"type": "integer"},
        "tasa_no_show_media": {"type": "number"},
        "tasa_cancelacion_media": {"type": "number"},
    },
}

DATE_DESC = "Fecha en formato YYYY-MM-DD."

TOOL_DEFINITIONS = [
    {
        "name": "get_revenue_forecast",
        "description": (
            "Prevision de facturacion diaria de La Roca para una fecha (modelo ML real, "
            "nunca lo calcules tu). Si la fecha ya paso, es un backtest contra el dato real."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": DATE_DESC},
                "weather_overrides": _WEATHER_SCHEMA,
            },
            "required": ["date"],
        },
    },
    {
        "name": "get_ticket_forecast",
        "description": "Prevision de ticket medio diario para una fecha (modelo ML real).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": DATE_DESC},
                "weather_overrides": _WEATHER_SCHEMA,
            },
            "required": ["date"],
        },
    },
    {
        "name": "get_historical_revenue",
        "description": "Facturacion, ticket medio y numero de tickets reales entre dos fechas (dato historico real).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": DATE_DESC},
                "date_to": {"type": "string", "description": DATE_DESC},
            },
            "required": ["date_from", "date_to"],
        },
    },
    {
        "name": "get_reservation_status",
        "description": "Reservas registradas en la app para una fecha (nombre, hora, comensales, estado).",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": DATE_DESC}},
            "required": ["date"],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": (
            "Meteorologia de una fecha: dato historico real si ya paso, prevision real si "
            "esta dentro del horizonte de la API, o climatologia (media historica) si no."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": DATE_DESC}},
            "required": ["date"],
        },
    },
    {
        "name": "get_events",
        "description": "Eventos locales relevantes cerca del restaurante para una fecha (dato real curado).",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": DATE_DESC}},
            "required": ["date"],
        },
    },
    {
        "name": "get_review_analysis",
        "description": (
            "Agregados reales sobre las 965 reseñas (Google + TripAdvisor): distribucion de "
            "sentimiento y top aspectos positivos/negativos, opcionalmente en un rango de fechas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": DATE_DESC},
                "date_to": {"type": "string", "description": DATE_DESC},
            },
        },
    },
    {
        "name": "get_ai_visibility",
        "description": (
            "Metricas de visibilidad del restaurante en LLMs (Mention Rate, Recommendation "
            "Rate...). TODAVIA EN MOCK -- la bateria de prompts no se ha ejecutado; devolvera "
            "valores nulos, dilo explicitamente si se usa."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query_set": {"type": "string"}},
        },
    },
    {
        "name": "get_competitor_analysis",
        "description": (
            "Comparativa frente a competidores locales. TODAVIA EN MOCK -- lista de "
            "competidores pendiente de confirmar; devolvera una lista vacia."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"competitor_ids": {"type": "array", "items": {"type": "string"}}},
        },
    },
    {
        "name": "run_revenue_scenario",
        "description": (
            "Simula como cambiaria la prevision de facturacion si cambiaran variables de "
            "entrada (p.ej. si lloviera). Es un 'model-based scenario', NUNCA presentar el "
            "resultado como causalidad ('esto hara que...') -- solo como simulacion del modelo."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": DATE_DESC + " Debe ser una fecha ya presente en el historico."},
                "weather_overrides": _WEATHER_SCHEMA,
                "event_overrides": _EVENT_SCHEMA,
            },
            "required": ["date"],
        },
    },
    {
        "name": "run_ticket_scenario",
        "description": "Igual que run_revenue_scenario pero para el ticket medio -- mismo aviso de 'model-based scenario'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": DATE_DESC + " Debe ser una fecha ya presente en el historico."},
                "weather_overrides": _WEATHER_SCHEMA,
                "event_overrides": _EVENT_SCHEMA,
            },
            "required": ["date"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": (
            "Busqueda documental (RAG) sobre la carta, las politicas del restaurante y el "
            "texto de reseñas reales. Usa esto SOLO para conocimiento textual/cualitativo -- "
            "nunca para una cifra o prediccion, eso es siempre otra tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "description": "Numero de fragmentos a devolver (por defecto 4)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_noshow_risk",
        "description": (
            "Probabilidad de no-show de UNA reserva concreta (no de un dia entero). Señal "
            "debil segun el propio notebook -- preséntalo siempre como indicio, nunca como "
            "certeza. 'origin' debe ser appmovil/moduloweb/software/terceros (nunca 'walk in')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reservation_date": {"type": "string", "description": DATE_DESC},
                "shift": {"type": "string", "enum": ["Comida", "Cena"]},
                "people": {"type": "integer"},
                "origin": {"type": "string", "enum": ["appmovil", "moduloweb", "software", "terceros"]},
                "es_grupo_grande": {"type": "boolean"},
                "zone": {"type": "string", "enum": ["Sala", "Terraza Cubierta"]},
                "antelacion_horas": {"type": "number", "description": "Obligatorio: la variable mas importante del modelo."},
                "reservas_mismo_dia_turno": {"type": "integer"},
            },
            "required": ["reservation_date", "shift", "people", "origin", "antelacion_horas"],
        },
    },
    {
        "name": "list_sales_articles",
        "description": (
            "Lista los 82 platos de carta que el modelo de ventas sabe predecir, con su "
            "article_code. Llama a esto primero si necesitas el codigo de un plato por su nombre."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_article_sales_forecast",
        "description": (
            "Prevision de unidades/dia de UN plato concreto para un periodo semanal (no "
            "diario). Necesita article_code -- usa list_sales_articles si no lo conoces."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article_code": {"type": "integer"},
                "period_start": {"type": "string", "description": "Lunes del periodo semanal (YYYY-MM-DD). Opcional: por defecto el ultimo periodo conocido."},
                "exogenous_overrides": _SALES_EXOGENOUS_SCHEMA,
            },
            "required": ["article_code"],
        },
    },
    {
        "name": "get_top_selling_articles",
        "description": (
            "Los platos con mas unidades/dia previstas para un periodo semanal -- util para "
            "orientar la compra de la semana."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period_start": {"type": "string", "description": "Lunes del periodo semanal (YYYY-MM-DD). Opcional."},
                "top_n": {"type": "integer", "description": "Por defecto 3."},
                "exogenous_overrides": _SALES_EXOGENOUS_SCHEMA,
            },
        },
    },
    {
        "name": "get_review_quality_trend",
        "description": (
            "Chequeo estadistico (no un modelo de ML) sobre si el sentimiento medio de las "
            "reseñas del ultimo mes ha caido de forma anomala frente al historico -- una "
            "alerta temprana de calidad, no una prediccion."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_DISPATCH = {
    "get_revenue_forecast": tools.get_revenue_forecast,
    "get_ticket_forecast": tools.get_ticket_forecast,
    "get_historical_revenue": tools.get_historical_revenue,
    "get_reservation_status": tools.get_reservation_status,
    "get_weather_forecast": tools.get_weather_forecast,
    "get_events": tools.get_events,
    "get_review_analysis": tools.get_review_analysis,
    "get_ai_visibility": tools.get_ai_visibility,
    "get_competitor_analysis": tools.get_competitor_analysis,
    "run_revenue_scenario": tools.run_revenue_scenario,
    "run_ticket_scenario": tools.run_ticket_scenario,
    "search_knowledge_base": tools.search_knowledge_base,
    "get_noshow_risk": tools.get_noshow_risk,
    "list_sales_articles": tools.list_sales_articles,
    "get_article_sales_forecast": tools.get_article_sales_forecast,
    "get_top_selling_articles": tools.get_top_selling_articles,
    "get_review_quality_trend": tools.get_review_quality_trend,
}

assert {t["name"] for t in TOOL_DEFINITIONS} == set(TOOL_DISPATCH)
