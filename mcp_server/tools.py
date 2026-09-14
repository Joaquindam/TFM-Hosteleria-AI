"""Logica de negocio detras de cada tool MCP (ver PROJECT_BRIEF.md secciones
9 y 10). Funciones planas y testables sin necesidad de un cliente MCP --
`server.py` solo las registra. Cada una devuelve un dict con un campo
`source` (`"model"`, `"historical_data"`, `"app"`, `"rag"` o `"mock"`) para
que quien las consuma (el LLM app, en la Fase 4) pueda citar de donde viene
cada dato -- nunca se presenta un mock como si fuera real.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backend import reservas_store
from ml.data_loader import load_gold
from ml.noshow_predictor import predict_noshow
from ml.paths import PROJECT_ROOT, SILVER_SNAP
from ml.revenue_predictor import predict_revenue
from ml.reviews_quality import detect_quality_drop
from ml.sales_predictor import list_available_articles, predict_sales, top_articles_for_period
from ml.scenarios import run_revenue_scenario as _run_revenue_scenario
from ml.scenarios import run_ticket_scenario as _run_ticket_scenario
from ml.ticket_predictor import predict_ticket
from rag.retriever import retrieve_context

_EVENTS_PATH = PROJECT_ROOT / "data" / "bronze" / "eventos_relevantes_pozuelo_la_roca_2025_2026.csv"

_WEATHER_FIELDS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
]


def get_revenue_forecast(date: str, weather_overrides: Optional[dict] = None) -> dict:
    """Previsión de facturación diaria (usa el modelo real, nunca lo calcula el LLM)."""
    return predict_revenue(date, weather_overrides=weather_overrides)


def get_ticket_forecast(date: str, weather_overrides: Optional[dict] = None) -> dict:
    """Previsión de ticket medio diario."""
    return predict_ticket(date, weather_overrides=weather_overrides)


def get_historical_revenue(date_from: str, date_to: str) -> dict:
    """Facturación, ticket medio y nº de tickets reales entre dos fechas (data/gold)."""
    gold = load_gold()
    mask = (gold["fecha"] >= pd.Timestamp(date_from)) & (gold["fecha"] <= pd.Timestamp(date_to))
    rows = gold.loc[mask, ["fecha", "facturacion", "ticket_medio", "num_tickets"]].copy()
    rows["fecha"] = rows["fecha"].dt.strftime("%Y-%m-%d")
    return {
        "date_from": date_from,
        "date_to": date_to,
        "days": rows.to_dict(orient="records"),
        "source": "historical_data",
    }


def get_reservation_status(date: str) -> dict:
    """Reservas registradas para `date` desde la app (data/gold/reservas.db)."""
    reservas = reservas_store.list_reservas_by_date(date, include_cancelled=False)
    return {
        "date": date,
        "reservas": reservas,
        "total_reservas": len(reservas),
        "total_comensales": sum(r["comensales"] for r in reservas),
        "source": "app",
    }


def get_weather_forecast(date: str) -> dict:
    """Meteorología de `date`. Orden: dato real del histórico si ya está en
    `data/gold` (`source: "historical_data"`); si no, previsión real de la
    API de Open-Meteo si la fecha está dentro de su horizonte
    (`source: "live_forecast"`); si tampoco, media climatológica de días
    similares del año en los ~4 años de datos reales (`source:
    "climatology"`, fiabilidad baja pero no inventada)."""
    gold = load_gold()
    target = pd.Timestamp(date)
    row = gold.loc[gold["fecha"] == target]

    if not row.empty:
        r = row.iloc[0]
        return {
            "date": date,
            **{f: (None if pd.isna(r[f]) else float(r[f])) for f in _WEATHER_FIELDS},
            "source": "historical_data",
        }

    from ml.weather_api import fetch_live_forecast
    live = fetch_live_forecast(target)
    if live is not None:
        return {"date": date, **{f: live[f] for f in _WEATHER_FIELDS}, "source": "live_forecast"}

    from ml.climatology import estimate_weather_climatology
    clima = estimate_weather_climatology(target)
    return {
        "date": date,
        **{f: clima[f] for f in _WEATHER_FIELDS},
        "source": "climatology",
        "n_dias_muestra": clima["n_dias_muestra"],
        "n_anios_muestra": clima["n_anios_muestra"],
    }


def get_events(date: str) -> dict:
    """Eventos locales relevantes para `date`
    (data/bronze/eventos_relevantes_pozuelo_la_roca_2025_2026.csv, dato real
    curado por la autora). MOCK solo si `date` cae fuera del rango cubierto
    por ese fichero."""
    events_df = pd.read_csv(_EVENTS_PATH, parse_dates=["fecha_inicio", "fecha_fin"])
    target = pd.Timestamp(date)

    covered = events_df["fecha_inicio"].min() <= target <= events_df["fecha_fin"].max()
    if not covered:
        return {"date": date, "events": [], "source": "mock"}

    mask = (events_df["fecha_inicio"] <= target) & (events_df["fecha_fin"] >= target)
    matched = events_df.loc[mask, ["nombre_evento", "categoria", "impacto_esperado",
                                    "direccion_demanda", "segmento_cliente"]]
    return {"date": date, "events": matched.to_dict(orient="records"), "source": "historical_data"}


def get_review_analysis(date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """Agregados reales sobre las reseñas ya depuradas y con sentimiento por
    aspecto (ABSA) en data/silver/snapshots/ -- no reprocesa las reseñas."""
    resenas = pd.read_parquet(SILVER_SNAP / "resenas_silver.parquet")
    absa = pd.read_parquet(SILVER_SNAP / "resenas_absa_silver.parquet")

    if date_from:
        resenas = resenas[resenas["review_date"] >= pd.Timestamp(date_from)]
        absa = absa[absa["review_date"] >= pd.Timestamp(date_from)]
    if date_to:
        resenas = resenas[resenas["review_date"] <= pd.Timestamp(date_to)]
        absa = absa[absa["review_date"] <= pd.Timestamp(date_to)]

    sentiment_counts = resenas["sentiment_source"].value_counts(dropna=True).to_dict()

    def _top_aspects(label: str) -> list[dict]:
        subset = absa[absa["absa_label"] == label]
        if subset.empty:
            return []
        agg = (
            subset.groupby("entity_name")["absa_score"]
            .agg(["mean", "count"])
            .sort_values("count", ascending=False)
            .head(5)
            .reset_index()
        )
        return [
            {"entidad": row["entity_name"], "menciones": int(row["count"]), "score_medio": round(float(row["mean"]), 3)}
            for _, row in agg.iterrows()
        ]

    return {
        "date_from": date_from,
        "date_to": date_to,
        "total_resenas": int(len(resenas)),
        "sentimiento": {str(k): int(v) for k, v in sentiment_counts.items()},
        "top_aspectos_positivos": _top_aspects("Positivo"),
        "top_aspectos_negativos": _top_aspects("Negativo"),
        "source": "historical_data",
    }


def get_ai_visibility(query_set: Optional[str] = None) -> dict:
    """Métricas de visibilidad en IAs (Mention Rate, Recommendation Rate...)
    -- ver PROJECT_BRIEF.md secciones 11 y 37. MOCK: la batería de prompts
    todavía no se ha ejecutado contra ningún LLM (Fase 5)."""
    return {
        "query_set": query_set or "default",
        "mention_rate": None,
        "recommendation_rate": None,
        "average_position": None,
        "attributes": [],
        "source": "mock",
    }


def get_competitor_analysis(competitor_ids: Optional[list] = None) -> dict:
    """Comparativa frente a competidores locales -- ver PROJECT_BRIEF.md
    sección 22 y 37. MOCK: lista de competidores pendiente de confirmar por
    la autora (Fase 5)."""
    return {"competitor_ids": competitor_ids or [], "competitors": [], "source": "mock"}


def run_revenue_scenario(date: str, weather_overrides: Optional[dict] = None,
                          event_overrides: Optional[dict] = None) -> dict:
    """Model-based scenario de facturación (nunca causal, ver sección 19)."""
    return _run_revenue_scenario(date, weather_overrides=weather_overrides, event_overrides=event_overrides)


def run_ticket_scenario(date: str, weather_overrides: Optional[dict] = None,
                         event_overrides: Optional[dict] = None) -> dict:
    """Model-based scenario de ticket medio (nunca causal, ver sección 19)."""
    return _run_ticket_scenario(date, weather_overrides=weather_overrides, event_overrides=event_overrides)


def search_knowledge_base(query: str, k: int = 4) -> dict:
    """RAG: recupera texto (carta, políticas, reseñas reales) relevante para
    `query`. Nunca devuelve cifras calculadas -- eso es siempre otra tool."""
    return {"query": query, "results": retrieve_context(query, k=k), "source": "rag"}


def get_noshow_risk(reservation_date: str, shift: str, people: int, origin: str,
                     es_grupo_grande: bool = False, zone: Optional[str] = None,
                     antelacion_horas: Optional[float] = None,
                     reservas_mismo_dia_turno: Optional[int] = None) -> dict:
    """Probabilidad de no-show de una reserva individual (ver
    ml/noshow_predictor.py). Señal débil segun el propio notebook -- usar
    como indicio, nunca como certeza. `antelacion_horas` es obligatorio."""
    return predict_noshow(
        reservation_date=reservation_date, shift=shift, people=people, origin=origin,
        es_grupo_grande=es_grupo_grande, zone=zone, antelacion_horas=antelacion_horas,
        reservas_mismo_dia_turno=reservas_mismo_dia_turno,
    )


def list_sales_articles() -> dict:
    """Los 82 artículos de carta que el modelo de ventas sabe predecir, con
    su article_code -- necesario para poder llamar a get_article_sales_forecast
    a partir del nombre de un plato en lenguaje natural."""
    return {"articles": list_available_articles(), "source": "model"}


def get_article_sales_forecast(article_code: int, period_start: Optional[str] = None,
                                exogenous_overrides: Optional[dict] = None) -> dict:
    """Predicción de unidades/día de un artículo de carta concreto para un
    periodo semanal (ver ml/sales_predictor.py). Usa list_sales_articles
    primero si no se conoce el article_code de un plato."""
    return predict_sales(article_code, period_start=period_start, exogenous_overrides=exogenous_overrides)


def get_top_selling_articles(period_start: Optional[str] = None, top_n: int = 3,
                              exogenous_overrides: Optional[dict] = None) -> dict:
    """Los `top_n` artículos con más unidades/día previstas para un periodo
    -- pensado para orientar la compra semanal (ver ml/sales_predictor.py)."""
    return top_articles_for_period(period_start=period_start, top_n=top_n, exogenous_overrides=exogenous_overrides)


def get_review_quality_trend() -> dict:
    """Alerta de caída de calidad en el sentimiento de las reseñas (ver
    notebooks/08_resenias.ipynb, Sección 4.1 y ml/reviews_quality.py). Es un
    chequeo estadístico sobre el sentimiento medio mensual ya calculado en
    Silver, no la salida de un modelo de ML entrenado."""
    return detect_quality_drop()
