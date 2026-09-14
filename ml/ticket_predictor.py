"""Adapter sobre los modelos de ticket medio ya entrenados
(results/models/mejor_modelo_ticket_medio*.joblib). No reentrena nada -- ver
PROJECT_BRIEF.md secciones 4 y 6, y CLAUDE.md ("nunca reentrenar").

Mismo esquema de dos modelos que ml/revenue_predictor.py: corto plazo
(notebook 07, con memoria reciente real) y largo plazo (notebook 10,
sin ella -- MAE test ~14.87€ frente a ~13.85€ del modelo corto).
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from ml.paths import TICKET_MODEL_PATH, TICKET_MODEL_LARGO_PLAZO_PATH
from ml.features import build_ticket_feature_table, TICKET_FEATURE_COLS, TICKET_FEATURE_COLS_LARGO_PLAZO

MAX_FUTURE_GAP_DAYS = 3


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(TICKET_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_model_largo_plazo():
    return joblib.load(TICKET_MODEL_LARGO_PLAZO_PATH)


def _to_timestamp(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def predict_ticket(date, weather_overrides: dict | None = None, event_overrides: dict | None = None) -> dict:
    """Predice el ticket medio de `date`. Misma logica de modos
    (historical / forecast / forecast_sin_historial_reciente) que
    predict_revenue -- ver ahi para el detalle completo.
    """
    target = _to_timestamp(date)

    base_table = build_ticket_feature_table()
    last_known = base_table["fecha"].max()

    if (base_table["fecha"] == target).any():
        table = base_table
        mode = "historical"
        used_recent_history = True
    else:
        cerca = target <= last_known + pd.Timedelta(days=MAX_FUTURE_GAP_DAYS)
        table = build_ticket_feature_table(
            extra_date=target, weather_overrides=weather_overrides, event_overrides=event_overrides
        )
        mode = "forecast" if cerca else "forecast_sin_historial_reciente"
        used_recent_history = cerca

    feature_cols = TICKET_FEATURE_COLS if used_recent_history else TICKET_FEATURE_COLS_LARGO_PLAZO
    model = _load_model() if used_recent_history else _load_model_largo_plazo()

    row = table.loc[table["fecha"] == target, feature_cols].tail(1)
    actual_row = table.loc[table["fecha"] == target, "ticket_medio"]
    actual = actual_row.iloc[-1] if len(actual_row) and pd.notna(actual_row.iloc[-1]) else None

    prediction = float(model.predict(row)[0])
    weather_source = table.attrs.get("weather_source")

    result = {
        "date": target.date().isoformat(),
        "prediction": round(prediction, 2),
        "actual": round(float(actual), 2) if actual is not None else None,
        "mode": mode,
        "model": "mejor_modelo_ticket_medio" if used_recent_history else "mejor_modelo_ticket_medio_largo_plazo",
        "source": "model",
        "weather_source": weather_source or "historical_data",
        "used_event_overrides": bool(event_overrides),
        "used_recent_history": used_recent_history,
    }
    avisos = []
    if not used_recent_history:
        avisos.append(
            "Sin historial reciente real para esta fecha -- se usa el modelo de largo plazo "
            "(entrenado sin variables de memoria). MAE test ~14.87€ frente a ~13.85€ del modelo "
            "de corto plazo (ver notebooks/10_prediccion_ticket_medio_largo_plazo.ipynb)."
        )
    if weather_source == "climatology":
        avisos.append(
            "Meteorologia estimada por climatologia (media de dias similares del anio en el "
            "historico real), no una prevision -- fiabilidad baja."
        )
    if avisos:
        result["aviso"] = " ".join(avisos)
    return result
