"""Escenarios "que pasaria si" sobre los modelos ya entrenados.

Importante (ver PROJECT_BRIEF.md seccion 19): esto es una simulacion del
modelo al cambiar inputs de entrada, NO un analisis causal -- el modelo
puede no haber visto esa combinacion de variables en el historico.
Cualquier texto generado sobre esto debe usar la expresion "model-based
scenario" y evitar lenguaje causal ("esto hara que...", "esto provocara...").
"""
from __future__ import annotations

import pandas as pd

from ml.revenue_predictor import _load_model as _load_revenue_model, predict_revenue
from ml.ticket_predictor import _load_model as _load_ticket_model, predict_ticket
from ml.features import (
    build_revenue_feature_table, build_ticket_feature_table,
    REVENUE_FEATURE_COLS, TICKET_FEATURE_COLS,
)


def _apply_overrides(row_df: pd.DataFrame, weather_overrides: dict | None, event_overrides: dict | None) -> pd.DataFrame:
    row_df = row_df.copy()
    idx = row_df.index[0]

    if weather_overrides:
        for k, v in weather_overrides.items():
            if k in row_df.columns:
                row_df.at[idx, k] = v
        if {"temperature_max", "temperature_min"} <= set(row_df.columns):
            row_df.at[idx, "rango_temperatura"] = row_df.at[idx, "temperature_max"] - row_df.at[idx, "temperature_min"]
        if "precipitation_mm" in row_df.columns:
            row_df.at[idx, "es_dia_lluvioso"] = int((row_df.at[idx, "precipitation_mm"] or 0) > 0)
        if "temperature_max" in row_df.columns:
            row_df.at[idx, "es_dia_muy_caluroso"] = int(row_df.at[idx, "temperature_max"] >= 30)
        if "temperature_min" in row_df.columns:
            row_df.at[idx, "es_dia_muy_frio"] = int(row_df.at[idx, "temperature_min"] <= 5)

    if event_overrides:
        for k, v in event_overrides.items():
            if k in row_df.columns:
                row_df.at[idx, k] = v

    return row_df


def _scenario(date, predict_fn, load_model_fn, build_table_fn, feature_cols,
              weather_overrides, event_overrides) -> dict:
    baseline = predict_fn(date)

    table = build_table_fn()
    target = pd.Timestamp(date).normalize()
    row_df = table.loc[table["fecha"] == target, feature_cols].tail(1)
    if row_df.empty:
        raise ValueError(
            f"No hay datos base para simular un escenario en {target.date()}: "
            "usa una fecha ya presente en el historico (los escenarios parten "
            "de un dia real conocido y le cambian una variable)."
        )

    scenario_row = _apply_overrides(row_df, weather_overrides, event_overrides)
    model = load_model_fn()
    scenario_prediction = float(model.predict(scenario_row)[0])

    return {
        "date": target.date().isoformat(),
        "baseline_prediction": baseline["prediction"],
        "scenario_prediction": round(scenario_prediction, 2),
        "difference": round(scenario_prediction - baseline["prediction"], 2),
        "weather_overrides": weather_overrides,
        "event_overrides": event_overrides,
        "type": "model-based scenario",
        "disclaimer": "Simulacion del modelo al cambiar inputs, no un analisis causal.",
    }


def run_revenue_scenario(date, weather_overrides: dict | None = None, event_overrides: dict | None = None) -> dict:
    return _scenario(date, predict_revenue, _load_revenue_model, build_revenue_feature_table,
                      REVENUE_FEATURE_COLS, weather_overrides, event_overrides)


def run_ticket_scenario(date, weather_overrides: dict | None = None, event_overrides: dict | None = None) -> dict:
    return _scenario(date, predict_ticket, _load_ticket_model, build_ticket_feature_table,
                      TICKET_FEATURE_COLS, weather_overrides, event_overrides)
