"""Adapter sobre los modelos de facturacion diaria ya entrenados
(results/models/mejor_modelo_facturacion_diaria*.joblib). No reentrena nada
-- ver PROJECT_BRIEF.md secciones 4 y 6, y CLAUDE.md ("nunca reentrenar").

Hay DOS modelos, para dos situaciones distintas:
- corto plazo (notebook 05): usa historial reciente real (facturacion de
  los ultimos dias/semanas) -- mas preciso, solo valido cuando ese
  historial existe de verdad (backtest, o el dia siguiente al ultimo dato).
- largo plazo (notebook 09): entrenado SIN esas variables de memoria, para
  fechas demasiado lejanas para tener un "hace 7 dias" real. Algo menos
  preciso (494€ MAE test frente a 457€), pero honesto: nunca sustituye el
  historial por un valor inventado.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from ml.paths import REVENUE_MODEL_PATH, REVENUE_MODEL_LARGO_PLAZO_PATH
from ml.features import build_revenue_feature_table, REVENUE_FEATURE_COLS, REVENUE_FEATURE_COLS_LARGO_PLAZO

MAX_FUTURE_GAP_DAYS = 3  # margen para saltar cierres cortos (lunes, puentes) con historial real


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(REVENUE_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_model_largo_plazo():
    return joblib.load(REVENUE_MODEL_LARGO_PLAZO_PATH)


def _to_timestamp(date) -> pd.Timestamp:
    return pd.Timestamp(date).normalize()


def predict_revenue(date, weather_overrides: dict | None = None, event_overrides: dict | None = None) -> dict:
    """Predice la facturacion de `date`.

    - Si `date` ya esta en el historico (data/gold): modo "historical"
      (backtest, se compara con el dato real ya conocido). Usa el modelo
      de corto plazo.
    - Si `date` es un dia futuro cercano (hasta MAX_FUTURE_GAP_DAYS tras el
      ultimo dato conocido): modo "forecast" -- modelo de corto plazo con
      el historial reciente real encadenado con normalidad.
    - Si `date` es un dia futuro mas lejano: modo
      "forecast_sin_historial_reciente" -- usa el modelo de largo plazo
      (notebook 09), entrenado sin variables de memoria. Menos preciso
      (ver docstring del modulo) pero sin inventar ningun dato.
    - Meteorologia para fechas futuras: si se aporta `weather_overrides`
      (7 variables), se usa tal cual. Si no se aporta nada, se calcula
      automaticamente una media climatologica (dias similares del anio en
      los ~4 anios de datos reales -- ver ml/climatology.py). Fiabilidad
      baja pero real, nunca inventada; se marca en `weather_source`.
    """
    target = _to_timestamp(date)

    base_table = build_revenue_feature_table()
    last_known = base_table["fecha"].max()

    if (base_table["fecha"] == target).any():
        table = base_table
        mode = "historical"
        used_recent_history = True
    else:
        cerca = target <= last_known + pd.Timedelta(days=MAX_FUTURE_GAP_DAYS)
        table = build_revenue_feature_table(
            extra_date=target, weather_overrides=weather_overrides, event_overrides=event_overrides
        )
        mode = "forecast" if cerca else "forecast_sin_historial_reciente"
        used_recent_history = cerca

    feature_cols = REVENUE_FEATURE_COLS if used_recent_history else REVENUE_FEATURE_COLS_LARGO_PLAZO
    model = _load_model() if used_recent_history else _load_model_largo_plazo()

    row = table.loc[table["fecha"] == target, feature_cols].tail(1)
    actual_row = table.loc[table["fecha"] == target, "facturacion"]
    actual = actual_row.iloc[-1] if len(actual_row) and pd.notna(actual_row.iloc[-1]) else None

    prediction = float(model.predict(row)[0])
    weather_source = table.attrs.get("weather_source")  # None si la fecha es historica (meteo real de gold)

    result = {
        "date": target.date().isoformat(),
        "prediction": round(prediction, 2),
        "actual": round(float(actual), 2) if actual is not None else None,
        "mode": mode,
        "model": "mejor_modelo_facturacion_diaria" if used_recent_history else "mejor_modelo_facturacion_diaria_largo_plazo",
        "source": "model",
        "weather_source": weather_source or "historical_data",
        "used_event_overrides": bool(event_overrides),
        "used_recent_history": used_recent_history,
    }
    avisos = []
    if not used_recent_history:
        avisos.append(
            "Sin historial reciente real para esta fecha -- se usa el modelo de largo plazo "
            "(entrenado sin variables de memoria de facturacion pasada). MAE test ~494€ frente "
            "a ~457€ del modelo de corto plazo (ver notebooks/09_prediccion_facturacion_largo_plazo.ipynb)."
        )
    if weather_source == "climatology":
        avisos.append(
            "Meteorologia estimada por climatologia (media de dias similares del anio en el "
            "historico real), no una prevision -- fiabilidad baja. Aporta weather_overrides "
            "con una prevision real si la tienes."
        )
    if avisos:
        result["aviso"] = " ".join(avisos)
    return result
