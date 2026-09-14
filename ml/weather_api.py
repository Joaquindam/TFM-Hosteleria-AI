"""Conexion a la API de prevision meteorologica en tiempo real de
Open-Meteo (gratuita, sin API key) -- https://open-meteo.com

Mismas coordenadas exactas que ya se usan en el historico
(data/bronze/open-meteo-pozuelo.csv): lat 40.45694, lon -3.8081665
(Pozuelo de Alarcón), para que el dato en vivo sea consistente con el
resto del pipeline.

Solo cubre un horizonte corto (la propia API de Open-Meteo no da mas alla
de ~16 dias reales de prevision) -- fuera de eso, o si falla la llamada
por cualquier motivo (sin internet, fecha fuera de rango, respuesta
vacia...), devuelve None sin lanzar excepcion: quien llama (ver
ml/features.py) cae entonces a la estimacion climatologica.
"""
from __future__ import annotations

import pandas as pd
import requests

LATITUDE = 40.45694
LONGITUDE = -3.8081665
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 5

_DAILY_FIELDS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "precipitation_hours", "wind_speed_10m_max", "sunshine_duration",
]


def fetch_live_forecast(date) -> dict | None:
    """Intenta traer la prevision real de Open-Meteo para `date`. Nunca
    lanza excepcion -- devuelve None si no hay dato (fecha demasiado
    lejana, fallo de red, respuesta inesperada), para que el llamante
    decida el respaldo (climatologia)."""
    target = pd.Timestamp(date).normalize()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": ",".join(_DAILY_FIELDS),
        "timezone": "Europe/Madrid",
        "start_date": target.date().isoformat(),
        "end_date": target.date().isoformat(),
    }
    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        tiempos = daily.get("time") or []
        if target.date().isoformat() not in tiempos:
            return None
        idx = tiempos.index(target.date().isoformat())

        valores = {
            "temperature_max": daily["temperature_2m_max"][idx],
            "temperature_min": daily["temperature_2m_min"][idx],
            "temperature_mean": daily["temperature_2m_mean"][idx],
            "precipitation_mm": daily["precipitation_sum"][idx],
            "precipitation_hours": daily["precipitation_hours"][idx],
            "wind_speed_max": daily["wind_speed_10m_max"][idx],
            "sunshine_duration_h": daily["sunshine_duration"][idx] / 3600.0,
        }
        if any(v is None for v in valores.values()):
            return None
        return valores
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError):
        return None
