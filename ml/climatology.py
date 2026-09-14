"""Estimacion climatologica de meteorologia (media historica de dias
similares del anio) para cuando no hay una prevision real ni el usuario
aporta una a mano.

No es una prevision meteorologica de verdad -- es "como suele ser el
tiempo estos dias", calculado sobre los ~4 anios de datos reales en
data/silver/snapshots/meteo_diaria_silver.parquet. Fiabilidad baja pero
real (no inventada), y siempre marcada como tal (`source: "climatology"`)
para que nadie la confunda con un dato observado o una prevision de una
API real.
"""
from __future__ import annotations

import pandas as pd

from ml.paths import METEO_DIARIA_SILVER

WEATHER_FIELDS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
]

DEFAULT_WINDOW_DAYS = 7  # +/- dias alrededor del mismo dia del anio, en todos los anios disponibles


def estimate_weather_climatology(date: pd.Timestamp, window_days: int = DEFAULT_WINDOW_DAYS) -> dict:
    """Media de cada variable meteorologica sobre todos los dias historicos
    dentro de +/- `window_days` del dia del anio de `date` (en cualquier
    anio disponible) -- una ventana centrada en la fecha, no solo el dia
    exacto, para no depender de un unico dato ruidoso.
    """
    meteo = pd.read_parquet(METEO_DIARIA_SILVER)
    meteo["date"] = pd.to_datetime(meteo["date"])

    target_doy = pd.Timestamp(date).dayofyear
    doy = meteo["date"].dt.dayofyear
    diff = (doy - target_doy).abs()
    diff = diff.combine(365 - diff, min)  # distancia circular (diciembre-enero son "cercanos")

    ventana = meteo.loc[diff <= window_days]
    if ventana.empty:
        ventana = meteo  # fallback extremo, no deberia pasar con datos anuales completos

    valores = {f: float(ventana[f].mean()) for f in WEATHER_FIELDS}
    return {
        **valores,
        "n_dias_muestra": int(len(ventana)),
        "n_anios_muestra": int(ventana["date"].dt.year.nunique()),
        "ventana_dias": window_days,
    }
