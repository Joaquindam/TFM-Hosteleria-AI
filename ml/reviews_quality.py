"""Replica la Seccion 4.1 de notebooks/08_resenias.ipynb: deteccion de
caida temporal de calidad en el sentimiento de las reseñas.

Agrega el sentimiento medio por reseña (ABSA, ya calculado en Silver) por
mes y compara el ultimo mes disponible contra la media historica -- una
alerta temprana, no una prediccion de un modelo ML. No reprocesa reseñas
ni recalcula ABSA (eso vive en el notebook), solo lee
data/silver/snapshots/resenas_absa_silver.parquet.
"""
from __future__ import annotations

import pandas as pd

from ml.paths import SILVER_SNAP

RESENAS_ABSA_SILVER = SILVER_SNAP / "resenas_absa_silver.parquet"
Z_THRESHOLD = 1.5  # mismo umbral que el notebook


def _monthly_quality_series() -> pd.Series:
    absa = pd.read_parquet(RESENAS_ABSA_SILVER)
    absa["review_date"] = pd.to_datetime(absa["review_date"])

    review_level = absa.groupby("review_id").agg(
        review_date=("review_date", "first"),
        avg_absa_score=("absa_score", "mean"),
    )
    monthly = (
        review_level.dropna()
        .assign(month=lambda d: d["review_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["avg_absa_score"].mean()
        .sort_index()
    )
    return monthly


def _series_to_records(monthly: pd.Series) -> list[dict]:
    return [{"month": m.date().isoformat(), "avg_absa_score": round(float(v), 3)} for m, v in monthly.items()]


def detect_quality_drop(z_threshold: float = Z_THRESHOLD) -> dict:
    """Mismo criterio que el notebook (Seccion 4.1): alerta si el ultimo mes
    cae mas de `z_threshold` desviaciones estandar por debajo de la media
    historica del sentimiento medio (ABSA) de las reseñas.
    """
    monthly = _monthly_quality_series()
    base = {"monthly_sentiment": _series_to_records(monthly), "source": "historical_data"}

    if len(monthly) < 4:
        return {**base, "is_alert": False, "reason": "serie demasiado corta para evaluar (<4 meses)"}

    history = monthly.iloc[:-1]
    last_month, last_value = monthly.index[-1], monthly.iloc[-1]
    historic_mean, historic_std = history.mean(), history.std()

    if historic_std == 0 or pd.isna(historic_std):
        return {**base, "is_alert": False, "reason": "sin variabilidad historica suficiente"}

    z_score = (last_value - historic_mean) / historic_std
    return {
        **base,
        "is_alert": bool(z_score < -z_threshold),
        "last_month": last_month.date().isoformat(),
        "last_value": round(float(last_value), 3),
        "historic_mean": round(float(historic_mean), 3),
        "z_score": round(float(z_score), 2),
    }
