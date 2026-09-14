from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, HTTPException

from ml.exceptions import MissingInputError
from ml.explainability import explain_revenue, explain_ticket
from ml.revenue_predictor import predict_revenue
from ml.ticket_predictor import predict_ticket

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/{target_date}")
def get_forecast(
    target_date: date_type,
    temperature_max: Optional[float] = None,
    temperature_min: Optional[float] = None,
    temperature_mean: Optional[float] = None,
    precipitation_mm: Optional[float] = None,
    precipitation_hours: Optional[float] = None,
    wind_speed_max: Optional[float] = None,
    sunshine_duration_h: Optional[float] = None,
):
    """Meteorologia como query params opcionales: solo hacen falta cuando la
    fecha es un dia futuro sin dato real todavia (ver ml/exceptions.py:
    MissingInputError). Para una fecha ya en el historico se ignoran -- el
    dato real ya esta en data/gold, nunca se sustituye por un override.
    """
    weather_overrides = {
        "temperature_max": temperature_max, "temperature_min": temperature_min,
        "temperature_mean": temperature_mean, "precipitation_mm": precipitation_mm,
        "precipitation_hours": precipitation_hours, "wind_speed_max": wind_speed_max,
        "sunshine_duration_h": sunshine_duration_h,
    }
    if all(v is None for v in weather_overrides.values()):
        weather_overrides = None

    try:
        revenue = predict_revenue(target_date, weather_overrides=weather_overrides)
        ticket = predict_ticket(target_date, weather_overrides=weather_overrides)
    except MissingInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"revenue": revenue, "ticket": ticket}


@router.get("/{target_date}/explain")
def get_forecast_explain(target_date: date_type):
    try:
        return {"revenue": explain_revenue(target_date), "ticket": explain_ticket(target_date)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
