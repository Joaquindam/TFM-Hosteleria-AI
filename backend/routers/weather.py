from datetime import date as date_type

from fastapi import APIRouter

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/{target_date}")
def get_weather(target_date: date_type):
    # MOCK - sin dato real: no hay integracion con una API de meteorologia
    # todavia (ver PROJECT_BRIEF.md seccion 18 y CLAUDE.md).
    return {
        "date": target_date.isoformat(),
        "temperature_max": None,
        "temperature_min": None,
        "temperature_mean": None,
        "precipitation_mm": None,
        "precipitation_hours": None,
        "wind_speed_max": None,
        "sunshine_duration_h": None,
        "source": "mock",
    }
