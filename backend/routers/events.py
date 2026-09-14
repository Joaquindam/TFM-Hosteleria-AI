from datetime import date as date_type

from fastapi import APIRouter

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/{target_date}")
def get_events(target_date: date_type):
    # MOCK - sin dato real: no hay fuente de eventos locales todavia
    # (ver PROJECT_BRIEF.md seccion 18 y CLAUDE.md).
    return {"date": target_date.isoformat(), "events": [], "source": "mock"}
