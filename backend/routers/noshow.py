from fastapi import APIRouter, HTTPException

from backend.schemas import NoShowRequest
from ml.exceptions import MissingInputError
from ml.noshow_predictor import predict_noshow

router = APIRouter(prefix="/forecast/noshow", tags=["noshow"])


@router.post("")
def get_noshow_prediction(payload: NoShowRequest):
    try:
        return predict_noshow(
            reservation_date=payload.reservation_date,
            shift=payload.shift,
            people=payload.people,
            origin=payload.origin,
            es_grupo_grande=payload.es_grupo_grande,
            zone=payload.zone,
            antelacion_horas=payload.antelacion_horas,
            reservas_mismo_dia_turno=payload.reservas_mismo_dia_turno,
        )
    except MissingInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
