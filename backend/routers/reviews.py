from fastapi import APIRouter

from ml.reviews_quality import detect_quality_drop

router = APIRouter(prefix="/reviews", tags=["reviews"])


@router.get("/quality-trend")
def get_quality_trend():
    """Alerta de caida de calidad en el sentimiento de las reseñas (ver
    notebooks/08_resenias.ipynb, Seccion 4.1). Es un chequeo estadistico
    (z-score) sobre el sentimiento medio mensual ya calculado en Silver,
    no la salida de un modelo de ML entrenado."""
    return detect_quality_drop()
