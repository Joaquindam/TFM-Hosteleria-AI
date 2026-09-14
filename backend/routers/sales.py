from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, HTTPException

from ml.exceptions import MissingInputError
from ml.features_sales import EXOGENOUS_OVERRIDE_FIELDS, load_sales_panel, next_period_bounds, period_for_date
from ml.sales_predictor import list_available_articles, predict_sales, top_articles_for_period

router = APIRouter(prefix="/forecast/sales", tags=["sales"])


@router.get("/articles")
def get_available_articles():
    """Los 82 articulos de carta que el modelo sabe predecir."""
    return {"articles": list_available_articles(), "source": "model"}


@router.get("/periods")
def get_available_periods():
    """Los periodos semanales conocidos (backtest) mas el siguiente
    (forecast) -- los periodos no son fechas libres, son semanas fijas del
    informe de ventas."""
    panel = load_sales_panel()
    known = sorted(panel["report_start"].dt.date.unique().tolist())
    next_start, next_end = next_period_bounds(panel)
    return {
        "known_period_starts": [d.isoformat() for d in known],
        "next_period_start": next_start.date().isoformat(),
        "next_period_end": next_end.date().isoformat(),
    }


@router.get("/top/{period_start}")
def get_top_sales_articles(
    period_start: date_type,
    top_n: int = 3,
    n_dias_festivo: Optional[int] = None,
    intensidad_evento_total: Optional[int] = None,
    temperature_mean_periodo: Optional[float] = None,
    precipitation_mm_total: Optional[float] = None,
    wind_speed_max_periodo: Optional[float] = None,
    sunshine_duration_h_total: Optional[float] = None,
    comensales_completada_periodo: Optional[int] = None,
    tasa_no_show_media: Optional[float] = None,
    tasa_cancelacion_media: Optional[float] = None,
):
    """Los `top_n` articulos con mas unidades/dia previstas para el periodo --
    pensado para orientar la compra semanal (ver ml/sales_predictor.py). Debe
    registrarse antes de \"/{article_code}/{period_start}\" para que \"top\" no
    se intente interpretar como un article_code.
    """
    overrides = {
        "n_dias_festivo": n_dias_festivo,
        "intensidad_evento_total": intensidad_evento_total,
        "temperature_mean_periodo": temperature_mean_periodo,
        "precipitation_mm_total": precipitation_mm_total,
        "wind_speed_max_periodo": wind_speed_max_periodo,
        "sunshine_duration_h_total": sunshine_duration_h_total,
        "comensales_completada_periodo": comensales_completada_periodo,
        "tasa_no_show_media": tasa_no_show_media,
        "tasa_cancelacion_media": tasa_cancelacion_media,
    }
    if all(v is None for v in overrides.values()):
        overrides = None
    elif any(overrides.get(f) is None for f in EXOGENOUS_OVERRIDE_FIELDS):
        raise HTTPException(status_code=422, detail="Si aportas variables exogenas, hay que aportarlas todas.")

    try:
        return top_articles_for_period(period_start=period_start, top_n=top_n, exogenous_overrides=overrides)
    except MissingInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{article_code}/{period_start}")
def get_sales_forecast(
    article_code: int,
    period_start: date_type,
    n_dias_festivo: Optional[int] = None,
    intensidad_evento_total: Optional[int] = None,
    temperature_mean_periodo: Optional[float] = None,
    precipitation_mm_total: Optional[float] = None,
    wind_speed_max_periodo: Optional[float] = None,
    sunshine_duration_h_total: Optional[float] = None,
    comensales_completada_periodo: Optional[int] = None,
    tasa_no_show_media: Optional[float] = None,
    tasa_cancelacion_media: Optional[float] = None,
):
    """`period_start` debe ser el lunes de un periodo semanal ya conocido,
    o el siguiente periodo tras el ultimo conocido. Los parametros
    exogenos son opcionales: solo hacen falta si ese periodo todavia no
    tiene (todos) sus dias en data/gold -- ver ml/features_sales.py.
    """
    overrides = {
        "n_dias_festivo": n_dias_festivo,
        "intensidad_evento_total": intensidad_evento_total,
        "temperature_mean_periodo": temperature_mean_periodo,
        "precipitation_mm_total": precipitation_mm_total,
        "wind_speed_max_periodo": wind_speed_max_periodo,
        "sunshine_duration_h_total": sunshine_duration_h_total,
        "comensales_completada_periodo": comensales_completada_periodo,
        "tasa_no_show_media": tasa_no_show_media,
        "tasa_cancelacion_media": tasa_cancelacion_media,
    }
    if all(v is None for v in overrides.values()):
        overrides = None
    elif any(overrides.get(f) is None for f in EXOGENOUS_OVERRIDE_FIELDS):
        raise HTTPException(status_code=422, detail="Si aportas variables exogenas, hay que aportarlas todas.")

    try:
        return predict_sales(article_code, period_start=period_start, exogenous_overrides=overrides)
    except MissingInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
