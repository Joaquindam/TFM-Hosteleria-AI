"""Adapter sobre el modelo de no-shows ya entrenado
(results/models/modelo_prediccion_noshows.joblib, notebook
06_pred_noshows.ipynb). No reentrena nada.

El artefacto guardado es un sklearn.pipeline.Pipeline completo:
    FunctionTransformer(log1p sobre antelacion_horas)
    -> StandardScaler
    -> LogisticRegression(class_weight='balanced')

Al ser un Pipeline, el modelo espera como entrada el DataFrame con las
features ya codificadas (shift/zone numerico, origin en one-hot) pero
SIN escalar y SIN la transformacion logaritmica: de eso se encarga el
propio Pipeline internamente. El adaptador reproduce la codificacion
categorica exactamente igual que en el notebook.

DIFERENCIA CLAVE con la version anterior (HistGradientBoosting): el
Pipeline (StandardScaler) NO acepta NaN. Por tanto, si la fecha pedida
no tiene datos meteorologicos (caso de fechas futuras), este adaptador
imputa con la mediana historica del meteo_diaria Silver, en lugar de
pasar None como hacia la version anterior.

Aviso importante: el formulario de reservas de la app (ver
backend/reservas_store.py) NO recoge todavia `origin`, `zone` ni
`antelacion_horas` -- son campos que el modelo necesita y que hoy solo se
pueden aportar a mano al llamar a este predictor, no derivarlos de
data/gold/reservas.db.
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from ml.exceptions import MissingInputError
from ml.paths import FESTIVOS_SILVER, METEO_DIARIA_SILVER, NOSHOW_MODEL_PATH, RESERVAS_SILVER

SHIFT_MAP = {"Comida": 0, "Cena": 1}
ZONE_MAP = {"Sala": 0, "Terraza Cubierta": 1}
ORIGIN_VALUES = ["appmovil", "moduloweb", "software", "terceros"]  # 'walk in' excluido: 0% no-show en el entrenamiento, el modelo nunca lo vio

FEATURE_ORDER = [
    "shift", "people", "es_grupo_grande", "zone", "antelacion_horas",
    "es_festivo", "temperature_mean", "precipitation_mm", "reservas_mismo_dia_turno",
    "origin_appmovil", "origin_moduloweb", "origin_software", "origin_terceros",
]

DEFAULT_ZONE = "Sala"  # moda de train, misma imputacion que en el notebook


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(NOSHOW_MODEL_PATH)


@lru_cache(maxsize=1)
def _meteo_medians() -> tuple[float, float]:
    """Mediana historica de temperature_mean y precipitation_mm en
    meteo_diaria Silver, calculada una sola vez. Se usa como fallback
    cuando la fecha pedida no tiene datos meteorologicos (fechas futuras).
    """
    meteo = pd.read_parquet(METEO_DIARIA_SILVER)
    return (
        float(meteo["temperature_mean"].median()),
        float(meteo["precipitation_mm"].median()),
    )


def _es_festivo(date: pd.Timestamp) -> int:
    festivos = pd.read_parquet(FESTIVOS_SILVER)
    festivos["fecha"] = pd.to_datetime(festivos["fecha"]).dt.normalize()
    return int((festivos["fecha"] == date.normalize()).any())


def _meteo(date: pd.Timestamp) -> tuple[float, float, str]:
    """Devuelve (temperature_mean, precipitation_mm, source).
    Nunca devuelve None: si la fecha no tiene dato real, usa la mediana
    historica y marca la fuente como 'median_fallback'.
    """
    meteo = pd.read_parquet(METEO_DIARIA_SILVER)
    row = meteo.loc[pd.to_datetime(meteo["date"]) == date.normalize()]
    if not row.empty:
        return (
            float(row["temperature_mean"].iloc[0]),
            float(row["precipitation_mm"].iloc[0]),
            "historical_data",
        )
    # Fallback: mediana historica. El Pipeline (StandardScaler) no acepta
    # NaN, a diferencia del HistGradientBoosting anterior.
    med_temp, med_prec = _meteo_medians()
    return med_temp, med_prec, "median_fallback"


def _reservas_mismo_dia_turno(date: pd.Timestamp, shift: str) -> int:
    """Cuenta reservas historicas (completada/no_show) del mismo dia+turno.
    Real si `date` esta en el historico de reservas_silver; 0 en caso
    contrario (no es mock, es la cuenta real de lo que hay -- que puede
    ser legitimamente 0 para una fecha sin reservas todavia)."""
    reservas = pd.read_parquet(RESERVAS_SILVER)
    universo = reservas[reservas["status_agrupado"].isin(["completada", "no_show"])]
    mask = (universo["reservation_date"].dt.normalize() == date.normalize()) & (universo["shift"] == shift)
    return int(mask.sum())


def predict_noshow(reservation_date, shift: str, people: int, origin: str,
                    es_grupo_grande: bool = False, zone: str | None = None,
                    antelacion_horas: float | None = None,
                    reservas_mismo_dia_turno: int | None = None) -> dict:
    """Probabilidad de que una reserva termine en no-show.

    `origin` debe ser uno de ORIGIN_VALUES -- el modelo nunca vio 'walk in'
    en el entrenamiento (0% no-show observado, canal excluido del universo
    de modelado), así que no se puede predecir para ese canal.
    `antelacion_horas` es obligatorio: es la variable mas importante del
    modelo, no se puede omitir ni inventar.
    """
    if shift not in SHIFT_MAP:
        raise ValueError(f"shift debe ser uno de {list(SHIFT_MAP)}, recibido: {shift!r}")
    if origin not in ORIGIN_VALUES:
        raise ValueError(
            f"origin debe ser uno de {ORIGIN_VALUES} -- el modelo no soporta 'walk in' "
            "(excluido del entrenamiento, 0% no-show observado en ese canal). Recibido: "
            f"{origin!r}"
        )
    if antelacion_horas is None:
        raise MissingInputError("antelacion_horas es obligatorio: es la variable mas importante del modelo.")

    zone = zone or DEFAULT_ZONE
    if zone not in ZONE_MAP:
        raise ValueError(f"zone debe ser uno de {list(ZONE_MAP)}, recibido: {zone!r}")

    date = pd.Timestamp(reservation_date)
    es_festivo = _es_festivo(date)
    temperature_mean, precipitation_mm, meteo_source = _meteo(date)

    if reservas_mismo_dia_turno is None:
        reservas_mismo_dia_turno = _reservas_mismo_dia_turno(date, shift)
        turno_source = "historical_data"
    else:
        turno_source = "override"

    # --- Codificacion categorica (identica al notebook) ---
    # El Pipeline espera las features ya codificadas:
    #   shift/zone como enteros 0/1
    #   origin como one-hot
    # El Pipeline se encarga internamente de:
    #   log1p(antelacion_horas) + StandardScaler + LogisticRegression
    row = {
        "shift": SHIFT_MAP[shift],
        "people": people,
        "es_grupo_grande": int(bool(es_grupo_grande)),
        "zone": ZONE_MAP[zone],
        "antelacion_horas": antelacion_horas,
        "es_festivo": es_festivo,
        "temperature_mean": temperature_mean,
        "precipitation_mm": precipitation_mm,
        "reservas_mismo_dia_turno": reservas_mismo_dia_turno,
        "origin_appmovil": 0, "origin_moduloweb": 0, "origin_software": 0, "origin_terceros": 0,
    }
    row[f"origin_{origin}"] = 1

    X = pd.DataFrame([row])[FEATURE_ORDER]
    model = _load_model()
    probabilidad = float(model.predict_proba(X)[0, 1])

    avisos = [
        "Senal debil: PR-AUC ~0.05, modestamente por encima de un baseline "
        "ingenuo -- usar como indicio, no como certeza."
    ]
    if meteo_source == "median_fallback":
        avisos.append(
            "Meteorologia estimada con la mediana historica (la fecha no tiene "
            "dato real en el historico): impacto minimo, la meteo casi no pesa "
            "en este modelo."
        )

    return {
        "reservation_date": date.date().isoformat(),
        "shift": shift,
        "probabilidad_no_show": round(probabilidad, 4),
        "reservas_mismo_dia_turno": reservas_mismo_dia_turno,
        "reservas_mismo_dia_turno_source": turno_source,
        "meteo_source": meteo_source,
        "model": "modelo_prediccion_noshows",
        "source": "model",
        "aviso": " ".join(avisos),
    }
