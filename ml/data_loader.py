"""Carga de los datos fuente usados por los notebooks 05 y 07.

Misma logica de lectura que los notebooks (mismos ficheros, mismo
tratamiento de fechas) para que las features que se calculen a partir de
aqui sean identicas a las que vieron los modelos en entrenamiento.
"""
import pandas as pd

from ml.paths import GOLD_MASTER_TABLE, RESERVAS_SILVER, FESTIVOS_SILVER, TICKETS_SILVER


def load_gold() -> pd.DataFrame:
    gold = pd.read_parquet(GOLD_MASTER_TABLE)
    gold["fecha"] = pd.to_datetime(gold["fecha"])
    gold = gold.sort_values("fecha").drop_duplicates("fecha", keep="last").reset_index(drop=True)
    return gold


def load_reservas_silver() -> pd.DataFrame:
    reservas = pd.read_parquet(RESERVAS_SILVER)
    reservas["reservation_date"] = pd.to_datetime(reservas["reservation_date"])
    reservas["created_date"] = pd.to_datetime(reservas["created_date"])
    return reservas


def load_festivos_silver() -> pd.DataFrame:
    festivos = pd.read_parquet(FESTIVOS_SILVER)
    festivos["fecha"] = pd.to_datetime(festivos["fecha"]).dt.normalize()
    return festivos


def load_tickets_silver() -> pd.DataFrame:
    tickets = pd.read_parquet(TICKETS_SILVER)
    tickets["fecha"] = pd.to_datetime(tickets["date"])
    return tickets
