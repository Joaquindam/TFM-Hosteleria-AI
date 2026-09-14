"""Explicabilidad de las predicciones, basada en la importancia por
permutacion ya calculada en el entrenamiento
(results/forecasting_final|ticket_medio_final/importancia_permutacion.csv).
No se recalcula SHAP en caliente -- ver PROJECT_BRIEF.md seccion 6.

Devuelve datos estructurados (variable, impacto, valor de este dia,
si es alto/bajo/tipico frente al historico). Traducirlo a lenguaje de
gerente es responsabilidad del Agent (Fase 4), no de esta capa.
"""
from __future__ import annotations

import pandas as pd

from ml.paths import FORECASTING_RESULTS_DIR, TICKET_RESULTS_DIR
from ml.features import build_revenue_feature_table, build_ticket_feature_table, REVENUE_FEATURE_COLS, TICKET_FEATURE_COLS

TOP_N_DRIVERS = 8


def _to_native(value):
    """Convierte escalares numpy (int64, float64, bool_...) a tipos nativos
    de Python -- FastAPI/pydantic no serializan numpy scalars directamente."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _describe_value(historical: pd.Series, value) -> dict:
    historical = historical.dropna()
    is_numeric = pd.api.types.is_numeric_dtype(historical)

    if is_numeric and value is not None and not isinstance(value, str) and not pd.isna(value):
        percentile = float((historical < value).mean() * 100)
        if percentile >= 75:
            level = "alto"
        elif percentile <= 25:
            level = "bajo"
        else:
            level = "tipico"
        return {"percentile": round(percentile, 1), "level": level}

    frecuencia = float((historical.astype(str) == str(value)).mean() * 100)
    return {"frequency_pct": round(frecuencia, 1), "level": None}


def _explain(date, table: pd.DataFrame, feature_cols: list[str], importance_path) -> dict:
    target = pd.Timestamp(date).normalize()
    row_df = table.loc[table["fecha"] == target, feature_cols]
    if row_df.empty:
        raise ValueError(f"No hay fila de features para {target.date()}. Solo se puede explicar una fecha ya presente en el historico (ver predict_revenue/predict_ticket para fechas futuras).")
    row = row_df.iloc[0]

    importancia = pd.read_csv(importance_path)
    top = importancia.sort_values("importancia_permutacion", ascending=False).head(TOP_N_DRIVERS)

    drivers = []
    for _, r in top.iterrows():
        var = r["variable"]
        if var not in table.columns:
            continue
        value = row.get(var)
        desc = _describe_value(table[var], value)
        drivers.append({
            "variable": var,
            "impacto_mae_eur": round(float(r["importancia_permutacion"]), 2),
            "valor_este_dia": None if pd.isna(value) else _to_native(value),
            **desc,
        })

    return {"date": target.date().isoformat(), "drivers": drivers}


def explain_revenue(date) -> dict:
    table = build_revenue_feature_table()
    return _explain(date, table, REVENUE_FEATURE_COLS, FORECASTING_RESULTS_DIR / "importancia_permutacion.csv")


def explain_ticket(date) -> dict:
    table = build_ticket_feature_table()
    return _explain(date, table, TICKET_FEATURE_COLS, TICKET_RESULTS_DIR / "importancia_permutacion.csv")
