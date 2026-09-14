"""Adapter sobre el modelo de ventas por articulo ya entrenado
(results/models/modelo_prediccion_ventas_articulo.joblib, notebook
04_prediccion_ventas.ipynb). No reentrena nada.

A diferencia de facturacion/ticket medio, este modelo predice a nivel
**articulo x periodo semanal** (no dia): units_por_dia de un plato
concreto de la carta, para uno de los 82 articulos con venta suficiente
para modelar (ver PROJECT_BRIEF.md, notas de integracion de la Fase 3).
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from ml.features_sales import (
    PERIOD_LENGTH_DAYS,
    SALES_FEATURE_COLS,
    article_static_features,
    build_sales_feature_table,
    load_sales_panel,
    next_period_bounds,
)
from ml.paths import SALES_MODEL_PATH


@lru_cache(maxsize=1)
def _load_model():
    return joblib.load(SALES_MODEL_PATH)


def list_available_articles() -> list[dict]:
    """Los 82 articulos que el modelo sabe predecir (con su nombre)."""
    panel = load_sales_panel()
    return (
        panel[["article_code", "article_name"]]
        .drop_duplicates()
        .sort_values("article_name")
        .to_dict(orient="records")
    )


def predict_sales(article_code: int, period_start=None, exogenous_overrides: dict | None = None) -> dict:
    """Predice `units_por_dia` (ritmo de venta diario) de `article_code`
    para el periodo semanal que empieza en `period_start`.

    - Si `period_start` coincide con uno de los 40 periodos ya conocidos:
      modo "historical" (backtest, se compara con la venta real).
    - Si es el periodo siguiente al ultimo conocido: modo "forecast" --
      usa el historial reciente real de ESE articulo (`lag_1`/`ewma_span3`
      encadenados de verdad). Hace falta `exogenous_overrides` con los 9
      agregados del periodo, salvo que esos dias ya existan en data/gold.
    - Cualquier periodo mas lejano: modo "forecast_sin_historial_reciente"
      -- ya no hay una "semana anterior" real para ese articulo, asi que
      `lag_1`/`ewma_span3` se sustituyen por la media historica de ese
      articulo (ver ml/features_sales.py). Sigue haciendo falta
      `exogenous_overrides` (eso nunca se sustituye solo).
    """
    model = _load_model()
    panel = load_sales_panel()

    static = article_static_features(article_code, panel)
    last_end = panel["report_end"].max()

    if period_start is None:
        target_start = pd.Timestamp(sorted(panel["report_start"].unique())[-1])
    else:
        target_start = pd.Timestamp(period_start)

    if (panel["report_start"] == target_start).any():
        table = build_sales_feature_table()
        mode = "historical"
        used_recent_history = True
    else:
        next_start, _ = next_period_bounds(panel)
        cerca = target_start == next_start
        period_end = target_start + pd.Timedelta(days=PERIOD_LENGTH_DAYS - 1)
        table = build_sales_feature_table(
            extra_period=(target_start, period_end), exogenous_overrides=exogenous_overrides,
            blank_history=not cerca,
        )
        mode = "forecast" if cerca else "forecast_sin_historial_reciente"
        used_recent_history = cerca

    row = table.loc[
        (table["article_code"] == article_code) & (table["report_start"] == target_start),
        SALES_FEATURE_COLS,
    ]
    if row.empty:
        raise ValueError(f"No hay datos suficientes para {article_code} en el periodo {target_start.date()}.")

    actual_row = table.loc[
        (table["article_code"] == article_code) & (table["report_start"] == target_start), "units_por_dia"
    ]
    actual = actual_row.iloc[-1] if len(actual_row) and pd.notna(actual_row.iloc[-1]) else None

    prediction = float(model.predict(row)[0])
    coverage = table.attrs.get("exogenous_coverage")

    result = {
        "article_code": article_code,
        "article_name": static["article_name"],
        "period_start": target_start.date().isoformat(),
        "period_end": (target_start + pd.Timedelta(days=PERIOD_LENGTH_DAYS - 1)).date().isoformat(),
        "prediction_units_por_dia": round(prediction, 3),
        "actual_units_por_dia": round(float(actual), 3) if actual is not None else None,
        "mode": mode,
        "model": "modelo_prediccion_ventas_articulo",
        "source": "model",
        "used_recent_history": used_recent_history,
    }
    if coverage is not None and coverage["dias_cubiertos"] < coverage["dias_periodo"]:
        result["aviso_cobertura"] = (
            f"Las variables exogenas del periodo se calcularon con solo "
            f"{coverage['dias_cubiertos']} de {coverage['dias_periodo']} dias reales "
            "(el resto del periodo todavia no esta en data/gold) -- aporta "
            "`exogenous_overrides` para una previsión con el periodo completo."
        )
    if not used_recent_history:
        result["aviso"] = (
            f"Sin historial reciente real de '{static['article_name']}' para este periodo -- "
            "lag_1/ewma_span3 se sustituyeron por la media historica de este articulo. "
            "Menos fiable que una prevision del periodo inmediatamente siguiente."
        )
    return result
