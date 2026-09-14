"""Ingenieria de variables para el modelo de ventas por articulo
(notebook 04_prediccion_ventas.ipynb). Reutiliza el panel ya guardado en
`data/gold/panel_prediccion_articulos.parquet` (construido por ese
notebook) y le anade encima la memoria temporal (`lag_1`, `ewma_span3`)
exactamente como hace la Seccion 12 del notebook -- esa parte no se guardo
en el parquet, asi que hay que recalcularla aqui, no inventarla distinto.
"""
from __future__ import annotations

import pandas as pd

from ml.data_loader import load_gold
from ml.exceptions import MissingInputError
from ml.paths import SALES_PANEL_TABLE

PERIOD_LENGTH_DAYS = 7  # todas las periodos son de 7 dias salvo el primero (arranque de captura)

SALES_NUMERIC_FEATURES = [
    "n_dias_festivo", "intensidad_evento_total", "temperature_mean_periodo",
    "precipitation_mm_total", "wind_speed_max_periodo", "sunshine_duration_h_total",
    "comensales_completada_periodo", "tasa_no_show_media", "tasa_cancelacion_media",
    "sugerencias", "lag_1", "ewma_span3",
]
SALES_CATEGORICAL_FEATURE = "article_code"
SALES_FEATURE_COLS = SALES_NUMERIC_FEATURES + [SALES_CATEGORICAL_FEATURE]

# Campos que hay que aportar para un periodo que todavia no existe en el panel
# (agregados a nivel de periodo, no por dia -- ver notebook seccion 9.2).
EXOGENOUS_OVERRIDE_FIELDS = [
    "n_dias_festivo", "intensidad_evento_total", "temperature_mean_periodo",
    "precipitation_mm_total", "wind_speed_max_periodo", "sunshine_duration_h_total",
    "comensales_completada_periodo", "tasa_no_show_media", "tasa_cancelacion_media",
]


def load_sales_panel() -> pd.DataFrame:
    return pd.read_parquet(SALES_PANEL_TABLE)


def _add_temporal_memory(panel: pd.DataFrame) -> pd.DataFrame:
    """lag_1 y ewma_span3, agrupados por articulo y con shift(1) antes de
    cualquier suavizado -- identico a la Seccion 12.1 del notebook."""
    panel = panel.sort_values(["article_code", "report_start"]).copy()
    g = panel.groupby("article_code", observed=True)["units_por_dia"]
    panel["lag_1"] = g.shift(1)
    panel["ewma_span3"] = g.transform(lambda s: s.shift(1).ewm(span=3, min_periods=1).mean())
    return panel


def article_static_features(article_code: int, panel: pd.DataFrame | None = None) -> dict:
    """`sugerencias` y `article_name` son constantes por articulo (ya se
    fijaron asi en el notebook con ffill/bfill) -- se leen del historico."""
    panel = panel if panel is not None else load_sales_panel()
    rows = panel.loc[panel["article_code"] == article_code]
    if rows.empty:
        raise ValueError(f"article_code {article_code} no esta en el panel de ventas conocido.")
    return {
        "article_name": str(rows["article_name"].iloc[0]),
        "sugerencias": int(rows["sugerencias"].mode().iloc[0]),
    }


def _aggregate_period_from_gold(period_start: pd.Timestamp, period_end: pd.Timestamp) -> dict:
    """Replica la Seccion 9.2 del notebook: agrega tabla_maestra_diaria en
    el rango [period_start, period_end]. Solo se usa cuando el periodo no
    tiene overrides explicitos y SI hay dias reales de gold en ese rango
    (backtest parcial); para un periodo totalmente futuro hacen falta
    `exogenous_overrides` (ver build_sales_feature_table).

    Incluye `_dias_cubiertos`/`_dias_periodo` para que quien llame pueda
    avisar si el agregado se calculo con el periodo incompleto (nunca se
    debe presentar una media de 4 dias como si fuera la semana entera sin
    decirlo)."""
    gold = load_gold()
    dias_periodo = (period_end - period_start).days + 1
    mask = (gold["fecha"] >= period_start) & (gold["fecha"] <= period_end)
    dias = gold.loc[mask]
    if dias.empty:
        raise MissingInputError(
            f"No hay dias de data/gold entre {period_start.date()} y {period_end.date()} "
            "para agregar las variables exogenas del periodo. Aporta `exogenous_overrides`."
        )
    return {
        "n_dias_festivo": int(dias["es_festivo"].sum()),
        "intensidad_evento_total": int(dias["intensidad_evento"].sum()),
        "temperature_mean_periodo": float(dias["temperature_mean"].mean()),
        "precipitation_mm_total": float(dias["precipitation_mm"].sum()),
        "wind_speed_max_periodo": float(dias["wind_speed_max"].max()),
        "sunshine_duration_h_total": float(dias["sunshine_duration_h"].sum()),
        "comensales_completada_periodo": int(dias["comensales_completada"].sum()),
        "tasa_no_show_media": float(dias["tasa_no_show"].mean()),
        "tasa_cancelacion_media": float(dias["tasa_cancelacion"].mean()),
        "_dias_cubiertos": len(dias),
        "_dias_periodo": dias_periodo,
    }


def next_period_bounds(panel: pd.DataFrame | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    panel = panel if panel is not None else load_sales_panel()
    last_end = panel["report_end"].max()
    start = last_end + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=PERIOD_LENGTH_DAYS - 1)
    return start, end


def period_for_date(date, panel: pd.DataFrame | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A que periodo semanal pertenece `date` -- para que la UI solo pida
    una fecha (no un periodo aparte) y esta funcion decida sola la semana.

    - Si `date` cae dentro de un periodo ya conocido, se devuelve ese
      periodo (backtest).
    - Si `date` es posterior al ultimo periodo conocido, se cuentan
      bloques de PERIOD_LENGTH_DAYS hacia delante desde ahi hasta
      encontrar el que contiene `date` (mismo criterio que usa el negocio
      para cerrar sus semanas de venta).
    """
    panel = panel if panel is not None else load_sales_panel()
    date = pd.Timestamp(date).normalize()

    periodos = panel[["report_start", "report_end"]].drop_duplicates().sort_values("report_start")
    match = periodos[(periodos["report_start"] <= date) & (date <= periodos["report_end"])]
    if not match.empty:
        row = match.iloc[0]
        return row["report_start"], row["report_end"]

    last_end = periodos["report_end"].max()
    cursor_start = last_end + pd.Timedelta(days=1)
    while date > cursor_start + pd.Timedelta(days=PERIOD_LENGTH_DAYS - 1):
        cursor_start += pd.Timedelta(days=PERIOD_LENGTH_DAYS)
    return cursor_start, cursor_start + pd.Timedelta(days=PERIOD_LENGTH_DAYS - 1)


def build_sales_feature_table(extra_period: tuple | None = None,
                               exogenous_overrides: dict | None = None,
                               blank_history: bool = False) -> pd.DataFrame:
    """Panel completo (82 articulos x periodos) con `lag_1`/`ewma_span3` ya
    calculados. Si `extra_period=(period_start, period_end)`, anade una fila
    por articulo para ese periodo nuevo -- usando `exogenous_overrides` si
    se aportan, o agregando data/gold directamente si el periodo ya tiene
    dias reales (backtest parcial).

    `blank_history=True`: para un periodo lejano (mas de uno por delante
    del ultimo conocido) no existe un "la semana pasada" real y encadenado
    -- en vez de heredar por error el ultimo valor real como si fuera la
    semana inmediatamente anterior, `lag_1`/`ewma_span3` se sustituyen por
    la media historica de ESE articulo en concreto (no una media global,
    que mezclaria platos con volumenes muy distintos entre si). El modelo
    Ridge no tolera NaN (a diferencia del de facturacion), asi que hace
    falta un valor explicito, no un hueco -- la prediccion deja de
    reflejar la tendencia reciente del articulo."""
    panel = load_sales_panel()
    coverage = None

    if extra_period is not None:
        period_start, period_end = extra_period
        if exogenous_overrides is not None:
            missing = [f for f in EXOGENOUS_OVERRIDE_FIELDS if exogenous_overrides.get(f) is None]
            if missing:
                raise MissingInputError(f"Faltan variables exogenas del periodo: {missing}")
            exog = exogenous_overrides
        else:
            exog = _aggregate_period_from_gold(period_start, period_end)
            coverage = {"dias_cubiertos": exog.pop("_dias_cubiertos"), "dias_periodo": exog.pop("_dias_periodo")}

        articulos = panel[["article_code", "article_name", "department_name_final",
                            "department_code_final", "sugerencias"]].drop_duplicates("article_code")
        nuevas_filas = articulos.copy()
        nuevas_filas["report_start"] = period_start
        nuevas_filas["report_end"] = period_end
        nuevas_filas["duracion_periodo"] = (period_end - period_start).days + 1
        nuevas_filas["units"] = None
        nuevas_filas["amount"] = None
        nuevas_filas["precio_unitario"] = None
        nuevas_filas["units_por_dia"] = None
        for k, v in exog.items():
            nuevas_filas[k] = v

        panel = pd.concat([panel, nuevas_filas], ignore_index=True)

    result = _add_temporal_memory(panel)

    if extra_period is not None and blank_history:
        medias_por_articulo = panel.groupby("article_code", observed=True)["units_por_dia"].mean()
        es_fila_nueva = result["report_start"] == extra_period[0]
        for idx in result.index[es_fila_nueva]:
            tipico = medias_por_articulo.get(result.at[idx, "article_code"])
            result.at[idx, "lag_1"] = tipico
            result.at[idx, "ewma_span3"] = tipico

    result.attrs["exogenous_coverage"] = coverage
    return result
