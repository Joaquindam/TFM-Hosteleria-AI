"""Ingenieria de variables identica a la de los notebooks 05 y 07.

Cualquier cambio aqui debe mantenerse coherente con esos notebooks: son la
fuente de verdad de como se entrenaron los modelos .joblib guardados en
results/models/. Ver PROJECT_BRIEF.md secciones 4 y 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data_loader import load_gold, load_reservas_silver, load_festivos_silver, load_tickets_silver
from ml.exceptions import MissingInputError

HORIZON = 1  # dias de antelacion (predecir el siguiente dia-servicio), igual que en los notebooks

REVENUE_FEATURE_COLS = [
    "dia_semana", "num_dia", "mes", "es_festivo", "festivo_nombre",
    "tiene_evento", "intensidad_evento", "impacto_evento", "direccion_evento",
    "categoria_evento", "cat_evento",
    "temperature_max", "temperature_min", "temperature_mean",
    "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
    "es_fin_de_semana", "es_lunes",
    "reservas_anticipadas", "comensales_anticipados", "grupos_grandes_anticipados", "antelacion_media_dias",
    "anio", "semana_anio", "trimestre", "dia_anio",
    "sin_dia_anio", "cos_dia_anio", "sin_dia_semana", "cos_dia_semana",
    "festivo_manana", "festivo_ayer", "es_puente",
    "rango_temperatura", "es_dia_lluvioso", "es_dia_muy_caluroso", "es_dia_muy_frio",
    "facturacion_1d_antes", "facturacion_3d_antes", "facturacion_7d_antes",
    "facturacion_14d_antes", "facturacion_21d_antes", "facturacion_28d_antes",
    "facturacion_media_3d", "facturacion_media_7d", "facturacion_std_7d", "facturacion_total_7d",
    "facturacion_media_14d", "facturacion_std_14d", "facturacion_total_14d",
    "facturacion_media_28d", "facturacion_std_28d", "facturacion_total_28d",
    "facturacion_media_90d", "facturacion_tendencia_7_28",
]

TICKET_FEATURE_COLS = [
    "es_festivo", "festivo_nombre", "tiene_evento", "intensidad_evento", "impacto_evento",
    "direccion_evento", "categoria_evento", "cat_evento",
    "temperature_max", "temperature_min", "temperature_mean",
    "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
    "reservas_anticipadas", "comensales_anticipados", "grupos_grandes_anticipados", "antelacion_media_dias",
    "dia_semana", "num_dia", "mes", "anio", "semana_anio", "trimestre", "dia_anio",
    "sin_dia_anio", "cos_dia_anio", "sin_dia_semana", "cos_dia_semana",
    "es_fin_de_semana", "es_lunes", "festivo_manana", "festivo_ayer", "es_puente",
    "rango_temperatura", "es_dia_lluvioso", "es_dia_muy_caluroso", "es_dia_muy_frio",
    "ticket_medio_1d_antes", "ticket_medio_3d_antes", "ticket_medio_7d_antes",
    "ticket_medio_14d_antes", "ticket_medio_21d_antes", "ticket_medio_28d_antes",
    "ticket_medio_media_3d", "ticket_medio_media_7d", "ticket_medio_media_14d", "ticket_medio_media_28d",
    "ticket_medio_std_7d", "ticket_medio_std_28d", "ticket_medio_tendencia_7_28",
    "facturacion_1d_antes", "facturacion_7d_antes", "facturacion_14d_antes", "facturacion_media_7d",
    "num_tickets_1d_antes", "num_tickets_7d_antes", "num_tickets_14d_antes", "num_tickets_media_7d",
    "ticket_mediano_1d_antes",
]

REVENUE_FEATURE_COLS_LARGO_PLAZO = [c for c in REVENUE_FEATURE_COLS if not c.startswith("facturacion_")]
TICKET_FEATURE_COLS_LARGO_PLAZO = [
    c for c in TICKET_FEATURE_COLS
    if not c.startswith(("ticket_medio_", "facturacion_", "num_tickets_")) and c != "ticket_mediano_1d_antes"
]

WEATHER_FIELDS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
]

EVENT_DEFAULTS = {
    "tiene_evento": 0, "intensidad_evento": 0,
    "impacto_evento": "sin_evento", "direccion_evento": "sin_evento",
    "categoria_evento": "sin_evento", "cat_evento": "Sin evento",
}


def _reservas_anticipadas(reservas: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    validas = reservas.dropna(subset=["reservation_date", "created_date"]).copy()
    validas["antelacion_dias"] = (validas["reservation_date"] - validas["created_date"]).dt.days
    conocidas = validas[validas["antelacion_dias"] >= horizon]
    return (
        conocidas.groupby("reservation_date")
        .agg(
            reservas_anticipadas=("reference_code", "count"),
            comensales_anticipados=("people", "sum"),
            grupos_grandes_anticipados=("es_grupo_grande", "sum"),
            antelacion_media_dias=("antelacion_dias", "mean"),
        )
        .reset_index()
        .rename(columns={"reservation_date": "fecha"})
    )


def _merge_reservas_anticipadas(df: pd.DataFrame, reservas: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    agg = _reservas_anticipadas(reservas, horizon)
    df = df.merge(agg, on="fecha", how="left")
    for c in ["reservas_anticipadas", "comensales_anticipados", "grupos_grandes_anticipados"]:
        df[c] = df[c].fillna(0)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["fecha"]
    df["anio"] = d.dt.year
    df["semana_anio"] = d.dt.isocalendar().week.astype(int)
    df["trimestre"] = d.dt.quarter
    df["dia_anio"] = d.dt.dayofyear
    df["sin_dia_anio"] = np.sin(2 * np.pi * df["dia_anio"] / 365.25)
    df["cos_dia_anio"] = np.cos(2 * np.pi * df["dia_anio"] / 365.25)
    df["sin_dia_semana"] = np.sin(2 * np.pi * df["num_dia"] / 7)
    df["cos_dia_semana"] = np.cos(2 * np.pi * df["num_dia"] / 7)
    return df


def _add_festivo_adjacency(df: pd.DataFrame, festivos: pd.DataFrame) -> pd.DataFrame:
    fechas_festivas = set(festivos["fecha"])
    d = df["fecha"]
    df["festivo_manana"] = (d + pd.Timedelta(days=1)).dt.normalize().isin(fechas_festivas).astype(int)
    df["festivo_ayer"] = (d - pd.Timedelta(days=1)).dt.normalize().isin(fechas_festivas).astype(int)
    df["es_puente"] = (
        ((d.dt.weekday == 0) & (df["festivo_manana"] == 1))
        | ((d.dt.weekday == 4) & (df["festivo_ayer"] == 1))
    ).astype(int)
    return df


def _add_weather_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["rango_temperatura"] = df["temperature_max"] - df["temperature_min"]
    df["es_dia_lluvioso"] = (df["precipitation_mm"].fillna(0) > 0).astype(int)
    df["es_dia_muy_caluroso"] = (df["temperature_max"] >= 30).astype(int)
    df["es_dia_muy_frio"] = (df["temperature_min"] <= 5).astype(int)
    return df


def _resolve_weather(date: pd.Timestamp, weather_overrides: dict | None) -> tuple[dict, str]:
    """Decide de donde sale la meteorologia de un dia futuro, en orden:
    1. `weather_overrides` explicito (las 7 variables) -> se usa tal cual
       -- source "override". Si se aporta incompleto, error (mezclar real
       y faltante a medias suele ser un fallo de quien llama).
    2. Prevision real de la API de Open-Meteo (ml/weather_api.py) -- solo
       funciona para los proximos dias (horizonte real de cualquier API de
       prevision) -- source "live_forecast".
    3. Si la API no tiene dato para esa fecha (demasiado lejana, sin
       internet, error) -> media climatologica de dias similares del anio
       en los ~4 anios de historico real (ver ml/climatology.py) --
       source "climatology". Baja fiabilidad pero real, nunca inventada.
    """
    if weather_overrides:
        missing = [f for f in WEATHER_FIELDS if weather_overrides.get(f) is None]
        if missing:
            raise MissingInputError(
                f"weather_overrides incompleto para {date.date()}: faltan {missing}. "
                "O se aportan las 7 variables, o ninguna (en ese caso se intenta la "
                "prevision real y, si no llega, una estimacion climatologica automatica)."
            )
        return {f: weather_overrides[f] for f in WEATHER_FIELDS}, "override"

    from ml.weather_api import fetch_live_forecast
    live = fetch_live_forecast(date)
    if live is not None:
        return live, "live_forecast"

    from ml.climatology import estimate_weather_climatology
    clima = estimate_weather_climatology(date)
    return {f: clima[f] for f in WEATHER_FIELDS}, "climatology"


def _build_future_raw_row(date: pd.Timestamp, festivos: pd.DataFrame,
                           weather_overrides: dict | None, event_overrides: dict | None) -> tuple[dict, str]:
    """Fila cruda con el mismo esquema que una fila de `gold`, para un dia
    que todavia no existe en el historico. No calcula nada que dependa de
    facturacion/tickets pasados: eso lo hace el pipeline de lags/rolling
    despues, sobre el historico real + esta fila.

    Devuelve (fila, weather_source) -- ver _resolve_weather.
    """
    weather, weather_source = _resolve_weather(date, weather_overrides)

    festivo_row = festivos.loc[festivos["fecha"] == date.normalize()]
    es_festivo = int(len(festivo_row) > 0)
    festivo_nombre = festivo_row["festivo_nombre"].iloc[0] if es_festivo else "sin_festivo"

    events = dict(EVENT_DEFAULTS)
    if event_overrides:
        events.update(event_overrides)

    row = {
        "fecha": date,
        "dia_semana": date.day_name(),
        "num_dia": date.dayofweek,
        "mes": date.month,
        "es_festivo": es_festivo,
        "festivo_nombre": festivo_nombre,
        "es_fin_de_semana": int(date.dayofweek in (5, 6)),
        "es_lunes": int(date.dayofweek == 0),
        "facturacion": np.nan,
        "num_tickets": np.nan,
        "ticket_medio": np.nan,
        "ticket_mediano": np.nan,
        **events,
        **weather,
    }
    return row, weather_source


def build_revenue_feature_table(extra_date: pd.Timestamp | None = None,
                                 weather_overrides: dict | None = None,
                                 event_overrides: dict | None = None) -> pd.DataFrame:
    """Replica la ingenieria de variables del notebook 05 sobre todo el
    historico de `gold`, opcionalmente con una fila extra al final para un
    dia que todavia no existe.

    Para fechas lejanas en el futuro (sin historial reciente real), quien
    llama debe usar `REVENUE_FEATURE_COLS_LARGO_PLAZO` en vez de
    `REVENUE_FEATURE_COLS` al leer la fila -- eso descarta las columnas de
    memoria (que aqui saldrian mal calculadas, heredando el ultimo dia real
    como si fuera "ayer") y usa en su lugar el modelo de largo plazo
    entrenado sin ellas (notebook 09, ver ml/revenue_predictor.py).

    Devuelve el DataFrame completo (con 'fecha', 'facturacion' real cuando
    se conoce, y las columnas de REVENUE_FEATURE_COLS), sin descartar filas
    por nulos -- el llamante decide que fila usar.
    """
    gold = load_gold()
    reservas = load_reservas_silver()
    festivos = load_festivos_silver()

    df = gold.copy()
    weather_source = None

    if extra_date is not None:
        row, weather_source = _build_future_raw_row(extra_date, festivos, weather_overrides, event_overrides)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)

    df = _merge_reservas_anticipadas(df, reservas, HORIZON)
    df = _add_calendar_features(df)
    df = _add_festivo_adjacency(df, festivos)
    df = _add_weather_derived(df)

    base = df["facturacion"].shift(HORIZON)
    for lag in [1, 3, 7, 14, 21, 28]:
        df[f"facturacion_{lag}d_antes"] = df["facturacion"].shift(HORIZON + lag - 1)
    for w in [3, 7, 14, 28, 90]:
        df[f"facturacion_media_{w}d"] = base.rolling(w, min_periods=max(2, w // 3)).mean()
        if w in (7, 14, 28):
            df[f"facturacion_std_{w}d"] = base.rolling(w, min_periods=max(2, w // 3)).std()
            df[f"facturacion_total_{w}d"] = base.rolling(w, min_periods=max(2, w // 3)).sum()
    df["facturacion_tendencia_7_28"] = df["facturacion_media_7d"] - df["facturacion_media_28d"]

    df.attrs["weather_source"] = weather_source
    return df


def build_ticket_feature_table(extra_date: pd.Timestamp | None = None,
                                weather_overrides: dict | None = None,
                                event_overrides: dict | None = None) -> pd.DataFrame:
    """Replica la ingenieria de variables del notebook 07. La serie base es
    la de tickets (reconstruida desde Silver), con `gold` aportando solo
    columnas de contexto -- igual que hace el notebook.

    Para fechas lejanas, usar `TICKET_FEATURE_COLS_LARGO_PLAZO` y el modelo
    de largo plazo (notebook 10) -- ver la misma nota en
    build_revenue_feature_table.
    """
    tickets = load_tickets_silver()
    gold = load_gold()
    reservas = load_reservas_silver()
    festivos = load_festivos_silver()

    ticket_diario = (
        tickets.groupby("fecha")
        .agg(
            facturacion=("document_total", "sum"),
            num_tickets=("document_id", "nunique"),
            ticket_mediano=("document_total", "median"),
            ticket_std=("document_total", "std"),
            ticket_max=("document_total", "max"),
        )
        .reset_index()
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    ticket_diario["ticket_medio"] = ticket_diario["facturacion"] / ticket_diario["num_tickets"]

    contexto_cols = [
        "fecha", "es_festivo", "festivo_nombre",
        "tiene_evento", "intensidad_evento", "impacto_evento", "direccion_evento",
        "categoria_evento", "cat_evento",
        "temperature_max", "temperature_min", "temperature_mean",
        "precipitation_mm", "precipitation_hours", "wind_speed_max", "sunshine_duration_h",
    ]
    df = ticket_diario.merge(gold[contexto_cols], on="fecha", how="left")

    weather_source = None
    if extra_date is not None:
        raw_row, weather_source = _build_future_raw_row(extra_date, festivos, weather_overrides, event_overrides)
        keep_keys = ["fecha", "facturacion", "num_tickets", "ticket_mediano"] + contexto_cols[1:]
        row = {k: v for k, v in raw_row.items() if k in keep_keys}
        row["ticket_medio"] = np.nan
        row["ticket_std"] = np.nan
        row["ticket_max"] = np.nan
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.sort_values("fecha").reset_index(drop=True)

    df = _merge_reservas_anticipadas(df, reservas, HORIZON)

    d = df["fecha"]
    df["dia_semana"] = d.dt.day_name()
    df["num_dia"] = d.dt.dayofweek
    df["mes"] = d.dt.month
    df = _add_calendar_features(df)
    df["es_fin_de_semana"] = df["num_dia"].isin([5, 6]).astype(int)
    df["es_lunes"] = (df["num_dia"] == 0).astype(int)
    df = _add_festivo_adjacency(df, festivos)
    df = _add_weather_derived(df)

    def lags_y_rolling(df, col, lags, windows, std_windows=None, prefix=None):
        prefix = prefix or col
        base = df[col].shift(HORIZON)
        for lag in lags:
            df[f"{prefix}_{lag}d_antes"] = df[col].shift(HORIZON + lag - 1)
        for w in windows:
            df[f"{prefix}_media_{w}d"] = base.rolling(w, min_periods=max(2, w // 3)).mean()
        for w in (std_windows or []):
            df[f"{prefix}_std_{w}d"] = base.rolling(w, min_periods=max(2, w // 3)).std()
        return df

    df = lags_y_rolling(df, "ticket_medio", lags=[1, 3, 7, 14, 21, 28], windows=[3, 7, 14, 28], std_windows=[7, 28])
    df["ticket_medio_tendencia_7_28"] = df["ticket_medio_media_7d"] - df["ticket_medio_media_28d"]
    df = lags_y_rolling(df, "facturacion", lags=[1, 7, 14], windows=[7])
    df = lags_y_rolling(df, "num_tickets", lags=[1, 7, 14], windows=[7])
    df["ticket_mediano_1d_antes"] = df["ticket_mediano"].shift(HORIZON)

    df.attrs["weather_source"] = weather_source
    return df
