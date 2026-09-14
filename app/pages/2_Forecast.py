from datetime import date

import pandas as pd
import requests
import streamlit as st

from config import BACKEND_URL
from theme import apply_theme

apply_theme("Predicción", "📊")
st.title("Predicción")

st.write(
    "Cualquier fecha ya presente en el historico se resuelve en modo "
    "**backtest** (prediccion vs. dato real). Para fechas futuras cercanas se usa "
    "el historial reciente real; para fechas lejanas (p.ej. dentro de varios meses) "
    "se usa un modelo entrenado sin esa memoria. La meteorologia, si no se aporta, "
    "se estima por la media de dias similares del año en el histórico "
    "real — fiabilidad baja, pero no inventada."
)

target_date = st.date_input("Fecha a predecir", value=date(2026, 6, 16))

resp = requests.get(f"{BACKEND_URL}/forecast/{target_date.isoformat()}", timeout=5)

if not resp.ok:
    st.error(resp.json().get("detail", f"Error del backend: {resp.status_code}"))
    st.stop()

data = resp.json()
revenue, ticket = data["revenue"], data["ticket"]

col1, col2 = st.columns(2)
col1.metric(
    "Facturacion prevista", f"{revenue['prediction']:.2f} €",
    delta=(f"{revenue['prediction'] - revenue['actual']:.2f} € vs real" if revenue["actual"] is not None else None),
)
col2.metric(
    "Ticket medio previsto", f"{ticket['prediction']:.2f} €",
    delta=(f"{ticket['prediction'] - ticket['actual']:.2f} € vs real" if ticket["actual"] is not None else None),
)

st.caption(f"Modo: {revenue['mode']} · Modelo: {revenue['model']} · Meteorología: {revenue['weather_source']}")
if revenue.get("aviso"):
    st.warning(revenue["aviso"])

if revenue["weather_source"] in ("climatology", "override"):
    with st.expander("Afinar con meteorología real (opcional)"):
        with st.form("weather_form"):
            c1, c2, c3, c4 = st.columns(4)
            wo = {}
            wo["temperature_max"] = c1.number_input("Temp. maxima (°C)", value=25.0)
            wo["temperature_min"] = c2.number_input("Temp. minima (°C)", value=15.0)
            wo["temperature_mean"] = c3.number_input("Temp. media (°C)", value=20.0)
            wo["precipitation_mm"] = c4.number_input("Precipitacion (mm)", value=0.0)
            c5, c6, c7 = st.columns(3)
            wo["precipitation_hours"] = c5.number_input("Horas de lluvia", value=0.0)
            wo["wind_speed_max"] = c6.number_input("Viento max (km/h)", value=10.0)
            wo["sunshine_duration_h"] = c7.number_input("Horas de sol", value=8.0)
            submitted = st.form_submit_button("Recalcular con esta meteorología")
        if submitted:
            resp2 = requests.get(f"{BACKEND_URL}/forecast/{target_date.isoformat()}", params=wo, timeout=5)
            if resp2.ok:
                data = resp2.json()
                revenue, ticket = data["revenue"], data["ticket"]
                st.success(
                    f"Recalculado — facturación: {revenue['prediction']:.2f} € · "
                    f"ticket medio: {ticket['prediction']:.2f} € (fuente meteo: {revenue['weather_source']})"
                )
            else:
                st.error(resp2.json().get("detail", "No se pudo recalcular."))

st.divider()
st.subheader("Principales factores (importancia por permutacion)")

if revenue["mode"] == "historical":
    explain_resp = requests.get(f"{BACKEND_URL}/forecast/{target_date.isoformat()}/explain", timeout=5)
    if explain_resp.ok:
        drivers = explain_resp.json()["revenue"]["drivers"]
        st.dataframe(pd.DataFrame(drivers), use_container_width=True, hide_index=True)
        st.caption(
            "impacto_mae_eur = cuanto empeora el error medio del modelo si se "
            "baraja esa variable al azar (mas alto = mas importante). "
            "level indica si el valor de este dia es alto/bajo/tipico frente al historico."
        )
    else:
        st.info("No hay explicabilidad disponible para esta fecha.")
else:
    st.info("La explicabilidad solo esta disponible para fechas ya presentes en el historico.")

st.divider()
st.subheader("Ventas por artículo (por semana)")
st.caption(
    "Modelo distinto al de facturación total: predice cuántas unidades/día se "
    "venderán de un plato concreto de la carta, a nivel de periodo semanal "
    "(no diario) — solo para los 82 artículos con venta suficiente para modelar."
)

articles_resp = requests.get(f"{BACKEND_URL}/forecast/sales/articles", timeout=5)
periods_resp = requests.get(f"{BACKEND_URL}/forecast/sales/periods", timeout=5)

if articles_resp.ok and periods_resp.ok:
    articles = articles_resp.json()["articles"]
    periods_info = periods_resp.json()

    article_options = {a["article_name"]: a["article_code"] for a in articles}
    period_options = periods_info["known_period_starts"] + [periods_info["next_period_start"]]

    c1, c2 = st.columns(2)
    article_name_sel = c1.selectbox("Artículo", sorted(article_options.keys()))
    period_sel = c2.selectbox(
        "Periodo (semana que empieza el)", period_options, index=len(period_options) - 2
    )
    article_code_sel = article_options[article_name_sel]

    sales_overrides = {}
    if period_sel == periods_info["next_period_start"]:
        st.info(
            f"Periodo futuro ({periods_info['next_period_start']} a "
            f"{periods_info['next_period_end']}): si esos días aún no están completos en "
            "data/gold, el resultado se marcará con un aviso de cobertura parcial."
        )

    col_articulo, col_top3 = st.columns(2)

    with col_articulo:
        sales_resp = requests.get(
            f"{BACKEND_URL}/forecast/sales/{article_code_sel}/{period_sel}",
            params=sales_overrides, timeout=5,
        )

        if sales_resp.status_code == 404:
            st.warning(sales_resp.json().get("detail"))
        elif sales_resp.status_code in (400, 422):
            st.error(sales_resp.json().get("detail"))
        elif sales_resp.ok:
            sd = sales_resp.json()
            colA, colB = st.columns(2)
            colA.metric(
                "Unidades/día previstas", f"{sd['prediction_units_por_dia']:.2f}",
                delta=(
                    f"{sd['prediction_units_por_dia'] - sd['actual_units_por_dia']:.2f} vs real"
                    if sd["actual_units_por_dia"] is not None else None
                ),
            )
            colB.metric("Periodo", f"{sd['period_start']} → {sd['period_end']}")
            st.caption(f"Modo: {sd['mode']} · Modelo: {sd['model']}")
            if sd.get("aviso"):
                st.warning(sd["aviso"])
            if sd.get("aviso_cobertura"):
                st.warning(sd["aviso_cobertura"])
        else:
            st.error(f"Error del backend: {sales_resp.status_code}")

    with col_top3:
        st.markdown("**Top 3 platos de la semana**")
        st.caption("Los 3 con más unidades/día previstas — orientativo para la compra en el supermercado.")
        top_resp = requests.get(
            f"{BACKEND_URL}/forecast/sales/top/{period_sel}",
            params={**sales_overrides, "top_n": 3}, timeout=5,
        )
        if top_resp.status_code == 404:
            st.warning(top_resp.json().get("detail"))
        elif top_resp.status_code in (400, 422):
            st.error(top_resp.json().get("detail"))
        elif top_resp.ok:
            td = top_resp.json()
            for i, art in enumerate(td["top_articles"], start=1):
                st.write(
                    f"**{i}. {art['article_name']}** — {art['prediction_units_por_dia']:.1f} ud/día "
                    f"(~{art['prediction_units_periodo']:.0f} ud en la semana)"
                )
            if td.get("aviso"):
                st.warning(td["aviso"])
            if td.get("aviso_cobertura"):
                st.warning(td["aviso_cobertura"])
        else:
            st.error(f"Error del backend: {top_resp.status_code}")
else:
    st.error("No se pudo cargar la lista de artículos/periodos desde el backend.")

st.divider()
st.subheader("Alerta de calidad (reseñas)")
st.caption(
    "Chequeo estadístico (no un modelo de ML) sobre el sentimiento medio mensual de las "
    "reseñas — ver notebooks/08_resenias.ipynb, Sección 4.1. Avisa si el último mes cae "
    "muy por debajo de la media histórica, como señal temprana de que algo se ha "
    "deteriorado (proveedor, personal...) antes de que se note en la facturación."
)

quality_resp = requests.get(f"{BACKEND_URL}/reviews/quality-trend", timeout=5)
if quality_resp.ok:
    qd = quality_resp.json()
    if qd.get("reason"):
        st.info(f"Sin alerta disponible: {qd['reason']}")
    else:
        if qd["is_alert"]:
            st.error(
                f"⚠ Caída de calidad detectada en {qd['last_month']}: sentimiento medio "
                f"{qd['last_value']} frente a una media histórica de {qd['historic_mean']} "
                f"(z-score {qd['z_score']})."
            )
        else:
            st.success(
                f"Sin caída de calidad relevante en {qd['last_month']}: sentimiento medio "
                f"{qd['last_value']} frente a una media histórica de {qd['historic_mean']} "
                f"(z-score {qd['z_score']})."
            )
        monthly = pd.DataFrame(qd["monthly_sentiment"]).set_index("month").tail(12)
        st.line_chart(monthly)
else:
    st.error("No se pudo cargar la alerta de calidad de reseñas.")

st.divider()
st.subheader("Riesgo de no-show de una reserva")
st.caption(
    "Modelo de clasificación a nivel de reserva individual (no de día). "
    "Aviso del propio notebook: señal débil (apenas por encima de un baseline "
    "ingenuo) — úsalo como indicio, no como certeza."
)
st.caption(
    "El formulario de Reservas de esta app todavía no recoge canal/zona/antelación, "
    "así que aquí se introducen a mano para poder consultar el modelo."
)

with st.form("noshow_form"):
    c1, c2, c3 = st.columns(3)
    ns_date = c1.date_input("Fecha de la reserva", value=date(2026, 6, 16), key="ns_date")
    ns_shift = c2.selectbox("Turno", ["Comida", "Cena"])
    ns_people = c3.number_input("Comensales", min_value=1, value=2, step=1)
    c4, c5, c6 = st.columns(3)
    ns_origin = c4.selectbox("Canal", ["software", "moduloweb", "appmovil", "terceros"])
    ns_zone = c5.selectbox("Zona", ["Sala", "Terraza Cubierta"])
    ns_antelacion = c6.number_input("Antelación (horas)", min_value=0.1, value=48.0)
    ns_grupo_grande = st.checkbox("Grupo grande")
    ns_submitted = st.form_submit_button("Calcular riesgo de no-show")

if ns_submitted:
    ns_resp = requests.post(
        f"{BACKEND_URL}/forecast/noshow",
        json={
            "reservation_date": ns_date.isoformat(), "shift": ns_shift, "people": int(ns_people),
            "origin": ns_origin, "zone": ns_zone, "antelacion_horas": ns_antelacion,
            "es_grupo_grande": ns_grupo_grande,
        },
        timeout=5,
    )
    if ns_resp.ok:
        nd = ns_resp.json()
        st.metric("Probabilidad de no-show", f"{nd['probabilidad_no_show'] * 100:.1f}%")
        st.caption(
            f"Reservas ya registradas ese día/turno: {nd['reservas_mismo_dia_turno']} "
            f"({nd['reservas_mismo_dia_turno_source']})"
        )
        st.warning(nd["aviso"])
    else:
        st.error(f"No se pudo calcular: {ns_resp.text}")
