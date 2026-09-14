from datetime import date, timedelta

import requests
import streamlit as st

from config import BACKEND_URL

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("Overview")

DEFAULT_DATE = date(2026, 6, 16)  # dia real dentro del historico (para que el KPI tenga dato de verdad)

target_date = st.date_input("Fecha de referencia", value=DEFAULT_DATE)

try:
    forecast_resp = requests.get(f"{BACKEND_URL}/forecast/{target_date.isoformat()}", timeout=5)
except requests.exceptions.RequestException:
    st.error(f"No se puede conectar al backend en {BACKEND_URL}.")
    st.stop()

if not forecast_resp.ok:
    st.error(forecast_resp.json().get("detail", f"Error del backend: {forecast_resp.status_code}"))
    st.stop()

data = forecast_resp.json()
revenue = data["revenue"]
ticket = data["ticket"]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Facturacion prevista",
    f"{revenue['prediction']:.2f} €",
    delta=(f"{revenue['prediction'] - revenue['actual']:.2f} € vs real" if revenue["actual"] is not None else None),
)
col2.metric(
    "Ticket medio previsto",
    f"{ticket['prediction']:.2f} €",
    delta=(f"{ticket['prediction'] - ticket['actual']:.2f} € vs real" if ticket["actual"] is not None else None),
)

reservations_resp = requests.get(f"{BACKEND_URL}/reservations/{target_date.isoformat()}", timeout=5)
reservas_data = reservations_resp.json() if reservations_resp.ok else {"total_reservas": 0, "total_comensales": 0}
col3.metric("Reservas registradas en la app", reservas_data["total_reservas"],
            help=f"{reservas_data['total_comensales']} comensales")

if revenue["mode"] == "historical":
    st.caption(
        f"Modo backtest: {target_date} ya esta en el historico. Se compara la "
        "prediccion del modelo con el dato real de ese dia."
    )
else:
    st.caption(f"Modo: {revenue['mode']} · Meteorología: {revenue['weather_source']}")
if revenue.get("aviso"):
    st.warning(revenue["aviso"])

st.divider()
st.subheader("Facturacion real vs. prevista (ultimos dias del historico)")

sample_dates = [DEFAULT_DATE + timedelta(days=i) for i in range(-5, 6)]
rows = []
for d in sample_dates:
    r = requests.get(f"{BACKEND_URL}/forecast/{d.isoformat()}", timeout=5)
    if r.ok:
        rev = r.json()["revenue"]
        rows.append({"fecha": d.isoformat(), "real": rev["actual"], "prevista": rev["prediction"]})

if rows:
    import pandas as pd
    chart_df = pd.DataFrame(rows).set_index("fecha")
    st.line_chart(chart_df)
else:
    st.write("No hay dias abiertos del restaurante en ese rango para graficar.")
