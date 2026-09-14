from datetime import date

import pandas as pd
import requests
import streamlit as st

from config import BACKEND_URL
from theme import apply_theme

apply_theme("Reservas", "📅")
st.title("Reservas")

st.caption(
    "Entrada manual: no hay integracion con un sistema de reservas externo "
    "(ver PROJECT_BRIEF.md seccion 5.1). Cancelar marca la reserva como "
    "'cancelada' -- nunca se borra el registro (ver CLAUDE.md)."
)

target_date = st.date_input("Fecha", value=date(2026, 6, 16))

st.subheader("Anadir reserva")
with st.form("nueva_reserva", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    hora = c1.text_input("Hora (HH:MM)", value="21:00")
    comensales = c2.number_input("Comensales", min_value=1, value=2, step=1)
    nombre = c3.text_input("Nombre / referencia")
    notas = st.text_input("Notas (opcional)")
    submitted = st.form_submit_button("Guardar reserva")

if submitted:
    resp = requests.post(
        f"{BACKEND_URL}/reservations",
        json={
            "fecha": target_date.isoformat(), "hora": hora, "comensales": int(comensales),
            "nombre": nombre or None, "notas": notas or None,
        },
        timeout=5,
    )
    if resp.ok:
        st.success("Reserva guardada.")
    else:
        st.error(f"No se pudo guardar: {resp.text}")

st.divider()
st.subheader(f"Reservas del {target_date.isoformat()}")

resp = requests.get(f"{BACKEND_URL}/reservations/{target_date.isoformat()}", timeout=5)
if not resp.ok:
    st.error(f"No se puede conectar al backend en {BACKEND_URL}.")
    st.stop()

data = resp.json()
reservas = data["reservas"]

st.metric("Total comensales confirmados/pendientes", data["total_comensales"])

if not reservas:
    st.write("No hay reservas registradas para este dia.")
else:
    for r in reservas:
        cols = st.columns([1, 1, 2, 2, 2, 1])
        cols[0].write(r["hora"] or "-")
        cols[1].write(f"{r['comensales']} pax")
        cols[2].write(r["nombre"] or "-")
        cols[3].write(r["estado"])
        nuevo_estado = cols[4].selectbox(
            "Cambiar estado", ["pendiente", "confirmada", "no_show", "completada", "cancelada"],
            index=["pendiente", "confirmada", "no_show", "completada", "cancelada"].index(r["estado"]),
            key=f"estado_{r['id']}", label_visibility="collapsed",
        )
        if cols[5].button("Aplicar", key=f"aplicar_{r['id']}"):
            update_resp = requests.put(
                f"{BACKEND_URL}/reservations/{r['id']}", json={"estado": nuevo_estado}, timeout=5
            )
            if update_resp.ok:
                st.rerun()
            else:
                st.error(f"No se pudo actualizar: {update_resp.text}")
