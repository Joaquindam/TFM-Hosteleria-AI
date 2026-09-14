import requests
import streamlit as st

from config import BACKEND_URL
from theme import LOGO_PATH, apply_theme

apply_theme("Restaurant Intelligence")

st.image(str(LOGO_PATH), width=280)
st.title("Restaurant Intelligence")
st.write(
    "Prototipo academico de aplicación sobre el pipeline de datos y los modelos ya "
    "entrenados del restaurante. Usa el menu de la izquierda para navegar "
    "entre Overview, Forecast y Reservas. A predecir!!!!!!! :)"
)

try:
    resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if resp.ok:
        st.success(f"Backend conectado en {BACKEND_URL}")
    else:
        st.error(f"Backend respondio con estado {resp.status_code}")
except requests.exceptions.RequestException:
    st.error(
        f"No se puede conectar al backend en {BACKEND_URL}. "
        "Arrancalo con: uvicorn backend.main:app --reload"
    )

st.info(
    "Los datos de meteorologia y eventos son **DEMO DATA** (mock) en esta fase: "
    "todavia no hay integracion con una API real. Las previsiones de facturacion "
    "y ticket medio usan los modelos ya entrenados en el TFM (no reentrenados aqui)."
)
