"""Estilo visual inspirado en https://www.larocamadrid.es/ (colores y logo
reales del restaurante). Los colores del tema en si viven en
.streamlit/config.toml (asi Streamlit los aplica a botones, inputs, etc. de
forma nativa); este modulo solo añade el logo/favicon reales y un par de
detalles que config.toml no cubre.
"""
from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"

PRIMARY_ACCENT = "#918772"  # oliva/taupe -- acento real de larocamadrid.es
DARK_TEXT = "#262220"


def apply_theme(page_title: str, page_icon: str | None = None) -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else (page_icon or "🍽️"),
        layout="wide",
    )
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH), size="large")

    st.markdown(
        f"""
        <style>
        h1, h2, h3 {{ color: {DARK_TEXT}; }}
        [data-testid="stMetricValue"] {{ color: {PRIMARY_ACCENT}; }}
        hr {{ border-color: {PRIMARY_ACCENT}33; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
