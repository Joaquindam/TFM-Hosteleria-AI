import requests
import streamlit as st

from config import BACKEND_URL
from theme import apply_theme

apply_theme("AI Advisor", "💬")
st.title("AI Advisor")

st.caption(
    "Chat con el asesor de gestión del restaurante: combina previsiones reales (modelos ML), "
    "datos operativos (reservas, meteo, eventos) y conocimiento documental (carta, políticas, "
    "reseñas). Nunca inventa cifras — cada respuesta cita de dónde salió el dato."
)

if "advisor_messages" not in st.session_state:
    st.session_state.advisor_messages = []

for msg in st.session_state.advisor_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            st.caption("Fuentes: " + ", ".join(msg["sources"]))

prompt = st.chat_input("Pregunta algo, p.ej. \"¿cómo estaremos este sábado?\"")

if prompt:
    st.session_state.advisor_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.advisor_messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/llm_app/chat",
                    json={"message": prompt, "history": history},
                    timeout=60,
                )
            except requests.exceptions.RequestException:
                st.error(f"No se puede conectar al backend en {BACKEND_URL}.")
                st.stop()

        if resp.status_code == 503:
            detail = resp.json().get("detail", "")
            if "GROQ_API_KEY" in detail and "Falta" in detail:
                st.warning(
                    "El AI Advisor todavía no está configurado: falta `GROQ_API_KEY` en el "
                    "archivo `.env` de la raíz del proyecto (ver `.env.example`). Añádela y "
                    "reinicia el backend para activar el chat."
                )
            else:
                st.warning(f"El AI Advisor no ha podido responder ahora mismo: {detail}")
            st.stop()
        elif not resp.ok:
            st.error(f"Error del backend: {resp.status_code}")
            st.stop()

        data = resp.json()
        st.write(data["reply"])
        if data.get("sources"):
            st.caption("Fuentes: " + ", ".join(data["sources"]))

    st.session_state.advisor_messages.append({
        "role": "assistant", "content": data["reply"], "sources": data.get("sources", []),
    })
