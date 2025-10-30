"""Streamlit frontend for the Insurance Chatbot."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("INSURANCE_CHATBOT_API_URL", "http://localhost:8000/chat")


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hola, soy el asistente de seguros. Pregúntame sobre coberturas, "
                    "exclusiones o límites de tus pólizas."
                ),
            }
        ]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Configuración")
    api_url = st.sidebar.text_input("URL del backend", value=DEFAULT_API_URL)
    top_k = st.sidebar.slider("Documentos a recuperar (top_k)", min_value=1, max_value=10, value=4)
    enable_web_search = st.sidebar.checkbox("Habilitar búsqueda web", value=False)
    debug = st.sidebar.checkbox("Modo debug (mostrar pasos/chunks)", value=False)
    language = st.sidebar.selectbox("Idioma de respuesta", options=["es", "en"], index=0)

    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Limpiar chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.last_response = None
        st.sidebar.success("Chat limpiado.")

    return {
        "api_url": api_url.rstrip("/"),
        "top_k": top_k,
        "enable_web_search": enable_web_search,
        "debug": debug,
        "language": language,
    }

def render_chat(messages: List[Dict[str, str]]) -> None:
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message("assistant" if role == "system" else role):
            st.markdown(content)


def call_backend(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(api_url, json=payload, timeout=45)
    response.raise_for_status()
    return response.json()

def main() -> None:
    st.set_page_config(page_title="Insurance Chatbot", page_icon="💬", layout="wide")
    st.title("Insurance Chatbot")
    st.caption("Interfaz demo con recuperador y agente LangGraph/LLM.")

    init_session()
    config = render_sidebar()

    # Chat actual
    render_chat(st.session_state.messages)

    # Input del usuario
    if prompt := st.chat_input("Escribe tu consulta sobre pólizas de seguro"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_chat(st.session_state.messages[-1:])

        payload = {
            "messages": st.session_state.messages,
            "top_k": config["top_k"],
            "enable_web_search": config["enable_web_search"],
            "debug": config["debug"],         
            "language": config["language"],   
        }

        try:
            response = call_backend(config["api_url"], payload)
            st.session_state.last_response = response
        except requests.RequestException as exc:
            error_message = (
                "No se pudo contactar al backend. Verifica que el servicio esté "
                f"corriendo en {config['api_url']}.\n\nDetalles: {exc}"
            )
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)
            return

        # Respuesta del asistente
        answer = response.get("answer", "No se recibió respuesta del backend.")
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Fuentes (con metadatos)
        sources = response.get("sources", []) or []
        if sources:
            with st.expander("📚 Fuentes utilizadas", expanded=True):
                for i, s in enumerate(sources, start=1):
                    title = s.get("title") or s.get("file_name") or "Sin título"
                    snippet = (s.get("snippet") or "").strip()
                    url = s.get("url")
                    meta_bits = []
                    if s.get("page") is not None:
                        meta_bits.append(f"p. **{s['page']}**")
                    if s.get("chunk_id"):
                        meta_bits.append(f"chunk **{s['chunk_id']}**")
                    if s.get("score") is not None:
                        try:
                            meta_bits.append(f"score **{float(s['score']):.3f}**")
                        except Exception:
                            pass
                    meta = " — " + ", ".join(meta_bits) if meta_bits else ""

                    st.markdown(f"**{i}. {title}**{meta}")
                    if snippet:
                        st.markdown(f"> {snippet}")
                    if url:
                        st.markdown(f"[Enlace]({url})")
                    st.markdown("---")

        render_chat(st.session_state.messages[-2:])

        # Modo debug (solo si se activó y el backend devolvió 'debug')
        if config["debug"] and st.session_state.last_response and st.session_state.last_response.get("debug"):
            with st.expander("🛠️ Detalles técnicos (debug)"):
                st.json(st.session_state.last_response.get("debug"))


if __name__ == "__main__":
    main()