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
        "debug": bool(debug),
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

    prompt = st.chat_input("Escribe tu consulta sobre pólizas de seguro")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": st.session_state.messages,
            "top_k": config["top_k"],
            "enable_web_search": config["enable_web_search"],
            "debug": bool(config["debug"]),
            "language": config["language"],
        }

        try:
            response = call_backend(config["api_url"], payload)
            st.session_state.last_response = response
            answer = response.get("answer", "No se recibió respuesta del backend.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.RequestException as exc:
            error_message = (
                "No se pudo contactar al backend. Verifica que el servicio esté "
                f"corriendo en {config['api_url']}.\n\nDetalles: {exc}"
            )
            st.session_state.last_response = {"error": str(exc)}
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)

        rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
        if rerun:
            rerun()

    render_chat(st.session_state.messages)

    last_response = st.session_state.last_response or {}

    sources = last_response.get("sources", []) or []
    if sources:
        with st.expander("📚 Fuentes utilizadas", expanded=True):
            for idx, source in enumerate(sources, start=1):
                title = source.get("title") or source.get("file_name") or "Sin título"
                page = source.get("page")
                score = source.get("score")
                snippet = (source.get("snippet") or "").strip()
                url = source.get("url")

                header_parts = [f"**{idx}. {title}**"]
                meta_bits: List[str] = []
                if page is not None:
                    meta_bits.append(f"p. **{page}**")
                chunk_id = source.get("chunk_id")
                if chunk_id:
                    meta_bits.append(f"chunk **{chunk_id}**")
                if score is not None:
                    try:
                        meta_bits.append(f"score **{float(score):.3f}**")
                    except Exception:
                        meta_bits.append(f"score {score}")

                if meta_bits:
                    header_parts.append(" — " + ", ".join(meta_bits))

                st.markdown("".join(header_parts))

                if snippet:
                    st.markdown(f"> {snippet}")
                if url:
                    st.markdown(f"[Enlace]({url})")
                st.markdown("")

    if config["debug"] and last_response.get("debug"):
        with st.expander("🛠️ Detalles Técnicos (Debug)"):
            steps = last_response["debug"].get("steps")
            st.json(steps if steps is not None else last_response["debug"])


if __name__ == "__main__":
    main()
