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
                    "👋 Hola, soy el asistente de seguros. Preguntame sobre coberturas, "
                    "exclusiones o límites de tus pólizas."
                ),
            }
        ]


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Configuración")
    api_url = st.sidebar.text_input("URL del backend", value=DEFAULT_API_URL)
    top_k = st.sidebar.slider("Documentos a recuperar", min_value=1, max_value=10, value=3)
    enable_web_search = st.sidebar.checkbox("Habilitar búsqueda web (mock)")

    return {
        "api_url": api_url.rstrip("/"),
        "top_k": top_k,
        "enable_web_search": enable_web_search,
    }


def render_chat(messages: List[Dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def call_backend(
    api_url: str, payload: Dict[str, Any]
) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="Insurance Chatbot", page_icon="💬")
    st.title("Insurance Chatbot")
    st.caption("Interfaz demo con mocks de recuperador y formateador de respuestas.")

    init_session()
    config = render_sidebar()

    render_chat(st.session_state.messages)

    if prompt := st.chat_input("Escribí tu consulta sobre pólizas de seguro"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_chat(st.session_state.messages[-1:])

        payload = {
            "messages": st.session_state.messages,
            "top_k": config["top_k"],
            "enable_web_search": config["enable_web_search"],
        }

        try:
            response = call_backend(config["api_url"], payload)
        except requests.RequestException as exc:
            error_message = (
                "No se pudo contactar al backend. Verificá que el servicio esté "
                f"corriendo en {config['api_url']}.\n\nDetalles: {exc}"
            )
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.error(error_message)
            return

        answer = response.get("answer", "No se recibió respuesta del backend.")
        st.session_state.messages.append({"role": "assistant", "content": answer})

        if sources := response.get("sources", []):
            formatted_sources = "\n".join(
                f"**{idx + 1}. {source.get('title', 'Sin título')}**\n"
                f"{source.get('snippet', '')}\n"
                f"{source.get('url', '')}".strip()
                for idx, source in enumerate(sources)
            )
            source_block = (
                "### Fuentes\n"
                "Los siguientes documentos fueron utilizados como contexto:\n\n"
                f"{formatted_sources}"
            )
            st.session_state.messages.append({"role": "assistant", "content": source_block})

        render_chat(st.session_state.messages[-2:])

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            with st.expander("Detalles técnicos"):
                st.json(response)


if __name__ == "__main__":
    main()
