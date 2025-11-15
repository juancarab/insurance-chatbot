from typing import Any, Dict, List
from datetime import datetime
import streamlit as st
from core.config import AppConfig
from core.api import call_backend_api
from core.state import format_timestamp

def build_message_html(message: Dict[str, Any], show_timestamp: bool = True) -> str:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    ts_iso = message.get("timestamp")
    ts_txt = ""
    if show_timestamp and ts_iso:
        try:
            ts_txt = format_timestamp(datetime.fromisoformat(ts_iso))
        except Exception:
            ts_txt = ""

    wrapper_cls = "user" if role == "user" else "assistant"
    bubble_cls  = "user-message" if role == "user" else "assistant-message"
    ts_html = f"<div class='message-timestamp'>{ts_txt}</div>" if ts_txt else ""

    return (
        f"<div class='message-wrapper {wrapper_cls}'>"
        f"  <div class='message-bubble {bubble_cls}'>"
        f"    <div>{content}</div>"
        f"    {ts_html}"
        f"  </div>"
        f"</div>"
    )

def typing_indicator():
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    """, unsafe_allow_html=True)

def handle_user_query(text: str, config: AppConfig, chat_placeholder=None) -> None:
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "timestamp": datetime.now().isoformat()
    })
    st.session_state.total_queries += 1

    if chat_placeholder is not None:
        chat_html = "".join(
            build_message_html(m, config.show_timestamps)
            for m in st.session_state.messages
        )
        chat_placeholder.markdown(
            f'<div class="chat-container">{chat_html}</div>',
            unsafe_allow_html=True,
        )

    response = call_backend_api(
        api_url=config.api_url,
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        top_k=config.top_k,
        enable_web_search=config.enable_web_search,
        debug=config.debug,
        language=config.language,
    )
    st.session_state.last_response = response

    if "error" in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ {response['error']}",
            "timestamp": datetime.now().isoformat(),
        })
    else:
        answer = response.get("answer", "No response was received from the server.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
        })

    if chat_placeholder is not None:
        chat_html = "".join(
            build_message_html(m, config.show_timestamps)
            for m in st.session_state.messages
        )
        chat_placeholder.markdown(
            f'<div class="chat-container">{chat_html}</div>',
            unsafe_allow_html=True,
        )

    st.rerun()

def render_chat_area(config: AppConfig):
    chat_placeholder = st.empty()

    chat_html = "".join(
        build_message_html(m, config.show_timestamps)
        for m in st.session_state.messages
    )
    chat_placeholder.markdown(
        f'<div class="chat-container">{chat_html}</div>',
        unsafe_allow_html=True,
    )

    prompt = st.chat_input("💬 Write your insurance query...")
    if prompt:
        handle_user_query(prompt, config, chat_placeholder)

