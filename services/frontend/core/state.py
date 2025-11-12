from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import streamlit as st

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = [{
            "role": "assistant",
            "content": "👋 ¡Hola! Soy tu asistente especializado en seguros. ¿En qué puedo ayudarte hoy?",
            "timestamp": datetime.now().isoformat()
        }]
    if "last_response" not in st.session_state:
        st.session_state.last_response: Optional[Dict[str, Any]] = None
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()

def format_timestamp(ts: Optional[datetime] = None) -> str:
    """
    ES:
    - 'ahora' si < 60s
    - 'hace N min' si < 60 min
    - 'HH:MM · hoy' si es hoy
    - 'HH:MM · ayer' si fue ayer
    - 'dd MMM · HH:MM' en otros casos
    """
    now = datetime.now()
    dt = ts or now
    delta = now - dt

    if delta.total_seconds() < 60:
        return "ahora"
    if delta < timedelta(hours=1):
        mins = int(delta.total_seconds() // 60)
        return f"hace {mins} min"

    if dt.date() == now.date():
        return dt.strftime("%H:%M · hoy")
    if dt.date() == (now - timedelta(days=1)).date():
        return dt.strftime("%H:%M · ayer")
    return dt.strftime("%d %b · %H:%M")