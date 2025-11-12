import streamlit as st
from typing import Tuple
from core.api import check_api_status, DEFAULT_API_URL
from core.config import AppConfig, Theme

def render_sidebar() -> AppConfig:
    st.sidebar.markdown("## ⚙️ Configuración")

    api_url = st.sidebar.text_input(
        "🔗 URL del Backend",
        value=DEFAULT_API_URL,
        help="URL del endpoint del chatbot (/chat)"
    )

    if check_api_status(api_url):
        st.sidebar.markdown('<div class="status-indicator status-online">✅ API Online</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-indicator status-error">❌ API Offline</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔍 Búsqueda")
    top_k = st.sidebar.slider("Documentos a recuperar", 1, 10, 4)
    enable_web_search = st.sidebar.checkbox("🌐 Habilitar búsqueda web", value=False)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🎨 Interfaz")
    theme = st.sidebar.selectbox("Tema", options=[Theme.LIGHT, Theme.DARK],
                                 format_func=lambda x: "☀️ Claro" if x == Theme.LIGHT else "🌙 Oscuro")
    language = st.sidebar.selectbox("Idioma de respuesta", options=["es", "en"],
                                    format_func=lambda x: "🇪🇸 Español" if x == "es" else "🇺🇸 English")
    show_timestamps = st.sidebar.checkbox("⏰ Mostrar timestamps", value=True)
    auto_scroll = st.sidebar.checkbox("📜 Auto-scroll", value=True)

    st.sidebar.markdown("---")

    with st.sidebar.expander("🔧 Opciones Avanzadas"):
        debug = st.checkbox("Modo Debug", value=False)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📈 Estadísticas de Sesión")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total consultas", st.session_state.get("total_queries", 0))
    with col2:
        from datetime import datetime
        session_start = st.session_state.get("session_start", datetime.now())
        duration = datetime.now() - session_start
        st.metric("Duración", f"{duration.seconds // 60}m")

    if st.sidebar.button("🗑️ Limpiar Conversación", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.last_response = None
        st.session_state.total_queries = 0
        st.rerun()

    return AppConfig(
        api_url=api_url.rstrip("/"),
        top_k=top_k,
        enable_web_search=enable_web_search,
        debug=debug,
        language=language,
        theme=theme,
        auto_scroll=auto_scroll,
        show_timestamps=show_timestamps
    )
