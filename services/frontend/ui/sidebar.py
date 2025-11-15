import streamlit as st
from typing import Tuple
from core.api import check_api_status, DEFAULT_API_URL
from core.config import AppConfig, Theme

def render_sidebar() -> AppConfig:
    st.sidebar.markdown("## ⚙️ Configuración")

    api_url = st.sidebar.text_input(
        "🔗 Backend URL",
        value=DEFAULT_API_URL,
        help="Chatbot endpoint URL (/chat)"
    )

    if check_api_status(api_url):
        st.sidebar.markdown('<div class="status-indicator status-online">✅ API Online</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-indicator status-error">❌ API Offline</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🔍 Search")
    top_k = st.sidebar.slider("Documents to retrieve", 1, 10, 4)
    enable_web_search = st.sidebar.checkbox("🌐 Enable web search", value=False)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 🎨 Interface")
    theme = st.sidebar.selectbox(
        "Tema",
        options=[Theme.DARK, Theme.LIGHT],
        format_func=lambda x: "☀️ Light" if x == Theme.LIGHT else "🌙 Dark",
    )
    language = st.sidebar.selectbox("Answer Language", options=["es", "en"],
                                    format_func=lambda x: "🇪🇸 Español" if x == "es" else "🇺🇸 English")
    show_timestamps = st.sidebar.checkbox("⏰ Show timestamps", value=True)
    auto_scroll = st.sidebar.checkbox("📜 Auto-scroll", value=True)

    st.sidebar.markdown("---")

    with st.sidebar.expander("🔧 Advanced settings"):
        debug = st.checkbox("Debug Mode", value=False)

    st.sidebar.markdown("---")

    st.sidebar.markdown("### 📈 Session Statistics")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total queries", st.session_state.get("total_queries", 0))
    with col2:
        from datetime import datetime
        session_start = st.session_state.get("session_start", datetime.now())
        duration = datetime.now() - session_start
        st.metric("Duration", f"{duration.seconds // 60}m")

    if st.sidebar.button("🗑️ Clean conversation", use_container_width=True):
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
