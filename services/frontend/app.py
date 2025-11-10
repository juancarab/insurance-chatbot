"""Streamlit frontend mejorado para Insurance Chatbot con diseño moderno y componentes dinámicos."""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import requests
import streamlit as st

# ========================== CONFIGURACIÓN ==========================

DEFAULT_API_URL = os.getenv("INSURANCE_CHATBOT_API_URL", "http://localhost:8000/chat")

class Theme(Enum):
    LIGHT = "light"
    DARK = "dark"

@dataclass
class AppConfig:
    """Configuración centralizada de la aplicación."""
    api_url: str
    top_k: int
    enable_web_search: bool
    debug: bool
    language: str
    theme: Theme
    auto_scroll: bool
    show_timestamps: bool

# ========================== ESTILOS Y TEMAS ==========================

def get_theme_styles(theme: Theme) -> str:
    """Retorna los estilos CSS según el tema seleccionado."""
    
    base_styles = """
    <style>
    /* Reset y configuración base */
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Contenedor principal mejorado */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    /* Fondo general suave */
    html, body, [data-testid="stAppViewContainer"]{
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%) !important;
        color: #E2E8F0 !important;
    }

}

    /* Header mejorado */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        color: white;
        text-align: center;
    }
    
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .app-header p {
        margin-top: 0.5rem;
        opacity: 0.95;
        font-size: 1.1rem;
    }
    
    /* Chat container mejorado */
    .chat-container{
        background: linear-gradient(180deg, #FCFEFF 0%, #F3F6FA 100%);
        border:1px solid #E2E8F0;
        border-radius:20px;
        padding:1.5rem;
        box-shadow: 0 6px 28px rgba(2,6,23,.06);
        min-height: 480px;
        margin-bottom: 1rem;
        color:#0B1220;
    }
    
    /* Mensajes mejorados */
    .message-wrapper{
        display:flex;                 /* <-- nuevo: contenedor flex */
        margin-bottom: 1.2rem;
        animation: fadeInUp 0.3s ease;
    }
    /* alineación por rol */
    .message-wrapper.assistant{ justify-content:flex-start; }
    .message-wrapper.user{ justify-content:flex-end; }

    @keyframes fadeInUp{
        from{ opacity:0; transform: translateY(10px); }
        to{   opacity:1; transform: translateY(0); }
    }

    .message-bubble{
        display:inline-block;
        max-width:78%;                /* 80% estaba bien; 78% queda más fino */
        padding:1rem 1.25rem;
        border-radius:18px;
        font-size:.95rem;
        line-height:1.5;
        word-wrap:break-word;
        position:relative;
    }
    
    .user-message {
        background: white;
        color: #1a202c;
        border: 2px solid #667eea;
        margin-left: auto;
        text-align: left;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.15);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-right: auto;
        box-shadow: 0 2px 15px rgba(102, 126, 234, 0.25);
    }
    
    /* Timestamp como chip */
    .message-timestamp{
        display:inline-block;
        margin-top:.40rem;
        padding:.15rem .55rem;
        background:#EEF2FF;
        color:#475569;
        border:1px solid #E2E8F0;
        border-radius:999px;
        font-size:.76rem;
        line-height:1;
    }
    .message-wrapper.user .message-timestamp{ align-self:flex-end; margin-right:6px; }
    .message-wrapper.assistant .message-timestamp{ align-self:flex-start; margin-left:6px; }

    /* Source cards mejoradas */
    .source-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border-color: #667eea;
    }
    
    .source-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .source-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .source-title {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .source-snippet {
        color: #4a5568;
        font-size: 0.9rem;
        line-height: 1.6;
        margin: 0.75rem 0;
        padding: 0.75rem;
        background: #f7fafc;
        border-radius: 8px;
        border-left: 3px solid #667eea;
    }
    
    .source-meta {
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .source-tag {
        display: inline-block;
        background: #eef2ff;
        color: #667eea;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Metrics dashboard */
    .metrics-container {
        background: linear-gradient(135deg, #f6f9fc 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Loading animation */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        padding: 1rem;
        gap: 0.4rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: typingAnimation 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typingAnimation {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.5;
        }
        30% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Sidebar personalizado */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .status-online {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* Scrollbar personalizado */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46a0 100%);
    }
    </style>
    """
    
    if theme == Theme.DARK:
        dark_overrides = """
        <style>
        /* Dark theme overrides */
        .stApp {
            background: #0f172a;
            color: #e2e8f0;
        }
        
        .chat-container {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-color: #334155;
        }
        
        .user-message {
            background: #1e293b;
            color: #f1f5f9;
            border-color: #667eea;
        }
        
        .source-card {
            background: #1e293b;
            border-color: #334155;
            color: #e2e8f0;
        }
        
        .source-snippet {
            background: #0f172a;
            color: #cbd5e1;
        }
        
        .metrics-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-color: #334155;
        }
        
        .metric-item {
            background: #0f172a;
            border-color: #334155;
        }
        
        .metric-label {
            color: #94a3b8;
        }
        
        .metric-value {
            color: #f1f5f9;
        }
        </style>
        """
        return base_styles + dark_overrides
    
    return base_styles

# ========================== FUNCIONES DE UTILIDAD ==========================

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 ¡Hola! Soy tu asistente especializado en seguros. Puedo ayudarte con preguntas sobre coberturas, exclusiones, límites de pólizas y más. ¿En qué puedo ayudarte hoy?"
        }]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()

def check_api_status(api_url: str) -> bool:
    """Verifica el estado del backend."""
    try:
        response = requests.get(f"{api_url.replace('/chat', '')}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def format_timestamp(ts: Optional[datetime] = None) -> str:
    """
    Devuelve tiempos legibles en ES:
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


def build_message_html(message: Dict[str, Any], show_timestamp: bool = True) -> str:
    """Devuelve el HTML de una burbuja, con wrapper por rol para alinear."""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")
    wrapper_cls = "user" if role == "user" else "assistant"
    bubble_cls = "user-message" if role == "user" else "assistant-message"
    ts = f"<div class='message-timestamp'>{timestamp}</div>" if show_timestamp and timestamp else ""
    return f"<div class='message-wrapper {wrapper_cls}'><div class='message-bubble {bubble_cls}'>{content}</div>{ts}</div>"

def render_typing_indicator() -> None:
    """Renderiza indicador de escritura."""
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    """, unsafe_allow_html=True)

def build_message_html(message: Dict[str, Any], show_timestamp: bool = False) -> str:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    wrapper_cls = "user" if role == "user" else "assistant"
    bubble_cls  = "user-message" if role == "user" else "assistant-message"
    return f"<div class='message-wrapper {wrapper_cls}'><div class='message-bubble {bubble_cls}'>{content}</div></div>"
    
    if role == "user":
        col1, col2 = st.columns([4, 1])
        with col2:
            st.markdown(f"""
                <div class="message-wrapper">
                    <div class="message-bubble user-message">
                        {content}
                    </div>
                    {f'<div class="message-timestamp">{timestamp}</div>' if show_timestamp and timestamp else ''}
                </div>
            """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"""
                <div class="message-wrapper">
                    <div class="message-bubble assistant-message">
                        {content}
                    </div>
                    {f'<div class="message-timestamp">{timestamp}</div>' if show_timestamp and timestamp else ''}
                </div>
            """, unsafe_allow_html=True)

def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Renderiza las fuentes con diseño mejorado."""
    if not sources:
        st.info("📚 No se encontraron fuentes específicas para esta consulta.")
        return
    
    st.markdown("### 📚 Fuentes Consultadas")
    
    for idx, source in enumerate(sources, 1):
        title = source.get("title") or source.get("file_name") or "Documento sin título"
        snippet = (source.get("snippet") or "").strip()
        url = source.get("url")
        page = source.get("page")
        chunk_id = source.get("chunk_id")
        score = source.get("score")
        
        meta_tags = []
        if page:
            meta_tags.append(f'<span class="source-tag">📄 Página {page}</span>')
        if chunk_id:
            meta_tags.append(f'<span class="source-tag">🔖 Chunk {chunk_id}</span>')
        if score is not None:
            try:
                score_val = float(score)
                meta_tags.append(f'<span class="source-tag">⭐ {score_val:.2f}</span>')
            except:
                pass
        
        st.markdown(f"""
            <div class="source-card">
                <div class="source-title">
                    <span class="source-number">{idx}</span>
                    {title}
                </div>
                {f'<div class="source-snippet">{snippet}</div>' if snippet else ''}
                <div class="source-meta">
                    {' '.join(meta_tags)}
                    {f'<a href="{url}" target="_blank" style="color: #667eea; text-decoration: none;">🔗 Ver documento</a>' if url else ''}
                </div>
            </div>
        """, unsafe_allow_html=True)

def render_metrics(response: Dict[str, Any]) -> None:
    """Renderiza métricas del sistema."""
    usage = response.get("usage", {})
    
    st.markdown("""
        <div class="metrics-container">
            <h4 style="margin-bottom: 1rem;">📊 Métricas de la Consulta</h4>
    """, unsafe_allow_html=True)
    
    metrics = {
        "Documentos recuperados": usage.get("retrieved_documents", 0),
        "Formateador": usage.get("formatter", "N/A"),
        "Búsqueda web": "Activa" if usage.get("web_search_enabled") else "Inactiva",
        "Idioma": usage.get("language", "es").upper(),
        "Top-K": usage.get("top_k", 0)
    }
    
    for label, value in metrics.items():
        st.markdown(f"""
            <div class="metric-item">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_debug_info(debug_data: Dict[str, Any]) -> None:
    """Renderiza información de debug."""
    with st.expander("🔧 Información de Debug", expanded=False):
        st.json(debug_data)

def call_backend_api(config: AppConfig, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Llama al backend y maneja errores."""
    payload = {
        "messages": messages,
        "top_k": config.top_k,
        "enable_web_search": config.enable_web_search,
        "debug": config.debug,
        "language": config.language,
    }
    
    try:
        response = requests.post(config.api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Timeout: El servidor tardó demasiado en responder"}
    except requests.exceptions.ConnectionError:
        return {"error": "No se pudo conectar al servidor backend"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Error HTTP: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Error inesperado: {str(e)}"}

def render_sidebar() -> AppConfig:
    """Renderiza sidebar con configuración mejorada."""
    st.sidebar.markdown("## ⚙️ Configuración")
    
    # Status del API
    api_url = st.sidebar.text_input(
        "🔗 URL del Backend",
        value=DEFAULT_API_URL,
        help="URL del endpoint del chatbot"
    )
    
    # Verificar status
    api_status = check_api_status(api_url)
    if api_status:
        st.sidebar.markdown('<div class="status-indicator status-online">✅ API Online</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-indicator status-error">❌ API Offline</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Configuración de búsqueda
    st.sidebar.markdown("### 🔍 Búsqueda")
    top_k = st.sidebar.slider(
        "Documentos a recuperar",
        min_value=1,
        max_value=10,
        value=4,
        help="Número de fragmentos relevantes a recuperar"
    )
    
    enable_web_search = st.sidebar.checkbox(
        "🌐 Habilitar búsqueda web",
        value=False,
        help="Permite buscar información adicional en la web"
    )
    
    st.sidebar.markdown("---")
    
    # Configuración de interfaz
    st.sidebar.markdown("### 🎨 Interfaz")
    
    theme = st.sidebar.selectbox(
        "Tema",
        options=[Theme.LIGHT, Theme.DARK],
        format_func=lambda x: "☀️ Claro" if x == Theme.LIGHT else "🌙 Oscuro"
    )
    
    language = st.sidebar.selectbox(
        "Idioma de respuesta",
        options=["es", "en"],
        format_func=lambda x: "🇪🇸 Español" if x == "es" else "🇺🇸 English"
    )
    
    show_timestamps = st.sidebar.checkbox(
        "⏰ Mostrar timestamps",
        value=True
    )
    
    auto_scroll = st.sidebar.checkbox(
        "📜 Auto-scroll",
        value=True,
        help="Desplazar automáticamente a los nuevos mensajes"
    )
    
    st.sidebar.markdown("---")
    
    # Opciones avanzadas
    with st.sidebar.expander("🔧 Opciones Avanzadas"):
        debug = st.checkbox(
            "Modo Debug",
            value=False,
            help="Muestra información técnica detallada"
        )
    
    st.sidebar.markdown("---")
    
    # Estadísticas de sesión
    st.sidebar.markdown("### 📈 Estadísticas de Sesión")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total consultas", st.session_state.total_queries)
    with col2:
        session_duration = datetime.now() - st.session_state.session_start
        st.metric("Duración", f"{session_duration.seconds // 60}m")
    
    # Acciones
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Limpiar Conversación", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.last_response = None
        st.session_state.total_queries = 0
        st.rerun()
    
    if st.sidebar.button("💾 Exportar Conversación", use_container_width=True):
        # Aquí podrías implementar exportación a JSON/PDF
        st.sidebar.info("Función próximamente disponible")
    
    # ... deja igual el resto de controles ...
    return AppConfig(
        api_url=api_url.rstrip("/"),
        top_k=top_k,
        enable_web_search=enable_web_search,
        debug=debug,
        language=language,
        theme=theme,
        auto_scroll=auto_scroll,
        show_timestamps=False   # fijo (sin hora)
    )


def render_tips_card():
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px; padding: 1.5rem; color: white; margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">💡 Tips para mejores respuestas</h3>
            <ul style="margin-bottom: 0;">
                <li>Sé específico en tus preguntas</li>
                <li>Pregunta sobre coberturas, exclusiones o límites</li>
                <li>Haz preguntas de seguimiento</li>
                <li>Activa la búsqueda web para información actualizada</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def handle_user_query(text: str, config: AppConfig) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.total_queries += 1

    response = call_backend_api(
        config,
        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    )
    st.session_state.last_response = response

    if "error" in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error: {response['error']}"
        })
    else:
        answer = response.get("answer", "No se recibió respuesta del servidor.")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


def render_suggestions(config: AppConfig):
    st.markdown("### 🎯 Preguntas Sugeridas")
    suggestions = [
        "¿Qué cubre la póliza de hogar básica?",
        "¿Cuáles son las exclusiones del seguro de auto?",
        "¿Cuál es el límite de cobertura médica?",
        "¿Cómo presento un reclamo?",
        "¿Qué documentos necesito para una reclamación?",
    ]
    for s in suggestions:
        if st.button(f"💬 {s}", key=f"sugg_{s}", use_container_width=True):
            handle_user_query(s, config)

# ========================== APLICACIÓN PRINCIPAL ==========================

def main():
    """Función principal de la aplicación."""
    st.set_page_config(
        page_title="Insurance Chatbot - Asistente Inteligente",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar estado
    init_session_state()
    
    # Obtener configuración del sidebar
    config = render_sidebar()
    
    # Aplicar tema
    st.markdown(get_theme_styles(config.theme), unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="app-header">
            <h1>🤖 Insurance Chatbot</h1>
            <p>Tu asistente inteligente para consultas sobre pólizas de seguro</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Layout principal de dos columnas
    col_chat, col_info = st.columns([2, 1])
    
    with col_chat:
        # Contenedor del chat
        # Después (mensajes quedan DENTRO de la caja)
        chat_html = "".join(build_message_html(m, config.show_timestamps) for m in st.session_state.messages)
        st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)

        # Input del usuario
        if prompt := st.chat_input("💬 Escribe tu consulta sobre seguros..."):
            handle_user_query(prompt, config)

            
            # Incrementar contador
            st.session_state.total_queries += 1
            
            
            # Mostrar indicador de escritura
            with st.spinner(""):
                render_typing_indicator()
                
                # Llamar al backend
                response = call_backend_api(
                    config,
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                )
                
                # Guardar respuesta
                st.session_state.last_response = response
                
                # Procesar respuesta
                if "error" in response:
                    error_msg = f"❌ Error: {response['error']}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": format_timestamp()
                    })
                    st.error(error_msg)
                else:
                    answer = response.get("answer", "No se recibió respuesta del servidor.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": format_timestamp()
                    })
                
                # Recargar para mostrar la respuesta
                st.rerun()
    
    # Columna de información
    with col_info:
        render_tips_card()
        render_suggestions(config)

    if st.session_state.last_response:
        resp = st.session_state.last_response
        if resp.get("sources"): render_sources(resp["sources"])
        if resp.get("usage"):   render_metrics(resp)
        if config.debug and resp.get("debug"): render_debug_info(resp["debug"])

    
    # Footer con información adicional
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="text-align: center; color: #718096;">
                <small>🔒 Tus datos están seguros</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="text-align: center; color: #718096;">
                <small>⚡ Powered by AI</small>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="text-align: center; color: #718096;">
                <small>📊 Análisis en tiempo real</small>
            </div>
        """, unsafe_allow_html=True)

# ========================== PUNTO DE ENTRADA ==========================

if __name__ == "__main__":
    main()
                