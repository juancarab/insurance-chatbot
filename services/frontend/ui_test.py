"""Insurance Chatbot — Pro UI (clean, fluid, minimal)"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

# -------------------- Config --------------------
DEFAULT_API_URL = os.getenv("INSURANCE_CHATBOT_API_URL", "http://localhost:8000/chat")
APP_TITLE = "Insurance Chatbot"
APP_TAGLINE = "Asistente de seguros con recuperación y agente LLM"

# -------------------- CSS --------------------
BASE_CSS = """
<style>
:root {
  --accent: {ACCENT};
  --accent-2: {ACCENT_2};
  --bg: #0c111c;
  --card: rgba(255,255,255,0.04);
  --line: rgba(255,255,255,0.10);
  --ring: {RING};
  --text: #e6eaf2;
  --muted: #a7b0c0;
}
.stApp {{
  background:
    radial-gradient(1100px 500px at 10% -10%, rgba(127,90,240,.10), transparent 55%),
    radial-gradient(900px 450px at 100% 0%, rgba(80,130,255,.10), transparent 55%),
    linear-gradient(180deg, #0a0f19 0%, #0a0f19 60%, #0b1220 100%);
}}
.block-container {{ padding-top: 1rem; max-width: 1150px; }}

.app-hero {{
  position: relative;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--line);
  padding: 18px 20px;
  overflow: hidden;
}}
.app-hero::after {{
  content: "";
  position: absolute; inset: -1px;
  background:
    radial-gradient(600px 180px at 20% 0%, rgba(255,255,255,0.05), transparent 60%),
    radial-gradient(500px 150px at 90% -10%, rgba(127,90,240,0.22), transparent 70%);
  pointer-events: none;
}}
.app-title {{ font-weight: 800; letter-spacing: .2px; font-size: 1.6rem; color: var(--text); }}
.app-sub {{ font-size: .95rem; color: var(--muted); }}

.badge {{
  display:inline-flex; gap:.4rem; align-items:center;
  border-radius: 999px; padding: .26rem .6rem;
  font-size:.76rem; color:#E7E9FF;
  background: rgba(127,90,240,.14);
  border:1px solid rgba(127,90,240,.35);
}}
.chip {{
  display:inline-flex; gap:.35rem; align-items:center;
  border-radius:999px; padding:.22rem .55rem; font-size:.74rem;
  color: var(--muted); border:1px dashed var(--line);
  background: rgba(255,255,255,0.02);
}}
.kbd {{ background:#0f1523; border:1px solid var(--line); padding:.1rem .4rem; border-radius:6px; font-size:.75rem; }}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg,#0c111c,.45,#0c111c00);
  border-right: 1px solid var(--line);
}}
.sidebar-card {{
  background: rgba(255,255,255,0.03);
  border:1px solid var(--line);
  border-radius: 14px; padding: 12px;
}}

.stChatMessage {{ padding: 0; }}
.bubble {{
  position: relative;
  border-radius: 14px; padding: 12px 14px; margin: 6px 0 12px;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border: 1px solid var(--line);
  box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}}
.bubble.assistant {{
  border-image: linear-gradient(90deg, var(--accent), transparent) 1;
}}
.bubble.user {{
  border-image: linear-gradient(90deg, rgba(56,189,248,.8), transparent) 1;
  background: linear-gradient(180deg, rgba(56,189,248,.10), rgba(255,255,255,0.02));
}}

.source-grid {{
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}}
.source-card {{
  border-radius: 14px; padding: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--line);
  transition: transform .15s ease, border .15s ease, background .15s ease;
}}
.source-card:hover {{
  transform: translateY(-2px);
  border-color: var(--ring);
  background: rgba(255,255,255,0.04);
}}
.source-title {{ font-weight: 700; color: var(--text); font-size: .98rem; }}
.meta {{ color: var(--muted); font-size: .8rem; }}
.quote {{ border-left: 3px solid var(--line); padding-left: .6rem; margin-top: .2rem; color: var(--text); opacity:.95; }}

.quick-row {{ display:flex; gap:.5rem; flex-wrap: wrap; }}
.quick-btn {{
  all: unset; cursor: pointer;
  font-size:.85rem; color:#dfe6ff;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  border:1px solid var(--line); border-radius: 999px; padding:.45rem .7rem;
}}
.quick-btn:hover {{ border-color: var(--ring); }}
</style>
"""

def inject_css(accent: str = "#7F5AF0") -> None:
    accent2 = "#8EA2FF"
    ring = "rgba(126, 90, 240, 0.42)"
    # ⚠️ reemplazamos SOLO los placeholders que pusimos y dejamos
    # las llaves del CSS intactas (no usamos .format).
    css = (
        BASE_CSS
        .replace("{ACCENT}", accent)
        .replace("{ACCENT_2}", accent2)
        .replace("{RING}", ring)
    )
    st.markdown(css, unsafe_allow_html=True)

# -------------------- State --------------------
def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hola, soy tu **asistente de seguros**. Pregúntame sobre **coberturas**, "
                    "**exclusiones** o **límites**. También puedo **citar fuentes**."
                ),
            }
        ]
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

# -------------------- UI Pieces --------------------
def header() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:14px;">
            <div>
              <div class="app-title">💬 Insurance Chatbot</div>
              <div class="app-sub">Asistente de seguros con recuperación y agente LLM</div>
            </div>
            <div style="display:flex;gap:.5rem;flex-wrap:wrap;">
              <span class="badge">RAG</span>
              <span class="badge">Agente</span>
              <span class="badge">UI Minimal</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def sidebar() -> Dict[str, Any]:
    st.sidebar.header("⚙️ Configuración")
    with st.sidebar.container():
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        api_url = st.text_input("URL del backend", value=DEFAULT_API_URL)
        top_k = st.slider("Documentos a recuperar (top_k)", 1, 10, 4)
        col1, col2 = st.columns(2)
        with col1:
            enable_web_search = st.checkbox("Búsqueda web", value=False)
        with col2:
            debug = st.checkbox("Modo debug", value=False)
        language = st.selectbox("Idioma de respuesta", ["es", "en"], index=0)
        st.markdown("---")
        accent = st.color_picker("Color de acento", "#7F5AF0", help="Personaliza el brillo/acento de la UI.")
        st.markdown("---")
        if st.button("🧹 Limpiar chat", use_container_width=True, type="secondary"):
            st.session_state.messages = st.session_state.messages[:1]
            st.session_state.last_response = None
            st.toast("Chat limpiado", icon="🧽")
        st.markdown("</div>", unsafe_allow_html=True)
    return {
        "api_url": api_url.rstrip("/"),
        "top_k": top_k,
        "enable_web_search": enable_web_search,
        "debug": bool(debug),
        "language": language,
        "accent": accent,
    }

def quick_actions() -> None:
    st.write("")
    st.markdown("**Sugerencias rápidas**")
    cols = st.columns([1,1,1,1])
    prompts = [
        "¿Qué cubre la póliza de autos contra terceros?",
        "Explica exclusiones típicas en pólizas de hogar.",
        "¿Cómo elegir suma asegurada adecuada para vida?",
        "Compara deducible vs coaseguro con ejemplo."
    ]
    for c, p in zip(cols, prompts):
        with c:
            if st.button(p, key=f"quick_{hash(p)}"):
                st.session_state.messages.append({"role": "user", "content": p})
                st.session_state["__trigger_request__"] = True

def render_chat(messages: List[Dict[str, str]]) -> None:
    for m in messages:
        role = "assistant" if m.get("role") in ("assistant","system") else "user"
        avatar = "🛡️" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(f'<div class="bubble {role}">', unsafe_allow_html=True)
            st.markdown(m.get("content",""))
            st.markdown("</div>", unsafe_allow_html=True)

def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources: return
    with st.expander("📚 Fuentes utilizadas", expanded=True):
        st.markdown('<div class="source-grid">', unsafe_allow_html=True)
        for i, s in enumerate(sources, 1):
            title = s.get("title") or s.get("file_name") or "Sin título"
            snippet = (s.get("snippet") or "").strip()
            meta_bits = []
            if s.get("page") is not None: meta_bits.append(f"p. **{s['page']}**")
            if s.get("chunk_id"): meta_bits.append(f"chunk **{s['chunk_id']}**")
            if s.get("score") is not None:
                try: meta_bits.append(f"score **{float(s['score']):.3f}**")
                except: meta_bits.append(f"score {s['score']}")
            meta = " — " + ", ".join(meta_bits) if meta_bits else ""
            st.markdown('<div class="source-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="source-title">{i}. {title}</div>', unsafe_allow_html=True)
            if meta: st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)
            if snippet: st.markdown(f'<div class="quote">“{snippet}”</div>', unsafe_allow_html=True)
            url = s.get("url")
            if url: st.link_button("Abrir enlace", url, use_container_width=True, type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_debug(resp: Dict[str, Any]) -> None:
    if "debug" not in resp: return
    with st.expander("🛠️ Debug"):
        tabs = st.tabs(["Pasos", "Respuesta completa"])
        with tabs[0]:
            st.json(resp["debug"].get("steps") or resp["debug"])
        with tabs[1]:
            st.json(resp)

# -------------------- Backend call --------------------
def call_backend(api_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(api_url, json=payload, timeout=45)
    r.raise_for_status()
    return r.json()

# -------------------- App --------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
    init_session()
    cfg = sidebar()
    inject_css(cfg["accent"])
    header()

    # Quick actions (optional, non-saturating)
    quick_actions()

    # Chat input
    user_prompt = st.chat_input("Escribe tu consulta sobre pólizas… (Shift+Enter = salto de línea)")

    trigger = False
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        trigger = True
    if st.session_state.pop("__trigger_request__", False):
        trigger = True

    if trigger:
        payload = {
            "messages": st.session_state.messages,
            "top_k": cfg["top_k"],
            "enable_web_search": cfg["enable_web_search"],
            "debug": bool(cfg["debug"]),
            "language": cfg["language"],
        }
        try:
            with st.status("Consultando backend…", state="running") as s:
                resp = call_backend(cfg["api_url"], payload)
                st.session_state.last_response = resp
                answer = resp.get("answer", "No se recibió respuesta del backend.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                s.update(label="Respuesta recibida", state="complete")
        except requests.RequestException as e:
            msg = f"❌ No se pudo contactar al backend **{cfg['api_url']}**\n\n**Detalle:** {e}"
            st.session_state.last_response = {"error": str(e)}
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.error(msg)
            st.toast("Verifica FastAPI y la URL.", icon="⚠️")

        rerun = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
        if rerun: rerun()

    # Transcript
    render_chat(st.session_state.messages)

    # Sources + Debug
    last = st.session_state.last_response or {}
    render_sources(last.get("sources", []) or [])
    if cfg["debug"]:
        render_debug(last)

    st.markdown("---")
    st.caption(
        "Consejo: usa <span class='kbd'>Shift</span>+<span class='kbd'>Enter</span> para saltos de línea. "
        "Personaliza el **color de acento** en el sidebar.",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
