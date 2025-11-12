import streamlit as st
from typing import Callable

def render_tips_card():
    st.markdown("""
        <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    border-radius:16px;padding:1.5rem;color:#fff;margin-bottom:1rem;">
            <h3 style="margin-top:0;">💡 Tips para mejores respuestas</h3>
            <ul style="margin-bottom:0;">
                <li>Sé específico en tus preguntas</li>
                <li>Pregunta sobre coberturas, exclusiones o límites</li>
                <li>Haz preguntas de seguimiento</li>
                <li>Activa la búsqueda web para información actualizada</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def render_suggestions(on_click: Callable[[str], None]):
    st.markdown("### 🎯 Preguntas Sugeridas")
    suggestions = [
        "¿Qué cubre la póliza de hogar básica?",
        "¿Cuáles son las exclusiones del seguro de auto?",
        "¿Cuál es el límite de cobertura médica?",
        "¿Cómo presento un reclamo?",
        "¿Qué documentos necesito para una reclamación?",
    ]
    for s in suggestions:
        if st.button(f"💬 {s}", key=f"sugg_{hash(s)}", use_container_width=True):
            on_click(s)
