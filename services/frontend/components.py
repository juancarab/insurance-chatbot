"""Componentes reutilizables para el frontend."""
import streamlit as st
from typing import List, Dict, Any

def render_chat_history_export(messages: List[Dict[str, Any]]) -> None:
    """Renderiza opciones de exportación del historial."""
    import json
    from datetime import datetime
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "total_messages": len(messages)
    }
    
    # Botón de descarga JSON
    st.download_button(
        label="📥 Descargar conversación (JSON)",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def render_feedback_widget() -> None:
    """Widget para recopilar feedback del usuario."""
    with st.expander("📝 ¿Cómo fue tu experiencia?"):
        rating = st.slider("Califica la respuesta", 1, 5, 3)
        feedback = st.text_area("Comentarios adicionales (opcional)")
        if st.button("Enviar feedback"):
            st.success("¡Gracias por tu feedback!")

def render_quick_actions(config: Any) -> None:
    """Renderiza acciones rápidas."""
    cols = st.columns(4)
    
    with cols[0]:
        if st.button("🔄 Nueva consulta", use_container_width=True):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()
    
    with cols[1]:
        if st.button("📋 Copiar última", use_container_width=True):
            if st.session_state.last_response:
                st.write("Respuesta copiada al portapapeles")
    
    with cols[2]:
        if st.button("🔊 Leer en voz alta", use_container_width=True):
            st.info("Función próximamente")
    
    with cols[3]:
        if st.button("📧 Enviar por email", use_container_width=True):
            st.info("Función próximamente")