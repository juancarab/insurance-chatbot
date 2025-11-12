from typing import Any, Dict, List
import streamlit as st

def render_sources(sources: List[Dict[str, Any]]) -> None:
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
        if page: meta_tags.append(f'<span class="source-tag">📄 Página {page}</span>')
        if chunk_id: meta_tags.append(f'<span class="source-tag">🔖 Chunk {chunk_id}</span>')
        if score is not None:
            try:
                meta_tags.append(f'<span class="source-tag">⭐ {float(score):.2f}</span>')
            except Exception:
                pass
        st.markdown(f"""
            <div class="source-card">
                <div class="source-title"><span class="source-number">{idx}</span>{title}</div>
                {f'<div class="source-snippet">{snippet}</div>' if snippet else ''}
                <div class="source-meta">{' '.join(meta_tags)} {f'<a href="{url}" target="_blank" style="color:#667eea;text-decoration:none;">🔗 Ver documento</a>' if url else ''}</div>
            </div>
        """, unsafe_allow_html=True)

def render_metrics(response: Dict[str, Any]) -> None:
    usage = response.get("usage", {})
    st.markdown('<div class="metrics-container"><h4 style="margin-bottom:1rem;">📊 Métricas de la Consulta</h4>', unsafe_allow_html=True)
    metrics = {
        "Documentos recuperados": usage.get("retrieved_documents", 0),
        "Formateador": usage.get("formatter", "N/A"),
        "Búsqueda web": "Activa" if usage.get("web_search_enabled") else "Inactiva",
        "Idioma": usage.get("language", "es").upper(),
        "Top-K": usage.get("top_k", 0),
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
    with st.expander("🔧 Información de Debug", expanded=False):
        st.json(debug_data)
