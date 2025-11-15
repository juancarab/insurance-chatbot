import streamlit as st
from typing import Callable

def render_tips_card():
    st.markdown("""
        <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                    border-radius:16px;padding:1.5rem;color:#fff;margin-bottom:1rem;">
            <h3 style="margin-top:0;">💡 Tips para mejores respuestas</h3>
            <ul style="margin-bottom:0;">
                <li>Be specific in your questions.</li>
                <li>Question about coverage, exclusions, or limits</li>
                <li>Ask follow-up questions</li>
                <li>Activate web search for updated information</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def render_suggestions(on_click: Callable[[str], None]):
    st.markdown("### 🎯 Suggested Questions")
    suggestions = [
        "What does basic home insurance cover?",
        "What are the exclusions from auto insurance?",
        "What is the medical coverage limit?",
        "How do I file a claim?",
        "What documents do I need for a claim?",
    ]
    for s in suggestions:
        if st.button(f"💬 {s}", key=f"sugg_{hash(s)}", use_container_width=True):
            on_click(s)
