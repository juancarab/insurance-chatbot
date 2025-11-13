from .config import Theme

def get_theme_styles(theme: Theme) -> str:
    base = """
    <style>
    .stApp{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;}
    .block-container{padding-top:2rem;padding-bottom:2rem;max-width:1400px;}
    html,body,[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#0F172A 0%,#111827 100%)!important;color:#E2E8F0!important;}

    /* Header */
    .app-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:20px;padding:2rem;margin-bottom:2rem;
        box-shadow:0 10px 30px rgba(102,126,234,.3);color:white;text-align:center;}
    .app-header h1{font-size:2.5rem;font-weight:700;margin:0;letter-spacing:-.5px;}
    .app-header p{margin-top:.5rem;opacity:.95;font-size:1.1rem;}

    /* Chat */
    .chat-container{background:linear-gradient(180deg,#FCFEFF 0%,#F3F6FA 100%);border:1px solid #E2E8F0;border-radius:20px;
        padding:1.5rem;box-shadow:0 6px 28px rgba(2,6,23,.06);min-height:480px;margin-bottom:1rem;color:#0B1220;}
    .message-wrapper{display:flex;margin-bottom:1.2rem;animation:fadeInUp .3s ease;}
    .message-wrapper.assistant{justify-content:flex-start;}
    .message-wrapper.user{justify-content:flex-end;}
    @keyframes fadeInUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
    .message-bubble{display:inline-block;padding:1rem 1.25rem;border-radius:18px;font-size:.95rem;line-height:1.5;word-wrap:break-word;position:relative;}
    .user-message{background:#fff;color:#1a202c;border:2px solid #667eea;margin-left:auto;text-align:left;box-shadow:0 2px 10px rgba(102,126,234,.15);}
    .assistant-message{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;margin-right:auto;box-shadow:0 2px 15px rgba(102,126,234,.25);}
    .message-timestamp{
    display:block;
    margin-top:.40rem;
    padding:.15rem .55rem;
    color:#EEF2FF;
    border-radius:999px;
    font-size:.76rem;
    line-height:1;
    }

    .message-wrapper.user .message-timestamp{
        margin-right:0;
    }

    .message-wrapper.assistant .message-timestamp{
        margin-left:0;
    }

    /* Sources */
    .source-card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:1.25rem;margin-bottom:1rem;transition:all .3s ease;position:relative;overflow:hidden;}
    .source-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.1);border-color:#667eea;}
    .source-card::before{content:'';position:absolute;left:0;top:0;height:100%;width:3px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);}
    .source-number{display:inline-block;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;width:24px;height:24px;border-radius:50%;text-align:center;line-height:24px;font-size:.85rem;font-weight:600;margin-right:.5rem;}
    .source-title{font-weight:600;color:#2d3748;margin-bottom:.5rem;font-size:1rem;}
    .source-snippet{color:#4a5568;font-size:.9rem;line-height:1.6;margin:.75rem 0;padding:.75rem;background:#f7fafc;border-radius:8px;border-left:3px solid #667eea;}
    .source-meta{display:flex;gap:1rem;align-items:center;flex-wrap:wrap;}
    .source-tag{display:inline-block;background:#eef2ff;color:#667eea;padding:.25rem .75rem;border-radius:999px;font-size:.75rem;font-weight:500;}

    /* Metrics */
    .metrics-container{background:linear-gradient(135deg,#f6f9fc 0%,#ffffff 100%);border-radius:16px;padding:1.25rem;border:1px solid #e2e8f0;margin-bottom:1rem;}
    .metric-item{display:flex;justify-content:space-between;align-items:center;padding:.75rem;background:#fff;border-radius:10px;margin-bottom:.75rem;border:1px solid #e2e8f0;}
    .metric-label{color:#718096;font-size:.85rem;font-weight:500;}
    .metric-value{color:#2d3748;font-weight:700;font-size:1.1rem;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}

    /* Typing */
    .typing-indicator{display:inline-flex;align-items:center;padding:1rem;gap:.4rem;}
    .typing-dot{width:8px;height:8px;background:#667eea;border-radius:50%;animation:typingAnimation 1.4s infinite;}
    .typing-dot:nth-child(2){animation-delay:.2s;}
    .typing-dot:nth-child(3){animation-delay:.4s;}
    @keyframes typingAnimation{0%,60%,100%{transform:translateY(0);opacity:.5;}30%{transform:translateY(-10px);opacity:1;}}

    /* Sidebar */
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a202c 0%,#2d3748 100%);}
    section[data-testid="stSidebar"] .stButton>button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:10px;padding:.5rem 1rem;font-weight:500;transition:all .3s ease;}
    section[data-testid="stSidebar"] .stButton>button:hover{transform:translateY(-2px);box-shadow:0 4px 15px rgba(102,126,234,.4);}

    .status-indicator{display:inline-flex;align-items:center;gap:.5rem;padding:.5rem 1rem;border-radius:999px;font-size:.9rem;font-weight:500;}
    .status-online{background:rgba(16,185,129,.1);color:#10b981;border:1px solid #10b981;}
    .status-error{background:rgba(239,68,68,.1);color:#ef4444;border:1px solid #ef4444;}

    ::-webkit-scrollbar{width:8px;height:8px;}
    ::-webkit-scrollbar-track{background:#f1f5f9;border-radius:10px;}
    ::-webkit-scrollbar-thumb{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:10px;}
    ::-webkit-scrollbar-thumb:hover{background:linear-gradient(135deg,#5a67d8 0%,#6b46a0 100%);}
    </style>
    """
    dark = """
    <style>
    .chat-container{background:linear-gradient(180deg,#1e293b 0%,#0f172a 100%);border-color:#334155;}
    .user-message{background:#1e293b;color:#f1f5f9;border-color:#667eea;}
    .source-card{background:#1e293b;border-color:#334155;color:#e2e8f0;}
    .source-snippet{background:#0f172a;color:#cbd5e1;}
    .metrics-container{background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);border-color:#334155;}
    .metric-item{background:#0f172a;border-color:#334155;}
    .metric-label{color:#94a3b8;}
    </style>
    """
    return base + (dark if theme == Theme.DARK else "")