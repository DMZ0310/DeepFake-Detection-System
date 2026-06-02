"""
DeepFake Shield - Main Application Entry Point
===============================================
Run with:  streamlit run main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from app.config import APP_TITLE, APP_ICON, VERSION

# ── Page Config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title=f"{APP_TITLE} v{VERSION}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #060d18 0%, #0a0e1a 50%, #06101c 100%);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050b14 0%, #071220 100%) !important;
    border-right: 1px solid #1a2840 !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: #80deea !important;
}

/* Main content container */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0d1b2a, #122038);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label {
    color: #607d8b !important;
    font-size: 0.78rem !important;
}
[data-testid="stMetricValue"] {
    color: #e0f7fa !important;
    font-weight: 700 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0277bd, #01579b) !important;
    color: #e0f7fa !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0288d1, #0277bd) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(2,119,189,0.4) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0277bd, #006db3) !important;
    font-size: 1.05rem !important;
    padding: 0.6rem 1.5rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0d1b2a !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #0277bd, #80deea) !important;
}

/* Info/warning/success boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a1220 !important;
    border-radius: 8px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #607d8b !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0277bd, #01579b) !important;
    color: #e0f7fa !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: #80deea !important;
}

/* Checkboxes */
.stCheckbox > label {
    color: #90a4ae !important;
}

/* Code blocks */
.stCode {
    background: #0a1220 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #0d1b2a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e0f7fa !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #060d18; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0277bd; }
</style>
""", unsafe_allow_html=True)


# ── Lazy page imports ──────────────────────────────────────────────────────────
from app.pages import home, detect, performance, analytics, live, about


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div style="text-align:center; padding: 1.2rem 0 1rem;">
            <div style="font-size:2.8rem; margin-bottom:0.3rem;">🛡️</div>
            <h2 style="
                color:#e0f7fa; font-size:1.3rem; font-weight:800;
                margin:0; letter-spacing:-0.5px;
            ">DeepFake Shield</h2>
            <div style="
                color:#546e7a; font-size:0.7rem; letter-spacing:1px;
                margin-top:0.2rem;
            ">v{VERSION} · AI DETECTION SYSTEM</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#1a2840; margin:0.5rem 0 1rem;">', unsafe_allow_html=True)

        nav_options = [
            "🏠  Home",
            "🔍  Upload & Detect",
            "📊  Model Performance",
            "📈  Analytics",
            "📷  Live Detection",
            "ℹ️  About",
        ]

        selected = st.radio(
            "Navigation",
            nav_options,
            label_visibility="collapsed",
        )

        st.markdown('<hr style="border-color:#1a2840; margin:1rem 0;">', unsafe_allow_html=True)

        # Status indicators
        _render_sidebar_status()

    return selected


def _render_sidebar_status():
    """Show quick-status pills in sidebar."""
    from pathlib import Path
    from app.config import MODEL_PATH

    model_ok = Path(MODEL_PATH).exists()
    
    checks = [
        ("Model Weights",  model_ok,  "Loaded" if model_ok else "Not trained"),
        ("Face Detector",  True,      "MediaPipe"),
        ("GPU / CPU",      False,     "CPU Mode"),
    ]
    
    st.markdown('<div style="font-size:0.72rem; color:#455a64; margin-bottom:0.4rem;">SYSTEM STATUS</div>', unsafe_allow_html=True)
    
    for name, ok, label in checks:
        dot   = "🟢" if ok else "🟡"
        color = "#a5d6a7" if ok else "#ffa726"
        st.markdown(f"""
        <div style="
            display:flex; justify-content:space-between; align-items:center;
            font-size:0.78rem; color:#546e7a; margin-bottom:0.3rem;
        ">
            <span>{dot} {name}</span>
            <span style="color:{color}; font-size:0.7rem;">{label}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#1a2840; margin:0.8rem 0;">', unsafe_allow_html=True)

    # Quick links
    st.markdown("""
    <div style="font-size:0.72rem; color:#455a64; margin-bottom:0.4rem;">QUICK LINKS</div>
    <div style="font-size:0.78rem;">
        <a href="https://github.com" target="_blank" style="color:#80deea; text-decoration:none;">📦 GitHub Repo</a><br>
        <a href="http://localhost:8000/docs" target="_blank" style="color:#80deea; text-decoration:none;">🚀 FastAPI Docs</a><br>
        <a href="https://github.com/ondyari/FaceForensics" target="_blank" style="color:#80deea; text-decoration:none;">📁 Dataset</a>
    </div>
    """, unsafe_allow_html=True)


# ── Router ─────────────────────────────────────────────────────────────────────
def main():
    selected = render_sidebar()

    page_map = {
        "🏠  Home":              home.render,
        "🔍  Upload & Detect":   detect.render,
        "📊  Model Performance": performance.render,
        "📈  Analytics":         analytics.render,
        "📷  Live Detection":    live.render,
        "ℹ️  About":             about.render,
    }

    renderer = page_map.get(selected, home.render)
    renderer()


if __name__ == "__main__":
    main()
