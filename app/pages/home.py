"""
DeepFake Detection System - Home Page
"""

import streamlit as st


def render():
    """Render the Home / Landing page."""

    # ── Hero Section ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <div style="font-size:5rem; margin-bottom:0.5rem;">🛡️</div>
        <h1 style="
            font-size: 3.2rem;
            font-weight: 900;
            background: linear-gradient(135deg, #e0f7fa 0%, #80deea 40%, #ff8a65 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem;
            letter-spacing: -1px;
        ">DeepFake Shield</h1>
        <p style="color:#90a4ae; font-size:1.2rem; margin-bottom:2rem; font-weight:300;">
            AI-Powered Video Authenticity Detection &amp; Prevention System
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats Row ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("🎯", "95%+", "Detection Accuracy"),
        ("⚡", "<30s",  "Average Analysis Time"),
        ("🔬", "CNN-LSTM", "Model Architecture"),
        ("🌐", "FaceForensics++", "Training Dataset"),
    ]
    for col, (icon, value, label) in zip([col1, col2, col3, col4], stats):
        col.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #0d1b2a, #1a2744);
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        ">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="color:#80deea; font-size:1.4rem; font-weight:800;">{value}</div>
            <div style="color:#607d8b; font-size:0.75rem; margin-top:0.2rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Cards ──────────────────────────────────────────────────────────
    st.markdown("### 🔧 System Capabilities")
    
    features = [
        ("🎬", "Video Analysis",
         "Upload any video format (MP4, AVI, MOV, MKV) for automated DeepFake detection using "
         "state-of-the-art CNN-LSTM deep learning."),
        ("👁️", "Face Detection",
         "MediaPipe-powered real-time face detection and landmark extraction for precise "
         "facial region analysis."),
        ("🧠", "CNN-LSTM Model",
         "EfficientNetB0 feature extractor combined with Bidirectional LSTM for temporal "
         "pattern analysis across video frames."),
        ("📊", "Confidence Scores",
         "Detailed per-frame and aggregate confidence scores with interactive probability "
         "timeline visualization."),
        ("🗺️", "Explainable AI",
         "Grad-CAM heatmaps highlight which facial regions contributed most to the "
         "deepfake classification decision."),
        ("📄", "Report Generation",
         "Export professional PDF or JSON detection reports with full analysis summary "
         "suitable for audit trails."),
        ("📷", "Live Detection",
         "Real-time webcam-based deepfake detection using your device camera with "
         "instant visual feedback."),
        ("🚀", "REST API",
         "FastAPI-powered REST endpoint for programmatic integration with external "
         "systems and automated pipelines."),
    ]

    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(145deg, #0d1b2a, #112036);
                border: 1px solid #1e3a5f;
                border-left: 3px solid #80deea;
                border-radius: 10px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.8rem;
            ">
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                    <span style="font-size:1.4rem;">{icon}</span>
                    <strong style="color:#e0f7fa; font-size:0.95rem;">{title}</strong>
                </div>
                <p style="color:#78909c; font-size:0.83rem; margin:0; line-height:1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ──────────────────────────────────────────────────────────
    st.markdown("### ⚙️ How It Works")
    
    steps = [
        ("1", "Upload Video", "Select a video file from your device"),
        ("2", "Frame Extraction", "System extracts key frames intelligently"),
        ("3", "Face Detection", "MediaPipe detects and crops faces"),
        ("4", "CNN Features", "EfficientNet extracts spatial features"),
        ("5", "LSTM Analysis", "Temporal patterns analyzed across frames"),
        ("6", "Verdict", "Confidence-scored REAL or FAKE decision"),
    ]
    
    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="text-align:center;">
            <div style="
                width:40px; height:40px;
                background: linear-gradient(135deg, #0277bd, #80deea);
                border-radius:50%;
                display:flex; align-items:center; justify-content:center;
                margin:0 auto 0.5rem;
                font-weight:900; color:#fff; font-size:1rem;
                line-height:40px;
            ">{num}</div>
            <div style="color:#e0f7fa; font-size:0.78rem; font-weight:600; margin-bottom:0.2rem;">{title}</div>
            <div style="color:#607d8b; font-size:0.68rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick Start CTA ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d2137, #0a3d5c);
        border: 1px solid #0277bd;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
    ">
        <h3 style="color:#80deea; margin-bottom:0.6rem;">Ready to Detect DeepFakes?</h3>
        <p style="color:#90a4ae; font-size:0.9rem; margin-bottom:0;">
            Navigate to <strong style="color:#e0f7fa;">Upload &amp; Detect</strong> from the sidebar to analyze your first video.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "⚠️ **Important:** This system is for educational and research purposes. "
        "Results should be reviewed by qualified professionals. "
        "No automated system achieves 100% accuracy on all video types.",
        icon="ℹ️"
    )
