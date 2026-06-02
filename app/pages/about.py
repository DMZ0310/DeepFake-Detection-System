"""
DeepFake Detection System - About / Info Page
"""

import streamlit as st
from app.config import VERSION


def render():
    st.markdown(f"""
    <h2 style="color:#80deea; font-weight:800; margin-bottom:0.2rem;">
        ℹ️ About DeepFake Shield
    </h2>
    <p style="color:#607d8b; margin-bottom:1.5rem;">
        System information, dataset details, and references.
    </p>
    """, unsafe_allow_html=True)

    # Version badge
    st.markdown(f"""
    <div style="
        display:inline-block;
        background: linear-gradient(90deg, #0277bd, #01579b);
        border-radius:20px; padding:0.3rem 1rem;
        color:#e0f7fa; font-size:0.85rem; font-weight:600;
        margin-bottom:1.5rem;
    ">
        v{VERSION} &nbsp;·&nbsp; Production Ready &nbsp;·&nbsp; CNN-LSTM Architecture
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Info ──────────────────────────────────────────────────────────
    st.markdown("### 📁 Dataset: FaceForensics++")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(145deg, #0d1b2a, #1a2744);
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            padding: 1.3rem;
        ">
            <h4 style="color:#80deea; margin-bottom:0.8rem;">Dataset Overview</h4>
            <ul style="color:#90a4ae; font-size:0.88rem; line-height:2;">
                <li>1,000 original YouTube videos</li>
                <li>4,000 manipulated videos (4 methods)</li>
                <li>Total: ~5,000 videos, ~500K frames</li>
                <li>Manipulation methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures</li>
                <li>Multiple compression levels (raw, c23, c40)</li>
                <li>Benchmarked by top research labs worldwide</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(145deg, #0d1b2a, #1a2744);
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            padding: 1.3rem;
        ">
            <h4 style="color:#80deea; margin-bottom:0.8rem;">Download Instructions</h4>
            <p style="color:#90a4ae; font-size:0.85rem; line-height:1.7;">
                FaceForensics++ requires academic registration:
            </p>
            <ol style="color:#90a4ae; font-size:0.85rem; line-height:2;">
                <li>Visit: <code style="color:#80deea">github.com/ondyari/FaceForensics</code></li>
                <li>Fill out the research access form</li>
                <li>Use the provided download script</li>
                <li>Place videos in <code style="color:#80deea">dataset/real/</code> and <code style="color:#80deea">dataset/fake/</code></li>
                <li>Run <code style="color:#80deea">python train.py --prepare</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("### 🛠️ Technology Stack")
    
    tech_groups = {
        "Deep Learning": [
            ("TensorFlow 2.x", "Core DL framework"),
            ("Keras API",      "Model building & training"),
            ("EfficientNetB0", "CNN backbone (ImageNet pretrained)"),
            ("Bidirectional LSTM", "Temporal sequence modeling"),
        ],
        "Computer Vision": [
            ("OpenCV",       "Video I/O and image processing"),
            ("MediaPipe",    "Real-time face detection & landmarks"),
            ("Grad-CAM",     "Explainability heatmaps"),
        ],
        "Application": [
            ("Streamlit",    "Interactive web dashboard"),
            ("FastAPI",      "REST API endpoint"),
            ("fpdf2",        "PDF report generation"),
            ("Plotly",       "Interactive charts"),
        ],
        "Data": [
            ("NumPy",    "Numerical computing"),
            ("Pandas",   "Data manipulation & analytics"),
            ("Scikit-learn", "Evaluation metrics"),
        ],
    }
    
    cols = st.columns(len(tech_groups))
    for col, (group, items) in zip(cols, tech_groups.items()):
        items_html = "".join(
            f'<li><strong style="color:#e0f7fa">{name}</strong><br>'
            f'<span style="font-size:0.75rem; color:#546e7a">{desc}</span></li>'
            for name, desc in items
        )
        col.markdown(f"""
        <div style="
            background:#0d1b2a; border:1px solid #1e3a5f;
            border-radius:10px; padding:1rem;
        ">
            <h5 style="color:#80deea; margin-bottom:0.6rem;">{group}</h5>
            <ul style="color:#90a4ae; font-size:0.82rem; line-height:1.9; padding-left:1.2rem;">
                {items_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture Diagram ───────────────────────────────────────────────────
    st.markdown("### 🏗️ Model Architecture Diagram")
    
    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #0d1b2a, #0a1628);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.82rem;
        color: #80deea;
        line-height: 1.9;
        overflow-x: auto;
    ">
        <pre style="margin:0; color:#80deea;">
┌─────────────────────────────────────────────────────────┐
│              DeepFake Shield - CNN-LSTM                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  INPUT: Video Sequence (batch, 20, 224, 224, 3)          │
│                     │                                    │
│    ┌────────────────▼────────────────┐                  │
│    │   TimeDistributed CNN           │                   │
│    │   EfficientNetB0 (pretrained)   │                   │
│    │   → GlobalAvgPool2D             │                   │
│    │   → Dense(512) + BN + Drop(0.3) │                   │
│    └────────────────┬────────────────┘                  │
│                     │ (batch, 20, 512)                   │
│    ┌────────────────▼────────────────┐                  │
│    │  Bidirectional LSTM (256)        │                  │
│    │  + LayerNorm                     │                  │
│    └────────────────┬────────────────┘                  │
│                     │ (batch, 20, 512)                   │
│    ┌────────────────▼────────────────┐                  │
│    │  LSTM (128) + LayerNorm          │                  │
│    └────────────────┬────────────────┘                  │
│                     │ (batch, 128)                       │
│    ┌────────────────▼────────────────┐                  │
│    │  Dense(256) + BN + Dropout(0.4) │                  │
│    │  Dense(64)  + Dropout(0.2)      │                  │
│    │  Dense(1) → Sigmoid             │                  │
│    └────────────────┬────────────────┘                  │
│                     │                                    │
│  OUTPUT:  0.0 = REAL │ 1.0 = FAKE                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
        </pre>
    </div>
    """, unsafe_allow_html=True)

    # ── References ────────────────────────────────────────────────────────────
    st.markdown("### 📚 References & Citations")
    
    refs = [
        ("FaceForensics++",
         "Rössler et al., ICCV 2019",
         "Learning to Detect Manipulated Facial Images"),
        ("MesoNet",
         "Afchar et al., WIFS 2018",
         "MesoNet: a Compact Facial Video Forgery Detection Network"),
        ("Grad-CAM",
         "Selvaraju et al., ICCV 2017",
         "Visual Explanations from Deep Networks via Gradient-based Localization"),
        ("EfficientNet",
         "Tan & Le, ICML 2019",
         "EfficientNet: Rethinking Model Scaling for CNNs"),
        ("MediaPipe",
         "Lugaresi et al., 2019",
         "MediaPipe: A Framework for Perceiving and Processing Reality"),
    ]
    
    for name, authors, title in refs:
        st.markdown(f"""
        <div style="
            background:#0d1b2a; border-left:3px solid #80deea;
            border-radius:0 8px 8px 0; padding:0.7rem 1rem;
            margin-bottom:0.5rem;
        ">
            <strong style="color:#e0f7fa">{name}</strong>
            <span style="color:#546e7a; font-size:0.8rem; margin-left:0.5rem">[{authors}]</span><br>
            <span style="color:#78909c; font-size:0.82rem; font-style:italic">{title}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── License & Disclaimer ───────────────────────────────────────────────────
    st.markdown("### ⚖️ License & Disclaimer")
    st.markdown("""
    <div style="
        background:linear-gradient(145deg,#0d1b2a,#1a1a0d);
        border:1px solid #f57f17; border-radius:10px; padding:1rem 1.3rem;
        font-size:0.85rem; color:#ffa726; line-height:1.7;
    ">
        <strong>⚠️ Research & Educational Use Only</strong><br>
        This system is provided for research, education, and awareness purposes. 
        The authors do not accept responsibility for misuse. Detection results should 
        always be reviewed by qualified professionals. No AI system is 100% accurate — 
        false positives and false negatives occur. Use responsibly and ethically.
    </div>
    """, unsafe_allow_html=True)
