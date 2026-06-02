"""
DeepFake Detection System - Upload & Detect Page
"""

import streamlit as st
import tempfile
import time
import os
import numpy as np
from pathlib import Path

from app.models.predictor import DeepFakePredictor
from app.utils.visualizer import (
    plot_frame_timeline,
    plot_confidence_gauge,
    plot_score_distribution,
    create_face_grid,
)
from app.utils.report_generator import DetectionReport
from app.config import SUPPORTED_FORMATS, MAX_UPLOAD_SIZE_MB


# ── Predictor Singleton ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_predictor() -> DeepFakePredictor:
    return DeepFakePredictor(lightweight=True)


# ── Page ───────────────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <h2 style="color:#80deea; font-weight:800; margin-bottom:0.2rem;">
        🔍 Upload &amp; Detect
    </h2>
    <p style="color:#607d8b; margin-bottom:1.5rem;">
        Upload a video and our AI will analyze it for DeepFake manipulation.
    </p>
    """, unsafe_allow_html=True)

    # ── Upload Section ────────────────────────────────────────────────────────
    col_upload, col_settings = st.columns([2, 1])

    with col_settings:
        st.markdown("#### ⚙️ Detection Settings")
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.3, max_value=0.8,
            value=0.5, step=0.05,
            help="Lower = more sensitive (more fake detections), Higher = more conservative"
        )
        show_faces    = st.checkbox("Show extracted faces", value=True)
        show_timeline = st.checkbox("Show frame timeline",  value=True)
        show_heatmap  = st.checkbox("Show score distribution", value=True)

    with col_upload:
        st.markdown("#### 📂 Select Video File")
        uploaded = st.file_uploader(
            label="",
            type=SUPPORTED_FORMATS,
            help=f"Max size: {MAX_UPLOAD_SIZE_MB} MB. Supported: {', '.join(SUPPORTED_FORMATS).upper()}"
        )

    # ── Preview & Analyze ─────────────────────────────────────────────────────
    if uploaded is not None:
        file_size_mb = uploaded.size / (1024 * 1024)
        
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB")
            return

        # File info bar
        st.markdown(f"""
        <div style="
            background:#0d1b2a; border:1px solid #1e3a5f; border-radius:8px;
            padding:0.7rem 1rem; margin:0.5rem 0;
            display:flex; align-items:center; gap:1rem; font-size:0.85rem;
        ">
            <span>📁 <strong style="color:#e0f7fa">{uploaded.name}</strong></span>
            <span style="color:#607d8b">|</span>
            <span style="color:#80deea">{file_size_mb:.1f} MB</span>
            <span style="color:#607d8b">|</span>
            <span style="color:#a5d6a7">{uploaded.type or 'video'}</span>
        </div>
        """, unsafe_allow_html=True)

        # Video preview
        st.video(uploaded)

        # Analyze button
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button(
            "🚀 Analyze for DeepFakes",
            type="primary",
            use_container_width=True,
        )

        if analyze_btn:
            _run_analysis(uploaded, threshold, show_faces, show_timeline, show_heatmap)
    else:
        # Placeholder
        st.markdown("""
        <div style="
            border: 2px dashed #1e3a5f;
            border-radius: 14px;
            padding: 3rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size:3rem; margin-bottom:1rem;">🎬</div>
            <p style="color:#546e7a; font-size:1rem;">
                Drag and drop a video file here, or click <strong>Browse files</strong>
            </p>
            <p style="color:#37474f; font-size:0.8rem;">
                Supports MP4, AVI, MOV, MKV, WEBM up to {max_mb} MB
            </p>
        </div>
        """.replace("{max_mb}", str(MAX_UPLOAD_SIZE_MB)), unsafe_allow_html=True)


# ── Analysis Runner ───────────────────────────────────────────────────────────

def _run_analysis(uploaded_file, threshold: float, show_faces: bool,
                  show_timeline: bool, show_heatmap: bool):
    """Execute the full detection pipeline and render results."""

    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Progress UI
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def _cb(p, s):
        progress_bar.progress(min(p, 1.0))
        status_text.markdown(
            f"<span style='color:#80deea; font-size:0.85rem;'>⏳ {s}</span>",
            unsafe_allow_html=True
        )

    try:
        predictor = get_predictor()
        predictor.threshold = threshold

        result = predictor.predict(tmp_path, progress_callback=_cb)

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return
    finally:
        os.unlink(tmp_path)
        progress_bar.empty()
        status_text.empty()

    # ── Store in session state for Results page ────────────────────────────
    st.session_state["last_result"]   = result
    st.session_state["last_filename"] = uploaded_file.name

    # ── Results Display ────────────────────────────────────────────────────
    _render_results(result, uploaded_file.name, show_faces, show_timeline, show_heatmap)


def _render_results(result: dict, filename: str, show_faces: bool,
                    show_timeline: bool, show_heatmap: bool):
    """Render detection results inline."""

    verdict    = result["verdict"]
    confidence = result["confidence"]
    is_fake    = result["is_fake"]
    color      = result["color"]

    # ── Verdict Banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="
        background: {'linear-gradient(135deg, #1a0a0a, #2d0f0f)' if is_fake else 'linear-gradient(135deg, #0a1a0d, #0d2e14)'};
        border: 2px solid {color};
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        text-align: center;
    ">
        <div style="font-size:3rem; margin-bottom:0.3rem;">
            {'🚨' if is_fake else '✅'}
        </div>
        <h2 style="color:{color}; font-size:2.5rem; font-weight:900; margin:0; letter-spacing:2px;">
            {verdict}
        </h2>
        <p style="color:#b0bec5; margin:0.3rem 0 0; font-size:1rem;">
            {confidence*100:.1f}% confidence · {result['processing_time_sec']:.1f}s analysis time
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Row ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Fake Score",      f"{result['fake_score']:.4f}"),
        ("Frames Analyzed", str(result["total_frames"])),
        ("Faces Found",     str(result["frames_with_faces"])),
        ("Duration",        f"{result['metadata'].get('duration_sec', 0):.1f}s"),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], metrics):
        col.metric(label, value)

    # ── Charts ─────────────────────────────────────────────────────────────
    chart_cols = st.columns([1, 1])

    with chart_cols[0]:
        gauge = plot_confidence_gauge(confidence, verdict, result["fake_score"])
        st.plotly_chart(gauge, use_container_width=True)

    with chart_cols[1]:
        if show_heatmap and result.get("per_sequence_scores"):
            dist = plot_score_distribution(result["per_sequence_scores"])
            st.plotly_chart(dist, use_container_width=True)

    if show_timeline and result.get("frame_scores"):
        timeline = plot_frame_timeline(result["frame_scores"], threshold=result["threshold"])
        st.plotly_chart(timeline, use_container_width=True)

    # ── Face Grid ──────────────────────────────────────────────────────────
    if show_faces and result.get("face_images"):
        st.markdown("#### 🖼️ Extracted Face Crops")
        grid_img = create_face_grid(result["face_images"], max_faces=12)
        if grid_img:
            st.image(grid_img, use_container_width=True, caption="Sample extracted faces (uniform scale)")

    # ── Report Download ────────────────────────────────────────────────────
    st.markdown("#### 📄 Download Report")
    dl1, dl2 = st.columns(2)

    report = DetectionReport(result, video_filename=filename)

    with dl1:
        import json
        json_str = json.dumps(report.to_dict(), indent=2, default=str)
        st.download_button(
            "⬇️ Download JSON Report",
            data=json_str,
            file_name=f"{report.report_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dl2:
        try:
            pdf_path = report.save_pdf()
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"{report.report_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF generation unavailable: {e}")
