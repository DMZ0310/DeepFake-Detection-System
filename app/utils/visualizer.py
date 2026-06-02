"""
DeepFake Detection System - Visualization Utilities
=====================================================
Charts, heatmaps, and visual summaries for the Streamlit dashboard.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional
from io import BytesIO
from PIL import Image

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Color Palette ──────────────────────────────────────────────────────────────
REAL_COLOR  = "#00e676"   # Vivid green
FAKE_COLOR  = "#ff1744"   # Vivid red
NEUTRAL     = "#90caf9"   # Light blue
BG_DARK     = "#0a0e1a"
GRID_COLOR  = "#1e2740"


# ── Frame Timeline ─────────────────────────────────────────────────────────────

def plot_frame_timeline(
    frame_scores: List[float],
    threshold: float = 0.5,
    title: str = "Frame-Level Fake Probability",
) -> go.Figure:
    """
    Interactive line chart showing per-frame fake probability over time.
    """
    n = len(frame_scores)
    x = list(range(n))

    fig = go.Figure()

    # Shaded FAKE zone
    fig.add_hrect(
        y0=threshold, y1=1.0,
        fillcolor=FAKE_COLOR, opacity=0.07,
        line_width=0,
        annotation_text="FAKE ZONE",
        annotation_position="top right",
        annotation_font_color=FAKE_COLOR,
        annotation_font_size=11,
    )

    # Threshold line
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=FAKE_COLOR, opacity=0.6,
        annotation_text=f"Threshold {threshold:.0%}",
        annotation_position="bottom right",
        annotation_font_color=FAKE_COLOR,
    )

    # Score area fill
    fig.add_trace(go.Scatter(
        x=x, y=frame_scores,
        fill="tozeroy",
        fillcolor="rgba(255,23,68,0.15)",
        line=dict(color=FAKE_COLOR, width=2),
        mode="lines",
        name="Fake Probability",
        hovertemplate="Frame %{x}<br>Score: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e0e0e0")),
        xaxis=dict(
            title="Frame Index",
            gridcolor=GRID_COLOR, color="#9e9e9e",
            showline=False,
        ),
        yaxis=dict(
            title="Fake Probability",
            range=[0, 1],
            gridcolor=GRID_COLOR, color="#9e9e9e",
            tickformat=".0%",
        ),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"),
        height=320,
        margin=dict(l=10, r=10, t=50, b=40),
        showlegend=False,
    )
    return fig


# ── Confidence Gauge ───────────────────────────────────────────────────────────

def plot_confidence_gauge(
    confidence: float,
    verdict: str,
    fake_score: float,
) -> go.Figure:
    """
    Circular gauge showing detection confidence.
    """
    color = FAKE_COLOR if verdict == "FAKE" else REAL_COLOR

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(confidence * 100, 1),
        delta={"reference": 50, "valueformat": ".1f"},
        title={"text": f"<b>{verdict}</b>", "font": {"size": 22, "color": color}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#555",
                "tickformat": ".0f",
                "nticks": 5,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": BG_DARK,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(0,230,118,0.08)"},
                {"range": [50, 100],"color": "rgba(255,23,68,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#fff", "width": 2},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ── Score Distribution ─────────────────────────────────────────────────────────

def plot_score_distribution(per_seq_scores: List[float]) -> go.Figure:
    """Histogram of per-sequence fake scores."""
    if not per_seq_scores:
        per_seq_scores = [0.5]

    df = pd.DataFrame({"score": per_seq_scores})
    median_score = float(np.median(per_seq_scores))
    color = FAKE_COLOR if median_score >= 0.5 else REAL_COLOR

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=per_seq_scores,
        nbinsx=20,
        marker_color=color,
        opacity=0.8,
        name="Sequence Scores",
    ))
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="#fff",
        annotation_text="Threshold", annotation_position="top",
    )
    fig.add_vline(
        x=median_score, line_dash="dot", line_color="#ffd740",
        annotation_text=f"Median {median_score:.2f}",
        annotation_position="top right",
        annotation_font_color="#ffd740",
    )
    fig.update_layout(
        title=dict(text="Sequence Score Distribution", font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(title="Fake Score", gridcolor=GRID_COLOR, color="#9e9e9e", range=[0, 1]),
        yaxis=dict(title="Count", gridcolor=GRID_COLOR, color="#9e9e9e"),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"),
        height=280,
        margin=dict(l=10, r=10, t=40, b=40),
        showlegend=False,
    )
    return fig


# ── Face Grid ──────────────────────────────────────────────────────────────────

def create_face_grid(
    face_images: List[np.ndarray],
    max_faces: int = 12,
    cols: int = 6,
    size: int = 96,
) -> Optional[Image.Image]:
    """
    Tile extracted face crops into a single grid image (PIL).
    """
    if not face_images:
        return None

    samples = face_images[::max(1, len(face_images) // max_faces)][:max_faces]
    rows = (len(samples) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * size), color=(15, 20, 35))

    for idx, face in enumerate(samples):
        try:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
            r, c = divmod(idx, cols)
            canvas.paste(pil, (c * size, r * size))
        except Exception as e:
            logger.warning(f"Face grid error at idx {idx}: {e}")

    return canvas


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ["REAL", "FAKE"],
) -> go.Figure:
    """Plot an interactive confusion matrix heatmap."""
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, BG_DARK], [0.5, "#1565c0"], [1, FAKE_COLOR]],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "#fff"},
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(title="Predicted", color="#9e9e9e"),
        yaxis=dict(title="Actual", color="#9e9e9e", autorange="reversed"),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"),
        height=300,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """Plot ROC curve with AUC annotation."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        fill="tozeroy", fillcolor="rgba(144,202,249,0.15)",
        line=dict(color=NEUTRAL, width=2),
        name=f"AUC = {auc:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(dash="dash", color="#555", width=1),
        name="Random Baseline",
    ))
    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(size=14, color="#e0e0e0")),
        xaxis=dict(title="False Positive Rate", gridcolor=GRID_COLOR, color="#9e9e9e"),
        yaxis=dict(title="True Positive Rate", gridcolor=GRID_COLOR, color="#9e9e9e", range=[0, 1.02]),
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"),
        height=350,
        margin=dict(l=10, r=10, t=50, b=50),
    )
    return fig


# ── Training History ───────────────────────────────────────────────────────────

def plot_training_history(history: Dict) -> go.Figure:
    """Plot training vs validation loss and accuracy."""
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "AUC"])

    for label, col, key in [("Train", NEUTRAL, "loss"), ("Val", FAKE_COLOR, "val_loss")]:
        if key in history:
            fig.add_trace(go.Scatter(x=epochs, y=history[key], name=label,
                                     line=dict(color=col)), row=1, col=1)
    for label, col, key in [("Train", NEUTRAL, "auc"), ("Val", FAKE_COLOR, "val_auc")]:
        if key in history:
            fig.add_trace(go.Scatter(x=epochs, y=history[key], name=label,
                                     line=dict(color=col), showlegend=False), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
        font=dict(color="#e0e0e0"), height=320,
        margin=dict(l=10, r=10, t=50, b=30),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(gridcolor=GRID_COLOR, color="#9e9e9e")})
    return fig


# ── GradCAM Overlay ────────────────────────────────────────────────────────────

def overlay_gradcam(face_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a face image.
    
    Args:
        face_img: BGR face image (H, W, 3) uint8
        heatmap:  Float heatmap (h, w) in [0, 1]
        alpha:    Blend factor
    
    Returns:
        Overlay image BGR (H, W, 3) uint8
    """
    heatmap_resized = cv2.resize(heatmap, (face_img.shape[1], face_img.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(face_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
