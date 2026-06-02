"""
DeepFake Detection System - Analytics Dashboard Page
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

from app.utils.visualizer import BG_DARK, REAL_COLOR, FAKE_COLOR, GRID_COLOR, NEUTRAL


def _generate_demo_history(n: int = 50) -> pd.DataFrame:
    """Generate demo scan history for the dashboard."""
    random.seed(42)
    np.random.seed(42)
    
    base_time = datetime.now() - timedelta(days=30)
    records = []
    
    for i in range(n):
        ts = base_time + timedelta(hours=random.randint(1, 720))
        is_fake = random.random() < 0.42
        score = np.random.beta(8, 2) if is_fake else np.random.beta(2, 8)
        records.append({
            "timestamp":  ts,
            "filename":   f"video_{i:03d}.mp4",
            "verdict":    "FAKE" if is_fake else "REAL",
            "confidence": round(float(score if is_fake else 1 - score), 4),
            "fake_score": round(float(score), 4),
            "duration_s": round(random.uniform(5, 120), 1),
            "frames":     random.randint(50, 500),
            "faces_pct":  round(random.uniform(0.6, 1.0), 2),
            "proc_time":  round(random.uniform(5, 35), 1),
        })
    
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def render():
    st.markdown("""
    <h2 style="color:#80deea; font-weight:800; margin-bottom:0.2rem;">
        📈 Analytics Dashboard
    </h2>
    <p style="color:#607d8b; margin-bottom:1.5rem;">
        Historical analysis trends, detection statistics, and usage insights.
    </p>
    """, unsafe_allow_html=True)

    st.info("📌 Showing **demo analytics** data. Real data populates as you analyze videos.", icon="ℹ️")

    # Load real session data or demo
    df = _get_analytics_data()

    # ── Top KPI Row ───────────────────────────────────────────────────────────
    total     = len(df)
    fake_cnt  = (df["verdict"] == "FAKE").sum()
    real_cnt  = total - fake_cnt
    avg_conf  = df["confidence"].mean()
    avg_time  = df["proc_time"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Scans",      str(total),         "🎬", "#80deea"),
        ("Fake Detected",    str(fake_cnt),       "🚨", FAKE_COLOR),
        ("Real Verified",    str(real_cnt),       "✅", REAL_COLOR),
        ("Avg Confidence",   f"{avg_conf:.1%}",   "🎯", "#ffd740"),
        ("Avg Proc. Time",   f"{avg_time:.1f}s",  "⚡", NEUTRAL),
    ]
    for col, (label, value, icon, color) in zip([k1, k2, k3, k4, k5], kpis):
        col.markdown(f"""
        <div style="
            background: linear-gradient(145deg, #0d1b2a, #1a2744);
            border: 1px solid #1e3a5f;
            border-top: 3px solid {color};
            border-radius: 10px;
            padding: 0.9rem;
            text-align: center;
        ">
            <div style="font-size:1.3rem;">{icon}</div>
            <div style="color:{color}; font-size:1.5rem; font-weight:800;">{value}</div>
            <div style="color:#546e7a; font-size:0.72rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend Chart ───────────────────────────────────────────────────────────
    col_trend, col_pie = st.columns([2, 1])

    with col_trend:
        st.markdown("#### 📅 Daily Detection Trend")
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby(["date", "verdict"]).size().unstack(fill_value=0).reset_index()
        
        trend_fig = go.Figure()
        if "FAKE" in daily.columns:
            trend_fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["FAKE"],
                name="FAKE", line=dict(color=FAKE_COLOR, width=2),
                fill="tozeroy", fillcolor="rgba(255,23,68,0.15)",
            ))
        if "REAL" in daily.columns:
            trend_fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["REAL"],
                name="REAL", line=dict(color=REAL_COLOR, width=2),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.15)",
            ))
        trend_fig.update_layout(
            xaxis=dict(gridcolor=GRID_COLOR, color="#9e9e9e"),
            yaxis=dict(gridcolor=GRID_COLOR, color="#9e9e9e", title="Scans"),
            paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
            font=dict(color="#e0e0e0"), height=280,
            margin=dict(l=10, r=10, t=10, b=40),
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    with col_pie:
        st.markdown("#### 🍩 Detection Split")
        fake_pct = fake_cnt / total * 100 if total > 0 else 50
        real_pct = 100 - fake_pct
        
        pie_fig = go.Figure(go.Pie(
            labels=["FAKE", "REAL"],
            values=[fake_pct, real_pct],
            hole=0.55,
            marker=dict(colors=[FAKE_COLOR, REAL_COLOR]),
            textinfo="label+percent",
            textfont=dict(color="#fff", size=12),
        ))
        pie_fig.add_annotation(
            text=f"<b>{fake_pct:.0f}%</b><br>Fake",
            showarrow=False, font=dict(size=14, color="#e0e0e0"),
        )
        pie_fig.update_layout(
            paper_bgcolor=BG_DARK, font=dict(color="#e0e0e0"),
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # ── Score Distribution + Confidence Histogram ─────────────────────────────
    col_sd, col_ch = st.columns(2)

    with col_sd:
        st.markdown("#### Fake Score Distribution by Class")
        fig = go.Figure()
        for verdict, color in [("FAKE", FAKE_COLOR), ("REAL", REAL_COLOR)]:
            subset = df[df["verdict"] == verdict]["fake_score"]
            fig.add_trace(go.Histogram(
                x=subset, nbinsx=20, name=verdict,
                marker_color=color, opacity=0.7,
            ))
        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="Fake Score", gridcolor=GRID_COLOR, color="#9e9e9e"),
            yaxis=dict(title="Count", gridcolor=GRID_COLOR, color="#9e9e9e"),
            paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
            font=dict(color="#e0e0e0"), height=260,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_ch:
        st.markdown("#### Processing Time Distribution")
        fig2 = go.Figure(go.Histogram(
            x=df["proc_time"], nbinsx=15,
            marker_color=NEUTRAL, opacity=0.8,
        ))
        fig2.update_layout(
            xaxis=dict(title="Seconds", gridcolor=GRID_COLOR, color="#9e9e9e"),
            yaxis=dict(title="Count", gridcolor=GRID_COLOR, color="#9e9e9e"),
            paper_bgcolor=BG_DARK, plot_bgcolor=BG_DARK,
            font=dict(color="#e0e0e0"), height=260,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Recent Scans Table ────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Recent Scan History")
    
    display_df = df.sort_values("timestamp", ascending=False).head(20).copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df["fake_score"] = display_df["fake_score"].apply(lambda x: f"{x:.4f}")
    display_df = display_df.rename(columns={
        "timestamp": "Time", "filename": "File",
        "verdict": "Verdict", "confidence": "Confidence",
        "fake_score": "Fake Score", "proc_time": "Proc (s)",
        "duration_s": "Duration (s)", "frames": "Frames",
    })
    
    st.dataframe(
        display_df[["Time", "File", "Verdict", "Confidence", "Fake Score", "Proc (s)", "Frames"]],
        use_container_width=True,
        hide_index=True,
    )

    # ── Download Analytics ────────────────────────────────────────────────────
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download Analytics CSV",
        data=csv,
        file_name="deepfake_analytics.csv",
        mime="text/csv",
    )


def _get_analytics_data() -> pd.DataFrame:
    """Return real scan history from session state or demo data."""
    # In a production system, this would come from a database
    # For now, use session state + demo
    if "scan_history" in st.session_state and st.session_state["scan_history"]:
        return pd.DataFrame(st.session_state["scan_history"])
    return _generate_demo_history(50)
