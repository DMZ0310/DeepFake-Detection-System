"""
DeepFake Detection System - Report Generator
=============================================
Generates PDF and JSON detection reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fpdf import FPDF

from app.utils.logger import get_logger
from app.config import REPORTS_DIR, VERSION

logger = get_logger(__name__)


class DetectionReport:
    """Generate formatted detection reports in PDF and JSON formats."""

    def __init__(self, result: Dict, video_filename: str = "video.mp4"):
        self.result = result
        self.video_filename = video_filename
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_id = f"DFD-{int(time.time())}"

    def to_dict(self) -> Dict:
        """Serialize result to a clean JSON-friendly dict."""
        return {
            "report_id":       self.report_id,
            "timestamp":       self.timestamp,
            "system_version":  VERSION,
            "video_file":      self.video_filename,
            "verdict":         self.result.get("verdict"),
            "confidence":      self.result.get("confidence"),
            "fake_score":      self.result.get("fake_score"),
            "threshold":       self.result.get("threshold"),
            "model_used":      self.result.get("model_used"),
            "processing_time": self.result.get("processing_time_sec"),
            "video_metadata":  self.result.get("metadata", {}),
            "frames_analyzed": self.result.get("total_frames"),
            "faces_detected":  self.result.get("frames_with_faces"),
            "summary": self._build_summary(),
        }

    def _build_summary(self) -> str:
        v = self.result
        return (
            f"The analyzed video '{self.video_filename}' was classified as "
            f"{v.get('verdict', 'UNKNOWN')} with {v.get('confidence', 0)*100:.1f}% confidence. "
            f"A total of {v.get('total_frames', 0)} frames were processed, "
            f"with {v.get('frames_with_faces', 0)} frames containing detectable faces. "
            f"Processing completed in {v.get('processing_time_sec', 0):.1f} seconds."
        )

    def save_json(self, path: Optional[str] = None) -> str:
        path = path or str(REPORTS_DIR / f"{self.report_id}.json")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"JSON report saved → {path}")
        return path

    def save_pdf(self, path: Optional[str] = None) -> str:
        """Generate a PDF report using fpdf2."""
        path = path or str(REPORTS_DIR / f"{self.report_id}.pdf")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        v = self.result
        verdict    = v.get("verdict", "UNKNOWN")
        confidence = v.get("confidence", 0)
        meta       = v.get("metadata", {})
        
        verdict_color = (220, 50, 50) if verdict == "FAKE" else (30, 180, 80)

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # ── Header ──────────────────────────────────────────────────────────
        pdf.set_fill_color(10, 14, 26)
        pdf.rect(0, 0, 210, 40, "F")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_xy(10, 8)
        pdf.cell(0, 12, "DeepFake Shield", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_x(10)
        pdf.cell(0, 7, "Detection Report  |  AI-Powered Video Analysis", ln=True)

        # ── Report Meta ─────────────────────────────────────────────────────
        pdf.set_text_color(40, 40, 40)
        pdf.set_fill_color(240, 242, 250)
        pdf.rect(0, 40, 210, 25, "F")
        pdf.set_xy(10, 44)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(90, 6, f"Report ID:  {self.report_id}", ln=False)
        pdf.cell(0, 6, f"Generated:  {self.timestamp}", ln=True)
        pdf.set_x(10)
        pdf.cell(90, 6, f"File:  {self.video_filename}", ln=False)
        pdf.cell(0, 6, f"System Version:  {VERSION}", ln=True)

        # ── Verdict Banner ──────────────────────────────────────────────────
        pdf.set_xy(10, 72)
        pdf.set_fill_color(*verdict_color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 30)
        pdf.cell(190, 22, f"  {verdict}  --  {confidence*100:.1f}% Confidence",
                 align="C", fill=True, ln=True)

        # ── Details Table ───────────────────────────────────────────────────
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(10, 102)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Detection Details", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_fill_color(245, 247, 255)

        rows = [
            ("Fake Score (raw)",    f"{v.get('fake_score', 0):.4f}"),
            ("Threshold",           f"{v.get('threshold', 0.5):.2f}"),
            ("Model Used",          v.get("model_used", "N/A")),
            ("Processing Time",     f"{v.get('processing_time_sec', 0):.2f}s"),
            ("Frames Analyzed",     str(v.get("total_frames", 0))),
            ("Faces Detected",      str(v.get("frames_with_faces", 0))),
            ("Video Duration",      f"{meta.get('duration_sec', 0):.1f}s"),
            ("Resolution",          f"{meta.get('width', 0)}×{meta.get('height', 0)}"),
            ("FPS",                 f"{meta.get('fps', 0):.1f}"),
        ]

        for i, (k, val) in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                pdf.set_fill_color(245, 247, 255)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_x(10)
            pdf.cell(90, 7, f"  {k}", border=0, fill=True)
            pdf.cell(100, 7, val, border=0, fill=True, ln=True)

        # ── Summary ─────────────────────────────────────────────────────────
        pdf.set_xy(10, pdf.get_y() + 8)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(10)
        pdf.multi_cell(190, 6, self._build_summary())

        # ── Disclaimer ──────────────────────────────────────────────────────
        pdf.set_xy(10, 260)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.multi_cell(190, 5,
            "DISCLAIMER: This report is generated by an automated AI system. "
            "Results should be reviewed by a qualified professional before taking any action. "
            "False positives and false negatives are possible. "
            "DeepFake Shield v" + VERSION
        )

        pdf.output(path)
        logger.info(f"PDF report saved → {path}")
        return path
