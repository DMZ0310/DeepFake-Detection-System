"""
DeepFake Detection System - Standalone Prediction Script
=========================================================
Usage:
    python predict.py --video path/to/video.mp4
    python predict.py --video video.mp4 --threshold 0.6
    python predict.py --video video.mp4 --output results.json
    python predict.py --video video.mp4 --report pdf
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.models.predictor import DeepFakePredictor
from app.utils.report_generator import DetectionReport
from app.utils.logger import get_logger
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD, LOGS_DIR

logger = get_logger(__name__, log_file=str(LOGS_DIR / "predict.log"))


def predict(args):
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    logger.info(f"Analyzing: {video_path}")

    predictor = DeepFakePredictor(
        model_path=args.model,
        threshold=args.threshold,
        lightweight=not args.full_model,
    )

    def _progress(p, s):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"\r  [{bar}] {p:.0%}  {s:<40}", end="", flush=True)

    result = predictor.predict(str(video_path), progress_callback=_progress)
    print()  # newline after progress bar

    # ── Print Summary ──────────────────────────────────────────────────────
    v     = result["verdict"]
    conf  = result["confidence"]
    score = result["fake_score"]
    time_ = result["processing_time_sec"]

    border = "=" * 55
    print(f"\n{border}")
    print(f"  DeepFake Shield — Detection Result")
    print(f"{border}")
    print(f"  File      : {video_path.name}")
    print(f"  Verdict   : {'🚨 ' if v=='FAKE' else '✅ '}{v}")
    print(f"  Confidence: {conf:.1%}")
    print(f"  Fake Score: {score:.4f}")
    print(f"  Threshold : {result['threshold']}")
    print(f"  Frames    : {result['total_frames']} analyzed, {result['frames_with_faces']} with faces")
    print(f"  Duration  : {result['metadata'].get('duration_sec', 0):.1f}s video")
    print(f"  Proc Time : {time_}s")
    print(f"{border}\n")

    # ── Save Output ────────────────────────────────────────────────────────
    report = DetectionReport(result, video_filename=video_path.name)

    if args.output:
        out_path = args.output
        if out_path.endswith(".json") or args.report == "json":
            saved = report.save_json(out_path)
        else:
            saved = report.save_json(out_path)
        print(f"  JSON saved → {saved}")

    if args.report == "pdf":
        pdf_path = report.save_pdf()
        print(f"  PDF saved  → {pdf_path}")
    elif args.report == "json" and not args.output:
        json_path = report.save_json()
        print(f"  JSON saved → {json_path}")

    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="DeepFake Shield — Video Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --video my_video.mp4
  python predict.py --video my_video.mp4 --threshold 0.6 --report pdf
  python predict.py --video my_video.mp4 --output results/result.json
        """
    )
    p.add_argument("--video",       required=True, help="Path to input video file")
    p.add_argument("--model",       default=MODEL_PATH, help="Path to .h5 model weights")
    p.add_argument("--threshold",   type=float, default=CONFIDENCE_THRESHOLD,
                   help="Detection threshold (0.0–1.0, default 0.5)")
    p.add_argument("--output",      default=None, help="Save result to JSON file")
    p.add_argument("--report",      choices=["json", "pdf", "none"], default="none",
                   help="Generate report: json | pdf | none")
    p.add_argument("--full-model",  action="store_true", dest="full_model",
                   help="Use full EfficientNet model (slower, more accurate)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict(args)
    # Exit code: 1 = FAKE, 0 = REAL (useful for shell scripting)
    sys.exit(1 if result["is_fake"] else 0)
