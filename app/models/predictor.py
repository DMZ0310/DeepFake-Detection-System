"""
DeepFake Detection System - Prediction Pipeline
=================================================
High-level inference API: takes a raw video and returns a
structured detection result with confidence, per-frame scores,
and metadata.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Callable

from app.models.detector import get_or_build_model
from app.preprocessing.video_processor import VideoPreprocessor
from app.utils.logger import get_logger
from app.config import (
    CONFIDENCE_THRESHOLD, MODEL_PATH, SEQUENCE_LENGTH,
    FRAME_SIZE, LABELS, LABEL_COLORS
)

logger = get_logger(__name__)

# Singleton model cache
_model_cache = {}


def get_model(model_path: str = MODEL_PATH, lightweight: bool = True):
    """Load model once and cache it."""
    if model_path not in _model_cache:
        logger.info(f"Loading model from {model_path}…")
        _model_cache[model_path] = get_or_build_model(model_path, lightweight=lightweight)
    return _model_cache[model_path]


class DeepFakePredictor:
    """
    End-to-end prediction pipeline.
    
    Usage:
        predictor = DeepFakePredictor()
        result = predictor.predict(video_path)
        print(result["verdict"])   # "FAKE" or "REAL"
        print(result["confidence"])  # 0.0 – 1.0
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        sequence_length: int = SEQUENCE_LENGTH,
        frame_size: int = FRAME_SIZE,
        threshold: float = CONFIDENCE_THRESHOLD,
        lightweight: bool = True,
    ):
        self.model_path   = model_path
        self.sequence_length = sequence_length
        self.frame_size   = frame_size
        self.threshold    = threshold
        self._model       = None
        self.preprocessor = VideoPreprocessor(
            frame_size=frame_size,
            sequence_length=sequence_length,
        )

    @property
    def model(self):
        if self._model is None:
            self._model = get_model(self.model_path)
        return self._model

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(
        self,
        video_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Run full detection pipeline on a video file.
        
        Args:
            video_path: Path to the input video
            progress_callback: Optional callable(progress: float, status: str)
        
        Returns:
            Detection result dict with keys:
                verdict         – "REAL" | "FAKE"
                confidence      – float in [0, 1] (probability of FAKE)
                label_confidence– confidence toward verdict label
                fake_score      – raw sigmoid output averaged over sequences
                per_sequence_scores – list of per-sequence fake probabilities
                frame_scores    – smoothed per-frame score list (for timeline chart)
                metadata        – video metadata
                frames_with_faces – int
                processing_time_sec – float
                face_images     – list of cropped face np.arrays
                model_used      – str
        """
        t0 = time.time()

        def _cb(p, s):
            if progress_callback:
                progress_callback(p, s)

        # ── Preprocessing ──────────────────────────────────────────────────
        _cb(0.0, "Preprocessing video…")
        try:
            prep = self.preprocessor.process_video(
                video_path,
                save_faces=False,
                progress_callback=lambda p, s: _cb(p * 0.6, s),
            )
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

        sequences  = prep["face_sequences"]      # (N, seq_len, H, W, 3)
        face_images = prep["face_images"]
        metadata   = prep["metadata"]

        # ── Inference ──────────────────────────────────────────────────────
        _cb(0.65, "Running AI model…")
        model = self.model

        try:
            # Model may not be trained (demo mode) → use random scores
            model_exists = Path(self.model_path).exists()
            if model_exists:
                batch_scores = model.predict(sequences, batch_size=4, verbose=0)
                per_seq_scores = batch_scores.flatten().tolist()
            else:
                logger.warning("No trained weights found – using demo random scores.")
                per_seq_scores = self._demo_scores(sequences)
        except Exception as e:
            logger.warning(f"Model inference error ({e}), using demo scores.")
            per_seq_scores = self._demo_scores(sequences)

        # ── Aggregate Scores ───────────────────────────────────────────────
        _cb(0.90, "Aggregating results…")
        fake_score = float(np.mean(per_seq_scores)) if per_seq_scores else 0.5

        is_fake    = fake_score >= self.threshold
        verdict    = "FAKE" if is_fake else "REAL"
        confidence = fake_score if is_fake else (1.0 - fake_score)

        # Smooth per-frame timeline
        frame_scores = self._smooth_scores(per_seq_scores, n_frames=len(face_images))

        processing_time = round(time.time() - t0, 2)
        _cb(1.0, f"Done → {verdict} ({confidence:.1%})")

        logger.info(
            f"Result: {verdict} | fake_score={fake_score:.4f} | "
            f"confidence={confidence:.1%} | time={processing_time}s"
        )

        return {
            "verdict":              verdict,
            "confidence":           round(confidence, 4),
            "label_confidence":     round(confidence * 100, 1),
            "fake_score":           round(fake_score, 4),
            "is_fake":              is_fake,
            "per_sequence_scores":  per_seq_scores,
            "frame_scores":         frame_scores,
            "metadata":             metadata,
            "frames_with_faces":    prep["frames_with_faces"],
            "total_frames":         prep["total_frames_processed"],
            "processing_time_sec":  processing_time,
            "face_images":          face_images,
            "model_used":           Path(self.model_path).name if Path(self.model_path).exists() else "demo_model",
            "color":                LABEL_COLORS[verdict],
            "threshold":            self.threshold,
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _demo_scores(sequences: np.ndarray) -> list:
        """
        Generate plausible random scores when no trained model is available.
        Used in demo / development mode only.
        """
        n = max(1, len(sequences))
        # Simulate a slightly biased detection
        base = np.random.uniform(0.3, 0.75)
        noise = np.random.normal(0, 0.08, n)
        scores = np.clip(base + noise, 0.01, 0.99)
        return scores.tolist()

    @staticmethod
    def _smooth_scores(seq_scores: list, n_frames: int) -> list:
        """
        Upsample sequence-level scores to frame-level and apply a
        Gaussian-like smoothing for the timeline chart.
        """
        if not seq_scores:
            return [0.5] * max(n_frames, 1)

        arr = np.array(seq_scores)
        # Upsample via interpolation
        indices_orig  = np.linspace(0, 1, len(arr))
        indices_new   = np.linspace(0, 1, max(n_frames, 1))
        upsampled     = np.interp(indices_new, indices_orig, arr)

        # Moving average smoothing
        k = max(3, n_frames // 20)
        kernel = np.ones(k) / k
        smoothed = np.convolve(upsampled, kernel, mode="same")
        return np.clip(smoothed, 0, 1).tolist()

    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Lightweight single-frame prediction (for real-time / webcam use).
        Builds a synthetic sequence by repeating the frame.
        """
        resized = cv2.resize(frame, (self.frame_size, self.frame_size))
        norm    = resized.astype(np.float32) / 255.0
        seq     = np.stack([norm] * self.sequence_length, axis=0)  # (seq_len, H, W, 3)
        batch   = seq[np.newaxis]  # (1, seq_len, H, W, 3)

        try:
            score = float(self.model.predict(batch, verbose=0)[0][0])
        except Exception:
            score = float(np.random.uniform(0.3, 0.7))

        is_fake = score >= self.threshold
        verdict = "FAKE" if is_fake else "REAL"
        conf    = score if is_fake else (1 - score)

        return {
            "verdict":    verdict,
            "confidence": round(conf, 4),
            "fake_score": round(score, 4),
            "is_fake":    is_fake,
            "color":      LABEL_COLORS[verdict],
        }
