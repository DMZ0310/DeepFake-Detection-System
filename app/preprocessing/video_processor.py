"""
DeepFake Detection System - Video Preprocessor
================================================
Handles video loading, frame extraction, face detection,
face cropping, and preprocessing pipeline.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import tempfile
import os

from app.utils.logger import get_logger
from app.config import (
    FRAME_SIZE, MAX_FRAMES, MIN_FACE_SIZE,
    FACE_PADDING, OUTPUTS_DIR, SEQUENCE_LENGTH
)

logger = get_logger(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


class VideoPreprocessor:
    """
    End-to-end video preprocessing pipeline for deepfake detection.
    
    Pipeline:
        1. Load video → extract frames
        2. Detect faces in each frame
        3. Crop and align faces
        4. Normalize pixel values
        5. Build temporal sequences for LSTM
    """

    def __init__(
        self,
        frame_size: int = FRAME_SIZE,
        sequence_length: int = SEQUENCE_LENGTH,
        max_frames: int = MAX_FRAMES,
        face_padding: float = FACE_PADDING,
    ):
        self.frame_size = frame_size
        self.sequence_length = sequence_length
        self.max_frames = max_frames
        self.face_padding = face_padding
        
        # MediaPipe detectors (lazy-loaded)
        self._face_detector = None
        self._face_mesh = None

    # ── Face Detector ──────────────────────────────────────────────────────────

    @property
    def face_detector(self):
        if self._face_detector is None:
            self._face_detector = mp_face_detection.FaceDetection(
                model_selection=1,       # full-range model
                min_detection_confidence=0.4
            )
        return self._face_detector

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.4
            )
        return self._face_mesh

    # ── Frame Extraction ───────────────────────────────────────────────────────

    def extract_frames(
        self,
        video_path: str,
        sample_rate: int = 1
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every N-th frame (1 = every frame)
        
        Returns:
            Tuple of (list of BGR frames, video metadata dict)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
            "video_path": str(video_path),
        }
        logger.info(f"Video: {width}x{height} @ {fps:.1f}fps | {total_frames} frames | {duration:.1f}s")

        frames: List[np.ndarray] = []
        frame_idx = 0
        
        # Adaptive sampling to stay within max_frames budget
        if total_frames > 0 and (total_frames // max(sample_rate, 1)) > self.max_frames:
            sample_rate = max(1, total_frames // self.max_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frames.append(frame)
            frame_idx += 1
            if len(frames) >= self.max_frames:
                break

        cap.release()
        logger.info(f"Extracted {len(frames)} frames (sample_rate={sample_rate})")
        metadata["extracted_frames"] = len(frames)
        return frames, metadata

    # ── Face Detection ─────────────────────────────────────────────────────────

    def detect_face(
        self,
        frame: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the largest face in a frame.
        
        Args:
            frame: BGR image array
        
        Returns:
            Bounding box (x, y, w, h) or None if no face found
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb)
        
        if not results.detections:
            return None

        # Pick detection with highest confidence
        best = max(results.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box

        # Convert relative → absolute with padding
        pad = self.face_padding
        x = max(0, int((bb.xmin - pad) * w))
        y = max(0, int((bb.ymin - pad) * h))
        x2 = min(w, int((bb.xmin + bb.width  + pad) * w))
        y2 = min(h, int((bb.ymin + bb.height + pad) * h))
        bw, bh = x2 - x, y2 - y

        if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
            return None
        return (x, y, bw, bh)

    def crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """Crop and resize a face region to target size."""
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (self.frame_size, self.frame_size),
                          interpolation=cv2.INTER_AREA)
        return face

    # ── Landmark Extraction ────────────────────────────────────────────────────

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 478 facial landmarks using MediaPipe FaceMesh.
        
        Returns:
            (478, 3) float32 array of (x, y, z) normalized coords, or None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    # ── Full Pipeline ──────────────────────────────────────────────────────────

    def process_video(
        self,
        video_path: str,
        save_faces: bool = False,
        progress_callback=None
    ) -> Dict:
        """
        Full preprocessing pipeline: video → face sequences ready for model.
        
        Args:
            video_path: Path to the input video
            save_faces: Whether to save cropped face images to disk
            progress_callback: Optional callable(progress: float, status: str)
        
        Returns:
            dict with keys:
                - face_sequences: np.ndarray (N, seq_len, H, W, 3) normalized
                - face_images: list of cropped face arrays (for display)
                - landmarks: list of landmark arrays
                - metadata: video metadata dict
                - frames_with_faces: count of frames where face was detected
                - total_frames_processed: total frames examined
        """
        def _cb(p, s):
            if progress_callback:
                progress_callback(p, s)
            logger.debug(f"[{p:.0%}] {s}")

        _cb(0.0, "Loading video…")
        frames, metadata = self.extract_frames(video_path)
        
        if not frames:
            raise ValueError("No frames could be extracted from the video.")

        face_images: List[np.ndarray] = []
        landmarks_list: List[Optional[np.ndarray]] = []
        
        total = len(frames)
        _cb(0.1, f"Processing {total} frames…")

        for i, frame in enumerate(frames):
            if i % max(1, total // 20) == 0:
                _cb(0.1 + 0.6 * (i / total), f"Detecting faces… frame {i+1}/{total}")
            
            bbox = self.detect_face(frame)
            if bbox:
                face = self.crop_face(frame, bbox)
                face_images.append(face)
                
                # Optionally extract landmarks from the face crop
                lm = self.extract_landmarks(face)
                landmarks_list.append(lm)

                if save_faces:
                    out_path = OUTPUTS_DIR / "faces" / f"face_{i:05d}.jpg"
                    cv2.imwrite(str(out_path), face)
            else:
                # Fallback: resize whole frame if no face found
                fallback = cv2.resize(frame, (self.frame_size, self.frame_size))
                face_images.append(fallback)
                landmarks_list.append(None)

        _cb(0.75, "Building temporal sequences…")
        sequences = self._build_sequences(face_images)
        
        _cb(0.95, "Finalizing…")
        metadata["frames_with_faces"] = sum(1 for l in landmarks_list if l is not None)
        metadata["total_frames_processed"] = len(frames)

        logger.info(
            f"Pipeline complete: {len(face_images)} faces → "
            f"{len(sequences)} sequences of length {self.sequence_length}"
        )

        _cb(1.0, "Done.")
        return {
            "face_sequences": sequences,
            "face_images": face_images,
            "landmarks": landmarks_list,
            "metadata": metadata,
            "frames_with_faces": metadata["frames_with_faces"],
            "total_frames_processed": len(frames),
        }

    def _build_sequences(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Convert a list of face images into overlapping temporal sequences.
        
        Args:
            face_images: List of (H, W, 3) uint8 arrays
        
        Returns:
            np.ndarray of shape (N, sequence_length, H, W, 3) float32 in [0, 1]
        """
        n = len(face_images)
        seq_len = self.sequence_length

        if n == 0:
            return np.zeros((1, seq_len, self.frame_size, self.frame_size, 3), dtype=np.float32)

        # Pad if fewer frames than one sequence
        if n < seq_len:
            pad = [face_images[-1]] * (seq_len - n)
            face_images = face_images + pad
            n = seq_len

        # Normalize
        norm = [f.astype(np.float32) / 255.0 for f in face_images]

        # Sliding window with stride 1
        sequences = []
        for start in range(n - seq_len + 1):
            seq = np.stack(norm[start : start + seq_len], axis=0)  # (seq_len, H, W, 3)
            sequences.append(seq)

        return np.array(sequences, dtype=np.float32)  # (N, seq_len, H, W, 3)

    # ── Utility ────────────────────────────────────────────────────────────────

    def draw_detection(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        label: str = "",
        confidence: float = 0.0,
        is_fake: bool = False,
    ) -> np.ndarray:
        """Draw detection bounding box and label on a frame."""
        out = frame.copy()
        color = (0, 80, 255) if is_fake else (0, 230, 100)   # BGR

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
            
            tag = f"{label} {confidence:.1%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(out, tag, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return out

    def __del__(self):
        if self._face_detector:
            self._face_detector.close()
        if self._face_mesh:
            self._face_mesh.close()
