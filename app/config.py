"""
DeepFake Detection System - Configuration
==========================================
Central configuration for all system components.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
APP_DIR = BASE_DIR / "app"
DATASET_DIR = BASE_DIR / "dataset"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for d in [DATASET_DIR, OUTPUTS_DIR, REPORTS_DIR, WEIGHTS_DIR, LOGS_DIR,
          OUTPUTS_DIR / "frames", OUTPUTS_DIR / "faces", OUTPUTS_DIR / "reports",
          DATASET_DIR / "real", DATASET_DIR / "fake"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", str(WEIGHTS_DIR / "deepfake_model.h5"))
FRAME_SIZE = int(os.getenv("FRAME_SIZE", 224))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 20))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# CNN feature output shape (before LSTM)
CNN_FEATURE_DIM = 512

# ─── Training Configuration ───────────────────────────────────────────────────
EPOCHS = int(os.getenv("EPOCHS", 50))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", 0.8))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", 0.1))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", 0.1))

# ─── Video Processing ─────────────────────────────────────────────────────────
MAX_FRAMES = 150          # Maximum frames to extract per video
MIN_FACE_SIZE = 60        # Minimum face size in pixels
FACE_PADDING = 0.2        # Padding around detected face (fraction)
SUPPORTED_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 500))

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "zoom_range": 0.1,
    "brightness_range": [0.8, 1.2],
}

# ─── Labels ───────────────────────────────────────────────────────────────────
LABELS = {0: "REAL", 1: "FAKE"}
LABEL_COLORS = {"REAL": "#00ff88", "FAKE": "#ff4757"}

# ─── API Configuration ────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ─── UI Configuration ─────────────────────────────────────────────────────────
APP_TITLE = "DeepFake Shield"
APP_SUBTITLE = "AI-Powered Deepfake Detection & Prevention System"
APP_ICON = "🛡️"
VERSION = "2.0.0"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = str(LOGS_DIR / "deepfake_detection.log")
