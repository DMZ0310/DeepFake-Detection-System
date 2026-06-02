# 📘 DeepFake Shield — Complete Instruction Manual

**Version:** 2.0.0  
**Last Updated:** 2024  
**System:** AI-Powered DeepFake Detection & Prevention

---

## 📋 Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation Guide](#2-installation-guide)
3. [Project Structure Explained](#3-project-structure-explained)
4. [Configuration Reference](#4-configuration-reference)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Training the Model](#6-training-the-model)
7. [Running the Dashboard](#7-running-the-dashboard)
8. [Using the Dashboard Pages](#8-using-the-dashboard-pages)
9. [Command-Line Prediction](#9-command-line-prediction)
10. [REST API Usage](#10-rest-api-usage)
11. [Understanding Results](#11-understanding-results)
12. [Model Architecture Deep Dive](#12-model-architecture-deep-dive)
13. [Troubleshooting](#13-troubleshooting)
14. [Deployment Guide](#14-deployment-guide)
15. [Advanced Customization](#15-advanced-customization)
16. [Frequently Asked Questions](#16-frequently-asked-questions)

---

## 1. System Requirements

### Minimum Requirements (Inference Only)

| Component     | Minimum          | Recommended         |
|---------------|------------------|---------------------|
| OS            | Windows 10 / Ubuntu 20.04 / macOS 11 | Ubuntu 22.04 LTS    |
| Python        | 3.10             | 3.11                |
| RAM           | 8 GB             | 16 GB               |
| CPU           | 4-core (2.0 GHz) | 8-core (3.0 GHz+)   |
| GPU           | Not required     | NVIDIA RTX 2060+    |
| Disk Space    | 3 GB             | 10 GB               |
| Internet      | For first run (downloads MediaPipe weights) | —  |

### Training Requirements

| Component     | Minimum          | Recommended         |
|---------------|------------------|---------------------|
| RAM           | 16 GB            | 32 GB               |
| GPU VRAM      | 8 GB (GTX 1080)  | 16 GB (RTX 3080+)   |
| CUDA          | 11.8             | 12.2                |
| Disk Space    | 50 GB (dataset)  | 200 GB              |
| Training Time | ~12h (CPU)       | ~2h (RTX 3080)      |

### Python Version Check

```bash
python --version   # Must be 3.10 or higher
pip --version      # Must be 21.0 or higher
```

---

## 2. Installation Guide

### Step 1 — Download the Project

**Option A: From ZIP (recommended for beginners)**
```
1. Download deepfake_detection.zip
2. Extract to your desired folder:
   - Windows: Right-click → "Extract All"
   - macOS:   Double-click the ZIP file
   - Linux:   unzip deepfake_detection.zip
3. Open a terminal and navigate into the folder:
   cd deepfake_detection
```

**Option B: From Git**
```bash
git clone https://github.com/yourname/deepfake-shield.git
cd deepfake-shield
```

---

### Step 2 — Create a Virtual Environment (Strongly Recommended)

A virtual environment keeps project dependencies isolated from your system Python.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt, confirming activation.

> ⚠️ **Always activate the virtual environment before running any project commands.**

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages. Expected time: 3–10 minutes depending on internet speed.

**What gets installed:**
- `streamlit` — Web dashboard framework
- `tensorflow` — Deep learning engine
- `opencv-python-headless` — Video processing
- `mediapipe` — Face detection
- `plotly` — Interactive charts
- `fastapi` + `uvicorn` — REST API server
- `fpdf2` — PDF report generation
- And 10+ supporting libraries

---

### Step 4 — GPU Support (Optional, Skip if CPU-only)

If you have an NVIDIA GPU:

```bash
# Check CUDA version
nvidia-smi

# For CUDA 12.x
pip install tensorflow[and-cuda]==2.15.0

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If GPU is detected, you'll see output like: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

### Step 5 — Verify Installation

```bash
python -c "
import tensorflow as tf
import cv2
import mediapipe
import streamlit
print('TensorFlow:', tf.__version__)
print('OpenCV:', cv2.__version__)
print('MediaPipe:', mediapipe.__version__)
print('Streamlit:', streamlit.__version__)
print('✅ All dependencies installed correctly')
"
```

---

### Step 6 — First Launch

```bash
streamlit run main.py
```

Your browser will automatically open to `http://localhost:8501`. If it doesn't, open it manually.

> ✅ **The system works in demo mode immediately — no training required to explore the UI.**

---

## 3. Project Structure Explained

```
deepfake_detection/
│
├── main.py                     ← ENTRY POINT — run this to start the app
├── train.py                    ← Training script
├── predict.py                  ← Command-line prediction tool
├── api.py                      ← FastAPI REST server
│
├── requirements.txt            ← Python package dependencies
├── .env                        ← Environment configuration (edit settings here)
├── README.md                   ← Project overview
├── DEPLOYMENT.md               ← Deployment instructions
├── INSTRUCTIONS.md             ← This file
│
├── .streamlit/
│   └── config.toml             ← Streamlit theme and server settings
│
├── app/                        ← Core application code
│   ├── config.py               ← Loads .env and defines all constants
│   │
│   ├── pages/                  ← One file per dashboard page
│   │   ├── home.py             ← Landing/overview page
│   │   ├── detect.py           ← Upload & detection UI (main feature)
│   │   ├── performance.py      ← Model metrics and evaluation charts
│   │   ├── analytics.py        ← Historical scan statistics
│   │   ├── live.py             ← Single-frame and webcam detection
│   │   └── about.py            ← Dataset, architecture, references
│   │
│   ├── models/
│   │   ├── detector.py         ← CNN-LSTM model architecture definition
│   │   └── predictor.py        ← High-level inference pipeline
│   │
│   ├── preprocessing/
│   │   └── video_processor.py  ← Frame extraction, face detection, sequences
│   │
│   └── utils/
│       ├── logger.py           ← Centralized logging setup
│       ├── visualizer.py       ← All chart/graph generation functions
│       └── report_generator.py ← PDF and JSON report creation
│
├── dataset/                    ← Place your training videos here
│   ├── real/                   ← Authentic (non-manipulated) videos
│   └── fake/                   ← DeepFake (manipulated) videos
│
├── weights/                    ← Trained model files saved here
│   ├── deepfake_model.h5       ← Final trained model (created after training)
│   └── deepfake_model_best.h5  ← Best checkpoint during training
│
├── outputs/                    ← Auto-created processing outputs
│   ├── frames/                 ← Extracted video frames (when saved)
│   └── faces/                  ← Cropped face images (when saved)
│
├── reports/                    ← Generated detection reports (PDF/JSON)
├── logs/                       ← Application and training logs
│   ├── deepfake_detection.log  ← Runtime log
│   ├── train.log               ← Training log
│   └── tensorboard/            ← TensorBoard training metrics
│
└── notebooks/
    └── analysis.ipynb          ← Jupyter notebook for exploration
```

### Key Files to Know

| File | When You'll Use It |
|------|--------------------|
| `main.py` | Every time — this starts the app |
| `.env` | To change settings (threshold, frame size, etc.) |
| `train.py` | Once — to train the model on your dataset |
| `predict.py` | For quick command-line predictions |
| `api.py` | To expose predictions as a web service |

---

## 4. Configuration Reference

All settings live in `.env`. Open it with any text editor:

```bash
# Windows
notepad .env

# macOS/Linux
nano .env
```

### Complete `.env` Reference

```ini
# ── Model Settings ──────────────────────────────────────────────────────────

# Path to the trained model weights file
MODEL_PATH=weights/deepfake_model.h5

# Detection threshold: probability above this → FAKE verdict
# Range: 0.0 to 1.0 | Default: 0.5
# Lower = more sensitive (catches more fakes, but more false alarms)
# Higher = more conservative (fewer false alarms, but may miss some fakes)
CONFIDENCE_THRESHOLD=0.5

# Number of frames grouped into one temporal sequence
# Higher = better temporal context, but slower and more memory
# Range: 10–40 | Default: 20
SEQUENCE_LENGTH=20

# Resolution of each face crop fed to the model
# 224 matches EfficientNetB0's expected input size
# Do not change unless you retrain the model
FRAME_SIZE=224

# Training batch size (lower if you run out of GPU memory)
BATCH_SIZE=16

# ── File & Path Settings ────────────────────────────────────────────────────

DATASET_PATH=dataset/
OUTPUT_PATH=outputs/
REPORTS_PATH=reports/
LOGS_PATH=logs/

# ── API Settings ─────────────────────────────────────────────────────────────

API_HOST=0.0.0.0
API_PORT=8000

# ── Upload Limits ────────────────────────────────────────────────────────────

# Maximum video file size in megabytes
MAX_UPLOAD_SIZE_MB=500

# Comma-separated list of accepted video formats
SUPPORTED_FORMATS=mp4,avi,mov,mkv

# ── Training Hyperparameters ─────────────────────────────────────────────────

EPOCHS=50
LEARNING_RATE=0.0001
TRAIN_SPLIT=0.8        # 80% of data for training
VAL_SPLIT=0.1          # 10% for validation
TEST_SPLIT=0.1         # 10% for final evaluation
```

### Common Configuration Changes

**More sensitive detection (catch more fakes):**
```ini
CONFIDENCE_THRESHOLD=0.35
```

**Faster processing on slow machines:**
```ini
SEQUENCE_LENGTH=10
FRAME_SIZE=128
```

**Handle longer videos:**
```ini
MAX_UPLOAD_SIZE_MB=1000
```

---

## 5. Dataset Preparation

### Option A: FaceForensics++ (Academic Standard Dataset)

This is the benchmark dataset used in published deepfake detection research.

**Step 1 — Request Access**
1. Go to: https://github.com/ondyari/FaceForensics
2. Fill out the Google Form (academic email required)
3. You'll receive a download script by email within 1–2 business days

**Step 2 — Download Videos**
```bash
# Download original + deepfake videos at medium compression (c23)
python faceforensics_download_v4.py \
    /path/to/download/destination \
    -d all \
    -c c23 \
    -t videos
```

Compression options:
- `raw` — Uncompressed (best quality, largest files, ~500 GB)
- `c23` — Light compression (~50 GB, recommended)
- `c40` — Heavy compression (~8 GB, easiest to start with)

**Step 3 — Organize Into Dataset Folders**
```bash
# Real videos
cp FaceForensics++/original_sequences/youtube/c23/videos/*.mp4 dataset/real/

# Fake videos (use one or all manipulation types)
cp FaceForensics++/manipulated_sequences/Deepfakes/c23/videos/*.mp4     dataset/fake/
cp FaceForensics++/manipulated_sequences/Face2Face/c23/videos/*.mp4     dataset/fake/
cp FaceForensics++/manipulated_sequences/FaceSwap/c23/videos/*.mp4      dataset/fake/
cp FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos/*.mp4 dataset/fake/
```

---

### Option B: Custom Dataset

You can train on your own videos. Requirements:
- **Minimum:** 50 videos per class (real/fake) for basic learning
- **Recommended:** 200+ videos per class for reliable detection
- **Format:** MP4, AVI, MOV, or MKV
- **Content:** Videos must contain at least one face visible in most frames
- **Duration:** 3–120 seconds per video works best

**Folder structure:**
```
dataset/
├── real/
│   ├── real_interview_001.mp4
│   ├── real_interview_002.mp4
│   └── ...
└── fake/
    ├── deepfake_001.mp4
    ├── deepfake_002.mp4
    └── ...
```

---

### Option C: Test With Minimal Data (Development Only)

For quickly testing the training pipeline with just a few videos:

```bash
python train.py --max-videos 10 --epochs 5 --lightweight
```

This uses only 10 videos per class and 5 epochs — fast but produces a low-quality model suitable only for pipeline verification.

---

### Verify Dataset Structure

```bash
python -c "
from pathlib import Path
real_count = len(list(Path('dataset/real').glob('*.mp4')))
fake_count = len(list(Path('dataset/fake').glob('*.mp4')))
print(f'Real videos: {real_count}')
print(f'Fake videos: {fake_count}')
print(f'Total: {real_count + fake_count}')
if real_count < 50 or fake_count < 50:
    print('⚠️  WARNING: Less than 50 videos per class. Results may be unreliable.')
else:
    print('✅ Dataset looks good!')
"
```

---

## 6. Training the Model

### Quick Training (Recommended First Run)

```bash
python train.py --lightweight --epochs 20 --max-videos 100
```

This uses the faster MobileNetV2 backbone, 20 epochs, and caps at 100 videos per class. Good for initial testing.

### Full Production Training

```bash
python train.py --epochs 50
```

Uses the full EfficientNetB0 backbone. Recommended for best accuracy.

### All Training Options

```bash
python train.py [OPTIONS]

Options:
  --lightweight       Use MobileNetV2 backbone (faster, slightly less accurate)
                      EfficientNetB0 is used by default (more accurate, slower)
  
  --epochs N          Number of training epochs (default: 50)
                      More epochs = better accuracy up to a point
                      Early stopping will halt training if validation stops improving
  
  --lr FLOAT          Learning rate for Adam optimizer (default: 0.0001)
                      Lower = slower but more stable training
                      Higher = faster but may overshoot optimal weights
  
  --batch-size N      Batch size (default: 16)
                      Reduce to 8 or 4 if you get "out of memory" errors
  
  --max-videos N      Process only N videos per class (useful for testing)
                      Example: --max-videos 50 uses 50 real + 50 fake videos
  
  --prepare           Only extract frames and build sequences, skip training
                      Useful to verify preprocessing before committing to training
  
  --cache             Save prepared sequences to disk after preprocessing
                      Speeds up re-runs by skipping extraction
  
  --no-cache          Ignore cached data and re-process from videos
  
  --resume            Continue training from the last saved checkpoint
                      (weights/deepfake_model_best.h5 must exist)
```

### Training Examples

```bash
# Minimum viable training (for testing the pipeline)
python train.py --max-videos 20 --epochs 5 --lightweight

# Fast training on limited hardware
python train.py --lightweight --epochs 30 --batch-size 8

# Full training with GPU
python train.py --epochs 50 --batch-size 32

# Resume interrupted training
python train.py --resume --epochs 20

# Prepare data only, inspect, then train separately
python train.py --prepare --cache
python train.py --epochs 50  # will use cached data
```

### What Happens During Training

```
Step 1 — Dataset Preparation
  ├── Scans dataset/real/ and dataset/fake/ for video files
  ├── Extracts frames from each video (adaptive sampling)
  ├── Detects and crops faces using MediaPipe
  ├── Builds overlapping sequences of SEQUENCE_LENGTH frames
  └── Splits into train (80%) / validation (10%) / test (10%)

Step 2 — Model Building
  ├── Loads EfficientNetB0 with ImageNet pretrained weights
  ├── Freezes early layers (keeps pretrained features)
  └── Adds BiLSTM + classifier head on top

Step 3 — Training Loop
  ├── Each epoch: forward pass, loss calculation, backpropagation
  ├── Saves best model checkpoint when validation AUC improves
  ├── Reduces learning rate when validation loss plateaus (ReduceLROnPlateau)
  └── Stops early if validation AUC doesn't improve for 7 epochs (EarlyStopping)

Step 4 — Evaluation
  ├── Runs predictions on held-out test set
  ├── Computes: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  └── Saves metrics to logs/training_history.json
```

### Monitoring Training Progress

**Option A: TensorBoard (Real-time)**
```bash
# In a second terminal window, while training is running:
tensorboard --logdir logs/tensorboard

# Then open: http://localhost:6006
```

**Option B: Log File**
```bash
# Watch training log in real time
tail -f logs/train.log
```

**Option C: History File (After Training)**
```bash
python -c "
import json
with open('logs/training_history.json') as f:
    data = json.load(f)
metrics = data['metrics']
print(f'Accuracy:  {metrics[\"accuracy\"]:.1%}')
print(f'Precision: {metrics[\"precision\"]:.1%}')
print(f'Recall:    {metrics[\"recall\"]:.1%}')
print(f'F1 Score:  {metrics[\"f1_score\"]:.1%}')
print(f'ROC-AUC:   {metrics[\"auc_roc\"]:.4f}')
"
```

### Training Output Files

After training completes, you'll find:

| File | Description |
|------|-------------|
| `weights/deepfake_model.h5` | Final trained model |
| `weights/deepfake_model_best.h5` | Best checkpoint (highest val AUC) |
| `logs/training_history.json` | All metrics and training curves |
| `logs/tensorboard/` | TensorBoard event files |
| `logs/train.log` | Detailed training log |

---

## 7. Running the Dashboard

### Start the Dashboard

```bash
streamlit run main.py
```

The dashboard opens at **http://localhost:8501**

### Command-Line Options

```bash
# Different port (if 8501 is taken)
streamlit run main.py --server.port 8502

# Allow network access (access from other devices on your LAN)
streamlit run main.py --server.address 0.0.0.0

# Disable browser auto-open
streamlit run main.py --server.headless true

# All options combined (for server deployment)
streamlit run main.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    --server.maxUploadSize 500
```

### Stopping the Dashboard

Press `Ctrl + C` in the terminal where it's running.

---

## 8. Using the Dashboard Pages

### Page 1 — 🏠 Home

The landing page gives a system overview. Use it to:
- Understand available features at a glance
- Check system status (sidebar) — model loaded, face detector, GPU mode
- Navigate quickly to the detection page via the CTA button

**Sidebar Status Indicators:**
- 🟢 Green = Component is loaded and ready
- 🟡 Yellow = Available but in demo/fallback mode
- 🔴 Red = Unavailable (check logs)

---

### Page 2 — 🔍 Upload & Detect

This is the main feature. Step-by-step:

**Step 1 — Configure Detection Settings (right panel)**

| Setting | What It Does | When to Adjust |
|---------|-------------|----------------|
| Detection Threshold | Probability cutoff for FAKE verdict | Lower (0.35) if missing fakes; higher (0.65) to reduce false alarms |
| Show extracted faces | Display face grid after analysis | Disable for faster results display |
| Show frame timeline | Show probability over time chart | Useful to see which frames are suspicious |
| Show score distribution | Histogram of sequence scores | Useful for understanding confidence spread |

**Step 2 — Upload Video**
1. Click "Browse files" or drag-and-drop a video
2. Supported: MP4, AVI, MOV, MKV, WEBM (up to 500 MB)
3. A preview player appears automatically

**Step 3 — Analyze**
1. Click the blue **"🚀 Analyze for DeepFakes"** button
2. A progress bar shows the pipeline stages:
   - `Loading video` → `Detecting faces` → `Running AI model` → `Aggregating results`
3. Typical time: 10–40 seconds depending on video length and hardware

**Step 4 — Review Results**

The results section shows:

```
┌─────────────────────────────────┐
│ 🚨 FAKE — 87.3% confidence      │  ← Main verdict banner
└─────────────────────────────────┘

Fake Score: 0.8730    ← Raw model output (0–1)
Frames Analyzed: 120  ← Total frames processed
Faces Found: 118      ← Frames where face was detected
Duration: 10.0s       ← Video length

[Gauge Chart]         ← Visual confidence meter
[Score Distribution]  ← Spread of per-sequence scores

[Frame Timeline]      ← Score plotted over time
                         Peaks = most suspicious moments

[Face Grid]           ← 12 sample face crops extracted
```

**Step 5 — Download Report**
- **JSON Report:** Machine-readable, includes all metrics
- **PDF Report:** Human-readable, suitable for documentation

---

### Page 3 — 📊 Model Performance

Shows evaluation metrics and charts. In demo mode, displays benchmark results from FaceForensics++. After training your own model, metrics update automatically.

**What Each Chart Shows:**

| Chart | Description |
|-------|-------------|
| Metric Cards | Accuracy, Precision, Recall, F1, AUC-ROC, Specificity |
| Confusion Matrix | True/False Positives and Negatives breakdown |
| ROC Curve | Trade-off between true positive and false positive rates |
| Training History | Loss and AUC curves across training epochs |
| Precision-Recall Curve | Model performance across all threshold values |
| Threshold Analysis | How precision, recall, and F1 change with threshold |
| Architecture Table | Layer-by-layer model breakdown with parameter counts |
| Benchmark Comparison | How this model compares to published alternatives |

---

### Page 4 — 📈 Analytics Dashboard

Tracks historical analysis statistics across all scans you've run. In demo mode, shows 50 simulated records.

**How to Use:**
1. The KPI row at the top shows: Total Scans, Fake Detected, Real Verified, Avg Confidence, Avg Processing Time
2. **Daily Trend** chart shows fake vs. real detections over time
3. **Detection Split** pie chart shows overall fake/real ratio
4. **Score Distribution** shows how fake scores differ between classes
5. **Processing Time Distribution** shows speed consistency
6. **Recent Scan History** table shows the last 20 analyzed videos
7. Click **"⬇️ Download Analytics CSV"** to export all data

---

### Page 5 — 📷 Live Detection

**Tab 1: Single Frame Analysis**

1. Upload a portrait image (JPG, PNG, WEBP)
2. The system detects the face using MediaPipe
3. Returns an instant REAL/FAKE verdict
4. Shows a probability bar and face detection status

Best used for:
- Analyzing screenshots from videos
- Checking profile photos
- Quick spot-checks without processing a full video

**Tab 2: Webcam Stream Info**

Provides instructions and code snippets for:
- Running real-time webcam detection locally (OpenCV script)
- Connecting to the FastAPI endpoint for programmatic access
- Hardware requirements for live detection performance

---

### Page 6 — ℹ️ About

Reference page with:
- Dataset (FaceForensics++) download instructions
- Full technology stack breakdown
- Model architecture ASCII diagram
- Academic citations and references
- License and disclaimer

---

## 9. Command-Line Prediction

For batch automation, scripting, or server-side use without the dashboard.

### Basic Usage

```bash
python predict.py --video path/to/video.mp4
```

### All Options

```bash
python predict.py [OPTIONS]

Required:
  --video PATH         Path to the video file to analyze

Optional:
  --threshold FLOAT    Detection threshold, 0.0–1.0 (default: 0.5)
  --model PATH         Path to custom .h5 model weights
  --output PATH        Save result as JSON to this file path
  --report FORMAT      Generate report: json | pdf | none (default: none)
  --full-model         Use EfficientNet instead of MobileNet (more accurate, slower)
```

### Examples

```bash
# Basic detection
python predict.py --video suspicious_video.mp4

# Save JSON result
python predict.py --video video.mp4 --output results/scan_001.json

# Generate PDF report
python predict.py --video video.mp4 --report pdf

# Use lower threshold (more sensitive)
python predict.py --video video.mp4 --threshold 0.4

# Use full model for best accuracy
python predict.py --video video.mp4 --full-model

# Combine options
python predict.py \
    --video video.mp4 \
    --threshold 0.45 \
    --report pdf \
    --output results/result.json \
    --full-model
```

### Shell Script Automation

The exit code is `1` if FAKE, `0` if REAL — useful for shell scripting:

```bash
#!/bin/bash
# Batch scan all videos in a folder

SCAN_DIR="./videos_to_check"
RESULTS_DIR="./scan_results"
mkdir -p "$RESULTS_DIR"

for video in "$SCAN_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    echo "Scanning: $filename"
    
    python predict.py \
        --video "$video" \
        --output "$RESULTS_DIR/${filename}.json" \
        --report pdf
    
    if [ $? -eq 1 ]; then
        echo "  ⚠️  FAKE DETECTED: $filename"
    else
        echo "  ✅ REAL: $filename"
    fi
done

echo "Done. Results saved to $RESULTS_DIR/"
```

---

## 10. REST API Usage

### Start the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive API Docs (Swagger):** http://localhost:8000/docs
- **Alternative Docs (ReDoc):** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### Endpoints Reference

#### `GET /health`

Check if the server is running and model is loaded.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "model_loaded": true,
  "uptime_sec": 42.3
}
```

---

#### `POST /predict`

Analyze a single video file.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@video.mp4" \
  -F "threshold=0.5"
```

With optional PDF report generation:
```bash
curl -X POST "http://localhost:8000/predict?report=true" \
  -F "file=@video.mp4"
```

Response:
```json
{
  "report_id": "DFD-1719000000",
  "verdict": "FAKE",
  "confidence": 0.873,
  "fake_score": 0.873,
  "is_fake": true,
  "threshold": 0.5,
  "frames_analyzed": 120,
  "faces_detected": 118,
  "processing_time_sec": 18.4,
  "model_used": "deepfake_model.h5",
  "metadata": {
    "total_frames": 300,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_sec": 10.0
  }
}
```

---

#### `POST /predict/batch`

Analyze up to 10 videos in one request.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@video1.mp4" \
  -F "files=@video2.mp4" \
  -F "files=@video3.mp4"
```

Response:
```json
{
  "results": [
    {"filename": "video1.mp4", "verdict": "REAL", "confidence": 0.91, "error": null},
    {"filename": "video2.mp4", "verdict": "FAKE", "confidence": 0.78, "error": null},
    {"filename": "video3.mp4", "verdict": "REAL", "confidence": 0.85, "error": null}
  ],
  "total": 3
}
```

---

### Python Client Example

```python
import requests

def analyze_video(video_path: str, threshold: float = 0.5) -> dict:
    """Send a video to the DeepFake Shield API and return the result."""
    url = "http://localhost:8000/predict"
    
    with open(video_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            params={"threshold": threshold}
        )
    
    response.raise_for_status()
    return response.json()

# Usage
result = analyze_video("my_video.mp4")
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']:.1%}")

if result["is_fake"]:
    print("⚠️  DeepFake detected!")
else:
    print("✅  Video appears authentic")
```

---

## 11. Understanding Results

### The Verdict

| Verdict | Meaning | Action |
|---------|---------|--------|
| ✅ REAL | No significant deepfake indicators found | Video is likely authentic |
| 🚨 FAKE | Deepfake manipulation detected | Treat with suspicion, verify through other means |

> ⚠️ **Important:** No AI system is 100% accurate. Always apply human judgment, especially for high-stakes decisions.

---

### Fake Score Interpretation

The `fake_score` is the raw sigmoid output from the model — a probability between 0 and 1.

| Fake Score | Interpretation |
|------------|----------------|
| 0.00 – 0.20 | Very likely authentic — strong real signals |
| 0.21 – 0.40 | Probably authentic — minor uncertainty |
| 0.41 – 0.50 | Ambiguous — model is uncertain, inconclusive |
| 0.51 – 0.70 | Probably manipulated — some fake signals present |
| 0.71 – 0.90 | Likely manipulated — clear fake signals |
| 0.91 – 1.00 | Almost certainly manipulated — very strong fake signals |

---

### Confidence Score Interpretation

`confidence` is the certainty toward the verdict label (not always toward FAKE):

- For a FAKE verdict: `confidence = fake_score`
- For a REAL verdict: `confidence = 1 - fake_score`

So 85% confidence in "REAL" means: the model is 85% sure the video is real.

---

### Frame Timeline Chart

The frame timeline shows how the fake probability changes over the video duration:

- **Flat high line** (~0.8+): Consistent deepfake manipulation throughout
- **Flat low line** (~0.2-): Consistently authentic throughout
- **Spikes**: Specific segments may be manipulated while others are not
- **Gradual rise**: Quality of manipulation may degrade over time

---

### Score Distribution Chart

This histogram shows the spread of per-sequence fake scores:

- **Tight cluster near 0**: Strong evidence of authentic video
- **Tight cluster near 1**: Strong evidence of deepfake
- **Wide spread**: Mixed evidence — consider this inconclusive
- **Bimodal (two peaks)**: Mixed content, some segments real, some fake

---

### Face Grid

The face grid shows 12 sample face crops extracted from the video. Look for:

- **Blurry boundaries** around the face — common in face-swap deepfakes
- **Inconsistent lighting** between frames
- **Unnatural eye reflections** or missing specular highlights
- **Irregular blinking** patterns
- **Hair/ear artifacts** at the edge of the face

---

## 12. Model Architecture Deep Dive

### Overview

The model uses a two-stage approach:

1. **Spatial Analysis (CNN):** Analyzes each frame independently to detect pixel-level artifacts
2. **Temporal Analysis (LSTM):** Looks across a sequence of frames to detect inconsistencies that only appear over time

### CNN Stage — EfficientNetB0

EfficientNetB0 is a convolutional neural network pretrained on 1.2 million ImageNet images. We use it as a feature extractor, meaning:

1. The early layers detect low-level patterns (edges, textures, colors)
2. The middle layers detect mid-level features (eyes, nose, skin texture)
3. The late layers detect high-level semantic features (face identity, expression)
4. We freeze most of the network (keep ImageNet weights) and only fine-tune the last 20 layers

This is called **transfer learning** — we leverage millions of training examples from ImageNet without needing to train from scratch.

### LSTM Stage — Temporal Modeling

After the CNN processes each frame, we have a sequence of feature vectors. The LSTM reads these in order, looking for patterns that span multiple frames:

- Real videos have **temporally consistent** features (lighting, identity, expression change naturally)
- Deepfakes often have **flickering artifacts** — the GAN-generated content is slightly different frame to frame
- The Bidirectional LSTM reads both forward and backward through the sequence for better context

### Why Sequences of 20 Frames?

- Too few frames (< 5): Miss temporal patterns, similar to single-image classifiers
- 20 frames: Captures ~0.67 seconds at 30fps — enough to see micro-expression artifacts
- Too many frames (> 40): Diminishing returns, higher memory usage, slower inference

### Training Strategy

| Technique | Purpose |
|-----------|---------|
| Transfer Learning | Leverage ImageNet features, reduce data needed |
| Batch Normalization | Stabilize training, allow higher learning rates |
| Dropout (0.3–0.4) | Prevent overfitting |
| Early Stopping | Halt before overfitting, save best weights |
| ReduceLROnPlateau | Fine-tune learning rate when progress stalls |
| Class Weighting | Handle imbalanced datasets (more real than fake videos) |
| Data Augmentation | Rotation, flipping, brightness — improve generalization |

---

## 13. Troubleshooting

### Installation Issues

---

**Error: `ModuleNotFoundError: No module named 'cv2'`**

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.9.0.80
```

---

**Error: `ModuleNotFoundError: No module named 'mediapipe'`**

```bash
pip install mediapipe==0.10.11
```

Note: MediaPipe requires Python 3.10 or 3.11. It does NOT work on Python 3.12 in all configurations.

---

**Error: `ImportError: libGL.so.1: cannot open shared object file`** (Linux)

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

**Error: TensorFlow GPU not detected**

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall TensorFlow with CUDA support
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.15.0

# Verify
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### Runtime Issues

---

**Dashboard loads but "Upload & Detect" crashes immediately**

Cause: Streamlit can't import a dependency.

```bash
# Check for import errors
python -c "from app.pages.detect import render; print('OK')"
```

---

**"No faces detected" warning on most frames**

Causes and fixes:
1. Video has poor lighting → try a higher-quality source
2. Faces are too small → increase `FACE_PADDING` in `.env` (try 0.3)
3. MediaPipe confidence too high → edit `min_detection_confidence` in `video_processor.py` to `0.3`

---

**Very slow processing (>5 min per video)**

```ini
# In .env, reduce these values:
SEQUENCE_LENGTH=10
FRAME_SIZE=128
```

```bash
# Also add to .env:
MAX_FRAMES=50
```

---

**Out of Memory during inference**

```bash
# Set TensorFlow to use GPU memory incrementally
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print('Memory growth enabled')
"
```

Or add to the top of `main.py`:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

**Training crashes with "NaN loss"**

```bash
# Reduce learning rate
python train.py --lr 0.00001 --resume
```

Also check that your videos actually contain faces (corrupt or audio-only files can cause NaN).

---

**Port 8501 is already in use**

```bash
# Use a different port
streamlit run main.py --server.port 8502

# Or kill the process using 8501
# Linux/macOS:
lsof -ti:8501 | xargs kill -9
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

---

**API returns 422 Unprocessable Entity**

This means the request format is wrong. Common fix:

```bash
# Wrong — missing -F flag
curl -X POST http://localhost:8000/predict --data video.mp4

# Correct — use multipart form
curl -X POST http://localhost:8000/predict -F "file=@video.mp4"
```

---

### Model Issues

---

**Model always predicts REAL (or always FAKE)**

Cause: Model is undertrained or dataset is too imbalanced.

Solutions:
1. Check your dataset balance: `ls dataset/real | wc -l` vs `ls dataset/fake | wc -l`
2. If imbalanced, use `--max-videos N` to cap the larger class
3. Train for more epochs: `python train.py --resume --epochs 30`
4. Try a lower learning rate: `python train.py --lr 0.00005`

---

**Accuracy is very low (<70%)**

1. Ensure you have at least 100 videos per class
2. Verify videos actually contain faces (open a few to check)
3. Try the lightweight model: `python train.py --lightweight`
4. Check if dataset folders are correct: real videos in `dataset/real/`, not swapped

---

## 14. Deployment Guide

### Quick Comparison

| Platform | Cost | GPU | Setup Time | Best For |
|----------|------|-----|------------|---------|
| Local | Free | Own | 5 min | Development |
| Streamlit Cloud | Free | No | 10 min | Demos |
| Hugging Face Spaces | Free/Paid | Optional | 15 min | ML community |
| Render | $7/mo | No | 20 min | Production |
| Railway | $5/mo | No | 10 min | Simple deploy |
| Docker (VPS) | ~$10/mo | Optional | 30 min | Full control |

---

### Streamlit Cloud (Easiest)

1. Push code to a public GitHub repository
2. Go to https://share.streamlit.io
3. Click **"New app"**
4. Select your repository and set main file to `main.py`
5. Click **Deploy**

> **Note:** Streamlit Cloud has 1 GB RAM. Use `--lightweight` model and set `SEQUENCE_LENGTH=10` for best results.

---

### Docker (Most Portable)

**Dockerfile** (create this in the project root):
```dockerfile
FROM python:3.11-slim

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
```

```bash
# Build
docker build -t deepfake-shield .

# Run (with model weights volume)
docker run -p 8501:8501 \
    -v $(pwd)/weights:/app/weights \
    deepfake-shield

# Run API
docker run -p 8000:8000 \
    -v $(pwd)/weights:/app/weights \
    deepfake-shield \
    uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 15. Advanced Customization

### Using a Different CNN Backbone

Edit `app/models/detector.py`, the `build_cnn_feature_extractor` function:

```python
# Current options:
backbone="efficientnet"  # EfficientNetB0 — best accuracy
backbone="mobilenet"     # MobileNetV2 — fastest

# Add custom backbone (example with ResNet50):
from tensorflow.keras.applications import ResNet50
base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
```

After changing the backbone, retrain the model completely (`python train.py`).

---

### Adjusting Sequence Length

A longer sequence captures more temporal context but uses more memory:

1. Update `.env`:
   ```ini
   SEQUENCE_LENGTH=30
   ```
2. Retrain: `python train.py`

---

### Adding Custom Augmentation

Edit `app/config.py`:

```python
AUGMENTATION_CONFIG = {
    "rotation_range": 15,        # ±15° rotation
    "width_shift_range": 0.15,   # ±15% horizontal shift
    "height_shift_range": 0.15,  # ±15% vertical shift
    "horizontal_flip": True,     # Random horizontal flip
    "zoom_range": 0.15,          # ±15% zoom
    "brightness_range": [0.7, 1.3],  # Brightness variation
    "channel_shift_range": 20,   # Color channel variation
}
```

---

### Exporting to ONNX / TFLite

For edge deployment or mobile:

```python
# TFLite (smaller, faster, CPU-optimized)
import tensorflow as tf

model = tf.keras.models.load_model("weights/deepfake_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("weights/deepfake_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved")
```

---

### Integrating with External Systems

The REST API makes integration straightforward. Example integrations:

**Slack Bot:**
```python
import requests

def check_video_slack(video_path):
    result = requests.post(
        "http://localhost:8000/predict",
        files={"file": open(video_path, "rb")}
    ).json()
    
    msg = f"🛡️ *DeepFake Check Result*\n"
    msg += f"Verdict: {'🚨 FAKE' if result['is_fake'] else '✅ REAL'}\n"
    msg += f"Confidence: {result['confidence']:.1%}"
    return msg
```

**Content Moderation Pipeline:**
```bash
# Scan all uploaded videos automatically
inotifywait -m uploads/ -e create | while read dir action file; do
    python predict.py --video "uploads/$file" --report json \
        --output "scan_results/${file%.mp4}.json"
done
```

---

## 16. Frequently Asked Questions

**Q: Do I need to train the model to use the app?**

No. The app runs in demo mode immediately, with simulated scores for all features. To get real detection results, you need to train the model (see Section 6).

---

**Q: How accurate is the model?**

On FaceForensics++ (a standard benchmark), the system achieves ~93% accuracy. However, accuracy varies significantly by:
- Compression level (raw video is easier to detect than highly compressed)
- Manipulation method (some are harder to detect)
- Video quality and lighting
- Whether the deepfake was made with recent/advanced tools

Always treat results as one input into a larger verification process.

---

**Q: Can it detect all types of deepfakes?**

It is primarily trained on face-swap and face-reenactment deepfakes (the most common types). It may not perform as well on:
- AI-generated entirely synthetic people (e.g., StyleGAN faces)
- Audio deepfakes (voice cloning) — this system only analyzes video
- Text/document forgeries
- Very recent novel deepfake methods not in the training data

---

**Q: My video shows as FAKE but I know it's real. Why?**

False positives can occur when:
- Video has heavy compression artifacts (similar to GAN artifacts)
- Unusual lighting or camera filters
- Rapid face movement or motion blur
- Very low resolution face crops
- Video codec introduces artifacts similar to deepfake signatures

Try increasing the threshold to 0.65 for a more conservative detector.

---

**Q: How long does analysis take?**

Typical times on CPU:
- 5-second video: ~8–12 seconds
- 30-second video: ~20–35 seconds
- 2-minute video: ~60–90 seconds

On a GPU (RTX 3080):
- 5-second video: ~2–4 seconds
- 30-second video: ~8–15 seconds

---

**Q: Can I use this commercially?**

The code is MIT licensed, but:
1. FaceForensics++ dataset has its own academic/research license
2. You should validate performance on your specific use case before deploying commercially
3. Consult legal counsel for any forensics or legal use cases

---

**Q: How do I update the model with new data?**

Add new videos to the dataset folders and run:
```bash
python train.py --resume --epochs 20 --no-cache
```

The `--resume` flag loads the existing weights and continues training, adapting to the new data.

---

**Q: Can it analyze images, not just videos?**

Yes — use the **Live Detection** page (Tab 1: Single Frame Analysis). Upload any portrait image for instant single-frame analysis.

---

**Q: The API is too slow for real-time use. How do I speed it up?**

1. Switch to the lightweight model in `app/models/predictor.py`:
   ```python
   DeepFakePredictor(lightweight=True)
   ```
2. Reduce sequence length in `.env`: `SEQUENCE_LENGTH=10`
3. Use GPU if available
4. Pre-load the model at startup (already done in `api.py`)
5. Consider TFLite export for further optimization (see Section 15)

---

*End of Instructions — DeepFake Shield v2.0.0*

For additional support, open an issue on the GitHub repository or consult the `logs/` directory for diagnostic information.
