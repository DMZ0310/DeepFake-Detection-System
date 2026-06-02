# 🚀 DeepFake Shield — Deployment Guide

Complete instructions for deploying DeepFake Shield to various platforms.

---

## 1. Local Development

```bash
# Install
pip install -r requirements.txt

# Run dashboard
streamlit run main.py

# Run API (separate terminal)
uvicorn api:app --reload --port 8000

# Train model
python train.py --lightweight --epochs 20
```

---

## 2. Streamlit Cloud (Free Tier)

**Best for:** Quick demos, sharing, no-cost hosting.

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_NAME/deepfake-shield.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set **Main file path**: `main.py`
   - Click "Deploy"

3. **Add Secrets** (optional, for API keys)
   - In App Settings → Secrets:
   ```toml
   CONFIDENCE_THRESHOLD = "0.5"
   ```

4. **Limitations on Free Tier**
   - 1 GB RAM (no heavy model training)
   - App sleeps after inactivity
   - Use lightweight model: set `lightweight=True` in predictor

---

## 3. Hugging Face Spaces

**Best for:** ML community visibility, GPU access.

### Steps

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - SDK: **Streamlit**
   - Hardware: CPU Basic (free) or T4 GPU ($0.60/hr)

2. **Upload Files**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_NAME/deepfake-shield
   cp -r deepfake_detection/* deepfake-shield/
   cd deepfake-shield
   git add .
   git commit -m "Add DeepFake Shield"
   git push
   ```

3. **Model Weights (Large Files)**
   ```bash
   # Use Git LFS for .h5 files
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes weights/
   git commit -m "Add model weights"
   git push
   ```

4. **Environment Variables**
   - In Space Settings → Variables and Secrets:
     - `MODEL_PATH` = `weights/deepfake_model.h5`

---

## 4. Render

**Best for:** Always-on production deployment with API.

### Streamlit Dashboard

Create `render.yaml`:
```yaml
services:
  - type: web
    name: deepfake-shield-dashboard
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run main.py --server.port $PORT --server.headless true --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### FastAPI Backend

```yaml
services:
  - type: web
    name: deepfake-shield-api
    env: python
    plan: starter  # $7/mo for always-on
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
```

---

## 5. Docker Compose (Full Stack)

`docker-compose.yml`:
```yaml
version: "3.9"

services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./weights:/app/weights
      - ./outputs:/app/outputs
      - ./reports:/app/reports
    environment:
      - MODEL_PATH=weights/deepfake_model.h5
    command: streamlit run main.py --server.address 0.0.0.0

  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    depends_on:
      - dashboard
```

`Dockerfile`:
```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
```

```bash
docker-compose up --build
```

---

## 6. AWS EC2

```bash
# Launch Ubuntu 22.04 instance (t3.medium or better)
# SSH in, then:

sudo apt update && sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

git clone https://github.com/YOUR_NAME/deepfake-shield.git
cd deepfake-shield
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run with screen (persists after SSH disconnect)
screen -S deepfake
streamlit run main.py --server.port 80 --server.address 0.0.0.0
# Ctrl+A, D to detach

# Open port 80 in EC2 Security Group
```

---

## 7. Google Cloud Run (Serverless)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/deepfake-shield

# Deploy
gcloud run deploy deepfake-shield \
  --image gcr.io/YOUR_PROJECT/deepfake-shield \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --port 8501
```

---

## 8. Performance Optimization for Production

### Model Optimization
```python
# Convert to TFLite for faster CPU inference
import tensorflow as tf

model = tf.keras.models.load_model("weights/deepfake_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("weights/deepfake_model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Caching Strategy
```python
# In Streamlit, cache model and preprocessor:
@st.cache_resource
def get_predictor():
    return DeepFakePredictor(lightweight=True)
```

### Frame Sampling
- Set `MAX_FRAMES=50` in `.env` for faster processing
- Use `sample_rate=3` to analyze every 3rd frame
- Adequate for most deepfakes which have consistent artifacts

---

## 9. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `weights/deepfake_model.h5` | Path to model weights |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection threshold |
| `SEQUENCE_LENGTH` | `20` | Frames per LSTM sequence |
| `FRAME_SIZE` | `224` | Face crop resolution |
| `MAX_FRAMES` | `150` | Max frames extracted per video |
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `50` | Training epochs |
| `LEARNING_RATE` | `0.0001` | Adam optimizer LR |
| `MAX_UPLOAD_SIZE_MB` | `500` | Upload limit |
| `API_PORT` | `8000` | FastAPI port |

---

## 10. Health Checks

```bash
# Dashboard
curl http://localhost:8501/_stcore/health

# API
curl http://localhost:8000/health
# Expected: {"status":"healthy","version":"2.0.0",...}
```
