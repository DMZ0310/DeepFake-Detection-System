"""
DeepFake Detection System - Model Training Script
==================================================
Usage:
    python train.py                    # Train from scratch
    python train.py --lightweight      # Train lightweight model (MobileNetV2)
    python train.py --prepare          # Only prepare dataset (no training)
    python train.py --epochs 30        # Override epochs
    python train.py --resume           # Resume from checkpoint

Expects dataset in:
    dataset/real/  → real video frames or videos
    dataset/fake/  → deepfake video frames or videos
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import (
    DATASET_DIR, WEIGHTS_DIR, SEQUENCE_LENGTH, FRAME_SIZE,
    BATCH_SIZE, EPOCHS, TRAIN_SPLIT, VAL_SPLIT, LOGS_DIR
)
from app.models.detector import (
    build_deepfake_detector, build_lightweight_model,
    get_callbacks, save_model
)
from app.preprocessing.video_processor import VideoPreprocessor
from app.utils.logger import get_logger

logger = get_logger(__name__, log_file=str(LOGS_DIR / "train.log"))


# ── Dataset Preparation ────────────────────────────────────────────────────────

def prepare_dataset(
    dataset_dir: Path = DATASET_DIR,
    sequence_length: int = SEQUENCE_LENGTH,
    frame_size: int = FRAME_SIZE,
    max_videos: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load videos from dataset/real and dataset/fake, extract face sequences,
    and return (X, y) arrays ready for training.
    
    Returns:
        X: np.ndarray (N, sequence_length, frame_size, frame_size, 3)
        y: np.ndarray (N,) binary labels: 0=REAL, 1=FAKE
    """
    preprocessor = VideoPreprocessor(
        frame_size=frame_size,
        sequence_length=sequence_length,
    )

    X_all, y_all = [], []

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = dataset_dir / folder
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            continue

        # Collect video files
        video_files = []
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            video_files.extend(folder_path.glob(ext))
        
        if max_videos:
            video_files = video_files[:max_videos]

        logger.info(f"Processing {len(video_files)} {folder} videos…")

        for i, video_path in enumerate(video_files):
            try:
                result = preprocessor.process_video(str(video_path))
                seqs = result["face_sequences"]  # (N, seq_len, H, W, 3)
                
                X_all.append(seqs)
                y_all.extend([label] * len(seqs))
                
                logger.info(
                    f"  [{i+1}/{len(video_files)}] {video_path.name}: "
                    f"{len(seqs)} sequences"
                )
            except Exception as e:
                logger.error(f"  Failed {video_path.name}: {e}")

    if not X_all:
        raise ValueError(
            "No data loaded. Please add videos to dataset/real/ and dataset/fake/"
        )

    X = np.concatenate(X_all, axis=0)
    y = np.array(y_all, dtype=np.float32)
    
    logger.info(f"Dataset: {X.shape[0]} sequences | REAL={np.sum(y==0)} | FAKE={np.sum(y==1)}")
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = 42,
) -> Tuple:
    """Shuffle and split dataset into train/val/test."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    n = len(X)
    n_train = int(n * train_split)
    n_val   = int(n * (train_split + val_split))
    
    X_train, y_train = X[:n_train],      y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:],        y[n_val:]

    logger.info(f"Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Run full evaluation and return metrics dict."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    logger.info("Evaluating on test set…")
    y_pred_raw = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    y_pred     = (y_pred_raw.flatten() >= 0.5).astype(int)
    y_true     = y_test.astype(int)

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":   float(roc_auc_score(y_true, y_pred_raw.flatten())),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            logger.info(f"  {k:12s}: {v:.4f}")
    logger.info("=" * 50)

    return metrics


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(args):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    logger.info("=" * 60)
    logger.info("DeepFake Shield — Training Pipeline")
    logger.info("=" * 60)

    # ── Prepare Data ────────────────────────────────────────────────────────
    logger.info("Step 1: Preparing dataset…")
    
    cache_path = DATASET_DIR / "prepared_data.npz"
    
    if cache_path.exists() and not args.no_cache:
        logger.info(f"Loading cached data from {cache_path}")
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
    else:
        X, y = prepare_dataset(
            max_videos=args.max_videos,
            sequence_length=SEQUENCE_LENGTH,
            frame_size=FRAME_SIZE,
        )
        if args.cache:
            np.savez_compressed(str(cache_path), X=X, y=y)
            logger.info(f"Dataset cached to {cache_path}")

    if args.prepare_only:
        logger.info("--prepare flag set. Exiting after dataset preparation.")
        return

    # ── Split ────────────────────────────────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    # ── Build Model ──────────────────────────────────────────────────────────
    logger.info("Step 2: Building model…")
    if args.lightweight:
        model = build_lightweight_model()
    else:
        model = build_deepfake_detector(learning_rate=args.lr)
    
    model.summary()

    # Resume from checkpoint?
    if args.resume:
        ckpt = str(WEIGHTS_DIR / "deepfake_model_best.h5")
        if Path(ckpt).exists():
            model.load_weights(ckpt)
            logger.info(f"Resumed from checkpoint: {ckpt}")
        else:
            logger.warning("No checkpoint found, training from scratch")

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Step 3: Training…")
    t0 = time.time()
    
    # Class weights to handle imbalance
    n_real = np.sum(y_train == 0)
    n_fake = np.sum(y_train == 1)
    total  = n_real + n_fake
    class_weights = {
        0: total / (2 * n_real + 1e-8),
        1: total / (2 * n_fake + 1e-8),
    }
    logger.info(f"Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1,
    )

    train_time = round(time.time() - t0, 1)
    logger.info(f"Training completed in {train_time}s")

    # ── Save ─────────────────────────────────────────────────────────────────
    model_path = str(WEIGHTS_DIR / "deepfake_model.h5")
    save_model(model, model_path)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    logger.info("Step 4: Evaluating…")
    metrics = evaluate_model(model, X_test, y_test)

    # Save training history
    history_path = str(LOGS_DIR / "training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "history": {k: [float(v) for v in vals]
                        for k, vals in history.history.items()},
            "metrics": metrics,
            "train_time_sec": train_time,
            "epochs": args.epochs,
        }, f, indent=2)
    
    logger.info(f"History saved → {history_path}")
    logger.info("Training pipeline complete. ✅")
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeepFake Shield Training Script")
    p.add_argument("--lightweight",  action="store_true", help="Use MobileNetV2 backbone")
    p.add_argument("--prepare",      dest="prepare_only", action="store_true", help="Prepare dataset only")
    p.add_argument("--resume",       action="store_true", help="Resume from checkpoint")
    p.add_argument("--no-cache",     dest="no_cache", action="store_true")
    p.add_argument("--cache",        action="store_true", help="Save prepared data to disk")
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE, dest="batch_size")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--max-videos",   type=int, default=None, dest="max_videos",
                   help="Limit videos per class (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
