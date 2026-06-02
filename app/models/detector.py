"""
DeepFake Detection System - CNN-LSTM Model Architecture
=========================================================
Combines a CNN feature extractor with an LSTM temporal classifier
for end-to-end deepfake video detection.

Architecture:
    Input (seq_len, H, W, 3)
      → TimeDistributed(EfficientNetB0)   [CNN Feature Extractor]
      → TimeDistributed(GlobalAvgPool2D)
      → TimeDistributed(Dense 512)
      → Bidirectional LSTM (256)
      → LSTM (128)
      → Dense + Dropout
      → Sigmoid Output (REAL=0 / FAKE=1)
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from app.utils.logger import get_logger
from app.config import (
    FRAME_SIZE, SEQUENCE_LENGTH, CNN_FEATURE_DIM,
    LEARNING_RATE, WEIGHTS_DIR, LOGS_DIR
)

logger = get_logger(__name__)


# ─── Model Builder ─────────────────────────────────────────────────────────────

def build_cnn_feature_extractor(
    input_shape: Tuple[int, int, int] = (FRAME_SIZE, FRAME_SIZE, 3),
    trainable_layers: int = 20,
    backbone: str = "efficientnet",
) -> Model:
    """
    Build CNN feature extractor using a pretrained backbone.
    
    Args:
        input_shape: Single frame shape (H, W, C)
        trainable_layers: Number of top backbone layers to fine-tune
        backbone: 'efficientnet' or 'mobilenet'
    
    Returns:
        Keras Model that maps (H, W, 3) → (feature_vector,)
    """
    inp = Input(shape=input_shape, name="frame_input")

    if backbone == "efficientnet":
        base = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        base = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )

    # Freeze most layers, fine-tune the top N
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base.layers[-trainable_layers:]:
        layer.trainable = True

    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(CNN_FEATURE_DIM, activation="relu", name="cnn_features")(x)
    x = layers.BatchNormalization(name="bn_features")(x)
    x = layers.Dropout(0.3, name="drop_features")(x)

    return Model(inputs=inp, outputs=x, name=f"cnn_{backbone}")


def build_deepfake_detector(
    sequence_length: int = SEQUENCE_LENGTH,
    frame_size: int = FRAME_SIZE,
    backbone: str = "efficientnet",
    lstm_units: Tuple[int, int] = (256, 128),
    dropout_rate: float = 0.4,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Build the full CNN-LSTM deepfake detection model.
    
    Args:
        sequence_length: Number of frames per temporal sequence
        frame_size: Spatial resolution of each frame
        backbone: CNN backbone ('efficientnet' | 'mobilenet')
        lstm_units: (first_lstm_units, second_lstm_units)
        dropout_rate: Dropout probability in classifier head
        learning_rate: Adam optimizer learning rate
    
    Returns:
        Compiled Keras model
    """
    input_shape = (sequence_length, frame_size, frame_size, 3)
    inp = Input(shape=input_shape, name="video_sequence")

    # ── CNN Feature Extraction (applied per-frame) ──────────────────────────
    cnn = build_cnn_feature_extractor(
        input_shape=(frame_size, frame_size, 3),
        backbone=backbone
    )
    x = layers.TimeDistributed(cnn, name="time_cnn")(inp)
    # x shape: (batch, seq_len, CNN_FEATURE_DIM)

    # ── Temporal Modeling ────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True, dropout=0.2),
        name="bilstm"
    )(x)
    x = layers.LayerNormalization(name="ln1")(x)

    x = layers.LSTM(lstm_units[1], return_sequences=False, dropout=0.2, name="lstm2")(x)
    x = layers.LayerNormalization(name="ln2")(x)

    # ── Classifier Head ──────────────────────────────────────────────────────
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)

    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop2")(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    # Output: 0 = REAL, 1 = FAKE

    model = Model(inputs=inp, outputs=out, name="DeepFakeDetector")

    # ── Compile ──────────────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    logger.info(f"Model built: {model.count_params():,} parameters")
    return model


# ─── Training Callbacks ────────────────────────────────────────────────────────

def get_callbacks(model_name: str = "deepfake_model") -> list:
    """Return standard training callbacks."""
    ckpt_path = str(WEIGHTS_DIR / f"{model_name}_best.h5")
    log_dir   = str(LOGS_DIR / "tensorboard")

    return [
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=7,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
    ]


# ─── Lightweight Demo Model ────────────────────────────────────────────────────

def build_lightweight_model(
    sequence_length: int = SEQUENCE_LENGTH,
    frame_size: int = FRAME_SIZE,
) -> Model:
    """
    Smaller model for demo / CPU inference (MobileNetV2 backbone).
    Faster but slightly less accurate than the full model.
    """
    return build_deepfake_detector(
        sequence_length=sequence_length,
        frame_size=frame_size,
        backbone="mobilenet",
        lstm_units=(128, 64),
        dropout_rate=0.3,
        learning_rate=LEARNING_RATE,
    )


# ─── Model I/O ─────────────────────────────────────────────────────────────────

def save_model(model: Model, path: str = None) -> str:
    """Save model weights to disk."""
    path = path or str(WEIGHTS_DIR / "deepfake_model.h5")
    model.save(path)
    logger.info(f"Model saved → {path}")
    return path


def load_model(path: str = None) -> Optional[Model]:
    """
    Load a saved model. Returns None if the file doesn't exist.
    Falls back to building a fresh (untrained) model.
    """
    path = path or str(WEIGHTS_DIR / "deepfake_model.h5")
    if not Path(path).exists():
        logger.warning(f"No weights found at {path}. Building fresh model.")
        return None
    try:
        model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded ← {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def get_or_build_model(path: str = None, lightweight: bool = False) -> Model:
    """
    Load existing model or build a new one.
    
    Args:
        path: Path to saved .h5 weights
        lightweight: Use MobileNetV2 backbone if True
    
    Returns:
        Keras model (loaded or freshly built)
    """
    model = load_model(path)
    if model is None:
        logger.info("Building new model…")
        if lightweight:
            model = build_lightweight_model()
        else:
            model = build_deepfake_detector()
    return model


# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam(
    model: Model,
    frame: np.ndarray,
    layer_name: str = "cnn_efficientnet",
    class_index: int = 0,
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single frame.
    
    Args:
        model: Trained Keras model
        frame: Preprocessed frame (1, H, W, 3) float32
        layer_name: Target conv layer name
        class_index: Output class index (0 = FAKE for sigmoid output)
    
    Returns:
        Heatmap as (H, W) float32 in [0, 1]
    """
    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(frame)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}")
        return np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
