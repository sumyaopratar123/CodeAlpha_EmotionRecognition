import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = "data/features.npz"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_cnn.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "label_classes.npy"
)

os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOAD FEATURES
# =========================================================

print("Loading features...")

data = np.load(
    DATA_PATH,
    allow_pickle=True
)

X = data["X"]
y = data["y"]

print("Original X:", X.shape)
print("Original classes:", np.unique(y))


# =========================================================
# SELECT EMOTIONS
# =========================================================

selected = [
    "angry",
    "happy",
    "sad"
]

mask = np.isin(
    y,
    selected
)

X = X[mask]
y = y[mask]

print("\nSelected classes:")
print(np.unique(y, return_counts=True))

print("Filtered X:", X.shape)


# =========================================================
# CREATE DELTA FEATURES
# =========================================================

print("\nCreating MFCC + Delta + Delta-Delta features...")

mfcc = X[..., 0]

delta = np.zeros_like(mfcc)

delta_delta = np.zeros_like(mfcc)

for i in range(len(mfcc)):

    delta[i] = librosa.feature.delta(
        mfcc[i],
        width=9,
        order=1
    )

    delta_delta[i] = librosa.feature.delta(
        mfcc[i],
        width=9,
        order=2
    )


# Shape:
# MFCC       = (samples, 40, 174)
# Delta      = (samples, 40, 174)
# Delta2     = (samples, 40, 174)

X = np.stack(
    [
        mfcc,
        delta,
        delta_delta
    ],
    axis=-1
)

print("New feature shape:", X.shape)


# =========================================================
# PER-SAMPLE NORMALIZATION
# =========================================================

print("\nNormalizing features...")

mean = np.mean(
    X,
    axis=(1, 2),
    keepdims=True
)

std = np.std(
    X,
    axis=(1, 2),
    keepdims=True
)

X = (
    X - mean
) / (
    std + 1e-8
)

X = X.astype(
    np.float32
)


# =========================================================
# LABEL ENCODING
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

classes = encoder.classes_

print("\nClasses:")
print(classes)

np.save(
    LABEL_PATH,
    classes
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=SEED,
    stratify=y_encoded
)

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# =========================================================
# CNN MODEL
# =========================================================

model = Sequential([

    Input(
        shape=X_train.shape[1:]
    ),

    # -----------------------------
    # BLOCK 1
    # -----------------------------

    Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.20),


    # -----------------------------
    # BLOCK 2
    # -----------------------------

    Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.25),


    # -----------------------------
    # BLOCK 3
    # -----------------------------

    Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.30),


    # -----------------------------
    # BLOCK 4
    # -----------------------------

    Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling2D(
        (2, 2)
    ),

    Dropout(0.35),


    # -----------------------------
    # CLASSIFIER
    # -----------------------------

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.40),

    Dense(
        3,
        activation="softmax"
    )
])


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nModel created successfully.")

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# =========================================================
# TRAIN
# =========================================================

print("\nStarting training...\n")

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=60,

    batch_size=16,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# =========================================================
# EVALUATION
# =========================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"\nTest Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Loss: {test_loss:.4f}"
)


# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=classes,
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)


# =========================================================
# SAVE MODEL
# =========================================================

model.save(
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

print(
    f"Labels saved to: {LABEL_PATH}"
)


# =========================================================
# SAVE GRAPHS
# =========================================================

import matplotlib.pyplot as plt

os.makedirs(
    "screenshots",
    exist_ok=True
)


# Accuracy
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Speech Emotion Recognition - Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "screenshots/training_accuracy.png"
)

plt.close()


# Loss
plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Speech Emotion Recognition - Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "screenshots/training_loss.png"
)

plt.close()


print(
    "\nTraining graphs saved."
)

print(
    "\n========== TRAINING COMPLETED =========="
)