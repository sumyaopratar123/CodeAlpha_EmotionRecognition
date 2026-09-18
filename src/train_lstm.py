import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "data/RAVDESS"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_lstm.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_label_classes.npy"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

SELECTED_EMOTIONS = [
    "angry",
    "happy",
    "sad"
]


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine features
    features = np.concatenate(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=0
    )

    # Transpose for LSTM
    # Shape becomes:
    # time_steps x features

    features = features.T

    return features.astype(
        np.float32
    )


# =========================================================
# LOAD DATA
# =========================================================

print(
    "Loading RAVDESS dataset..."
)

X = []
y = []

count = 0

for actor in sorted(
    os.listdir(DATASET_PATH)
):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(
        os.listdir(actor_path)
    ):

        if not filename.endswith(".wav"):
            continue

        emotion_code = filename.split("-")[2]

        emotion = EMOTIONS.get(
            emotion_code
        )

        if emotion not in SELECTED_EMOTIONS:
            continue

        file_path = os.path.join(
            actor_path,
            filename
        )

        try:

            features = extract_features(
                file_path
            )

            X.append(
                features
            )

            y.append(
                emotion
            )

            count += 1

            if count % 50 == 0:

                print(
                    f"Processed: {count}"
                )

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )


print(
    "\nDataset loading completed!"
)

print(
    "Total samples:",
    len(X)
)


# =========================================================
# PAD SEQUENCES
# =========================================================

print(
    "\nPadding sequences..."
)

max_length = max(
    feature.shape[0]
    for feature in X
)

feature_size = X[0].shape[1]

X_padded = np.zeros(
    (
        len(X),
        max_length,
        feature_size
    ),
    dtype=np.float32
)

for i, feature in enumerate(X):

    length = min(
        feature.shape[0],
        max_length
    )

    X_padded[
        i,
        :length,
        :
    ] = feature[
        :length,
        :
    ]


X = X_padded


print(
    "X shape:",
    X.shape
)


# =========================================================
# NORMALIZATION
# =========================================================

print(
    "\nNormalizing features..."
)

mean = np.mean(
    X,
    axis=(0, 1),
    keepdims=True
)

std = np.std(
    X,
    axis=(0, 1),
    keepdims=True
)

X = (
    X - mean
) / (
    std + 1e-8
)


# =========================================================
# LABEL ENCODING
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    y
)

classes = encoder.classes_

print(
    "\nClasses:",
    classes
)

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


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# =========================================================
# LSTM MODEL
# =========================================================

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    ),

    Dropout(
        0.30
    ),

    Bidirectional(
        LSTM(
            32
        )
    ),

    Dropout(
        0.30
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.30
    ),

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

    metrics=[
        "accuracy"
    ]
)


print(
    "\nLSTM model created!"
)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=8,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=1e-6,

    verbose=1
)


# =========================================================
# TRAIN
# =========================================================

print(
    "\nStarting LSTM training...\n"
)

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=40,

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

print(
    "\nEvaluating model..."
)

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print(
    f"\nTest Accuracy: {test_accuracy:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {test_loss:.4f}"
)


# =========================================================
# PREDICTION
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

print(
    "\nClassification Report:\n"
)

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

print(
    "\nConfusion Matrix:\n"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


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
# TRAINING GRAPHS
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
    "LSTM Speech Emotion Recognition - Accuracy"
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
    "screenshots/lstm_training_accuracy.png"
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
    "LSTM Speech Emotion Recognition - Loss"
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
    "screenshots/lstm_training_loss.png"
)

plt.close()


print(
    "\nTraining graphs saved."
)

print(
    "\n========== LSTM TRAINING COMPLETED =========="
)