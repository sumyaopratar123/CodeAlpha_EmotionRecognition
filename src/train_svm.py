import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = "data/RAVDESS"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_svm.joblib"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_label_classes.npy"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

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

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Delta
    delta = librosa.feature.delta(
        mfcc
    )

    # Delta Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = []

    # MFCC statistics
    features.extend(
        np.mean(mfcc, axis=1)
    )

    features.extend(
        np.std(mfcc, axis=1)
    )

    # Delta statistics
    features.extend(
        np.mean(delta, axis=1)
    )

    features.extend(
        np.std(delta, axis=1)
    )

    # Delta-Delta statistics
    features.extend(
        np.mean(delta2, axis=1)
    )

    features.extend(
        np.std(delta2, axis=1)
    )

    return np.array(
        features,
        dtype=np.float32
    )


# =========================================================
# LOAD DATASET
# =========================================================

print("Loading RAVDESS dataset...")

X = []
y = []

total = 0

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

        parts = filename.split("-")

        emotion_code = parts[2]

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

            X.append(features)

            y.append(emotion)

            total += 1

            if total % 50 == 0:

                print(
                    f"Processed: {total}"
                )

        except Exception as e:

            print(
                f"Error: {filename} -> {e}"
            )


X = np.array(X)

y = np.array(y)

print("\nDataset loaded!")

print(
    "X shape:",
    X.shape
)

print(
    "Classes:",
    np.unique(y, return_counts=True)
)


# =========================================================
# LABEL ENCODING
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

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

    random_state=42,

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
# SVM PIPELINE
# =========================================================

model = Pipeline([

    (
        "scaler",

        StandardScaler()
    ),

    (
        "svm",

        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )
])


# =========================================================
# TRAIN
# =========================================================

print(
    "\nTraining SVM..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed!"
)


# =========================================================
# TEST
# =========================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nTest Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
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

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

print(
    f"Labels saved to: {LABEL_PATH}"
)

print(
    "\n========== SVM TRAINING COMPLETED =========="
)