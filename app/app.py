import os
import tempfile

import joblib
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "models/emotion_lstm.keras"
LABEL_PATH = "models/lstm_label_classes.npy"


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🎙️ Emotion Recognition from Speech")

st.write(
    "Upload a speech audio file or record your voice "
    "to detect the emotion using a Bi-LSTM deep learning model."
)

st.info(
    "Supported emotions: Angry, Happy, Sad"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    labels = np.load(
        LABEL_PATH,
        allow_pickle=True
    )

    return model, labels


try:

    model, labels = load_model()

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


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

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine
    features = np.concatenate(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=0
    )

    # Time x Features
    features = features.T

    # Same padding used during training
    max_length = 220

    if features.shape[0] < max_length:

        pad_width = (
            max_length - features.shape[0]
        )

        features = np.pad(
            features,
            (
                (0, pad_width),
                (0, 0)
            ),
            mode="constant"
        )

    else:

        features = features[
            :max_length,
            :
        ]

    # Normalize
    mean = np.mean(
        features,
        axis=0,
        keepdims=True
    )

    std = np.std(
        features,
        axis=0,
        keepdims=True
    )

    features = (
        features - mean
    ) / (
        std + 1e-8
    )

    return features.astype(
        np.float32
    )


# =========================================================
# AUDIO INPUT
# =========================================================

st.subheader("🎤 Record Your Voice")

recorded_audio = st.audio_input(
    "Click here and speak"
)


st.subheader("📁 Or Upload Audio")

uploaded_audio = st.file_uploader(
    "Upload a speech audio file",
    type=[
        "wav",
        "mp3",
        "m4a"
    ]
)


# =========================================================
# SELECT AUDIO
# =========================================================

audio_file = None

if recorded_audio is not None:

    audio_file = recorded_audio

elif uploaded_audio is not None:

    audio_file = uploaded_audio


# =========================================================
# PREDICTION
# =========================================================

if audio_file is not None:

    st.audio(
        audio_file
    )

    if st.button(
        "🔍 Predict Emotion",
        type="primary"
    ):

        with st.spinner(
            "Analyzing speech..."
        ):

            temp_path = None

            try:

                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:

                    temp_file.write(
                        audio_file.getvalue()
                    )

                    temp_path = temp_file.name


                # Extract features
                features = extract_features(
                    temp_path
                )

                # Add batch dimension
                features = np.expand_dims(
                    features,
                    axis=0
                )


                # Prediction
                probabilities = model.predict(
                    features,
                    verbose=0
                )[0]


                predicted_index = np.argmax(
                    probabilities
                )

                predicted_emotion = labels[
                    predicted_index
                ]

                confidence = (
                    probabilities[
                        predicted_index
                    ] * 100
                )


                # =================================================
                # RESULT
                # =================================================

                st.success(
                    f"🎯 Predicted Emotion: "
                    f"{str(predicted_emotion).upper()}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


                # =================================================
                # PROBABILITY
                # =================================================

                st.subheader(
                    "📊 Emotion Probabilities"
                )

                for emotion, probability in zip(
                    labels,
                    probabilities
                ):

                    st.write(
                        f"**{str(emotion).capitalize()}** "
                        f"- {probability * 100:.2f}%"
                    )

                    st.progress(
                        float(probability)
                    )


                # =================================================
                # DETAILS
                # =================================================

                st.subheader(
                    "🔬 Model Details"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.write(
                        "**Model:** Bi-LSTM"
                    )

                    st.write(
                        "**Feature:** MFCC"
                    )

                with col2:

                    st.write(
                        "**Additional Features:** "
                        "Delta + Delta-Delta"
                    )

                    st.write(
                        "**Classes:** 3"
                    )


            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

            finally:

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(
                        temp_path
                    )


# =========================================================
# INFORMATION
# =========================================================

st.divider()

st.subheader(
    "ℹ️ About this Project"
)

st.write(
    "This project recognizes human emotions from speech "
    "using speech signal processing and deep learning."
)

st.write(
    "MFCC, Delta and Delta-Delta features are extracted "
    "from the audio and processed using a Bidirectional "
    "LSTM neural network."
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CodeAlpha Machine Learning Internship | "
    "Emotion Recognition from Speech"
)

st.caption(
    "Support :- 24x7 Help Desk :- sumitade324@gmail.com"
)