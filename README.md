# 🎙️ Emotion Recognition from Speech

A machine learning and deep learning project that recognizes human emotions from speech audio using MFCC-based speech signal processing and a Bidirectional LSTM neural network.

## 🎯 Objective

The objective of this project is to recognize human emotions from speech audio, specifically:

- Angry
- Happy
- Sad

The project uses MFCC features along with Delta and Delta-Delta features and applies a Bidirectional LSTM deep learning model for emotion classification.

## ✨ Features

- 🎤 Record speech directly from the browser
- 📁 Upload WAV, MP3, or M4A audio
- 🎵 Speech feature extraction using MFCC
- 📈 Delta and Delta-Delta feature extraction
- 🧠 Bidirectional LSTM deep learning model
- 📊 Emotion probability visualization
- 🎯 Emotion prediction with confidence
- 🌐 Streamlit web application
- 📊 Model performance evaluation
- 🔬 SVM baseline for comparison

## 📚 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The dataset contains speech recordings representing different emotions.

For this project, three emotions were selected:

- Angry
- Happy
- Sad

The original RAVDESS audio dataset is not included in this repository because of its large size.

## 🧠 Methodology

The overall workflow is:

```text
Speech Audio
     ↓
Audio Preprocessing
     ↓
MFCC Extraction
     ↓
Delta Features
     ↓
Delta-Delta Features
     ↓
Feature Combination
     ↓
Bidirectional LSTM
     ↓
Emotion Classification
     ↓
Prediction + Confidence