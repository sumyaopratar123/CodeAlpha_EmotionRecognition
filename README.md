🎙️ Emotion Recognition from Speech

A machine learning and deep learning project that recognizes human emotions from speech audio using MFCC-based speech signal processing and a Bidirectional LSTM neural network.

🎯 Objective

The objective of this project is to recognize human emotions from speech audio, specifically:

- Angry
- Happy
- Sad

The project uses MFCC features along with Delta and Delta-Delta features and applies a Bidirectional LSTM deep learning model for emotion classification.

✨ Features

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

📚 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The dataset contains speech recordings representing different emotions.

For this project, three emotions were selected:

- Angry
- Happy
- Sad

The original RAVDESS audio dataset is not included in this repository because of its large size.

🧠 Methodology

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
🔬 Feature Extraction

The following speech features are extracted:

MFCC

Mel-Frequency Cepstral Coefficients are used to represent important characteristics of the speech signal.

Delta

First-order temporal derivatives of MFCC features are calculated to capture changes in speech characteristics over time.

Delta-Delta

Second-order temporal derivatives are calculated to capture higher-level temporal variations.

The final feature representation contains:

40 MFCC
+ 40 Delta
+ 40 Delta-Delta
= 120 features
🧠 Deep Learning Model

The final deep learning model uses a Bidirectional LSTM architecture.

Architecture:

Input
 ↓
Bidirectional LSTM (64)
 ↓
Dropout (0.30)
 ↓
Bidirectional LSTM (32)
 ↓
Dropout (0.30)
 ↓
Dense (64, ReLU)
 ↓
Dropout (0.30)
 ↓
Dense (3, Softmax)

Total trainable parameters:

140,291
📊 Model Results
Bi-LSTM

Test Accuracy:

54.31%

Classification performance:

Emotion	Precision	Recall	F1-Score
Angry	0.64	0.71	0.68
Happy	0.40	0.26	0.31
Sad	0.53	0.67	0.59
SVM Baseline

An SVM model was also trained as a baseline/comparison model.

Test Accuracy:

76.72%

The SVM is included for model comparison, while the Bidirectional LSTM satisfies the deep learning requirement of the project.

📊 Confusion Matrix

Bi-LSTM confusion matrix:

[[27,  5,  6],
 [12, 10, 17],
 [ 3, 10, 26]]
🛠️ Technologies Used
Python
NumPy
Pandas
Librosa
SoundFile
Scikit-learn
TensorFlow
Keras
Matplotlib
Seaborn
Streamlit
Joblib
📁 Project Structure
CodeAlpha_EmotionRecognition/
│
├── app/
│   └── app.py
│
├── models/
│   ├── emotion_cnn.keras
│   ├── emotion_lstm.keras
│   ├── emotion_svm.joblib
│   ├── label_classes.npy
│   ├── lstm_label_classes.npy
│   └── svm_label_classes.npy
│
├── screenshots/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── lstm_training_accuracy.png
│   └── lstm_training_loss.png
│
├── src/
│   ├── extract_features.py
│   ├── train_model.py
│   ├── train_svm.py
│   └── train_lstm.py
│
├── requirements.txt
├── .gitignore
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/sumyaopratar123/CodeAlpha_EmotionRecognition.git

Go to the project directory:

cd CodeAlpha_EmotionRecognition

Create a virtual environment:

python -m venv .venv

Activate the environment on Windows:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
▶️ Run the Streamlit Application

From the project root:

streamlit run app/app.py

The application will open in your browser.

🎤 Using the Application
Open the Streamlit application.
Record your voice using the microphone option or upload an audio file.
Click Predict Emotion.
The application extracts MFCC, Delta and Delta-Delta features.
The Bi-LSTM model processes the features.
The predicted emotion and confidence are displayed.
📸 Screenshots

Training accuracy and loss graphs are available in the screenshots directory.

🚀 Future Improvements
Train using all eight RAVDESS emotions.
Increase the size and diversity of the training dataset.
Experiment with CNN-LSTM and CRNN architectures.
Use data augmentation for speech.
Improve generalization to real-world microphone recordings.
Add more speech emotion datasets such as TESS and EMO-DB.
Improve inference preprocessing consistency.
Deploy the application for public access.
👨‍💻 Author

Sumit Ganesh Ade

Computer Science & Engineering

Internship

CodeAlpha Machine Learning Internship

Project

Emotion Recognition from Speech

Support

24x7 Help Desk: sumitade324@gmail.com

🔗 GitHub

https://github.com/sumyaopratar123/CodeAlpha_EmotionRecognition