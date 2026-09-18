import os
import numpy as np
import librosa

DATASET_PATH = "data/RAVDESS"
OUTPUT_PATH = "data/features.npz"

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

X = []
y = []

print("Starting MFCC feature extraction...")

for actor in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in sorted(os.listdir(actor_path)):
        if not filename.endswith(".wav"):
            continue

        try:
            # Emotion code is the 3rd value in RAVDESS filename
            emotion_code = filename.split("-")[2]
            emotion = EMOTIONS[emotion_code]

            file_path = os.path.join(actor_path, filename)

            # Load audio
            audio, sample_rate = librosa.load(file_path, sr=22050)

            # Extract 40 MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            # Fixed size for CNN
            max_length = 174

            if mfcc.shape[1] < max_length:
                pad_width = max_length - mfcc.shape[1]
                mfcc = np.pad(
                    mfcc,
                    ((0, 0), (0, pad_width)),
                    mode="constant"
                )
            else:
                mfcc = mfcc[:, :max_length]

            X.append(mfcc)
            y.append(emotion)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

X = np.array(X)
y = np.array(y)

# Add CNN channel dimension
X = X[..., np.newaxis]

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y
)

print("\nFeature extraction completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Saved to:", OUTPUT_PATH)