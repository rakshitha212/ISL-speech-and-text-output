import os
import json
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
dataset_dir = "dataset"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "gesture_model_lstm.h5")
labels_path = os.path.join(model_dir, "labels.json")

MAX_FRAMES = 30  # Increased to match SEQ_LEN in app.py
FEATURES = 150

# ==============================
# LOAD DATASET
# ==============================
print("[INFO] Loading dataset...")

X = []
y = []

classes = sorted([
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d))
])

print(f"[INFO] Detected {len(classes)} classes.")

for label_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Group files by video prefix
    video_groups = {}
    for file in os.listdir(class_dir):
        if file.endswith(".csv"):
            match = re.match(r"(.+)_f(\d+)\.csv", file)
            if match:
                prefix = match.group(1)
                frame_num = int(match.group(2))
                if prefix not in video_groups:
                    video_groups[prefix] = []
                video_groups[prefix].append((frame_num, file))
            else:
                # If it doesn't match the pattern, maybe it's a single file sequence?
                if "f" not in file:
                    prefix = file
                    if prefix not in video_groups:
                        video_groups[prefix] = [(0, file)]

    for prefix, frame_list in video_groups.items():
        frame_list.sort()
        
        sequence_data = []
        for _, filename in frame_list:
            file_path = os.path.join(class_dir, filename)
            try:
                # Some CSVs might have a header, some might not. 
                # Based on previous view, it has 0,1,2...149 as header.
                df = pd.read_csv(file_path)
                if not df.empty:
                    # Take all rows if it's a multi-row CSV, or just the first if it's single-row
                    data = df.values
                    for row in data:
                        if len(row) == FEATURES:
                            sequence_data.append(row)
            except:
                continue
        
        if len(sequence_data) < 3: 
            continue
            
        sequence = np.array(sequence_data)
        
        # ==============================
        # NORMALIZATION (Translation Invariant)
        # ==============================
        # Subtract first frame to get relative movement
        sequence = sequence - sequence[0]
        # Scale to [-1, 1] range
        sequence = sequence / (np.max(np.abs(sequence)) + 1e-6)

        # ==============================
        # FIX SEQUENCE LENGTH (Uniform Sampling)
        # ==============================
        if sequence.shape[0] > MAX_FRAMES:
            # Sample MAX_FRAMES evenly from the entire sequence
            # This ensures we see the start, middle, and end of the gesture
            indices = np.linspace(0, sequence.shape[0] - 1, MAX_FRAMES).astype(int)
            sequence = sequence[indices]
        else:
            padding = np.zeros((MAX_FRAMES - sequence.shape[0], FEATURES))
            sequence = np.vstack((sequence, padding))

        X.append(sequence)
        y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"[INFO] Total Samples: {len(X)}")

if len(X) == 0:
    print("[ERROR] No valid data found! Ensure CSVs are in dataset/[class]/video_fXX.csv format.")
    exit()

# ==============================
# DATA PREPARATION
# ==============================
with open(labels_path, "w") as f:
    json.dump(classes, f)

y_cat = to_categorical(y, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.1, random_state=42)

# ==============================
# BUILD IMPROVED MODEL
# ==============================
print("[INFO] Building Optimized LSTM Model...")
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_FRAMES, FEATURES)),
    BatchNormalization(),
    Dropout(0.2),
    
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================
# TRAIN MODEL
# ==============================
print("[INFO] Training...")
model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Final accuracy on test set: {acc * 100:.2f}%")

model.save(model_path)
print(f"[INFO] Model saved to {model_path}")