import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
import threading

# Global training state for the UI
training_state = {
    "is_training": False,
    "progress": 0,
    "message": "Idle",
    "error": None,
    "reload_model": False
}

class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / self.total_epochs) * 100)
        training_state["progress"] = progress
        acc = logs.get('accuracy', 0)
        loss = logs.get('loss', 0)
        training_state["message"] = f"Training epoch {epoch+1}/{self.total_epochs} (Loss: {loss:.4f}, Acc: {acc:.4f})"

def train_custom_model_thread():
    try:
        training_state["is_training"] = True
        training_state["progress"] = 0
        training_state["message"] = "Starting training..."
        training_state["error"] = None
        training_state["reload_model"] = False
        
        dataset_dir = "dataset_custom"
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "gesture_model_lstm.h5")
        labels_path = os.path.join(model_dir, "labels.json")

        # Delete old model so stale weights never contaminate inference
        if os.path.exists(model_path):
            os.remove(model_path)
            print("[TRAIN] Deleted old model before retraining.")
        if os.path.exists(labels_path):
            os.remove(labels_path)

        MAX_FRAMES = 30
        FEATURES = 150

        training_state["message"] = "Loading dataset..."
        
        X = []
        y = []

        active_file = os.path.join(dataset_dir, "active_words.json")
        if os.path.exists(active_file):
            with open(active_file, "r") as f:
                classes = json.load(f)
        else:
            classes = sorted([
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
            ])

        if len(classes) == 0:
            raise Exception("No active classes found for training.")

        for label_idx, class_name in enumerate(classes):
            class_dir = os.path.join(dataset_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".npy"):
                    file_path = os.path.join(class_dir, file)
                    sequence = np.load(file_path)
                    
                    if sequence.shape == (MAX_FRAMES, FEATURES):
                        X.append(sequence)
                        y.append(label_idx)

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            raise Exception("No valid .npy data found! Please record some gestures.")

        with open(labels_path, "w") as f:
            json.dump(classes, f)

        y_cat = to_categorical(y, num_classes=len(classes))
        
        # Handle cases with very few samples gracefully
        test_size = 0.1 if len(X) > 10 else 0.0
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y_cat, y_cat

        training_state["message"] = "Building model..."
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(MAX_FRAMES, FEATURES)),
            Dropout(0.5),
            LSTM(32, return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(len(classes), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        training_state["message"] = "Training..."
        
        epochs = 100
        progress_callback = ProgressCallback(epochs)
        
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(32, max(1, len(X_train) // 2)),
            validation_data=(X_test, y_test) if test_size > 0 else None,
            verbose=0,
            callbacks=[progress_callback]
        )

        training_state["message"] = "Saving model..."
        model.save(model_path)
        
        training_state["progress"] = 100
        training_state["message"] = "Training complete! Reloading model..."
        training_state["reload_model"] = True
        
    except Exception as e:
        training_state["error"] = str(e)
        training_state["message"] = f"Error: {str(e)}"
    finally:
        training_state["is_training"] = False

def start_training():
    if training_state["is_training"]:
        return False, "Training is already running."
    
    thread = threading.Thread(target=train_custom_model_thread, daemon=True)
    thread.start()
    return True, "Training started."

def get_training_status():
    return training_state