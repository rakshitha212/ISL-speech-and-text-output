# 🤟 Indian Sign Language Detection (ISLD)

> **Breaking barriers in communication with AI-powered, real-time Sign Language translation.**

A full-stack web application that provides two-way communication between Indian Sign Language (ISL) and spoken/written language — empowering the deaf and hard-of-hearing community through the power of computer vision and deep learning.

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [🧠 Innovation: Custom Data Collection & Training](#-innovation-custom-data-collection--training)
- [Technology Stack](#️-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [How to Use](#-how-to-use)
  - [Gesture to Speech](#1-gesture-to-speech)
  - [Speech to Gesture](#2-speech-to-gesture)
  - [Train Model (Custom Dataset)](#3-train-model-custom-dataset)
- [Model Architecture](#-model-architecture)
- [Default Gesture Classes](#-default-gesture-classes)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌐 About the Project

In India, over **63 million people** rely on sign language as their primary means of communication. Despite this, a significant communication gap exists between the hearing and hearing-impaired communities in everyday environments — hospitals, schools, government offices, and public spaces.

The **Indian Sign Language Detection (ISLD)** system bridges this gap by using:
- **MediaPipe** for precise, real-time hand and pose landmark extraction from a standard webcam.
- **LSTM Neural Networks** that analyse sequences of 30 frames to understand the temporal nature of gestures (movement over time, not just static poses).
- A unique **in-browser training pipeline** that allows anyone to build a personalized, high-accuracy model without touching a single line of code.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| ✋ **Gesture to Speech** | Real-time LSTM-based recognition of ISL gestures, converted to text and spoken aloud. |
| 🎤 **Speech to Gesture** | Spoken words transcribed via Google Speech API and displayed as ISL sign videos or letter animations. |
| 🧠 **Custom Model Training** | Record your own gesture dataset in-browser and retrain the LSTM model with a single click. |
| 🔄 **Hot Model Reload** | After training, the new model is automatically loaded into memory — no server restart needed. |
| ➕ **Dynamic Word Management** | Add or remove gesture classes at any time; choose to retain, archive, or delete associated data. |
| 🌐 **Google Translate** | Built-in Google Translate widget for multilingual accessibility. |
| 📱 **Responsive Design** | Clean, modern UI that works on desktops and tablets. |

---

## 🧠 Innovation: Custom Data Collection & Training

### The Problem with Pre-Trained Models

Traditional sign language recognition systems rely on large, pre-collected datasets trained by researchers. While these models can achieve high accuracy in controlled environments, they suffer from several real-world limitations:

- **Environment Mismatch**: Models trained in a studio with perfect lighting fail in typical home or office conditions with variable backgrounds, shadows, and camera angles.
- **Individual Variation**: Sign language, like spoken language, has regional and personal dialects. A model trained on one person's signing style may not generalize to another's.
- **Rigid Class Structure**: Adding a new word or gesture requires access to the original training data and a full model retraining pipeline — typically only possible for researchers.
- **High Latency**: Heavy, generic models have large inference times unsuitable for real-time interactive applications.

### Our Solution: In-Browser, User-Driven Training

This project introduces a **dynamic data collection and model training pipeline** directly accessible through the web UI — a significant departure from the traditional static model approach.

#### ✅ Advantages of Our Custom Training Approach

1. **Environment-Aware Accuracy**
   - Data is recorded in *your* environment, with *your* camera, lighting, and background. The model learns what it will actually see — eliminating the single biggest source of error in pre-trained models.

2. **Personalized to the Individual**
   - Each user's signing style, speed, and hand size is captured directly. The model becomes a personal interpreter, not a generic one.

3. **Zero-Code Extensibility**
   - New gesture classes can be added through the UI in seconds. No Python knowledge or command-line access is required to extend the model's vocabulary.

4. **Fast, Focused Models**
   - Training on 5–15 custom words produces a model orders of magnitude smaller and faster than a generic 40-class model, enabling smoother real-time inference.

5. **Dataset Ownership & Privacy**
   - All data stays on the local machine. No gesture data is uploaded to any external server.

6. **Active Learning Ready**
   - If a gesture is being misidentified, the user can simply record more samples for that class and retrain — a manual form of active learning that continuously improves accuracy.

7. **Neutral State Detection**
   - The required `Neutral` class teaches the model to recognize "no gesture is being performed," eliminating false positives during idle periods — a critical feature absent from most academic models.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.1 |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Computer Vision** | MediaPipe (Holistic Landmark Detection), OpenCV |
| **Deep Learning** | TensorFlow 2.15, Keras (LSTM) |
| **ML Utilities** | Scikit-Learn (train/test split) |
| **Speech Recognition** | Google Web Speech API (browser-side) |
| **Database** | SQLite3 (gesture-to-video mapping) |
| **Concurrency** | Python `threading` (background training) |

---

## 📂 Project Structure

```
ISL-speech-and-text-output/
│
├── app.py                      # Main Flask app — routes, camera loop, inference
├── train_model.py              # LSTM training module (runs in background thread)
│
├── model/
│   ├── gesture_model_lstm.h5   # Trained LSTM model weights
│   └── labels.json             # Active class label mapping
│
├── dataset_custom/             # User-recorded gesture sequences
│   ├── active_words.json       # Source of truth for active training classes
│   ├── Hello/                  # Each word gets its own folder
│   │   ├── seq_001.npy         # 30-frame landmark sequence (shape: 30×150)
│   │   └── seq_002.npy
│   └── Neutral/
│       └── seq_001.npy
│
├── static/
│   ├── style.css               # Application-wide styling
│   ├── script.js               # Shared JavaScript logic
│   └── gestures/               # MP4 videos and GIF animations for Sign Library
│
├── templates/
│   ├── index.html              # Home page
│   ├── gesture.html            # Gesture to Speech page
│   ├── speech.html             # Speech to Gesture page
│   ├── train.html              # Dynamic Dataset & Training UI
│   └── about.html              # About page
│
├── database/
│   └── isl.db                  # SQLite: maps spoken words to gesture video files
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10.x** (recommended — TensorFlow 2.15 compatibility)
- A working **webcam**
- A modern browser (Chrome or Edge recommended for Web Speech API)

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/ISL-speech-and-text-output.git
cd ISL-speech-and-text-output
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📖 How to Use

### 1. Gesture to Speech

1. Navigate to **Gesture to Speech** from the top navigation bar.
2. Click **Enable Camera** — your webcam feed will appear.
3. Perform an ISL gesture in front of the camera.
4. The detected word will appear in the **Detected Word** panel and be spoken aloud automatically.

> **Note:** You must train a custom model first (see below) to get accurate results.

---

### 2. Speech to Gesture

1. Navigate to **Speech to Gesture**.
2. Click **Enable Microphone** and speak a word clearly.
3. The system will:
   - Display the recognized text.
   - Play the matching ISL sign video from the library, **or**
   - Spell it out letter-by-letter using ISL finger-spelling GIFs if no video is found.

---

### 3. Train Model (Custom Dataset)

This is the core innovation of this project. Follow these steps to build your own high-accuracy model:

**Step 1 — Select or Add Words**
- Navigate to the **Train Model** page.
- Default words are pre-loaded. Add new words by typing in the input box and clicking **+**.
- Remove words using the 🗑️ icon. Choose to keep or permanently delete the associated data.

**Step 2 — Record Gesture Data**
- Select a word from the list.
- Click **Enable Camera**.
- Click **Record 30 Frames** and perform the gesture. The button stays disabled until recording is confirmed complete by the server.
- Repeat **20–30 times** per word for best accuracy.
- ⚠️ Always record the **Neutral** class (hands resting) — this is critical for the model to distinguish intent from inactivity.

**Step 3 — Train**
- Click **Start Training**. The button disables immediately and shows a live progress bar.
- Training runs in a background thread — the UI remains fully responsive.
- Upon completion, the new model is **automatically reloaded into memory** — no server restart needed.
- You can immediately test the new model on the Gesture to Speech page.

---

## 🔬 Model Architecture

The gesture recognition model is a **stacked LSTM network** designed to capture the temporal dynamics of sign language gestures.

```
Input: (30 frames × 150 landmarks)
  │
  ├─ LSTM(64 units, return_sequences=True)
  ├─ Dropout(0.5)
  ├─ LSTM(32 units)
  ├─ Dropout(0.5)
  ├─ Dense(64, activation='relu')
  └─ Dense(N_classes, activation='softmax')
```

**Key design decisions:**
- **150 landmarks per frame**: 33 pose + 21 left hand + 21 right hand keypoints × (x, y, z) coordinates.
- **30-frame window**: ~1 second of gesture at 30fps, capturing full motion arc.
- **Raw landmark features**: No shifting-baseline normalization — preserves absolute spatial relationships between hands and body.
- **Rolling prediction average**: The last 5 predictions are averaged to smooth output and eliminate single-frame noise.
- **Neutral class**: Prevents false positives by giving the model an explicit "idle" state.

---

## 🤙 Default Gesture Classes

The system initializes with the following default words. All can be removed and replaced with any words you choose:

| # | Word | # | Word |
|---|---|---|---|
| 1 | Hello | 7 | Please |
| 2 | Thank you | 8 | Sorry |
| 3 | Yes | 9 | Good |
| 4 | No | 10 | Bad |
| 5 | Help | 11 | Eat |
| 6 | Water | **+** | **Neutral** *(Required)* |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) — Google's real-time ML pipeline for landmark detection.
- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning framework.
- [Kaggle ISL Dataset](https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks) — Reference dataset for initial exploration.
- [Font Awesome](https://fontawesome.com/) — UI icons.
- [Google Translate API](https://cloud.google.com/translate) — Multilingual widget.

---

&copy; 2026 Indian Sign Language Detection Project · Jupiter King Technology · Karnataka, India
