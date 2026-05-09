# Indian Sign Language (ISL) Recognition & Synthesis System

🚀 **Real-Time Communication Bridge for the Deaf and Hard of Hearing**

This project is a comprehensive translation system designed to bridge the gap between spoken language and Indian Sign Language (ISL). It features two-way communication: converting hand gestures into spoken text/audio, and converting spoken words into high-quality sign language videos.

---


## dataset used 
https://www.kaggle.com/datasets/kaushikyh/indian-sign-language-words-with-landmarks



## 📚 Supported Dataset Words
The system supports a wide range of Indian Sign Language (ISL) signs. These are categorized into real-time recognition classes (LSTM-based) and common synthesis phrases.

### ✋ Gesture Recognition Classes (41)
These words are used to train the LSTM model for real-time camera-based detection:

*   **Animals:** Animal, Bird, Cat, Cow, Dog, Fish, Horse, Mouse
*   **Time & Days:** Afternoon, Evening, Friday, Monday, Month, Morning, Hour, Minute
*   **Adjectives & Emotions:** Bad, Beautiful, Big, Cheap, Cold, Curved, Dry, Expensive, Famous, Fast, Flat, Good, Happy, Healthy, Hot, Light, Long, Loose, Loud
*   **Clothing & Objects:** Clothing, Dress, Hat
*   **People:** Blind, Deaf, Female

### 🎤 Speech Synthesis Phrases (Common)
In addition to the above, the system includes sign library entries for common interactions (accessible via Voice-to-Sign):

*   **Greetings:** Hello, Good Morning, Good Night, Thanks
*   **Basics:** Yes, No, Please, Sorry, Help
*   **Needs:** Water, Food, I Love You

### 🔠 Alphabet Finger-Spelling
The system supports full **A-Z finger-spelling** fallback. If a specific word is not in the library, the system will automatically spell it out letter-by-letter using ISL alphabet signs.


## 🌟 Key Features

### ✋ 1. Gesture to Speech (Recognition)
- **Real-Time Analysis**: Captures hand and pose landmarks using a webcam.
- **Deep Learning Core**: Uses a custom-trained **LSTM (Long Short-Term Memory)** neural network to recognize temporal movement patterns.
- **High Accuracy**: Trained on a custom dataset with 41+ classes (Animals, Days of the Week, Common Phrases, etc.).
- **Smart Filtering**: Includes confidence thresholding and stability logic to prevent "noisy" or flickering predictions.

### 🎤 2. Speech to Gesture (Synthesis)
- **Voice Recognition**: Converts spoken audio into text using the Google Speech Recognition API.
- **Video Library**: Plays high-quality **MP4 video recordings** of real ISL signs from the dataset.
- **Alphabet Fallback**: If a specific word video is missing, the system automatically spells it out letter-by-letter using ISL finger-spelling.

---

## 🏗️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | Python 3.10.6, Flask |
| **Frontend** | HTML5, Modern CSS (Glassmorphism), JavaScript |
| **Computer Vision** | MediaPipe (Landmark Extraction), OpenCV |
| **Deep Learning** | TensorFlow, Keras (LSTM Architecture) |
| **Speech Engine** | Google Speech Recognition API |
| **Database** | SQLite3 |
| **Processing** | FFmpeg (Video Optimization) |

---

## 📂 Project Structure

```text
ISL-Project/
├── app.py                # Main Flask Application
├── train_model.py        # LSTM Training Pipeline
├── process_videos.py     # Video conversion & DB sync utility
├── model/
│   ├── gesture_model_lstm.h5  # Trained LSTM weights
│   └── labels.json            # Class mapping
├── dataset/              # Raw data (Hand Landmark CSVs & MOV videos)
├── static/
│   ├── style.css         # Modern UI Design
│   ├── gestures/         # Processed MP4 and GIF sign library
│   └── script.js         # Frontend Logic
├── templates/            # HTML Pages (Home, Gesture, Speech, About)
├── database/
│   └── isl.db            # SQLite mapping for words to videos
└── requirements.txt      # Project Dependencies
```

---

## ⚙️ Setup & Installation

### 1. Environment Setup
**Requirement**: Python 3.10.6 (Recommended)

Create a virtual environment and install the required packages:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Preparation (Optional)
If you add new videos to the `dataset/` folder:
1. Run the video processor to generate web-friendly MP4s:
   ```bash
   python process_videos.py
   ```
2. Train the LSTM model on the new data:
   ```bash
   python train_model.py
   ```

### 3. Running the App
Start the Flask server:
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧠 Model Details

### Gesture Recognition (LSTM)
Unlike simple image classifiers, our model analyzes **sequences of movement** over 30 frames. 
- **Features**: 150 landmarks (Pose + Left Hand + Right Hand).
- **Normalization**: Translation-invariant normalization (relative to the first frame).
- **Architecture**: Multiple LSTM layers with Dropout and BatchNormalization for robust learning.

---

## 🎨 UI/UX Design
The application features a premium, modern design:
- **Responsive Layout**: Works on desktops and tablets.
- **Interactive UI**: Real-time feedback with "Listening..." animations and camera tracking status.
- **Theme**: A sleek Blue & White professional palette with clean typography.

---

&copy; 2025 Indian Sign Language Detection Project.
