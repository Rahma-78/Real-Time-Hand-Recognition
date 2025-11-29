
![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# Real-Time Hand Gesture Recognition
This project implements a real-time hand gesture recognition system using computer vision and machine learning techniques.

## 🎯 Project Overview

The project uses OpenCV for video capture and MediaPipe for hand detection, combined with a pre-trained SVM classifier to recognize different hand gestures in real-time. It provides an efficient and accurate solution for hand gesture recognition suitable for various applications including sign language recognition, human-computer interaction, and gesture-based control systems.

## ✨ Features

- **Real-time hand detection and tracking** using MediaPipe
- **Hand gesture classification** using optimized SVM classifier
- **Visual feedback** with hand landmarks and gesture labels
- **Pre-trained model** support for immediate use
- **Multiple ML models comparison** (SVM, Random Forest, KNN, etc.)
- **Comprehensive evaluation metrics** (accuracy, F1-score, precision, recall)


## 🎥 Demo

[▶️ **Click here to watch the full demo video**](https://drive.google.com/file/d/1f_LoaRfVtp_51Yo8v2JlwGalIsvZYi3a/view?usp=sharing)
## 📁 Project Structure

```bash
├── models/
│   └── svm_winner.pkl          # Pre-trained optimized SVM model
├── notebooks/
│   └── Project_ML.ipynb        # Data analysis, training, and evaluation
├── src/
│   └── real_time_demo.py       # Main application script
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

```
## 🚀 Installation

### Prerequisites

- Python 3.7 or higher

### Quick Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/real-time-hand-recognition.git
cd real-time-hand-recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```


## 🎮 Usage

### Running the Demo

Run the main demo script:
```bash
python src/real_time_demo.py
```

The script will:
- Access your webcam
- Detect hands in real-time
- Classify gestures using the pre-trained SVM model
- Display results with visual feedback
- Save the output as `output.mp4`

**Controls:**
- Press `q` to quit the application
- The output video will be saved as `output.mp4`

### Training Your Own Model

For model training and analysis, refer to the `notebooks/Project_ML.ipynb` Jupyter notebook, which contains:
- Data preprocessing steps
- Model training code for multiple algorithms
- Hyperparameter tuning with GridSearchCV
- Evaluation metrics and visualization
- Model comparison and selection

## 📋 Dependencies

- `opencv-python` - Computer vision and video processing
- `mediapipe` - Hand detection and tracking
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `joblib` - Model serialization
- `pandas` - Data manipulation (for notebook)
- `matplotlib` - Visualization (for notebook)
- `jupyter` - Interactive computing environment (for notebook)

## 🧠 Model Details

The project uses a Support Vector Machine (SVM) classifier that has been optimized through GridSearchCV hyperparameter tuning. Key model characteristics:

The model is saved as `svm_winner.pkl` and can be loaded directly for inference.

## 📊 Training Data

The model was trained on the HaGRID dataset, which contains various hand gestures. The training process includes:

1. **Data preprocessing**:
   - Normalization of hand landmarks
   - Centering at wrist position
   - Scaling based on middle finger position

2. **Feature extraction**:
   - 21 hand landmarks (x, y, z coordinates)
   - Total of 63 features per sample

3. **Model evaluation**:
   - Cross-validation
   - Multiple metrics (accuracy, F1-score, precision, recall)
   - Comparison with other algorithms (Random Forest, KNN, etc.)

