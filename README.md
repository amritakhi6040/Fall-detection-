# 🚨 Fall Detection System — MediaPipe + PAF + Multi-Model Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-purple?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab)

> A comprehensive, production-ready fall detection system that extracts **MediaPipe pose landmarks + Part Affinity Fields (PAF)** from videos and benchmarks **four classifiers** — LSTM, ANN, SVM, and XGBoost — with full evaluation metrics including ROC-AUC curves, confusion matrices, and real-time video inference with bounding box overlays.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Innovation — PAF Features](#key-innovation--paf-features)
- [System Pipeline](#system-pipeline)
- [Model Architectures](#model-architectures)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Real-Time Inference](#real-time-inference)
- [Training Details](#training-details)
- [Evaluation & Metrics](#evaluation--metrics)
- [Saved Models](#saved-models)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

---

## 🧠 Overview

This project goes beyond basic pose-based fall detection by introducing **Part Affinity Fields (PAF)** — directional limb vectors derived from MediaPipe skeleton connections — as additional features. These capture *how limbs are oriented and moving*, not just where joints are located, giving classifiers richer biomechanical context to distinguish a fall from normal activity.

Four machine learning models are trained and benchmarked on the same feature set:

| Model | Type | Input Shape |
|-------|------|-------------|
| **LSTM** | Deep Learning (Sequential) | `(30, FRAME_FEATURE_SIZE)` |
| **ANN** | Deep Learning (Feedforward) | `(FRAME_FEATURE_SIZE × 30,)` flattened |
| **SVM** | Classical ML | `(FRAME_FEATURE_SIZE × 30,)` flattened |
| **XGBoost** | Gradient Boosting | `(FRAME_FEATURE_SIZE × 30,)` flattened |

---

## 🔬 Key Innovation — PAF Features

Standard MediaPipe gives you **132 features/frame** (33 landmarks × 4 values each).

This project adds **Part Affinity Fields** on top:

```
For each limb connection (a → b):
  dx        = xb - xa          (horizontal direction)
  dy        = yb - ya          (vertical direction)
  length    = sqrt(dx² + dy²)  (limb length)
  conf      = visibility_a × visibility_b  (joint confidence)
```

With `32 limb pairs × 4 PAF features = 128 PAF features/frame`, the final combined feature vector is:

```
FRAME_FEATURE_SIZE = 132 (landmarks) + 128 (PAF) = 260 features/frame
Total per video    = 260 × 30 frames = 7,800 values
```

This allows models to understand body posture geometry and limb dynamics — critical signals for detecting falls.

---

## 🔄 System Pipeline

```
Video Files (.avi / .mp4 / .mov)
           ↓
  Extract 30 evenly-spaced frames
           ↓
  MediaPipe Pose Estimation
  → 33 landmarks × (x, y, z, vis) = 132 features/frame
           ↓
  PAF Computation (compute_paf_features_from_landmarks)
  → 32 limb pairs × (dx, dy, length, conf) = 128 features/frame
           ↓
  Combined Feature Vector: (30, 260) per video
           ↓
  ┌────────────────────────────────────────┐
  │  Sequential input → LSTM              │
  │  Flattened input  → ANN / SVM / XGB   │
  └────────────────────────────────────────┘
           ↓
  Binary Classification: ADL (0) | Fall (1)
```

---

## 🏗️ Model Architectures

### 1. LSTM Model
```
Input: (30, 260)
  ↓  Masking (mask_value=0.0)    ← ignores zero-padded frames
  ↓  LSTM (128 units, return_sequences=True)
  ↓  Dropout (0.3)
  ↓  LSTM (64 units)
  ↓  Dense (64, ReLU)
  ↓  Dropout (0.3)
  ↓  Dense (2, Softmax)
```

### 2. ANN Model
```
Input: (7800,)   ← flattened sequence
  ↓  Dense (256, ReLU)
  ↓  Dropout (0.3)
  ↓  Dense (128, ReLU)
  ↓  Dropout (0.3)
  ↓  Dense (2, Softmax)
```

### 3. SVM
- Kernel: **RBF**
- `probability=True` (enables confidence scores)
- Input: flattened pose+PAF feature vector
- Used for **real-time inference** in the sliding window detector

### 4. XGBoost
- `n_estimators=300`, `learning_rate=0.05`
- `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`
- `eval_metric='mlogloss'`

---

## 📂 Dataset Structure

```
video_1/
├── Subject_01/
│   ├── ADL/
│   │   ├── video_001.avi
│   │   └── video_002.mp4
│   └── Fall/
│       ├── video_001.avi
│       └── video_002.mp4
├── Subject_02/
│   ├── ADL/
│   └── Fall/
...
```

| Label | Class | Description |
|-------|-------|-------------|
| `0` | ADL | Activities of Daily Living (walking, sitting, bending) |
| `1` | Fall | Fall events (forward, backward, lateral, from chair) |

Supported formats: `.avi`, `.mp4`, `.mov`

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fall-detection-mediapipe-paf.git
cd fall-detection-mediapipe-paf
```

### 2. Install Dependencies

```bash
pip install tensorflow opencv-python mediapipe xgboost scikit-learn numpy matplotlib joblib
```

### 3. Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install mediapipe
```

Place your dataset at: `MyDrive/video_1/`

---

## 🚀 Usage

### Step 1: Extract Features (Landmarks + PAF)

```python
# For a single video — returns (30, 260) array
features = extract_pose_keypoints_with_paf("path/to/video.avi")
print(features.shape)  # (30, 260)
```

### Step 2: Build Full Dataset

```python
# Loads all subject videos, builds sequential and flattened versions
X_seq, X_flat, y_all = build_feature_dataset(dataset)

# X_seq  → shape (N, 30, 260) — for LSTM
# X_flat → shape (N, 7800)    — for ANN, SVM, XGBoost
```

### Step 3: Train/Test Split

```python
X_seq_train, X_seq_test, X_flat_train, X_flat_test, y_train, y_test = train_test_split(
    X_seq, X_flat, y_all, test_size=0.2, random_state=42, stratify=y_all
)
```

### Step 4: Train All Models

```python
# LSTM
lstm_model = build_lstm_model(num_classes=2)
lstm_model.fit(X_seq_train, y_train, epochs=100, batch_size=8, callbacks=[early_stop])
lstm_model.save("lstm_pose_model.h5")

# SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_flat_train, y_train)
joblib.dump(svm, "svm_pose_model.pkl")

# XGBoost
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6)
xgb.fit(X_flat_train, y_train)
joblib.dump(xgb, "xgb_pose_model.pkl")

# ANN
ann_model.fit(X_flat_train, y_train, epochs=50, batch_size=16)
```

---

## 🎥 Real-Time Inference

The system includes a **sliding window real-time fall detector** powered by the trained SVM model:

```python
detect_fall_realtime(
    video_path="path/to/video.mp4",
    output_path="annotated_output.mp4"   # optional: saves annotated video
)
```

**What it does frame-by-frame:**
- Runs MediaPipe pose estimation + PAF computation
- Maintains a rolling buffer of the last 30 frames
- Predicts every frame once the buffer fills
- Draws **color-coded bounding box** around the person:
  - 🟢 **Green** = ADL (normal activity)
  - 🔴 **Red** = Fall detected
- Overlays skeleton landmarks and limb connections
- Saves the annotated video to disk
- Compatible with both **Google Colab** (`cv2_imshow`) and local environments (`cv2.imshow`)

---

## 🔧 Training Details

| Parameter | LSTM | ANN | SVM | XGBoost |
|-----------|------|-----|-----|---------|
| Input type | Sequential `(30, 260)` | Flat `(7800,)` | Flat `(7800,)` | Flat `(7800,)` |
| Optimizer | Adam | Adam | — | — |
| Loss | Sparse CE | Sparse CE | — | mlogloss |
| Batch size | 8 | 16 | — | — |
| Max epochs | 100 | 50 | — | 300 trees |
| Early stopping | patience=10 | ✗ | — | — |
| Val split | 20% | 20% | — | — |
| Test split | 20% stratified | 20% stratified | 20% stratified | 20% stratified |

---

## 📊 Evaluation & Metrics

Each model is evaluated with:

- ✅ Confusion Matrix
- ✅ Classification Report (Precision, Recall, F1-Score per class)
- ✅ Overall Accuracy
- ✅ **ROC Curve + AUC Score** (LSTM & XGBoost)
- ✅ Training/Validation Accuracy & Loss curves (LSTM & ANN)

> ⚠️ Fill in your actual benchmark results after training:

| Model | Accuracy | AUC | F1 (Fall) |
|-------|----------|-----|-----------|
| LSTM | — | — | — |
| ANN | — | — | — |
| SVM (RBF) | — | — | — |
| XGBoost | — | — | — |

---

## 💾 Saved Models

After training, the following model files are generated:

| File | Model | Format |
|------|-------|--------|
| `lstm_pose_model.h5` | LSTM | Keras HDF5 |
| `svm_pose_model.pkl` | SVM | joblib pickle |
| `xgb_pose_model.pkl` | XGBoost | joblib pickle |

Load them for inference:

```python
from tensorflow.keras.models import load_model
import joblib

lstm = load_model("lstm_pose_model.h5")
svm  = joblib.load("svm_pose_model.pkl")
xgb  = joblib.load("xgb_pose_model.pkl")
```

---

## 📁 Project Structure

```
fall-detection-mediapipe-paf/
│
├── fall_detection_mediapipe.py    # Full pipeline: features + 4 models + real-time
├── lstm_pose_model.h5             # Saved LSTM weights (generated after training)
├── svm_pose_model.pkl             # Saved SVM model (generated after training)
├── xgb_pose_model.pkl             # Saved XGBoost model (generated after training)
├── README.md                      # Project documentation
│
└── dataset/                       # Your dataset (not tracked by git)
    └── video_1/
        └── Subject_XX/
            ├── ADL/
            └── Fall/
```

> Add a `.gitignore` to exclude `.h5`, `.pkl`, and dataset folders from version control.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥ 2.x | LSTM & ANN models |
| `mediapipe` | ≥ 0.10 | Pose landmark extraction |
| `opencv-python` | ≥ 4.x | Video I/O & real-time rendering |
| `xgboost` | ≥ 1.7 | Gradient boosting classifier |
| `scikit-learn` | ≥ 1.x | SVM, train/test split, metrics |
| `numpy` | ≥ 1.21 | Array operations |
| `matplotlib` | ≥ 3.4 | ROC curves & training plots |
| `joblib` | ≥ 1.x | Model serialization (SVM/XGB) |

---

## 🔭 Future Work

- [ ] Cache extracted PAF features as `.npy` to skip re-processing videos
- [ ] Add **Bidirectional LSTM** or **Temporal Transformer** on the PAF+landmark sequence
- [ ] Experiment with **attention mechanisms** over the 30-frame window
- [ ] Extend to **multi-person** fall detection (multiple pose tracks)
- [ ] Export LSTM to **TFLite** for mobile/embedded deployment
- [ ] Evaluate on standard benchmarks: UR Fall Detection Dataset, Le2i Fall Dataset
- [ ] Replace SVM sliding window inference with **LSTM streaming** for lower latency

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change or improve.

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Your Name**  
- GitHub: [@your-username](https://github.com/your-username)  
- Email: your.email@example.com

---

> ⭐ If this project helped your research or work, please give it a star — it helps others find it!
