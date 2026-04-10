# TruthLens — Deepfake Detector

> Detect AI-generated or manipulated images and videos instantly using a custom-trained Convolutional Neural Network.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (model is already included)
python inference.py

# 3. Open browser
# http://127.0.0.1:5000
```

No training required — the pre-trained model (`deepfake_detector_model.keras`) is bundled.

---

## What It Does

Upload any image or video and the model tells you in seconds whether it is **REAL** or **FAKE** (AI-generated / deepfake), along with a confidence score. For images, a **Grad-CAM heatmap** is overlaid to highlight which regions influenced the prediction.

---

## How It Works

```
User uploads image/video
        ↓
Flask receives file → saves to /uploads
        ↓
Image: resized to 128×128 RGB → CNN inference → Grad-CAM overlay
Video: up to 16 frames sampled → per-frame CNN inference → mean score
        ↓
Sigmoid output:  ≥ 0.5 → FAKE   |   < 0.5 → REAL
        ↓
Result + confidence % returned to UI
        ↓
Uploaded file deleted
```

---

## The Model

| Property | Value |
|---|---|
| Architecture | Custom CNN (3 Conv2D blocks) |
| Input size | 128 × 128 × 3 (RGB) |
| Output | Sigmoid (0 = Real, 1 = Fake) |
| Loss | Binary Crossentropy |
| Optimizer | Adam (lr = 0.0001) |
| Test Accuracy | ~88% |
| Test Precision | ~94% |
| Training data | ~100,000 images (Kaggle Deepfake & Real Images) |
| Framework | TensorFlow 2.x / Keras |

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** TensorFlow 2.x, Keras, OpenCV (Grad-CAM)
- **Frontend:** Vanilla HTML/CSS/JS (no framework, fully self-contained)

---

## Project Structure

```
TruthLens/
├── inference.py                   # Flask web app + inference logic (entry point)
├── train.py                       # Training pipeline (model already trained)
├── deepfake_detector_model.keras  # Pre-trained model weights
├── requirements.txt               # Python dependencies
├── templates/
│   └── index.html                 # Frontend UI
├── static/
│   └── k.ico                      # Favicon
├── samples/                       # Demo test images
├── uploads/                       # Temp folder (auto-cleaned after each request)
└── data/                          # Dataset directory (gitignored)
    └── Dataset/
        ├── Train/
        ├── Test/
        └── Validation/
```

---

## Supported Formats

| Type | Extensions |
|---|---|
| Images | PNG, JPG, JPEG |
| Videos | MP4, MOV, AVI, MKV, WEBM |

---

## Training Your Own Model

```bash
# Place the dataset at data/Dataset/ with Train/, Test/, Validation/ subdirs
python train.py
```

This saves `deepfake_detector_model.keras` to the project root.

---

## Credits

- **Dataset:** Trung-Nghia Le (Kaggle: deepfake-and-real-images)
- **Model & App:** TruthLens team
