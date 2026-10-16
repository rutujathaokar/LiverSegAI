# Liver Disease Segmentation from MRI using UNet

An end-to-end medical deep learning system for **semantic segmentation and morphological characterization of liver lesions** from axial abdominal Magnetic Resonance Imaging (MRI) scans.

Built with **PyTorch**, a clinical **Flask backend API**, and a modern medical dashboard (**HTML5, CSS3, Vanilla JavaScript, Chart.js**).

---

## 🌟 Key Highlights

- **Standard UNet Architecture**: 4-level symmetric encoder-decoder network with concatenative skip connections and 1x1 Sigmoid classification head.
- **Combined BCE + Dice Loss**: Formulated as $\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$ to tackle severe foreground/background class imbalance.
- **Large MRI Cohort (2,200 Slices)**: Formatted to 256×256 grayscale, normalized to $[0.0, 1.0]$, and strictly binarized. Split into **80% Train (1,760 slices)**, **10% Validation (220 slices)**, and **10% Test (220 slices)**.
- **Automated Kaggle API Pipeline**: `download_data.py` supports direct Kaggle downloading (`mrinmoves/cirrmri600`, CHAOS) via `kaggle.json`, plus a high-fidelity synthetic MRI generation fallback.
- **Morphological Tumor Classifier**: Extracts geometric and textural biomarkers (**Shape Solidity**, **Edge Roughness Index**, **Lesion Area in mm²**, and **Lesion Multiplicity**) to estimate **Benign (Non-Cancerous)** vs **Malignant (Cancerous)** risk with confidence scoring.
- **Interactive Medical Dashboard**: Dark clinical sidebar, clean white cards, 4-image quad view (Original MRI, Ground Truth Mask, UNet Prediction, Multi-channel Diagnostic Overlay), live Chart.js training curves, and print/PDF clinical report export.

---

## 🗂️ Project Directory Structure

```text
LiverSeg/
├── data/
│   ├── raw/                             # Downloaded raw archives (CirrMRI600+, CHAOS)
│   └── processed/                       # 256x256 preprocessed cohort
│       ├── train/                       # 1,760 slices (images/ and masks/)
│       ├── val/                         # 220 slices (images/ and masks/)
│       ├── test/                        # 220 slices (images/ and masks/)
│       └── metadata.json                # Cohort metadata, split counts, and distributions
├── model/
│   ├── __init__.py
│   ├── unet.py                          # PyTorch UNet architecture with skip connections
│   ├── metrics.py                       # Combined BCE+Dice loss, Dice, IoU, Precision, Recall
│   ├── tumor_classifier.py              # Clinical morphological feature extraction & scoring
│   ├── best_model.pth                   # Checkpoint of highest validation Dice score
│   ├── training_curves.png              # Loss trajectory and Dice progression plots
│   ├── training_history.json            # Epoch-by-epoch history for Chart.js dashboard
│   └── test_metrics.json                # Quantitative test set evaluation summary
├── static/
│   ├── css/
│   │   └── style.css                    # Clinical UI design system & responsive layout
│   └── js/
│       └── main.js                      # Chart.js integration, random sample picker, /predict API
├── templates/
│   └── index.html                       # Medical dashboard SPA template
├── download_data.py                     # Data acquisition and preprocessing script
├── train.py                             # UNet training script with early stopping
├── evaluate.py                          # Test evaluation and montage visual generator
├── app.py                               # Flask REST API server
├── requirements.txt                     # Python package dependencies
└── README.md                            # Complete documentation & run guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download / Prepare the Dataset
The script checks for your Kaggle API token (`~/.kaggle/kaggle.json`). If credentials are not yet configured, it automatically invokes the realistic abdominal MRI synthesis engine to generate 2,200 slices:
```bash
python download_data.py --slices 2200
```

> **Using your own Kaggle account:**
> 1. Download `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/settings).
> 2. Save it to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<Username>\.kaggle\kaggle.json`).
> 3. Run `python download_data.py --dataset anhoangvo/chaos-t1-and-t2` (or `andrewmvd/lits-png`).

### 4. Model Training & Serialization (Pickle .pkl)
The trained model is serialized into standard Python pickle format (`model/best_model.pkl` and `model/unet_model.pkl`) to prevent redundant re-training:
```bash
# Checks if best_model.pkl exists; skips retraining if already saved
python train.py

# Force retraining from scratch if needed
python train.py --retrain --epochs 30 --batch-size 16 --base-features 8
```
This saves:
- `model/best_model.pkl` (Pickled model bundle for instant reuse without retraining)
- `model/unet_model.pkl` (Direct pickled UNet model object)
- `model/best_model.pth` (PyTorch state checkpoint)
- `model/training_curves.png`
- `model/training_history.json`

### 5. Evaluate on the Test Set
Compute final Dice, IoU, Precision, Recall, Specificity, and generate a 4-sample comparison montage:
```bash
python evaluate.py
```
This outputs `model/test_metrics.json` and `model/test_predictions_sample.png`.

### 6. Launch the Medical Web Application
Start the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:5000`**

---

## 🔬 Clinical Methodology & Architecture

### 1. UNet Contracting-Expansive Architecture
```text
Input (1, 256, 256)
  │
  ├── DoubleConv [3x3, f=8] ─────────────── Skip 1 ──────────────┐
  │   ▼ MaxPool(2)                                               │
  ├── DoubleConv [3x3, f=16] ────────────── Skip 2 ────────┐     │
  │   ▼ MaxPool(2)                                         │     │
  ├── DoubleConv [3x3, f=32] ────────────── Skip 3 ──┐     │     │
  │   ▼ MaxPool(2)                                   │     │     │
  ├── DoubleConv [3x3, f=64] ──────── Skip 4 ──┐     │     │     │
  │   ▼ MaxPool(2)                             │     │     │     │
  └── Bottleneck DoubleConv [3x3, f=128]       │     │     │     │
      ▼ UpConv(2)                              │     │     │     │
      Cat + DoubleConv [3x3, f=64] ◄───────────┘     │     │     │
      ▼ UpConv(2)                                    │     │     │
      Cat + DoubleConv [3x3, f=32] ◄─────────────────┘     │     │
      ▼ UpConv(2)                                          │     │
      Cat + DoubleConv [3x3, f=16] ◄───────────────────────┘     │
      ▼ UpConv(2)                                                │
      Cat + DoubleConv [3x3, f=8]  ◄─────────────────────────────┘
      ▼ Conv 1x1 + Sigmoid
Output Mask (1, 256, 256)
```

### 2. Loss Formulation
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \left(1 - \frac{2 \sum (p_i \cdot g_i) + \epsilon}{\sum p_i + \sum g_i + \epsilon}\right)$$

### 3. Rule-Based Morphological Characterization

Since standard segmentation benchmarks do not feature histopathological labels, a clinical morphological analyzer evaluates the predicted lesion:

| Morphological Feature | Mathematical Definition | Clinical Interpretation |
| :--- | :--- | :--- |
| **Solidity** | $\frac{\text{Contour Area}}{\text{Convex Hull Area}}$ | High ($> 0.88$) $\implies$ Smooth/Benign.<br>Low ($< 0.80$) $\implies$ Infiltrative/Malignant. |
| **Edge Roughness Index** | $\frac{\text{Contour Perimeter}}{\text{Convex Hull Perimeter}}$ | High ($> 1.25$) $\implies$ Spiculation/Malignancy.<br>Low ($\approx 1.0$) $\implies$ Circumscribed/Benign. |
| **Lesion Multiplicity** | Count of disconnected components | Multiple discrete foci $\implies$ Multifocal HCC or Metastases. |
| **Volume / Area** | Pixels $\times (1.25\text{ mm})^2$ | Large lesion mass elevates clinical risk profile. |

> **Medical Disclaimer Banner**: *"Demo estimate - not a medical diagnosis. For investigational machine learning research and educational benchmarking only."*

---

## 📡 REST API Reference

### `GET /api/stats`
Returns global model metrics, slice counts, and training epoch history for Chart.js.

### `GET /api/random-test-sample`
Randomly selects an unseen slice from `data/processed/test/`, returning base64 encoded strings for the MRI and ground truth mask.

### `POST /api/predict`
Accepts `image` (file or base64) and optional `mask` (file or base64).
**Response:**
```json
{
  "success": true,
  "has_ground_truth": true,
  "images": {
    "original": "data:image/png;base64,...",
    "ground_truth": "data:image/png;base64,...",
    "predicted": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,..."
  },
  "metrics": {
    "dice": 0.924,
    "iou": 0.858,
    "precision": 0.941,
    "recall": 0.908,
    "accuracy": 0.994
  },
  "features": {
    "area_mm2": 1420.5,
    "total_pixels": 909,
    "solidity": 0.742,
    "roughness": 1.341,
    "lesion_count": 2
  },
  "classification": {
    "prediction": "Malignant (Cancerous)",
    "confidence": 92.4,
    "badge_class": "badge-malignant",
    "summary": "Morphological features demonstrate high-risk characteristics...",
    "reasons": [
      "Infiltrative/jagged lesion contour (Solidity: 0.74 < 0.80)",
      "Significant border spiculation (Roughness: 1.34)"
    ],
    "disclaimer": "Demo estimate - not a medical diagnosis."
  }
}
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
