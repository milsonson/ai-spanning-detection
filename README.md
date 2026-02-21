# AI Rotation Analysis Toolkit

Audio signal processing and machine learning analysis toolkit for multi-label prediction (shape, rotation speed, material) from WAV audio files.

## Overview

This toolkit provides a complete pipeline for:
- Audio feature extraction from WAV files (time-domain and frequency-domain features)
- Traditional machine learning classification/regression (Random Forest, SVM, Gradient Boosting, etc.)
- Deep learning with 1D CNN models (PyTorch-based, GPU acceleration supported)
- Graphical user interfaces for no-code usage
- Automated visualization of training history and prediction results

## Project Structure

```
.
├── wav_inspector.py              # Core audio feature extraction module
├── wav_inspector_gui.py          # GUI for audio feature extraction
├── ml_train_predict.py           # Traditional ML training/prediction CLI
├── ml_train_predict_gui.py       # GUI for traditional ML training/prediction
├── dl_train_predict.py           # Deep learning training/prediction CLI
├── dl_train_predict_gui.py       # GUI for deep learning training/prediction
├── output/                       # Extracted feature data (example)
├── train_models/                 # Directory for saved trained models
└── recordings/                   # Original WAV recording files
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- PyTorch 2.0+
- joblib

## Quick Start (GUI)

### 1. Feature Extraction
```bash
python3 wav_inspector_gui.py
```
Select WAV files or folders. Extracted features will be saved as CSV files.

### 2. Model Training (Traditional ML)
```bash
python3 ml_train_predict_gui.py
```
Select feature data folder, set label type (shape/speed/material), start training.

### 3. Model Training (Deep Learning)
```bash
python3 dl_train_predict_gui.py
```
Supports GPU acceleration, suitable for large-scale datasets.

## Data Format

### Sample Folder Naming Convention
```
{shape}_{direction}_{speed}_{material}
```

Example: `10_c_100_p` represents:
- Shape ID: 10
- Direction: c (clockwise)
- Speed: 100 (corresponds to 0.3536 rad/s)
- Material: p (plastic)

### Speed Mapping Table

| Gear | Period (s) | Angular Velocity (rad/s) |
|------|------------|--------------------------|
| 80   | 2.522      | 2.49                     |
| 100  | 1.433      | 4.38                     |
| 120  | 0.983      | 6.39                     |
| 140  | 0.733      | 8.57                     |
| 160  | 0.600      | 10.47                    |
| 180  | 0.495      | 12.70                    |
| 200  | 0.4326     | 14.52                    |
| 220  | 0.397      | 15.82                    |
| 240  | 0.3536     | 17.77                    |

Angular velocity is calculated as: $\omega = 2\pi / T$

## CLI Usage

### Feature Extraction
```bash
python3 wav_inspector.py \
  --input /path/to/audio.wav \
  --output /path/to/output \
  --bandpass --channel auto --peaks 10
```

### Traditional ML Training
```bash
python3 ml_train_predict.py train \
  --label shape \
  --source envelope_detrended \
  --train "/path/train" \
  --test "/path/test" \
  --out /path/to/models
```

### Traditional ML Prediction
```bash
python3 ml_train_predict.py predict \
  --model /path/to/model.joblib \
  --data "/path/test" \
  --out /path/to/predictions
```

### Deep Learning Training
```bash
python3 dl_train_predict.py train \
  --label speed \
  --source envelope_detrended \
  --test "/path/data" \
  --out /path/to/models \
  --epochs 80 --batch-size 32 --device cuda
```

### Deep Learning Prediction
```bash
python3 dl_train_predict.py predict \
  --model /path/to/best_model.pt \
  --data "/path/test" \
  --out /path/to/predictions \
  --device cuda
```

## Output Directory Structure

### Training Output
```
models/                 # Saved model files
├── model1.joblib
├── model2.joblib
└── ...
train_report.csv        # Raw evaluation metrics
train_report.md         # Human-readable training report
train_config.json       # Training configuration
plots/                  # Training curves (deep learning only)
├── train_history.png
└── validation_curves.png
```

### Prediction Output
```
predictions.csv         # Prediction results
test_report.md          # Test report
plots/                  # Visualizations
├── predict_plot.png
└── confusion_matrix.png
```

## Supported Label Types

| Label Type | Task Type | Description |
|------------|-----------|-------------|
| shape      | Classification | Object shape ID |
| speed      | Regression | Rotation speed (rad/s) |
| material   | Classification | Material type |

## Advanced Configuration

### Deep Learning Model Tuning
```bash
python3 dl_train_predict.py train \
  --label shape \
  --base-ch 64 \
  --blocks 4 \
  --kernel 7 \
  --dropout 0.2 \
  --scheduler-patience 5
```

### Feature Extraction Parameters
- **Bandpass filter**: 50-8000 Hz
- **Envelope cutoff frequency**: Configurable (default 20Hz)
- **FFT peak count**: Configurable (default 10)
- **Data slicing**: Supports sliding window slicing

## License

MIT License

## References

- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [SciPy](https://scipy.org/) - Scientific computing library
