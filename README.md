# Train & Predict Toolkit

This repo includes:
- `wav_inspector_gui.py`: GUI to extract features and CSV output from WAV files.
- `ml_train_predict.py`: CLI to train multiple models and run predictions.
- `ml_train_predict_gui.py`: GUI to train/predict using extracted CSV output.
- `dl_train_predict.py`: Deep learning CLI to train/predict with GPU acceleration.
- `dl_train_predict_gui.py`: GUI for deep learning training/prediction.

## Quick Start (GUI)
1) Generate samples (optional if you already have `output/`):
```
python3 wav_inspector_gui.py
```
2) Train and predict:
```
python3 ml_train_predict_gui.py
```
Deep learning (GPU):
```
python3 dl_train_predict_gui.py
```

## Training Output Layout
When you set an output folder in the training GUI or CLI:
- `models/` contains all saved model files.
- `train_report.csv` contains raw metrics in CSV form.
- `train_report.md` contains a readable summary table.
- `train_config.json` stores the training config.
- Deep learning output also includes:
  - `train_history.csv` for epoch-by-epoch loss/metric.
  - `plots/` with training curves and validation plots.

## Prediction Output Layout
When you set an output folder for prediction:
- `predictions.csv` contains `sample_id`, `predicted`, and `actual`.
- `plots/` contains `predict_plot.png` (scatter or bar chart).
- Deep learning output also includes:
  - `prediction_report.md`
  - `plots/` with confusion matrix or regression scatter/residuals (when ground truth exists).

## Data Format Assumptions
- Each sample folder must contain `summary.csv` and data CSVs (e.g., `raw.csv/`, `envelope.csv/`, `envelope_detrended.csv/`).
- The folder name encodes labels in the format:
  `shape_c_speed_material` (example: `10_c_100_p`).
- Labels:
  - `shape` (classification)
  - `speed` (regression, converted to angular speed in rad/s from gear codes)
  - `material` (classification)

Speed gear mapping (period in seconds, rad/s = 2*pi/period):
- 80 -> 2.522
- 100 -> 1.433
- 120 -> 0.983
- 140 -> 0.733
- 160 -> 0.600
- 180 -> 0.495
- 200 -> 0.4326
- 220 -> 0.397
- 240 -> 0.3536

## CLI Examples (Optional)
Train:
```
python3 ml_train_predict.py train --label shape --source envelope_detrended --train "/path/a;/path/b" --test "/path/c" --out /path/to/out
```
Predict:
```
python3 ml_train_predict.py predict --model /path/to/model.joblib --data "/path/x;/path/y" --out /path/to/predict_out
```

Deep learning train (GPU, auto split):
```
python3 dl_train_predict.py train --label speed --source envelope_detrended --test "/path/output" --out /path/to/out
```
Optional model/scheduler tuning (deep learning):
- `--base-ch`, `--blocks`, `--kernel`, `--dropout`, `--scheduler-patience`
Deep learning predict:
```
python3 dl_train_predict.py predict --model /path/to/out/models/best_model.pt --data "/path/output" --out /path/to/predict_out
```

## Notes
- The GUI accepts multiple files/folders and auto-detects valid sample folders.
- For random splits, select "All samples + random split" and set the test ratio.
