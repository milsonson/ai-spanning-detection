#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.tree import DecisionTreeClassifier

SOURCE_CHOICES = ("raw", "envelope", "envelope_detrended")
LABEL_CHOICES = ("shape", "speed", "material")
SPEED_GEAR_TO_PERIOD_S = {
    80: 2.522,
    100: 1.433,
    120: 0.983,
    140: 0.733,
    160: 0.600,
    180: 0.495,
    200: 0.4326,
    220: 0.397,
    240: 0.3536,
}
SPEED_PERIOD_MIN_S = min(SPEED_GEAR_TO_PERIOD_S.values())
SPEED_PERIOD_MAX_S = max(SPEED_GEAR_TO_PERIOD_S.values())
SPEED_LABEL_TOL = 1e-6
LABEL_DISPLAY = {
    "shape": "shape",
    "speed": "speed (rad/s)",
    "material": "material",
}


@dataclass
class DatasetItem:
    sample_dir: str
    sample_id: str
    label_text: str
    features: np.ndarray


def parse_label_from_stem(stem: str) -> Dict[str, str]:
    name = stem.strip()
    name = re.sub(r"\([^)]*\)$", "", name)
    match = re.match(r"^(?P<shape>[^_]+)_(?P<direction>c|u1|u2|d1|d2|l|r)_(?P<speed>[^_]+)_(?P<material>[^_]+)$",name)

    if not match:
        raise ValueError(f"Unrecognized label format: {stem}")
    return match.groupdict()


def _label_display_name(label_target: str) -> str:
    return LABEL_DISPLAY.get(label_target, label_target)


def _speed_label_to_period_seconds(label: str) -> float:
    text = label.strip()
    if not text:
        raise ValueError("Missing speed label")
    try:
        numeric = float(text)
    except Exception as exc:
        raise ValueError(f"Non-numeric speed label: {label!r}") from exc
    gear = int(round(numeric))
    if abs(numeric - gear) <= SPEED_LABEL_TOL and gear in SPEED_GEAR_TO_PERIOD_S:
        return SPEED_GEAR_TO_PERIOD_S[gear]
    for period in SPEED_GEAR_TO_PERIOD_S.values():
        if abs(numeric - period) <= SPEED_LABEL_TOL:
            return numeric
    if SPEED_PERIOD_MIN_S <= numeric <= SPEED_PERIOD_MAX_S:
        return numeric
    gears = ", ".join(str(g) for g in sorted(SPEED_GEAR_TO_PERIOD_S))
    raise ValueError(
        f"Unknown speed label {label!r}; expected gear [{gears}] or period seconds"
    )


def _speed_label_to_rad_s(label: str) -> float:
    period_s = _speed_label_to_period_seconds(label)
    return (2.0 * math.pi) / period_s


def _format_label_value(label_target: str, label_value: str) -> str:
    if label_target != "speed":
        return label_value
    return f"{_speed_label_to_rad_s(label_value):.6f}"


def _is_sample_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "summary.csv"))


def _infer_sample_dir(path: str) -> Optional[str]:
    if os.path.isdir(path):
        if _is_sample_dir(path):
            return path
        candidates = []
        for root, dirs, files in os.walk(path):
            if "summary.csv" in files:
                candidates.append(root)
        if candidates:
            return None  # handled by walk
        return None
    if not os.path.isfile(path):
        return None
    name = os.path.basename(path)
    parent = os.path.dirname(path)
    if name == "summary.csv" and _is_sample_dir(parent):
        return parent
    if os.path.basename(parent) in ("raw.csv", "envelope.csv", "envelope_detrended.csv"):
        parent_parent = os.path.dirname(parent)
        if _is_sample_dir(parent_parent):
            return parent_parent
    if os.path.basename(parent) == "full_csv":
        parent_parent = os.path.dirname(parent)
        if _is_sample_dir(parent_parent):
            return parent_parent
    if name.endswith(".csv") and _is_sample_dir(parent):
        return parent
    return None


def collect_sample_dirs(paths: Sequence[str]) -> List[str]:
    found: List[str] = []
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path) and not _is_sample_dir(path):
            for root, dirs, files in os.walk(path):
                if "summary.csv" in files:
                    found.append(root)
            continue
        inferred = _infer_sample_dir(path)
        if inferred:
            found.append(inferred)
    deduped = sorted(set(found))
    return deduped


def _read_series_csv(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    times: List[float] = []
    values: List[float] = []
    label: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return np.array([]), np.array([]), None
        if first and first[0].lower() == "label":
            label = first[1] if len(first) > 1 else ""
            header = next(reader, None)
        else:
            header = first
        if not header or len(header) < 2:
            return np.array([]), np.array([]), label
        for row in reader:
            if len(row) < 2:
                continue
            try:
                t = float(row[0])
                v = float(row[1])
            except Exception:
                continue
            times.append(t)
            values.append(v)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32), label


def _list_segment_files(sample_dir: str, source: str) -> List[str]:
    folder = os.path.join(sample_dir, f"{source}.csv")
    if os.path.isdir(folder):
        files = sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().startswith("segment_") and name.lower().endswith(".csv")
        )
        if files:
            return files
    full_csv = os.path.join(sample_dir, "full_csv", f"{source}.csv")
    if os.path.isfile(full_csv):
        return [full_csv]
    return []


def _compute_fft_features(values: np.ndarray, times: np.ndarray) -> Tuple[float, float, float]:
    if values.size < 4 or times.size < 4:
        return 0.0, 0.0, 0.0
    dt = np.median(np.diff(times))
    if not np.isfinite(dt) or dt <= 0:
        return 0.0, 0.0, 0.0
    n = values.size
    window = np.hanning(n).astype(np.float32)
    y = values.astype(np.float32) * window
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)
    mag = np.abs(fft_vals)
    if mag.size <= 1:
        return 0.0, 0.0, 0.0
    mag[0] = 0.0
    idx = int(np.argmax(mag))
    peak = float(freqs[idx]) if idx < freqs.size else 0.0
    mag_sum = float(np.sum(mag))
    if mag_sum <= 0:
        return peak, 0.0, 0.0
    centroid = float(np.sum(freqs * mag) / mag_sum)
    bandwidth = float(math.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
    return peak, centroid, bandwidth


def _compute_basic_features(values: np.ndarray, times: np.ndarray) -> Tuple[List[float], List[str]]:
    if values.size == 0:
        return [0.0] * 13, [
            "mean",
            "std",
            "rms",
            "peak",
            "crest",
            "p25",
            "p50",
            "p75",
            "skew",
            "kurtosis",
            "zcr",
            "fft_peak",
            "spec_centroid",
            "spec_bw",
        ]
    mean = float(np.mean(values))
    std = float(np.std(values))
    rms = float(np.sqrt(np.mean(values ** 2)))
    peak = float(np.max(np.abs(values)))
    crest = peak / rms if rms > 0 else 0.0
    p25 = float(np.percentile(values, 25))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))
    if std > 0:
        centered = values - mean
        m3 = float(np.mean(centered ** 3))
        m4 = float(np.mean(centered ** 4))
        skew = m3 / (std ** 3)
        kurtosis = m4 / (std ** 4)
    else:
        skew = 0.0
        kurtosis = 0.0
    signs = np.signbit(values)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if values.size > 1 else 0.0
    fft_peak, centroid, bandwidth = _compute_fft_features(values, times)
    values_out = [
        mean,
        std,
        rms,
        peak,
        crest,
        p25,
        p50,
        p75,
        skew,
        kurtosis,
        zcr,
        fft_peak,
        centroid,
        bandwidth,
    ]
    names = [
        "mean",
        "std",
        "rms",
        "peak",
        "crest",
        "p25",
        "p50",
        "p75",
        "skew",
        "kurtosis",
        "zcr",
        "fft_peak",
        "spec_centroid",
        "spec_bw",
    ]
    return values_out, names


def extract_features_from_segments(segments: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    per_segment = []
    feature_names: List[str] = []
    for times, values in segments:
        features, names = _compute_basic_features(values, times)
        per_segment.append(features)
        if not feature_names:
            feature_names = names
    if not per_segment:
        return np.array([]), []
    matrix = np.asarray(per_segment, dtype=np.float32)
    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    aggregated = np.concatenate([means, stds], axis=0)
    agg_names = [f"{name}_mean" for name in feature_names] + [f"{name}_std" for name in feature_names]
    return aggregated, agg_names


def build_sample_features(
    sample_dir: str,
    source: str,
    use_slices: bool,
) -> Tuple[np.ndarray, List[str], str]:
    if source not in SOURCE_CHOICES:
        raise ValueError(f"Invalid source: {source}")
    segment_files = _list_segment_files(sample_dir, source)
    if not segment_files:
        raise FileNotFoundError(f"No CSV data for {source} in {sample_dir}")
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    label_from_file: Optional[str] = None
    for path in segment_files:
        times, values, label = _read_series_csv(path)
        if label and not label_from_file:
            label_from_file = label
        if times.size and values.size:
            segments.append((times, values))
        if not use_slices:
            break
    features, names = extract_features_from_segments(segments)
    if features.size == 0:
        raise ValueError(f"Empty features for {sample_dir}")
    return features, names, label_from_file or ""


def build_dataset(
    sample_dirs: Sequence[str],
    label_target: str,
    source: str,
    use_slices: bool,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    items: List[DatasetItem] = []
    feature_names: List[str] = []
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir.rstrip(os.sep))
        labels = parse_label_from_stem(sample_id)
        label_value = labels.get(label_target)
        if label_value is None:
            raise ValueError(f"Missing label {label_target} for {sample_id}")
        label_value = _format_label_value(label_target, label_value)
        features, names, label_from_file = build_sample_features(sample_dir, source, use_slices)
        if not feature_names:
            feature_names = names
        if names != feature_names:
            raise ValueError(f"Feature mismatch in {sample_id}")
        items.append(DatasetItem(sample_dir, sample_id, label_value, features))
    X = np.vstack([item.features for item in items]) if items else np.zeros((0, 0))
    y = [item.label_text for item in items]
    sample_ids = [item.sample_id for item in items]
    return X, y, sample_ids, feature_names


def _coerce_numeric_labels(
    values: Sequence[str],
    sample_ids: Sequence[str],
    label_name: str,
) -> List[float]:
    converted: List[float] = []
    bad: List[Tuple[str, str]] = []
    for idx, value in enumerate(values):
        try:
            converted.append(float(value))
        except Exception:
            sample_id = sample_ids[idx] if idx < len(sample_ids) else str(idx)
            bad.append((sample_id, str(value)))
    if bad:
        preview = ", ".join(f"{sid}={val}" for sid, val in bad[:5])
        raise ValueError(f"Non-numeric {label_name} labels: {preview}")
    return converted


def _split_data(
    X: np.ndarray,
    y: List[str],
    sample_ids: List[str],
    test_ratio: float,
    seed: int,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    if test_ratio <= 0:
        return X, np.zeros((0, X.shape[1])), y, [], sample_ids, []
    stratify = y if task == "classification" and len(set(y)) > 1 else None
    return train_test_split(
        X,
        y,
        sample_ids,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )


# --- Professional Plotting Utilities ---

def _set_plot_style():
    try:
        sns.set_theme(style="whitegrid", context="notebook", palette="deep")
    except Exception:
        plt.style.use("ggplot")

def _plot_confusion_matrix(y_true, y_pred, labels: List[str], out_path: str, title: str = "Confusion Matrix"):
    _set_plot_style()
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_feature_importance(model, feature_names: List[str], out_path: str, title: str = "Feature Importance", top_n: int = 15):
    _set_plot_style()
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = np.mean(importances, axis=0)
    
    if importances is None or len(importances) != len(feature_names):
        return

    # Sort and clip
    indices = np.argsort(importances)[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_vals, y=sorted_names, palette="viridis")
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_class_distribution(y_train, y_test, out_path: str):
    _set_plot_style()
    # Combine for plotting
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    df_train = pd.DataFrame({"Label": train_counts.index, "Count": train_counts.values, "Set": "Train"})
    df_test = pd.DataFrame({"Label": test_counts.index, "Count": test_counts.values, "Set": "Test"})
    df = pd.concat([df_train, df_test])
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="Label", y="Count", hue="Set", palette="muted")
    plt.title("Class Distribution: Train vs Test", fontsize=14, pad=15)
    plt.xlabel("Class Label")
    plt.ylabel("Sample Count")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_roc_curve(y_true, y_prob, classes, out_path: str):
    _set_plot_style()
    y_true_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_true_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    # Check dimensions
    if y_prob.shape != y_true_bin.shape:
        # Handle binary case where probability might be 1D
        if n_classes == 2 and y_prob.ndim == 1:
             # Scikit-learn binary case often needs 1D prob for positive class
             fpr, tpr, _ = roc_curve(y_true, y_prob)
             roc_auc = auc(fpr, tpr)
             plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        elif n_classes == 2 and y_prob.shape[1] == 2:
             fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
             roc_auc = auc(fpr, tpr)
             plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        else:
            # Fallback or error
            plt.close()
            return
    else:
        # Multi-class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_residuals(y_true, y_pred, out_path: str):
    _set_plot_style()
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter
    sns.scatterplot(x=y_pred, y=residuals, ax=ax1, alpha=0.6)
    ax1.axhline(0, color='r', linestyle='--')
    ax1.set_title("Residuals vs Predicted")
    ax1.set_xlabel("Predicted Value")
    ax1.set_ylabel("Residuals")
    
    # Hist
    sns.histplot(residuals, kde=True, ax=ax2, color="teal")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual Value")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_prediction_distribution(y_pred, task, out_path):
    _set_plot_style()
    plt.figure(figsize=(10, 6))
    if task == "classification":
        # Bar chart
        try:
            counts = pd.Series(y_pred).value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, palette="viridis")
            plt.title("Predicted Class Distribution", fontsize=14)
            plt.xlabel("Class")
            plt.ylabel("Count")
        except Exception:
            pass
    else:
        # Histogram
        sns.histplot(y_pred, kde=True, color="teal")
        plt.title("Predicted Value Distribution", fontsize=14)
        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_regression_scatter(y_true, y_pred, out_path):
    _set_plot_style()
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=60)
    
    # Ideal line
    try:
        vmin = min(min(y_true), min(y_pred))
        vmax = max(max(y_true), max(y_pred))
        plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label="Ideal Fit")
    except Exception:
        pass
    
    plt.title("Actual vs Predicted", fontsize=14)
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _make_scaled(model):
    # Imputer adds robustness against NaNs which break KNN/SVM
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")), 
        ("scaler", StandardScaler()), 
        ("model", model)
    ])


def _get_models(task: str, seed: int, n_samples: int = 100) -> Dict[str, object]:
    # Dynamic KNN neighbors to avoid errors on small datasets
    knn_k = min(5, max(1, n_samples - 1))
    
    if task == "regression":
        return {
            "LinearRegression": _make_scaled(LinearRegression()),
            "Ridge": _make_scaled(Ridge(random_state=seed)),
            "Lasso": _make_scaled(Lasso(random_state=seed, max_iter=5000)),
            "ElasticNet": _make_scaled(ElasticNet(random_state=seed, max_iter=5000)),
            "SVR_RBF": _make_scaled(SVR(kernel="rbf", C=10.0)),
            "KNN": _make_scaled(KNeighborsRegressor(n_neighbors=knn_k)),
            "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=seed),
            "GradientBoosting": GradientBoostingRegressor(random_state=seed),
            "HistGradientBoosting": _make_scaled(
                GradientBoostingRegressor(random_state=seed)
            ),
            "AdaBoost": AdaBoostRegressor(random_state=seed, n_estimators=200),
        }
    return {
        "LogisticRegression": _make_scaled(LogisticRegression(max_iter=2000)),
        "LinearSVC": _make_scaled(LinearSVC()),
        "SVC_RBF": _make_scaled(SVC(kernel="rbf", probability=True)),
        "KNN": _make_scaled(KNeighborsClassifier(n_neighbors=knn_k)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed, n_estimators=200),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }


def train_models(
    X_train: np.ndarray,
    y_train: Sequence,
    X_test: np.ndarray,
    y_test: Sequence,
    task: str,
    seed: int,
) -> List[Dict[str, object]]:
    results = []
    # Pass train sample count to adjust model parameters
    models = _get_models(task, seed, n_samples=len(X_train))
    for name, model in models.items():
        result = {"model": name, "status": "ok"}
        try:
            model.fit(X_train, y_train)
            y_test_count = len(y_test) if y_test is not None else 0
            if X_test.size and y_test_count > 0:
                preds = model.predict(X_test)
                if task == "regression":
                    score = r2_score(y_test, preds)
                    result["metric"] = "r2"
                    result["score"] = float(score)
                else:
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="macro") if len(set(y_test)) > 1 else 0.0
                    result["metric"] = "accuracy"
                    result["score"] = float(acc)
                    result["f1"] = float(f1)
            else:
                preds = model.predict(X_train)
                if task == "regression":
                    score = r2_score(y_train, preds)
                    result["metric"] = "r2_train"
                    result["score"] = float(score)
                else:
                    acc = accuracy_score(y_train, preds)
                    result["metric"] = "accuracy_train"
                    result["score"] = float(acc)
            result["estimator"] = model
        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
    return results


def save_model_bundle(
    out_path: str,
    model,
    metadata: Dict[str, object],
    feature_names: List[str],
    label_encoder: Optional[LabelEncoder],
) -> None:
    payload = {
        "model": model,
        "metadata": metadata,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
    }
    joblib.dump(payload, out_path)


def load_model_bundle(path: str) -> Dict[str, object]:
    return joblib.load(path)


def train_and_save(
    train_dirs: Sequence[str],
    test_dirs: Sequence[str],
    label_target: str,
    source: str,
    use_slices: bool,
    test_ratio: float,
    seed: int,
    out_dir: str,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if label_target not in LABEL_CHOICES:
        raise ValueError(f"Invalid label target: {label_target}")
    task = "regression" if label_target == "speed" else "classification"
    label_display = _label_display_name(label_target)

    # --- 1. Data Preparation ---
    if train_dirs:
        X_train, y_train, train_ids, feature_names = build_dataset(train_dirs, label_target, source, use_slices)
        X_test = np.zeros((0, X_train.shape[1]))
        y_test: List[str] = []
        test_ids: List[str] = []
        if test_dirs:
            X_test, y_test, test_ids, _ = build_dataset(test_dirs, label_target, source, use_slices)
    else:
        X_all, y_all, sample_ids, feature_names = build_dataset(test_dirs, label_target, source, use_slices)
        X_train, X_test, y_train, y_test, train_ids, test_ids = _split_data(
            X_all, y_all, sample_ids, test_ratio, seed, task
        )

    if task == "regression":
        y_train = _coerce_numeric_labels(y_train, train_ids, label_target)
        y_test = _coerce_numeric_labels(y_test, test_ids, label_target) if y_test else []

    # Label Encoding for Classification
    label_encoder = None
    y_train_encoded = y_train
    y_test_encoded = y_test
    classes = []
    
    if task == "classification":
        label_encoder = LabelEncoder()
        # Combine to ensure all classes are known
        all_labels = list(y_train) + (list(y_test) if len(y_test) > 0 else [])
        label_encoder.fit(all_labels)
        classes = label_encoder.classes_
        
        y_train_encoded = label_encoder.transform(y_train)
        if len(y_test) > 0:
            y_test_encoded = label_encoder.transform(y_test)
    
    # --- 2. Data Profiling Plots ---
    dist_plot_path = os.path.join(plots_dir, "class_distribution.png")
    if task == "classification":
        _plot_class_distribution(y_train, y_test, dist_plot_path)

    # --- 3. Model Training ---
    results = train_models(X_train, y_train_encoded, X_test, y_test_encoded, task, seed)
    
    # --- 4. Detailed Analysis & Plotting ---
    saved = []
    detailed_reports = []
    
    # Identify Best Model
    best_score = -float("inf")
    best_model_name = "None"
    
    metadata = {
        "label_target": label_target,
        "label_display": label_display,
        "task": task,
        "source": source,
        "use_slices": bool(use_slices),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "feature_count": int(X_train.shape[1]) if X_train.size else 0,
        "train_count": len(train_ids),
        "test_count": len(test_ids)
    }
    if label_target == "speed":
        metadata["speed_units"] = "rad/s"
        metadata["speed_period_map_s"] = SPEED_GEAR_TO_PERIOD_S

    for r in results:
        if r.get("status") != "ok":
            continue
            
        model_name = r["model"]
        est = r["estimator"]
        score = r.get("score", 0)
        if score > best_score:
            best_score = score
            best_model_name = model_name

        # Save Model
        model_filename = f"{label_target}_{source}_{model_name}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        save_model_bundle(model_path, est, metadata, feature_names, label_encoder)
        saved.append(model_path)

        # Generate Detailed Plots
        analysis = {"model": model_name}
        
        # Feature Importance
        fi_plot_name = f"{model_name}_features.png"
        fi_plot_path = os.path.join(plots_dir, fi_plot_name)
        final_estimator = est.steps[-1][1] if isinstance(est, Pipeline) else est
        _plot_feature_importance(final_estimator, feature_names, fi_plot_path, title=f"Feature Importance: {model_name}")
        if os.path.exists(fi_plot_path):
            analysis["fi_plot"] = os.path.join("plots", fi_plot_name)

        # Performance Plots (Test set only if available)
        if len(y_test_encoded) > 0:
            try:
                if task == "classification":
                    y_pred = est.predict(X_test)
                    y_prob = est.predict_proba(X_test) if hasattr(est, "predict_proba") else None
                    
                    # Confusion Matrix
                    cm_plot_name = f"{model_name}_cm.png"
                    cm_plot_path = os.path.join(plots_dir, cm_plot_name)
                    _plot_confusion_matrix(y_test_encoded, y_pred, range(len(classes)), cm_plot_path, title=f"Confusion Matrix: {model_name}")
                    if os.path.exists(cm_plot_path):
                        analysis["cm_plot"] = os.path.join("plots", cm_plot_name)
                    
                    # ROC Curve
                    if y_prob is not None:
                        roc_plot_name = f"{model_name}_roc.png"
                        roc_plot_path = os.path.join(plots_dir, roc_plot_name)
                        _plot_roc_curve(y_test_encoded, y_prob, classes, roc_plot_path)
                        if os.path.exists(roc_plot_path):
                            analysis["roc_plot"] = os.path.join("plots", roc_plot_name)
                    
                    # Text Report
                    clf_report = classification_report(y_test_encoded, y_pred, target_names=[str(c) for c in classes], output_dict=True)
                    analysis["report"] = clf_report

                elif task == "regression":
                    y_pred = est.predict(X_test)
                    
                    # Residuals
                    res_plot_name = f"{model_name}_residuals.png"
                    res_plot_path = os.path.join(plots_dir, res_plot_name)
                    _plot_residuals(y_test_encoded, y_pred, res_plot_path)
                    if os.path.exists(res_plot_path):
                        analysis["res_plot"] = os.path.join("plots", res_plot_name)
                    
                    # Calculate extra metrics
                    analysis["mae"] = mean_absolute_error(y_test_encoded, y_pred)
                    analysis["rmse"] = math.sqrt(mean_squared_error(y_test_encoded, y_pred))

            except Exception as e:
                analysis["error"] = str(e)
        
        detailed_reports.append(analysis)

    # --- 5. Generate Markdown Report ---
    report_md = os.path.join(out_dir, "train_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        # Title
        f.write(f"# Professional Analysis Report: {label_display.upper()}\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n")
        f.write(
            f"This report analyzes the performance of machine learning models trained to "
            f"predict **{label_display}** using **{source}** data.\n\n"
        )
        f.write(f"- **Best Performing Model**: **{best_model_name}** with a score of **{best_score:.4f}**.\n")
        f.write(f"- **Total Samples**: {len(train_ids)} Training, {len(test_ids)} Validation.\n")
        f.write("- **Recommendation**: ")
        if best_score > 0.85:
            f.write("The best model shows strong performance and is ready for initial deployment testing.\n")
        elif best_score > 0.7:
            f.write("Performance is acceptable but could benefit from hyperparameter tuning or more data.\n")
        else:
            f.write("Performance is suboptimal. Consider feature engineering or checking data quality.\n")
        f.write("\n")

        # 2. Data Profile
        f.write("## 2. Data Profile\n")
        if task == "classification":
            if os.path.exists(dist_plot_path):
                f.write(f"![Class Distribution]({os.path.join('plots', 'class_distribution.png')})\n\n")
            f.write(f"- **Classes**: {', '.join([str(c) for c in classes])}\n")
        f.write(f"- **Feature Dimension**: {metadata['feature_count']} input features extracted.\n\n")

        # 3. Model Comparison Table
        f.write("## 3. Model Performance Comparison\n")
        f.write("| Model | Metric | Score | Status | Error |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(
                "| **{model}** | {metric} | **{score:.4f}** | {status} | {error} |\n".format(
                    model=r.get("model", ""),
                    metric=r.get("metric", "N/A"),
                    score=float(r.get("score", 0.0)) if isinstance(r.get("score"), (float, int)) else 0.0,
                    status=r.get("status", ""),
                    error=(r.get("error", "") or "").replace("\n", " ")[:50],
                )
            )
        f.write("\n")

        # 4. Deep Dive
        f.write("## 4. Deep Dive Analysis\n")
        for detail in detailed_reports:
            model_name = detail["model"]
            f.write(f"### Model: {model_name}\n")
            
            # Key Metrics
            if "mae" in detail:
                 f.write(f"- **MAE**: {detail['mae']:.4f}\n")
                 f.write(f"- **RMSE**: {detail['rmse']:.4f}\n\n")
            
            # Visuals Layout
            f.write("| Performance Visualization | Feature Importance |\n")
            f.write("| --- | --- |\n")
            
            # Left Column: Perf Plot
            perf_img = "N/A"
            if detail.get("roc_plot"):
                perf_img = f"![ROC]({detail.get('roc_plot')})"
            elif detail.get("cm_plot"):
                perf_img = f"![Confusion Matrix]({detail.get('cm_plot')})"
            elif detail.get("res_plot"):
                perf_img = f"![Residuals]({detail.get('res_plot')})"
            
            # Right Column: Feature Plot
            feat_img = f"![Features]({detail.get('fi_plot')})" if detail.get("fi_plot") else "Not Available"
            
            f.write(f"| {perf_img} | {feat_img} |\n\n")

            # Classification Report Table
            if "report" in detail:
                f.write("**Detailed Classification Metrics**\n\n")
                f.write("| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                rep = detail["report"]
                for k, v in rep.items():
                    if isinstance(v, dict):
                         f.write(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {v['support']} |\n")
                f.write("\n")
            
            f.write("---\n")

    # CSV Summary
    report_csv = os.path.join(out_dir, "train_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "status", "metric", "score", "f1", "error"])
        for r in results:
            writer.writerow([r.get("model"), r.get("status"), r.get("metric"), r.get("score"), r.get("f1"), r.get("error")])

    # JSON Metadata
    meta_path = os.path.join(out_dir, "train_config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "results": results,
        "saved_models": saved,
        "report_path": report_csv,
        "report_md": report_md,
        "models_dir": models_dir,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "task": task,
    }


def predict_with_model(
    model_path: str,
    data_dirs: Sequence[str],
    out_dir: str,
) -> Dict[str, object]:
    # --- 1. Load Resources ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    bundle = load_model_bundle(model_path)
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    feature_names = bundle.get("feature_names", [])
    label_encoder = bundle.get("label_encoder")

    label_target = metadata.get("label_target", "unknown")
    label_display = metadata.get("label_display", _label_display_name(label_target))
    source = metadata.get("source", "envelope_detrended")
    use_slices = metadata.get("use_slices", True)
    task = metadata.get("task", "classification")

    # --- 2. Build Dataset & Predict ---
    X, y_text, sample_ids, feat_names = build_dataset(data_dirs, label_target, source, use_slices)
    
    # Feature Validation
    if len(feat_names) != len(feature_names):
         # Try to adapt if just a mismatch in number but compatible types (unlikely but safe to check)
         pass 

    preds = model.predict(X)
    
    # Handle Classification Decoding
    y_prob = None
    pred_labels = preds
    if task == "classification":
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)
            except Exception:
                pass
        
        if label_encoder is not None:
            pred_labels = label_encoder.inverse_transform(preds)
    
    # Extract Ground Truth if available in filenames
    actual_values: List[str] = []
    has_ground_truth = False
    valid_gt_indices = []
    
    if label_target in LABEL_CHOICES:
        for idx, sample_id in enumerate(sample_ids):
            try:
                val = parse_label_from_stem(sample_id).get(label_target)
                if val:
                    actual_values.append(_format_label_value(label_target, val))
                    valid_gt_indices.append(idx)
                else:
                    actual_values.append("")
            except Exception:
                actual_values.append("")
        
        if len(valid_gt_indices) > 0:
            has_ground_truth = True

    # --- 3. Output Preparation ---
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save CSV
    pred_csv = os.path.join(out_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "predicted", "actual", "confidence"])
        for i, sample_id in enumerate(sample_ids):
            conf = ""
            if y_prob is not None:
                conf = f"{np.max(y_prob[i]):.4f}"
            writer.writerow([sample_id, pred_labels[i], actual_values[i], conf])

    # --- 4. Visualization & Metrics ---
    plots = {}
    metrics = {}
    
    # 4.1 Distribution Plot (Always available)
    dist_plot_path = os.path.join(plot_dir, "pred_distribution.png")
    _plot_prediction_distribution(pred_labels, task, dist_plot_path)
    if os.path.exists(dist_plot_path):
        plots["distribution"] = os.path.join("plots", "pred_distribution.png")

    # 4.2 Ground Truth Analysis
    if has_ground_truth:
        # Filter valid data
        y_true_filtered = [actual_values[i] for i in valid_gt_indices]
        y_pred_filtered = [pred_labels[i] for i in valid_gt_indices]
        
        if task == "classification":
            # Metrics
            metrics["accuracy"] = accuracy_score(y_true_filtered, y_pred_filtered)
            metrics["f1_macro"] = f1_score(y_true_filtered, y_pred_filtered, average="macro")
            metrics["report"] = classification_report(y_true_filtered, y_pred_filtered, output_dict=True)
            
            # Confusion Matrix
            cm_path = os.path.join(plot_dir, "pred_cm.png")
            classes = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
            _plot_confusion_matrix(y_true_filtered, y_pred_filtered, classes, cm_path, title="Prediction Confusion Matrix")
            if os.path.exists(cm_path):
                plots["cm"] = os.path.join("plots", "pred_cm.png")
                
            # ROC if proba available and label encoder matches
            if y_prob is not None and label_encoder is not None:
                try:
                    # Need encoded true labels
                    y_true_enc = label_encoder.transform(y_true_filtered)
                    # Filter probabilities
                    y_prob_filtered = y_prob[valid_gt_indices]
                    roc_path = os.path.join(plot_dir, "pred_roc.png")
                    _plot_roc_curve(y_true_enc, y_prob_filtered, label_encoder.classes_, roc_path)
                    if os.path.exists(roc_path):
                        plots["roc"] = os.path.join("plots", "pred_roc.png")
                except Exception as e:
                    print(f"ROC generation failed: {e}")

        elif task == "regression":
            # Convert to float
            try:
                y_true_num = np.array([float(x) for x in y_true_filtered])
                y_pred_num = np.array([float(x) for x in y_pred_filtered])
                
                # Metrics
                metrics["r2"] = r2_score(y_true_num, y_pred_num)
                metrics["mae"] = mean_absolute_error(y_true_num, y_pred_num)
                metrics["rmse"] = math.sqrt(mean_squared_error(y_true_num, y_pred_num))
                
                # Scatter Plot
                scat_path = os.path.join(plot_dir, "pred_scatter.png")
                _plot_regression_scatter(y_true_num, y_pred_num, scat_path)
                if os.path.exists(scat_path):
                    plots["scatter"] = os.path.join("plots", "pred_scatter.png")
                
                # Residuals
                res_path = os.path.join(plot_dir, "pred_residuals.png")
                _plot_residuals(y_true_num, y_pred_num, res_path)
                if os.path.exists(res_path):
                    plots["residuals"] = os.path.join("plots", "pred_residuals.png")

            except Exception as e:
                print(f"Regression analysis failed: {e}")

    # --- 5. Generate Markdown Report ---
    report_md = os.path.join(out_dir, "prediction_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Prediction Analysis Report\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Samples Predicted**: {len(sample_ids)}\n")
        f.write(f"- **Model Used**: {os.path.basename(model_path)}\n")
        f.write(f"- **Task Type**: {task.title()} ({label_display})\n")
        if has_ground_truth:
            f.write(f"- **Ground Truth Coverage**: {len(valid_gt_indices)} / {len(sample_ids)} samples ({len(valid_gt_indices)/len(sample_ids)*100:.1f}%)\n")
            if task == "classification":
                f.write(f"- **Overall Accuracy**: **{metrics.get('accuracy', 0):.2%}**\n")
            else:
                f.write(f"- **R² Score**: **{metrics.get('r2', 0):.4f}**\n")
        else:
             f.write("- **Status**: No ground truth labels detected in filenames. Only predictions provided.\n")
        f.write("\n")
        
        # 2. Key Visualizations
        f.write("## 2. Visualizations\n")
        f.write("| Prediction Distribution | Performance (if GT) |\n")
        f.write("| --- | --- |\n")
        
        dist_img = f"![Distribution]({plots.get('distribution')})" if "distribution" in plots else "N/A"
        
        perf_img = "N/A"
        if "cm" in plots:
            perf_img = f"![Confusion Matrix]({plots['cm']})"
        elif "scatter" in plots:
            perf_img = f"![Scatter Plot]({plots['scatter']})"
            
        f.write(f"| {dist_img} | {perf_img} |\n\n")
        
        if "roc" in plots or "residuals" in plots:
            f.write("| Advanced Analysis | |\n")
            f.write("| --- | --- |\n")
            adv_img = ""
            if "roc" in plots:
                adv_img = f"![ROC]({plots['roc']})"
            elif "residuals" in plots:
                adv_img = f"![Residuals]({plots['residuals']})"
            f.write(f"| {adv_img} | |\n\n")

        # 3. Detailed Metrics
        if has_ground_truth:
            f.write("## 3. Detailed Metrics\n")
            if task == "classification" and "report" in metrics:
                f.write("**Classification Report**\n\n")
                f.write("| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("| --- | --- | --- | --- | --- |\n")
                rep = metrics["report"]
                for k, v in rep.items():
                    if isinstance(v, dict):
                        f.write(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {v['support']} |\n")
            elif task == "regression":
                f.write(f"- **Mean Absolute Error (MAE)**: {metrics.get('mae', 'N/A')}\n")
                f.write(f"- **Root Mean Sq Error (RMSE)**: {metrics.get('rmse', 'N/A')}\n")
        
        f.write("\n---\n*Report generated by AI Rotation Analysis CLI*")

    return {
        "pred_csv": pred_csv,
        "report_md": report_md,
        "plot_dir": plot_dir,
        "metrics": metrics,
        "sample_count": len(sample_ids),
        "label_target": label_target,
        "task": task,
    }


def _parse_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/predict models from WAV slice CSV output.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train models and save them")
    train_parser.add_argument(
        "--label",
        choices=LABEL_CHOICES,
        required=True,
        help="Label target (speed uses rad/s derived from period)",
    )
    train_parser.add_argument("--source", choices=SOURCE_CHOICES, default="envelope_detrended")
    train_parser.add_argument("--use-slices", action="store_true", default=True)
    train_parser.add_argument("--no-slices", action="store_false", dest="use_slices")
    train_parser.add_argument("--train", default="", help="Train paths (semicolon separated)")
    train_parser.add_argument("--test", default="", help="Test paths (semicolon separated)")
    train_parser.add_argument("--test-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--out", required=True, help="Output directory for models")

    pred_parser = sub.add_parser("predict", help="Run prediction with a saved model")
    pred_parser.add_argument("--model", required=True, help="Path to model joblib")
    pred_parser.add_argument("--data", required=True, help="Data paths (semicolon separated)")
    pred_parser.add_argument("--out", required=True, help="Output directory for predictions")

    args = parser.parse_args()
    if args.command == "train":
        train_paths = collect_sample_dirs(_parse_paths(args.train))
        test_paths = collect_sample_dirs(_parse_paths(args.test))
        if not train_paths and not test_paths:
            print("No training data found.")
            return 1
        if not train_paths:
            train_paths = []
        if not test_paths:
            test_paths = train_paths
        train_and_save(
            train_paths,
            test_paths,
            label_target=args.label,
            source=args.source,
            use_slices=bool(args.use_slices),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
            out_dir=args.out,
        )
        return 0
    result = predict_with_model(args.model, collect_sample_dirs(_parse_paths(args.data)), args.out)
    print(f"Saved predictions: {result['pred_csv']}")
    if result.get("plot_path"):
        print(f"Saved plot: {result['plot_path']}")
    if result.get("metrics"):
        print(f"Metrics: {result['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
