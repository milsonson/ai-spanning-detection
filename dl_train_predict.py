#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# --- Plot Styling ---
def _set_plot_style():
    try:
        # Try to use a nice style if available, otherwise fallback to manual
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass
    
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1

_set_plot_style()

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
class SampleInfo:
    sample_dir: str
    sample_id: str
    label: object


@dataclass
class SegmentRecord:
    sample_id: str
    label: object
    segment_path: str


def parse_label_from_stem(stem: str) -> Dict[str, str]:
    name = stem.strip()
    name = re.sub(r"\([^)]*\)$", "", name)
    match = re.match(
        r"^(?P<shape>[^_]+)_(?P<direction>c|u1|u2|d1|d2|l|r)_(?P<speed>[^_]+)_(?P<material>[^_]+)$",
        name,
    )
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


def _normalize_label_value(label_target: str, label_value: str) -> object:
    if label_target != "speed":
        return label_value
    return float(_speed_label_to_rad_s(label_value))


def _is_sample_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "summary.csv"))


def _infer_sample_dir(path: str) -> Optional[str]:
    if os.path.isdir(path):
        if _is_sample_dir(path):
            return path
        candidates = []
        for root, _, files in os.walk(path):
            if "summary.csv" in files:
                candidates.append(root)
        if candidates:
            return None
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
            for root, _, files in os.walk(path):
                if "summary.csv" in files:
                    found.append(root)
            continue
        inferred = _infer_sample_dir(path)
        if inferred:
            found.append(inferred)
    return sorted(set(found))


def _parse_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


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
    return np.asarray(times, dtype=np.float32), np.asarray(values, dtype=np.float32), label


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


def _resample_series(values: np.ndarray, seq_len: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(seq_len, dtype=np.float32)
    if values.size == seq_len:
        return values.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=seq_len, dtype=np.float32)
    resampled = np.interp(x_new, x_old, values).astype(np.float32)
    return resampled


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values)) if values.size else 1.0
    if std < 1e-6:
        std = 1.0
    return (values - mean) / std


def _build_sample_infos(sample_dirs: Sequence[str], label_target: str) -> List[SampleInfo]:
    infos: List[SampleInfo] = []
    for sample_dir in sample_dirs:
        sample_id = os.path.basename(sample_dir.rstrip(os.sep))
        labels = parse_label_from_stem(sample_id)
        label_value = labels.get(label_target)
        if label_value is None:
            raise ValueError(f"Missing label {label_target} for {sample_id}")
        label_norm = _normalize_label_value(label_target, label_value)
        infos.append(SampleInfo(sample_dir=sample_dir, sample_id=sample_id, label=label_norm))
    return infos


def _split_sample_infos(
    infos: Sequence[SampleInfo],
    test_ratio: float,
    seed: int,
    task: str,
) -> Tuple[List[SampleInfo], List[SampleInfo]]:
    infos_list = list(infos)
    if test_ratio <= 0:
        return infos_list, []
    rng = random.Random(seed)
    if task != "classification":
        rng.shuffle(infos_list)
        split_idx = int(round(len(infos_list) * (1.0 - test_ratio)))
        return infos_list[:split_idx], infos_list[split_idx:]

    grouped: Dict[str, List[SampleInfo]] = {}
    for info in infos_list:
        grouped.setdefault(str(info.label), []).append(info)

    train: List[SampleInfo] = []
    test: List[SampleInfo] = []
    for group in grouped.values():
        rng.shuffle(group)
        n_test = int(round(len(group) * test_ratio))
        if test_ratio > 0 and n_test == 0 and len(group) > 1:
            n_test = 1
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    if not test and infos_list:
        rng.shuffle(infos_list)
        split_idx = int(round(len(infos_list) * (1.0 - test_ratio)))
        return infos_list[:split_idx], infos_list[split_idx:]

    return train, test


def _build_segment_records(
    infos: Sequence[SampleInfo],
    source: str,
    use_slices: bool,
) -> Tuple[List[SegmentRecord], Dict[str, object]]:
    records: List[SegmentRecord] = []
    sample_labels: Dict[str, object] = {}
    for info in infos:
        sample_labels[info.sample_id] = info.label
        segment_files = _list_segment_files(info.sample_dir, source)
        if not segment_files:
            raise FileNotFoundError(f"No {source} CSV files in {info.sample_dir}")
        if not use_slices:
            segment_files = [segment_files[0]]
        for path in segment_files:
            records.append(SegmentRecord(sample_id=info.sample_id, label=info.label, segment_path=path))
    return records, sample_labels


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    if y_true.size == 0 or num_classes <= 0:
        return 0.0
    f1s: List[float] = []
    for cls in range(num_classes):
        tp = float(np.sum((y_true == cls) & (y_pred == cls)))
        fp = float(np.sum((y_true != cls) & (y_pred == cls)))
        fn = float(np.sum((y_true == cls) & (y_pred != cls)))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def _plot_training_curves(
    epochs: List[int],
    train_losses: List[float],
    val_losses: List[float],
    metrics: List[float],
    metric_name: str,
    out_path: str,
) -> None:
    if not epochs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#4C72B0", marker=".")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#C44E52", marker=".")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Metric Curve
    axes[1].plot(epochs, metrics, label=metric_name, color="#55A868", marker=".")
    axes[1].set_title(f"Validation {metric_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(metric_name)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: str,
) -> None:
    if y_true.size == 0:
        return
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() * 0.6 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_class_performance(class_metrics: Dict[str, Dict[str, float]], out_path: str) -> None:
    if not class_metrics:
        return
    
    classes = list(class_metrics.keys())
    f1_scores = [m["f1"] for m in class_metrics.values()]
    precisions = [m["precision"] for m in class_metrics.values()]
    recalls = [m["recall"] for m in class_metrics.values()]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#4C72B0')
    rects2 = ax.bar(x, recalls, width, label='Recall', color='#55A868')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#C44E52')
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    if y_true.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, edgecolor='w', linewidth=0.5, color="#4C72B0")
    
    vmin = float(min(np.min(y_true), np.min(y_pred)))
    vmax = float(max(np.max(y_true), np.max(y_pred)))
    margin = (vmax - vmin) * 0.05
    vmin -= margin
    vmax += margin
    
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    
    ax.set_xlabel("Actual Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    if y_true.size == 0:
        return
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=30, edgecolor='w', linewidth=0.5, color="#8172B2")
    axes[0].axhline(0.0, color="r", ls="--", lw=2)
    axes[0].set_xlabel("Predicted Value")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Residual Distribution
    axes[1].hist(residuals, bins=30, color="#64B5CD", edgecolor='black', alpha=0.8)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


class SegmentDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SegmentRecord],
        seq_len: int,
        task: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        augment: bool = False,
        cache: bool = False,
        allow_unknown_label: bool = False,
    ) -> None:
        self.records = list(records)
        self.seq_len = int(seq_len)
        self.task = task
        self.class_to_idx = class_to_idx or {}
        self.augment = augment
        self.cache_enabled = cache
        self.cache: Dict[str, np.ndarray] = {}
        self.allow_unknown_label = allow_unknown_label

    def __len__(self) -> int:
        return len(self.records)

    def _load_series(self, path: str) -> np.ndarray:
        if self.cache_enabled and path in self.cache:
            return self.cache[path]
        _, values, _ = _read_series_csv(path)
        if values.size == 0:
            values = np.zeros(1, dtype=np.float32)
        if self.cache_enabled:
            self.cache[path] = values
        return values

    def _maybe_augment(self, values: np.ndarray) -> np.ndarray:
        if not self.augment:
            return values
        out = values
        if random.random() < 0.5:
            gain = random.uniform(0.85, 1.15)
            out = out * gain
        if random.random() < 0.5:
            noise = np.random.normal(0.0, 0.02, size=out.shape).astype(np.float32)
            out = out + noise
        if random.random() < 0.3:
            shift = random.randint(-self.seq_len // 10, self.seq_len // 10)
            out = np.roll(out, shift)
        return out

    def __getitem__(self, idx: int):
        record = self.records[idx]
        values = self._load_series(record.segment_path)
        values = _resample_series(values, self.seq_len)
        values = _standardize(values)
        values = self._maybe_augment(values)
        x = torch.from_numpy(values).unsqueeze(0)
        if self.task == "classification":
            label_text = str(record.label)
            if label_text not in self.class_to_idx and self.allow_unknown_label:
                label_idx = 0
            else:
                label_idx = self.class_to_idx[label_text]
            y = torch.tensor(label_idx, dtype=torch.long)
        else:
            y = torch.tensor(float(record.label), dtype=torch.float32)
        return x, y, record.sample_id


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel: int, dropout: float) -> None:
        super().__init__()
        
        # Calculate padding to maintain size (same padding behavior)
        # Output size = (Input + 2*pad - kernel) / stride + 1
        # We want Output * stride = Input.
        # Ideally for stride 1: Input + TotalPad - kernel + 1 = Input => TotalPad = kernel - 1
        
        if kernel % 2 == 0:
            # Even kernel: Need asymmetric padding (e.g., k=10, pad=9 -> 4 left, 5 right)
            pad_total = kernel - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            self.pad1 = nn.ConstantPad1d((pad_left, pad_right), 0.0)
            self.pad2 = nn.ConstantPad1d((pad_left, pad_right), 0.0)
            conv_pad = 0
        else:
            # Odd kernel: Symmetric padding works (e.g., k=7, pad=6 -> 3 left, 3 right)
            self.pad1 = nn.Identity()
            self.pad2 = nn.Identity()
            conv_pad = kernel // 2

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=conv_pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=conv_pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        out = out + self.skip(x)
        out = self.act(out)
        return out


class ConvNet1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        blocks: int,
        kernel: int,
        dropout: float,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
        )
        layers: List[nn.Module] = []
        channels = base_ch
        for idx in range(blocks):
            out_ch = channels * 2 if idx < blocks - 1 else channels
            stride = 2 if idx < blocks - 1 else 1
            layers.append(ResBlock(channels, out_ch, stride=stride, kernel=kernel, dropout=dropout))
            channels = out_ch
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.blocks(out)
        out = self.pool(out)
        out = out.squeeze(-1)
        out = self.head(out)
        return out


def _select_device(device_pref: str) -> torch.device:
    if device_pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _aggregate_predictions(
    sample_ids: List[str],
    preds: np.ndarray,
    probs: Optional[np.ndarray],
    task: str,
    class_names: Optional[List[str]],
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], List[float]]:
    grouped_preds: Dict[str, List[np.ndarray]] = {}
    for idx, sample_id in enumerate(sample_ids):
        if task == "classification" and probs is not None:
            grouped_preds.setdefault(sample_id, []).append(probs[idx])
        else:
            grouped_preds.setdefault(sample_id, []).append(np.array([preds[idx]], dtype=np.float32))

    agg_ids: List[str] = []
    agg_preds: List[float] = []
    agg_probs: List[np.ndarray] = []
    confidences: List[float] = []

    for sample_id, values in grouped_preds.items():
        agg_ids.append(sample_id)
        if task == "classification" and probs is not None:
            stacked = np.stack(values, axis=0)
            avg = np.mean(stacked, axis=0)
            agg_probs.append(avg)
            pred_idx = int(np.argmax(avg))
            agg_preds.append(pred_idx)
            confidences.append(float(np.max(avg)))
        else:
            stacked = np.stack(values, axis=0)
            avg = float(np.mean(stacked))
            agg_preds.append(avg)
            confidences.append(0.0)

    if task == "classification" and agg_probs:
        return agg_ids, np.array(agg_preds, dtype=np.int64), np.stack(agg_probs, axis=0), confidences
    return agg_ids, np.array(agg_preds, dtype=np.float32), None, confidences


def _predict_aggregate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    use_amp: bool,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray], List[float]]:
    model.eval()
    all_sample_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, _, sample_ids = batch
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                if task == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.detach().cpu().numpy())
                    all_preds.append(preds.detach().cpu().numpy())
                else:
                    outputs = outputs.squeeze(1)
                    all_preds.append(outputs.detach().cpu().numpy())
            all_sample_ids.extend(list(sample_ids))
    preds_array = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    probs_array = np.concatenate(all_probs, axis=0) if all_probs else None
    return _aggregate_predictions(all_sample_ids, preds_array, probs_array, task, None)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    criterion: nn.Module,
    sample_labels: Dict[str, object],
    class_names: Optional[List[str]],
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_sample_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x, y, sample_ids = batch
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                if task == "classification":
                    loss = criterion(outputs, y)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.detach().cpu().numpy())
                    all_preds.append(preds.detach().cpu().numpy())
                else:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, y)
                    all_preds.append(outputs.detach().cpu().numpy())
            total_loss += float(loss.item()) * x.shape[0]
            total_count += x.shape[0]
            all_sample_ids.extend(list(sample_ids))

    mean_loss = total_loss / total_count if total_count else 0.0

    if not all_sample_ids:
        return mean_loss, {}

    preds_array = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    probs_array = np.concatenate(all_probs, axis=0) if all_probs else None
    agg_ids, agg_preds, agg_probs, _ = _aggregate_predictions(
        all_sample_ids, preds_array, probs_array, task, class_names
    )

    metrics: Dict[str, float] = {}
    if task == "classification":
        if class_names is None:
            return mean_loss, metrics
        y_true = np.array([class_names.index(str(sample_labels[sid])) for sid in agg_ids], dtype=np.int64)
        y_pred = agg_preds.astype(np.int64)
        metrics["accuracy"] = _accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = _f1_macro(y_true, y_pred, len(class_names))
    else:
        y_true = np.array([float(sample_labels[sid]) for sid in agg_ids], dtype=np.float32)
        y_pred = agg_preds.astype(np.float32)
        metrics["r2"] = _r2_score(y_true, y_pred)
        metrics["mae"] = _mae(y_true, y_pred)
        metrics["rmse"] = _rmse(y_true, y_pred)

    return mean_loss, metrics


def _calculate_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for i, cls_name in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[cls_name] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def train_and_save(
    train_dirs: Sequence[str],
    test_dirs: Sequence[str],
    label_target: str,
    source: str,
    use_slices: bool,
    test_ratio: float,
    seed: int,
    out_dir: str,
    epochs: int = 80,
    batch_size: int = 32,
    seq_len: int = 2048,
    base_ch: int = 32,
    blocks: int = 4,
    kernel: int = 7,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    scheduler_patience: int = 3,
    device: str = "auto",
    num_workers: int = 2,
    augment: bool = True,
    amp: Optional[bool] = None,
    grad_clip: float = 1.0,
    cache: bool = False,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if label_target not in LABEL_CHOICES:
        raise ValueError(f"Invalid label target: {label_target}")
    if source not in SOURCE_CHOICES:
        raise ValueError(f"Invalid source: {source}")

    task = "regression" if label_target == "speed" else "classification"
    label_display = _label_display_name(label_target)

    train_dirs = list(train_dirs)
    test_dirs = list(test_dirs)

    if train_dirs:
        train_infos = _build_sample_infos(train_dirs, label_target)
        if test_dirs:
            test_infos = _build_sample_infos(test_dirs, label_target)
        else:
            train_infos, test_infos = _split_sample_infos(
                train_infos, test_ratio, seed, task
            )
    else:
        if not test_dirs:
            raise ValueError("No training data provided.")
        all_infos = _build_sample_infos(test_dirs, label_target)
        train_infos, test_infos = _split_sample_infos(
            all_infos, test_ratio, seed, task
        )

    if not train_infos:
        raise ValueError("Training split is empty.")

    train_records, train_sample_labels = _build_segment_records(
        train_infos, source, use_slices
    )
    test_records, test_sample_labels = (
        _build_segment_records(test_infos, source, use_slices)
        if test_infos
        else ([], {})
    )

    if task == "classification":
        all_labels = [str(info.label) for info in train_infos + test_infos]
        class_names = sorted(set(all_labels))
        if len(class_names) < 2:
            raise ValueError("Need at least 2 classes for classification.")
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    else:
        class_names = None
        class_to_idx = None

    _set_seed(seed)
    device_obj = _select_device(device)
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if amp is None:
        use_amp = device_obj.type == "cuda"
    else:
        use_amp = bool(amp) and device_obj.type == "cuda"

    train_dataset = SegmentDataset(
        train_records,
        seq_len=seq_len,
        task=task,
        class_to_idx=class_to_idx,
        augment=augment,
        cache=cache,
    )
    val_dataset = SegmentDataset(
        test_records,
        seq_len=seq_len,
        task=task,
        class_to_idx=class_to_idx,
        augment=False,
        cache=cache,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )

    out_dim = len(class_names) if task == "classification" else 1
    model = ConvNet1D(
        in_ch=1,
        base_ch=base_ch,
        blocks=blocks,
        kernel=kernel,
        dropout=dropout,
        out_dim=out_dim,
    ).to(device_obj)

    if task == "classification":
        class_counts = np.zeros(out_dim, dtype=np.float32)
        for info in train_infos:
            class_counts[class_to_idx[str(info.label)]] += 1.0
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
        class_weights = (
            class_weights / class_weights.sum() * out_dim
            if class_weights.sum() > 0
            else class_weights
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32).to(device_obj)
        )
    else:
        criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_factor = 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=scheduler_patience,
        factor=scheduler_factor,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history_csv = os.path.join(out_dir, "train_history.csv")
    best_model_path = os.path.join(models_dir, "best_model.pt")
    last_model_path = os.path.join(models_dir, "last_model.pt")
    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch_history: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    metric_history: List[float] = []
    metric_name = "f1_macro" if task == "classification" else "r2"

    if progress_callback:
        progress_callback({"type": "start", "epochs": epochs})

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "metric",
                "accuracy",
                "f1_macro",
                "r2",
                "mae",
                "rmse",
            ]
        )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if progress_callback and batch_idx % 2 == 0:
                 progress_callback({
                     "type": "batch",
                     "epoch": epoch,
                     "epochs": epochs,
                     "batch": batch_idx,
                     "total_batches": total_batches
                 })
            
            x, y, _ = batch
            x = x.to(device_obj)
            y = y.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                if task == "classification":
                    loss = criterion(outputs, y)
                else:
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item()) * x.shape[0]
            train_count += x.shape[0]

        train_loss = train_loss / train_count if train_count else 0.0
        if val_dataset and len(val_dataset) > 0:
            val_loss, val_metrics = _evaluate(
                model,
                val_loader,
                device_obj,
                task,
                criterion,
                test_sample_labels,
                class_names,
                use_amp,
            )
        else:
            val_loss = train_loss
            val_metrics = {}

        scheduler.step(val_loss)

        if not val_metrics:
            metric = -val_loss
            accuracy = f1_macro = ""
            r2 = mae = rmse = ""
            metric_name = "neg_val_loss"
        elif task == "classification":
            metric = float(
                val_metrics.get("f1_macro", val_metrics.get("accuracy", 0.0))
            )
            accuracy = val_metrics.get("accuracy", 0.0)
            f1_macro = val_metrics.get("f1_macro", 0.0)
            r2 = mae = rmse = ""
        else:
            metric = float(val_metrics.get("r2", -val_loss))
            accuracy = f1_macro = ""
            r2 = val_metrics.get("r2", 0.0)
            mae = val_metrics.get("mae", 0.0)
            rmse = val_metrics.get("rmse", 0.0)

        epoch_history.append(epoch)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        metric_history.append(float(metric))

        if progress_callback:
            progress_callback(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "epochs": epochs,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "metric": float(metric),
                }
            )

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{metric:.6f}",
                    f"{accuracy}" if accuracy != "" else "",
                    f"{f1_macro}" if f1_macro != "" else "",
                    f"{r2}" if r2 != "" else "",
                    f"{mae}" if mae != "" else "",
                    f"{rmse}" if rmse != "" else "",
                ]
            )

        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "base_ch": base_ch,
                        "blocks": blocks,
                        "kernel": kernel,
                        "dropout": dropout,
                        "out_dim": out_dim,
                    },
                    "metadata": {
                        "label_target": label_target,
                        "label_display": label_display,
                        "task": task,
                        "source": source,
                        "use_slices": bool(use_slices),
                        "seq_len": int(seq_len),
                        "classes": class_names or [],
                        "speed_units": "rad/s" if label_target == "speed" else "",
                        "speed_period_map_s": SPEED_GEAR_TO_PERIOD_S
                        if label_target == "speed"
                        else {},
                    },
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        if patience > 0 and bad_epochs >= patience:
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "base_ch": base_ch,
                "blocks": blocks,
                "kernel": kernel,
                "dropout": dropout,
                "out_dim": out_dim,
            },
            "metadata": {
                "label_target": label_target,
                "label_display": label_display,
                "task": task,
                "source": source,
                "use_slices": bool(use_slices),
                "seq_len": int(seq_len),
                "classes": class_names or [],
                "speed_units": "rad/s" if label_target == "speed" else "",
                "speed_period_map_s": SPEED_GEAR_TO_PERIOD_S
                if label_target == "speed"
                else {},
            },
        },
        last_model_path,
    )

    eval_plots: Dict[str, str] = {}
    eval_metrics: Dict[str, float] = {}
    train_curve_path = ""
    if epoch_history:
        train_curve_path = os.path.join(plots_dir, "training_curves.png")
        _plot_training_curves(
            epoch_history,
            train_losses,
            val_losses,
            metric_history,
            metric_name,
            train_curve_path,
        )

    if test_infos:
        eval_checkpoint = _load_checkpoint(best_model_path)
        eval_config = eval_checkpoint.get("model_config", {})
        eval_model = ConvNet1D(
            in_ch=1,
            base_ch=int(eval_config.get("base_ch", 32)),
            blocks=int(eval_config.get("blocks", 4)),
            kernel=int(eval_config.get("kernel", 7)),
            dropout=float(eval_config.get("dropout", 0.1)),
            out_dim=int(eval_config.get("out_dim", out_dim)),
        ).to(device_obj)
        eval_model.load_state_dict(eval_checkpoint.get("model_state", {}))
        eval_model.eval()

        agg_ids, agg_preds, _, _ = _predict_aggregate_loader(
            eval_model, val_loader, device_obj, task, use_amp
        )
        if agg_ids:
            if task == "classification" and class_names:
                y_true = np.array(
                    [
                        class_names.index(str(test_sample_labels[sid]))
                        for sid in agg_ids
                    ],
                    dtype=np.int64,
                )
                y_pred = agg_preds.astype(np.int64)
                eval_metrics["accuracy"] = _accuracy_score(y_true, y_pred)
                eval_metrics["f1_macro"] = _f1_macro(y_true, y_pred, len(class_names))
                
                cm_path = os.path.join(plots_dir, "val_confusion_matrix.png")
                _plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
                if os.path.exists(cm_path):
                    eval_plots["confusion_matrix"] = os.path.join(
                        "plots", "val_confusion_matrix.png"
                    )
                
                # New: Per-class performance
                per_class_metrics = _calculate_per_class_metrics(y_true, y_pred, class_names)
                per_class_path = os.path.join(plots_dir, "val_per_class.png")
                _plot_class_performance(per_class_metrics, per_class_path)
                if os.path.exists(per_class_path):
                    eval_plots["per_class"] = os.path.join("plots", "val_per_class.png")
                    
            elif task == "regression":
                y_true = np.array(
                    [float(test_sample_labels[sid]) for sid in agg_ids],
                    dtype=np.float32,
                )
                y_pred = agg_preds.astype(np.float32)
                eval_metrics["r2"] = _r2_score(y_true, y_pred)
                eval_metrics["mae"] = _mae(y_true, y_pred)
                eval_metrics["rmse"] = _rmse(y_true, y_pred)
                scatter_path = os.path.join(plots_dir, "val_scatter.png")
                residuals_path = os.path.join(plots_dir, "val_residuals.png")
                _plot_regression_scatter(y_true, y_pred, scatter_path)
                _plot_residuals(y_true, y_pred, residuals_path)
                if os.path.exists(scatter_path):
                    eval_plots["scatter"] = os.path.join("plots", "val_scatter.png")
                if os.path.exists(residuals_path):
                    eval_plots["residuals"] = os.path.join(
                        "plots", "val_residuals.png"
                    )

    report_md = os.path.join(out_dir, "train_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# Deep Learning Training Report: {label_display.upper()}\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Summary\n")
        f.write(f"- **Task**: {task}\n")
        f.write(f"- **Source**: {source}\n")
        f.write(f"- **Seq Length**: {seq_len}\n")
        f.write(
            f"- **Model**: base_ch={base_ch}, blocks={blocks}, kernel={kernel}, dropout={dropout}\n"
        )
        f.write(f"- **Epochs**: {epochs}\n")
        f.write(
            f"- **LR Scheduler**: ReduceLROnPlateau(patience={scheduler_patience}, factor={scheduler_factor})\n"
        )
        f.write(f"- **Best Epoch**: {best_epoch}\n")
        f.write(f"- **Best Metric**: {best_metric:.6f}\n")
        f.write(f"- **Train Samples**: {len(train_infos)}\n")
        f.write(f"- **Validation Samples**: {len(test_infos)}\n")
        f.write(f"- **History CSV**: {history_csv}\n\n")

        f.write("## 2. Training Curves\n")
        if train_curve_path and os.path.exists(train_curve_path):
            f.write(
                f"![Training Curves]({os.path.join('plots', 'training_curves.png')})\n\n"
            )
        else:
            f.write("No training curve available.\n\n")

        f.write("## 3. Validation Performance\n")
        if not test_infos:
            f.write("No validation set available.\n")
        else:
            if task == "classification":
                f.write(f"- **Accuracy**: {eval_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"- **F1 Macro**: {eval_metrics.get('f1_macro', 0):.4f}\n\n")
                f.write("### Plots\n")
                if eval_plots.get("confusion_matrix"):
                    f.write(
                        f"![Confusion Matrix]({eval_plots['confusion_matrix']})\n\n"
                    )
                if eval_plots.get("per_class"):
                    f.write(
                        f"![Per-Class Performance]({eval_plots['per_class']})\n"
                    )
            else:
                f.write(f"- **R2**: {eval_metrics.get('r2', 0):.4f}\n")
                f.write(f"- **MAE**: {eval_metrics.get('mae', 0):.4f}\n")
                f.write(f"- **RMSE**: {eval_metrics.get('rmse', 0):.4f}\n\n")
                f.write("### Plots\n")
                if eval_plots.get("scatter"):
                    f.write(f"![Scatter]({eval_plots['scatter']})\n\n")
                if eval_plots.get("residuals"):
                    f.write(f"![Residuals]({eval_plots['residuals']})\n")

    if progress_callback:
        progress_callback(
            {"type": "done", "best_epoch": best_epoch, "best_metric": best_metric}
        )

    config_path = os.path.join(out_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_target": label_target,
                "label_display": label_display,
                "task": task,
                "source": source,
                "use_slices": bool(use_slices),
                "seq_len": int(seq_len),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "base_ch": int(base_ch),
                "blocks": int(blocks),
                "kernel": int(kernel),
                "dropout": float(dropout),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "patience": int(patience),
                "scheduler_patience": int(scheduler_patience),
                "scheduler_factor": float(scheduler_factor),
                "device": str(device_obj),
                "classes": class_names or [],
            },
            f,
            indent=2,
        )

    return {
        "best_model": best_model_path,
        "last_model": last_model_path,
        "history_csv": history_csv,
        "report_md": report_md,
        "plots_dir": plots_dir,
        "train_ids": [info.sample_id for info in train_infos],
        "test_ids": [info.sample_id for info in test_infos],
        "task": task,
        "label_target": label_target,
    }


def _load_checkpoint(path: str) -> Dict[str, object]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.load(path, map_location="cpu")


def predict_with_model(
    model_path: str,
    data_dirs: Sequence[str],
    out_dir: str,
    device: str = "auto",
    num_workers: int = 2,
    amp: Optional[bool] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    checkpoint = _load_checkpoint(model_path)
    metadata = checkpoint.get("metadata", {})
    model_config = checkpoint.get("model_config", {})

    label_target = metadata.get("label_target", "unknown")
    label_display = metadata.get("label_display", _label_display_name(label_target))
    task = metadata.get("task", "classification")
    source = metadata.get("source", "envelope_detrended")
    use_slices = metadata.get("use_slices", True)
    seq_len = int(metadata.get("seq_len", 2048))
    class_names = metadata.get("classes") or []

    infos = _build_sample_infos(list(data_dirs), label_target)
    records, sample_labels = _build_segment_records(infos, source, use_slices)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)} if class_names else None

    dataset = SegmentDataset(
        records,
        seq_len=seq_len,
        task=task,
        class_to_idx=class_to_idx,
        augment=False,
        cache=False,
        allow_unknown_label=True,
    )

    device_obj = _select_device(device)
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if amp is None:
        use_amp = device_obj.type == "cuda"
    else:
        use_amp = bool(amp) and device_obj.type == "cuda"

    out_dim = model_config.get("out_dim", len(class_names) if task == "classification" else 1)
    model = ConvNet1D(
        in_ch=1,
        base_ch=int(model_config.get("base_ch", 32)),
        blocks=int(model_config.get("blocks", 4)),
        kernel=int(model_config.get("kernel", 7)),
        dropout=float(model_config.get("dropout", 0.1)),
        out_dim=int(out_dim),
    )
    model.load_state_dict(checkpoint.get("model_state", {}))
    model.to(device_obj)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_obj.type == "cuda",
    )

    all_sample_ids: List[str] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    
    total_batches = len(loader)
    if progress_callback:
        progress_callback({"type": "start", "total_batches": total_batches})

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if progress_callback and batch_idx % 2 == 0:
                progress_callback({"type": "batch", "batch": batch_idx, "total_batches": total_batches})
            
            x, _, sample_ids = batch
            x = x.to(device_obj)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                if task == "classification":
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs.detach().cpu().numpy())
                    all_preds.append(preds.detach().cpu().numpy())
                else:
                    outputs = outputs.squeeze(1)
                    all_preds.append(outputs.detach().cpu().numpy())
            all_sample_ids.extend(list(sample_ids))

    preds_array = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    probs_array = np.concatenate(all_probs, axis=0) if all_probs else None
    agg_ids, agg_preds, agg_probs, confidences = _aggregate_predictions(
        all_sample_ids, preds_array, probs_array, task, class_names
    )

    pred_csv = os.path.join(out_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "predicted", "actual", "confidence"])
        for idx, sample_id in enumerate(agg_ids):
            if task == "classification":
                pred_label = class_names[int(agg_preds[idx])] if class_names else str(agg_preds[idx])
            else:
                pred_label = f"{float(agg_preds[idx]):.6f}"
            actual = sample_labels.get(sample_id, "")
            if task == "classification":
                actual_val = str(actual) if actual != "" else ""
            else:
                actual_val = f"{float(actual):.6f}" if actual != "" else ""
            writer.writerow([sample_id, pred_label, actual_val, f"{confidences[idx]:.4f}"])

    metrics: Dict[str, float] = {}
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    if agg_ids:
        if task == "classification" and class_names:
            has_unknown = any(str(sample_labels[sid]) not in class_names for sid in agg_ids)
            if not has_unknown:
                y_true = np.array([class_names.index(str(sample_labels[sid])) for sid in agg_ids], dtype=np.int64)
                y_pred = agg_preds.astype(np.int64)
                metrics["accuracy"] = _accuracy_score(y_true, y_pred)
                metrics["f1_macro"] = _f1_macro(y_true, y_pred, len(class_names))
        elif task == "regression":
            y_true = np.array([float(sample_labels[sid]) for sid in agg_ids], dtype=np.float32)
            y_pred = agg_preds.astype(np.float32)
            metrics["r2"] = _r2_score(y_true, y_pred)
            metrics["mae"] = _mae(y_true, y_pred)
            metrics["rmse"] = _rmse(y_true, y_pred)

    plot_paths: Dict[str, str] = {}
    if y_true is not None and y_pred is not None:
        if task == "classification" and class_names:
            cm_path = os.path.join(plots_dir, "pred_confusion_matrix.png")
            _plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
            if os.path.exists(cm_path):
                plot_paths["confusion_matrix"] = os.path.join("plots", "pred_confusion_matrix.png")
        elif task == "regression":
            scatter_path = os.path.join(plots_dir, "pred_scatter.png")
            residuals_path = os.path.join(plots_dir, "pred_residuals.png")
            _plot_regression_scatter(y_true, y_pred, scatter_path)
            _plot_residuals(y_true, y_pred, residuals_path)
            if os.path.exists(scatter_path):
                plot_paths["scatter"] = os.path.join("plots", "pred_scatter.png")
            if os.path.exists(residuals_path):
                plot_paths["residuals"] = os.path.join("plots", "pred_residuals.png")

    report_md = os.path.join(out_dir, "prediction_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# Deep Learning Prediction Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. Summary\n")
        f.write(f"- **Label**: {label_display}\n")
        f.write(f"- **Task**: {task}\n")
        f.write(f"- **Source**: {source}\n")
        f.write(f"- **Seq Length**: {seq_len}\n")
        f.write(f"- **Samples**: {len(agg_ids)}\n")
        if metrics:
            if task == "classification":
                f.write(f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"- **F1 Macro**: {metrics.get('f1_macro', 0):.4f}\n")
            else:
                f.write(f"- **R2**: {metrics.get('r2', 0):.4f}\n")
                f.write(f"- **MAE**: {metrics.get('mae', 0):.4f}\n")
                f.write(f"- **RMSE**: {metrics.get('rmse', 0):.4f}\n")
        f.write("\n")

        f.write("## 2. Visualizations\n")
        if not plot_paths:
            f.write("No plots available.\n")
        else:
            if plot_paths.get("confusion_matrix"):
                f.write(f"![Confusion Matrix]({plot_paths['confusion_matrix']})\n")
            if plot_paths.get("scatter"):
                f.write(f"![Scatter]({plot_paths['scatter']})\n\n")
            if plot_paths.get("residuals"):
                f.write(f"![Residuals]({plot_paths['residuals']})\n")
    
    if progress_callback:
        progress_callback({"type": "done"})

    return {
        "pred_csv": pred_csv,
        "report_md": report_md,
        "metrics": metrics,
        "sample_count": len(agg_ids),
        "label_target": label_target,
        "task": task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep learning train/predict for rotation analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train a deep learning model")
    train_parser.add_argument("--label", choices=LABEL_CHOICES, required=True, help="Label target (speed uses rad/s)")
    train_parser.add_argument("--source", choices=SOURCE_CHOICES, default="envelope_detrended")
    train_parser.add_argument("--use-slices", action="store_true", default=True)
    train_parser.add_argument("--no-slices", action="store_false", dest="use_slices")
    train_parser.add_argument("--train", default="", help="Train paths (semicolon separated)")
    train_parser.add_argument("--test", default="", help="Test paths (semicolon separated)")
    train_parser.add_argument("--test-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--out", required=True, help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--seq-len", type=int, default=2048)
    train_parser.add_argument("--base-ch", type=int, default=32)
    train_parser.add_argument("--blocks", type=int, default=4)
    train_parser.add_argument("--kernel", type=int, default=7)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--scheduler-patience", type=int, default=3)
    train_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--augment", action="store_true", default=True)
    train_parser.add_argument("--no-augment", action="store_false", dest="augment")
    train_parser.add_argument("--amp", action="store_true", default=None)
    train_parser.add_argument("--no-amp", action="store_false", dest="amp")
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--cache", action="store_true", default=False)

    pred_parser = sub.add_parser("predict", help="Predict using a saved model")
    pred_parser.add_argument("--model", required=True, help="Path to model checkpoint .pt")
    pred_parser.add_argument("--data", required=True, help="Data paths (semicolon separated)")
    pred_parser.add_argument("--out", required=True, help="Output directory")
    pred_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    pred_parser.add_argument("--num-workers", type=int, default=2)
    pred_parser.add_argument("--amp", action="store_true", default=None)
    pred_parser.add_argument("--no-amp", action="store_false", dest="amp")

    args = parser.parse_args()

    if args.command == "train":
        train_paths = collect_sample_dirs(_parse_paths(args.train))
        test_paths = collect_sample_dirs(_parse_paths(args.test))
        if not train_paths and not test_paths:
            print("No training data found.")
            return 1
        train_and_save(
            train_paths,
            test_paths,
            label_target=args.label,
            source=args.source,
            use_slices=bool(args.use_slices),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
            out_dir=args.out,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            seq_len=int(args.seq_len),
            base_ch=int(args.base_ch),
            blocks=int(args.blocks),
            kernel=int(args.kernel),
            dropout=float(args.dropout),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            scheduler_patience=int(args.scheduler_patience),
            device=str(args.device),
            num_workers=int(args.num_workers),
            augment=bool(args.augment),
            amp=args.amp,
            grad_clip=float(args.grad_clip),
            cache=bool(args.cache),
        )
        return 0

    result = predict_with_model(
        model_path=args.model,
        data_dirs=collect_sample_dirs(_parse_paths(args.data)),
        out_dir=args.out,
        device=str(args.device),
        num_workers=int(args.num_workers),
        amp=args.amp,
    )
    print(f"Saved predictions: {result['pred_csv']}")
    if result.get("metrics"):
        print(f"Metrics: {result['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
