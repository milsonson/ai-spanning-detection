#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks, hilbert, spectrogram

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_BANDPASS_LOWCUT = 50.0
DEFAULT_BANDPASS_HIGHCUT = 8000.0
DEFAULT_BANDPASS_ORDER = 4
DEFAULT_SMOOTH_CUTOFF = 50.0
DEFAULT_ENV_CUTOFF = 20.0
DEFAULT_ENV_MAX_FREQ = 50.0
DEFAULT_ENV_TREND_CUTOFF = 0.5
DEFAULT_ENV_NORM_MODE = "detrended-max"
DEFAULT_CENTER_DURATION = 8.0
DEFAULT_DECIMATE_K = 10
DEFAULT_WINDOW_SIZE = 1.0
DEFAULT_HOP_SIZE = 0.5
DEFAULT_EXPORT_SLICES = True

CHANNEL_MODES = ("auto", "left", "right", "mean", "sum", "diff")
ENV_NORM_MODES = ("detrended-max", "envelope-max", "scale", "none")


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.size <= padlen:
        return x
    try:
        return filtfilt(b, a, x)
    except Exception:
        return x


def lowpass_filter(audio: np.ndarray, sr: int, cutoff: float, order: int = 2) -> np.ndarray:
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    if normal_cutoff <= 0 or normal_cutoff >= 1.0:
        return audio
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return _safe_filtfilt(b, a, audio)


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    lowcut: float = DEFAULT_BANDPASS_LOWCUT,
    highcut: float = DEFAULT_BANDPASS_HIGHCUT,
    order: int = DEFAULT_BANDPASS_ORDER,
) -> np.ndarray:
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    if low >= high or low <= 0 or high >= 1.0:
        return audio
    b, a = butter(order, [low, high], btype="band")
    return _safe_filtfilt(b, a, audio)


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.astype(np.float32).mean(axis=1)


def _channel_labels(channels: int) -> List[str]:
    if channels == 2:
        return ["left", "right"]
    return [f"ch{i + 1}" for i in range(channels)]


def _channel_rms(channel: np.ndarray) -> float:
    if channel.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(channel.astype(np.float64) ** 2)))


def compute_channel_stats(data: np.ndarray) -> dict:
    stats: dict = {"rms": [], "corr_lr": None}
    if data.ndim == 1:
        stats["rms"] = [_channel_rms(data)]
        return stats
    channels = data.shape[1]
    rms_list = []
    for idx in range(channels):
        rms_list.append(_channel_rms(data[:, idx]))
    stats["rms"] = rms_list
    if channels >= 2:
        left = data[:, 0].astype(np.float32)
        right = data[:, 1].astype(np.float32)
        if left.size and right.size and np.std(left) > 0 and np.std(right) > 0:
            stats["corr_lr"] = float(np.corrcoef(left, right)[0, 1])
        else:
            stats["corr_lr"] = 0.0
    return stats


def select_channel(data: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    if data.ndim == 1:
        return data.astype(np.float32), "mono"
    channels = data.shape[1]
    labels = _channel_labels(channels)
    channel_data = [data[:, idx].astype(np.float32) for idx in range(channels)]
    rms_list = [_channel_rms(ch) for ch in channel_data]

    mode = (mode or "auto").lower()
    if mode == "left":
        idx = 0
        return channel_data[idx], labels[idx]
    if mode == "right":
        idx = 1 if channels > 1 else 0
        return channel_data[idx], labels[idx]
    if mode == "mean":
        mono = np.mean(np.stack(channel_data, axis=1), axis=1)
        return mono, "mean"
    if mode == "sum":
        mono = np.sum(np.stack(channel_data, axis=1), axis=1)
        return mono, "sum"
    if mode == "diff":
        if channels >= 2:
            mono = 0.5 * (channel_data[0] - channel_data[1])
            return mono, "diff"
        return channel_data[0], labels[0]

    idx = int(np.argmax(rms_list)) if rms_list else 0
    return channel_data[idx], f"auto:{labels[idx]}"


def _normalize_audio(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data.astype(np.float32)
    if np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 1.0:
            audio = audio / max_abs
        return np.clip(audio, -1.0, 1.0)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        if scale == 0:
            return data.astype(np.float32)
        return data.astype(np.float32) / scale
    return data.astype(np.float32)


def _dtype_scale(data: np.ndarray, dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(max(abs(info.min), info.max)) or 1.0
    max_abs = float(np.max(np.abs(data))) if data.size else 0.0
    return max_abs if max_abs > 1.0 else 1.0


def decimate_audio(audio: np.ndarray, sr: float, k: int) -> Tuple[np.ndarray, float, int]:
    if audio.size == 0:
        return audio, sr, 1
    step = int(k)
    if step <= 1:
        return audio, sr, 1
    decimated = audio[::step].astype(np.float32)
    new_sr = float(sr) / float(step)
    return decimated, new_sr, step


def center_crop_audio(
    audio_raw: np.ndarray,
    audio_norm: np.ndarray,
    sr: float,
    duration_s: float,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    total = int(audio_raw.size)
    if total == 0 or sr <= 0:
        info = {
            "analysis_start_samples": 0,
            "analysis_end_samples": 0,
            "analysis_frames": 0,
            "analysis_start_s": 0.0,
            "analysis_end_s": 0.0,
            "analysis_duration_s": 0.0,
        }
        return audio_raw, audio_norm, info
    duration_s = max(0.0, float(duration_s))
    target_len = int(round(duration_s * sr)) if duration_s > 0 else total
    if target_len <= 0 or target_len > total:
        target_len = total
    start = max(0, (total - target_len) // 2)
    end = start + target_len
    trimmed_raw = audio_raw[start:end]
    trimmed_norm = audio_norm[start:end]
    info = {
        "analysis_start_samples": int(start),
        "analysis_end_samples": int(end),
        "analysis_frames": int(end - start),
        "analysis_start_s": start / float(sr),
        "analysis_end_s": end / float(sr),
        "analysis_duration_s": (end - start) / float(sr),
    }
    return trimmed_raw, trimmed_norm, info


def read_wav(
    path: str, channel_mode: str = "auto"
) -> Tuple[int, np.ndarray, np.ndarray, np.dtype, int, int, str, dict]:
    sr, data = wavfile.read(path)
    channels = 1 if data.ndim == 1 else data.shape[1]
    frames = data.shape[0]
    channel_stats = compute_channel_stats(data)
    mono, channel_label = select_channel(data, channel_mode)
    audio_raw = mono.astype(np.float32)
    audio = _normalize_audio(audio_raw)
    return sr, audio_raw, audio, data.dtype, channels, frames, channel_label, channel_stats


def compute_amp_freq_series(audio: np.ndarray, sr: float, apply_bandpass: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([]), np.array([]), np.array([])
    audio_bp = bandpass_filter(audio, sr) if apply_bandpass else audio
    analytic = hilbert(audio_bp)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    dphase = np.gradient(phase)
    inst_freq = (sr / (2.0 * np.pi)) * dphase
    try:
        amplitude = lowpass_filter(amplitude, sr, cutoff=DEFAULT_SMOOTH_CUTOFF, order=2)
        inst_freq = lowpass_filter(inst_freq, sr, cutoff=DEFAULT_SMOOTH_CUTOFF, order=2)
    except Exception:
        pass
    inst_freq = np.maximum(inst_freq, 0.0)
    t = np.arange(audio.size, dtype=np.float64) / float(sr)
    return t, amplitude.astype(np.float32), inst_freq.astype(np.float32)


def compute_envelope_series(
    audio: np.ndarray,
    sr: float,
    apply_bandpass: bool,
    env_cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([]), np.array([])
    audio_bp = bandpass_filter(audio, sr) if apply_bandpass else audio
    analytic = hilbert(audio_bp)
    envelope = np.abs(analytic)
    if env_cutoff and env_cutoff > 0:
        try:
            envelope = lowpass_filter(envelope, sr, cutoff=env_cutoff, order=2)
        except Exception:
            pass
    t = np.arange(audio.size, dtype=np.float64) / float(sr)
    return t, envelope.astype(np.float32)


def detrend_envelope(
    envelope: np.ndarray, sr: float, trend_cutoff: float
) -> Tuple[np.ndarray, np.ndarray, bool]:
    if envelope.size == 0 or trend_cutoff <= 0:
        return envelope, np.zeros_like(envelope), False
    nyquist = 0.5 * sr
    normal_cutoff = trend_cutoff / nyquist
    if normal_cutoff <= 0 or normal_cutoff >= 1.0:
        return envelope, np.zeros_like(envelope), False
    b, a = butter(2, normal_cutoff, btype="low", analog=False)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if envelope.size <= padlen:
        return envelope, np.zeros_like(envelope), False
    try:
        trend = filtfilt(b, a, envelope)
    except Exception:
        return envelope, np.zeros_like(envelope), False
    detrended = envelope - trend
    return detrended.astype(np.float32), trend.astype(np.float32), True


def normalize_envelope_series(
    envelope: np.ndarray,
    trend: np.ndarray,
    detrended: np.ndarray,
    mode: str,
    dtype_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mode = (mode or DEFAULT_ENV_NORM_MODE).lower()
    if mode not in ENV_NORM_MODES:
        mode = DEFAULT_ENV_NORM_MODE
    if mode == "none":
        return envelope, trend, detrended, 1.0
    if mode == "scale":
        scale = dtype_scale if dtype_scale > 0 else 1.0
    elif mode == "envelope-max":
        scale = float(np.max(np.abs(envelope))) if envelope.size else 0.0
    else:  # detrended-max
        scale = float(np.max(np.abs(detrended))) if detrended.size else 0.0
    if scale <= 0:
        return envelope, trend, detrended, 1.0
    return envelope / scale, trend / scale, detrended / scale, scale


def compute_fft(audio: np.ndarray, sr: float, apply_bandpass: bool) -> Tuple[np.ndarray, np.ndarray]:
    if audio.size == 0:
        return np.array([]), np.array([])
    audio_bp = bandpass_filter(audio, sr) if apply_bandpass else audio
    n = audio_bp.size
    window = np.hanning(n).astype(np.float32)
    y = audio_bp * window
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(fft_vals) / np.maximum(np.sum(window), 1e-8)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return freqs, mag_db


def compute_envelope_fft(envelope: np.ndarray, sr: float, max_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    if envelope.size == 0:
        return np.array([]), np.array([])
    env = envelope.astype(np.float32)
    env = env - float(np.mean(env))
    n = env.size
    window = np.hanning(n).astype(np.float32)
    y = env * window
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(fft_vals) / np.maximum(np.sum(window), 1e-8)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    if max_freq and max_freq > 0:
        limit = min(max_freq, sr / 2.0)
        mask = freqs <= limit
        freqs = freqs[mask]
        mag_db = mag_db[mask]
    return freqs, mag_db


def extract_fft_peaks(freqs: np.ndarray, mag_db: np.ndarray, top_n: int) -> List[Tuple[float, float]]:
    if freqs.size < 3:
        return []
    mags = mag_db.copy()
    mags[0] = -np.inf
    peaks, _ = find_peaks(mags)
    if peaks.size == 0:
        peaks = np.argsort(mags)[-top_n:]
    ranked = sorted(peaks, key=lambda idx: mags[idx], reverse=True)
    return [(float(freqs[idx]), float(mags[idx])) for idx in ranked[:top_n]]


def extract_envelope_peaks(freqs: np.ndarray, mag_db: np.ndarray, top_n: int) -> List[Tuple[float, float]]:
    if freqs.size < 3:
        return []
    mags = mag_db.copy()
    mags[0] = -np.inf
    peaks, _ = find_peaks(mags)
    if peaks.size == 0:
        peaks = np.argsort(mags)[-top_n:]
    ranked = sorted(peaks, key=lambda idx: mags[idx], reverse=True)
    return [(float(freqs[idx]), float(mags[idx])) for idx in ranked[:top_n]]
def decimate_for_plot(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or x.size <= max_points:
        return x, y
    step = int(math.ceil(x.size / float(max_points)))
    return x[::step], y[::step]


def write_summary_csv(path: str, rows: List[Tuple[str, str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value", "unit"])
        writer.writerows(rows)


def write_fft_peaks_csv(path: str, peaks: List[Tuple[float, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "frequency_hz", "magnitude_db"])
        for idx, (freq, mag_db) in enumerate(peaks, start=1):
            writer.writerow([idx, f"{freq:.3f}", f"{mag_db:.3f}"])


def write_series_csv(path: str, t: np.ndarray, values: np.ndarray, value_label: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", value_label])
        for i in range(t.size):
            writer.writerow([f"{t[i]:.9f}", f"{values[i]:.6f}"])


def write_amp_freq_csv(path: str, t: np.ndarray, amp: np.ndarray, freq: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "amplitude", "freq_hz"])
        for i in range(t.size):
            writer.writerow([f"{t[i]:.9f}", f"{amp[i]:.6f}", f"{freq[i]:.3f}"])


def write_envelope_csv(path: str, t: np.ndarray, envelope: np.ndarray) -> None:
    write_series_csv(path, t, envelope, "envelope")


def write_envelope_trend_csv(path: str, t: np.ndarray, trend: np.ndarray) -> None:
    write_series_csv(path, t, trend, "trend")


def write_envelope_detrended_csv(path: str, t: np.ndarray, detrended: np.ndarray) -> None:
    write_series_csv(path, t, detrended, "envelope_detrended")


def _label_row(label: str, width: int) -> List[str]:
    row = ["label", label]
    while len(row) < width:
        row.append("")
    return row


def split_time_series_to_folder(
    out_dir: str,
    folder_name: str,
    label: str,
    columns: List[np.ndarray],
    headers: List[str],
    formatters: List,
    window_size_s: float,
    hop_size_s: float,
    sr: float,
) -> int:
    if not columns or len(columns) != len(headers) or len(headers) != len(formatters):
        raise ValueError("Invalid split columns or headers")
    length = int(columns[0].size)
    if any(int(col.size) != length for col in columns):
        raise ValueError("Split columns length mismatch")
    window_samples = int(round(float(window_size_s) * sr))
    hop_samples = int(round(float(hop_size_s) * sr))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Invalid window or hop size for splitting")
    if length < window_samples:
        return 0

    folder_path = os.path.join(out_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    segment_count = 1 + (length - window_samples) // hop_samples
    width = max(3, len(str(segment_count - 1)))
    label_row = _label_row(label, len(headers))

    for idx in range(segment_count):
        start = idx * hop_samples
        end = start + window_samples
        file_name = f"segment_{idx:0{width}d}.csv"
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(label_row)
            writer.writerow(headers)
            for k in range(start, end):
                row = []
                for j in range(len(columns)):
                    value = columns[j][k]
                    if j == 0 and headers[j] == "time_s":
                        value = value - columns[j][start]
                    row.append(formatters[j](value))
                writer.writerow(row)
    return segment_count


def plot_waveform(out_path: str, t: np.ndarray, audio: np.ndarray, max_points: int, title: str) -> None:
    t_plot, audio_plot = decimate_for_plot(t, audio, max_points)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t_plot, audio_plot, color="#1f77b4", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_amp_freq(
    out_path: str,
    t: np.ndarray,
    amp: np.ndarray,
    freq: np.ndarray,
    max_points: int,
) -> None:
    t_plot, amp_plot = decimate_for_plot(t, amp, max_points)
    _, freq_plot = decimate_for_plot(t, freq, max_points)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)
    ax1.plot(t_plot, amp_plot, color="#2ca02c", lw=0.6)
    ax1.set_ylabel("Amplitude (env)")
    ax1.grid(True, ls="--", alpha=0.4)
    ax2.plot(t_plot, freq_plot, color="#d62728", lw=0.6)
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_envelope(out_path: str, t: np.ndarray, envelope: np.ndarray, max_points: int) -> None:
    t_plot, env_plot = decimate_for_plot(t, envelope, max_points)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t_plot, env_plot, color="#17becf", lw=0.6)
    ax.set_title("Envelope")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_envelope_trend(
    out_path: str,
    t: np.ndarray,
    envelope: np.ndarray,
    trend: np.ndarray,
    max_points: int,
) -> None:
    t_plot, env_plot = decimate_for_plot(t, envelope, max_points)
    _, trend_plot = decimate_for_plot(t, trend, max_points)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t_plot, env_plot, color="#17becf", lw=0.6, label="Envelope")
    ax.plot(t_plot, trend_plot, color="#bcbd22", lw=1.0, label="Trend")
    ax.set_title("Envelope Trend")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_envelope_detrended(
    out_path: str,
    t: np.ndarray,
    detrended: np.ndarray,
    max_points: int,
) -> None:
    t_plot, det_plot = decimate_for_plot(t, detrended, max_points)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t_plot, det_plot, color="#ff9896", lw=0.6)
    ax.set_title("Envelope (Detrended)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_envelope_spectrum(out_path: str, freqs: np.ndarray, mag_db: np.ndarray, max_freq: float) -> None:
    if freqs.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(freqs, mag_db, color="#ff7f0e", lw=0.6)
    if max_freq and max_freq > 0:
        ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Envelope Spectrum")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_fft(out_path: str, freqs: np.ndarray, mag_db: np.ndarray, sr: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(freqs, mag_db, color="#9467bd", lw=0.6)
    ax.set_xlim(0, sr / 2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_spectrogram(out_path: str, audio: np.ndarray, sr: int, apply_bandpass: bool) -> None:
    if audio.size == 0:
        return
    audio_bp = bandpass_filter(audio, sr) if apply_bandpass else audio
    nperseg = min(1024, audio_bp.size)
    if nperseg < 8:
        return
    noverlap = int(nperseg * 0.75)
    f, t, sxx = spectrogram(
        audio_bp,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    sxx = np.maximum(sxx, 1e-12)
    sxx_db = 20.0 * np.log10(sxx)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    im = ax.imshow(
        sxx_db,
        origin="lower",
        aspect="auto",
        extent=[t.min() if t.size else 0.0, t.max() if t.size else 0.0, f.min() if f.size else 0.0, f.max() if f.size else 0.0],
        cmap="magma",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_summary_rows(
    wav_path: str,
    sr: float,
    analysis_sr: float,
    decimate_k: int,
    decimate_ratio: float,
    dtype: np.dtype,
    channels: int,
    frames: int,
    label: str,
    center_duration_s: float,
    split_enabled: bool,
    window_size_s: float,
    hop_size_s: float,
    segment_count: int,
    segment_samples: int,
    analysis_start_s: float,
    analysis_end_s: float,
    analysis_frames: int,
    analysis_duration_s: float,
    audio: np.ndarray,
    freqs: np.ndarray,
    mag_db: np.ndarray,
    inst_freq: np.ndarray,
    peaks: List[Tuple[float, float]],
    envelope_peaks: List[Tuple[float, float]],
    apply_bandpass: bool,
    channel_mode: str,
    channel_label: str,
    channel_stats: dict,
    env_cutoff: float,
    env_max_freq: float,
    env_detrend_cutoff: float,
    detrend_applied: bool,
    env_norm_mode: str,
    env_norm_scale: float,
) -> List[Tuple[str, str, str]]:
    duration_s = frames / float(sr) if sr > 0 else 0.0
    mean = float(np.mean(audio)) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    crest = peak / rms if rms > 0 else 0.0
    crest_db = 20.0 * math.log10(crest) if crest > 0 else 0.0
    mag = 10 ** (mag_db / 20.0) if mag_db.size else np.array([])
    if mag.size and np.sum(mag) > 0:
        centroid = float(np.sum(freqs * mag) / np.sum(mag))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    else:
        centroid = 0.0
        bandwidth = 0.0
    valid_freq = inst_freq[np.isfinite(inst_freq) & (inst_freq > 0)]
    freq_mean = float(np.mean(valid_freq)) if valid_freq.size else 0.0
    freq_median = float(np.median(valid_freq)) if valid_freq.size else 0.0
    dominant_freq = peaks[0][0] if peaks else 0.0
    env_peak_freq = envelope_peaks[0][0] if envelope_peaks else 0.0
    env_peak_db = envelope_peaks[0][1] if envelope_peaks else 0.0
    env_period = 1.0 / env_peak_freq if env_peak_freq > 0 else 0.0
    rms_list = channel_stats.get("rms") or []
    rms_text = ",".join(f"{val:.3f}" for val in rms_list) if rms_list else ""
    corr_lr = channel_stats.get("corr_lr")
    corr_text = f"{corr_lr:.3f}" if corr_lr is not None else ""
    rows = [
        ("file_name", os.path.basename(wav_path), ""),
        ("label", label, ""),
        ("file_path", os.path.abspath(wav_path), ""),
        ("sample_rate", f"{sr:.3f}", "Hz"),
        ("analysis_sample_rate", f"{analysis_sr:.3f}", "Hz"),
        ("decimate_k", f"{decimate_k}", ""),
        ("decimate_ratio", f"{decimate_ratio:.6f}", ""),
        ("channels", f"{channels}", ""),
        ("sample_format", str(dtype), ""),
        ("frames", f"{frames}", "samples"),
        ("duration", f"{duration_s:.6f}", "s"),
        ("center_duration", f"{center_duration_s:.6f}", "s"),
        ("export_slices", "yes" if split_enabled else "no", ""),
        ("split_window_size", f"{window_size_s:.6f}", "s"),
        ("split_hop_size", f"{hop_size_s:.6f}", "s"),
        ("split_window_count", f"{segment_count}", ""),
        ("split_window_samples", f"{segment_samples}", "samples"),
        ("analysis_frames", f"{analysis_frames}", "samples"),
        ("analysis_duration", f"{analysis_duration_s:.6f}", "s"),
        ("analysis_start_s", f"{analysis_start_s:.6f}", "s"),
        ("analysis_end_s", f"{analysis_end_s:.6f}", "s"),
        ("channel_mode", channel_mode, ""),
        ("channel_selected", channel_label, ""),
        ("channel_rms", rms_text, ""),
        ("channel_corr_lr", corr_text, ""),
        ("mean_amplitude", f"{mean:.6f}", ""),
        ("rms_amplitude", f"{rms:.6f}", ""),
        ("peak_amplitude", f"{peak:.6f}", ""),
        ("crest_factor", f"{crest:.6f}", ""),
        ("crest_factor_db", f"{crest_db:.3f}", "dB"),
        ("dominant_freq", f"{dominant_freq:.3f}", "Hz"),
        ("spectral_centroid", f"{centroid:.3f}", "Hz"),
        ("spectral_bandwidth", f"{bandwidth:.3f}", "Hz"),
        ("inst_freq_mean", f"{freq_mean:.3f}", "Hz"),
        ("inst_freq_median", f"{freq_median:.3f}", "Hz"),
        ("envelope_peak_freq", f"{env_peak_freq:.3f}", "Hz"),
        ("envelope_peak_period", f"{env_period:.3f}", "s"),
        ("envelope_peak_db", f"{env_peak_db:.3f}", "dB"),
        ("envelope_cutoff", f"{env_cutoff:.3f}", "Hz"),
        ("envelope_max_freq", f"{env_max_freq:.3f}", "Hz"),
        ("envelope_detrend_cutoff", f"{env_detrend_cutoff:.3f}", "Hz"),
        ("envelope_detrend_applied", "yes" if detrend_applied else "no", ""),
        ("envelope_norm_mode", env_norm_mode, ""),
        ("envelope_norm_scale", f"{env_norm_scale:.6f}", ""),
        ("bandpass_enabled", "yes" if apply_bandpass else "no", ""),
    ]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a WAV file and generate CSV tables/slices.")
    parser.add_argument("wav_path", help="Path to a .wav file")
    parser.add_argument("--out", dest="out_dir", default=None, help="Output directory")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable bandpass filtering")
    parser.add_argument("--peaks", type=int, default=10, help="Number of FFT peaks to export")
    parser.add_argument(
        "--channel",
        choices=CHANNEL_MODES,
        default="auto",
        help="Channel mode: auto/left/right/mean/sum/diff",
    )
    parser.add_argument(
        "--env-cutoff",
        type=float,
        default=DEFAULT_ENV_CUTOFF,
        help="Envelope lowpass cutoff (Hz)",
    )
    parser.add_argument(
        "--env-max-freq",
        type=float,
        default=DEFAULT_ENV_MAX_FREQ,
        help="Envelope spectrum max frequency (Hz); set <=0 to skip",
    )
    parser.add_argument(
        "--env-detrend-cutoff",
        type=float,
        default=DEFAULT_ENV_TREND_CUTOFF,
        help="Envelope trend lowpass cutoff (Hz); set <=0 to disable detrend",
    )
    parser.add_argument(
        "--env-norm",
        choices=ENV_NORM_MODES,
        default=DEFAULT_ENV_NORM_MODE,
        help="Envelope normalization: detrended-max/envelope-max/scale/none",
    )
    parser.add_argument(
        "--decimate-k",
        type=int,
        default=DEFAULT_DECIMATE_K,
        help="Decimate by keeping every k-th sample (k>=1)",
    )
    parser.add_argument(
        "--center-duration",
        type=float,
        default=DEFAULT_CENTER_DURATION,
        help="Center duration (seconds) to keep for analysis",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=DEFAULT_WINDOW_SIZE,
        help="Window size in seconds for slice export",
    )
    parser.add_argument(
        "--hop-size",
        type=float,
        default=DEFAULT_HOP_SIZE,
        help="Hop size in seconds for slice export",
    )
    parser.add_argument(
        "--export-slices",
        action="store_true",
        default=DEFAULT_EXPORT_SLICES,
        help="Export sliced CSVs (folders raw.csv/envelope.csv/envelope_detrended.csv)",
    )
    parser.add_argument(
        "--no-slices",
        action="store_false",
        dest="export_slices",
        help="Disable exporting sliced CSVs",
    )
    return parser.parse_args()


def process_wav(
    wav_path: str,
    out_dir: Optional[str] = None,
    apply_bandpass: bool = True,
    peaks: int = 10,
    channel_mode: str = "auto",
    env_cutoff: float = DEFAULT_ENV_CUTOFF,
    env_max_freq: float = DEFAULT_ENV_MAX_FREQ,
    env_detrend_cutoff: float = DEFAULT_ENV_TREND_CUTOFF,
    env_norm_mode: str = DEFAULT_ENV_NORM_MODE,
    decimate_k: int = DEFAULT_DECIMATE_K,
    center_duration_s: float = DEFAULT_CENTER_DURATION,
    export_slices: bool = DEFAULT_EXPORT_SLICES,
    window_size_s: float = DEFAULT_WINDOW_SIZE,
    hop_size_s: float = DEFAULT_HOP_SIZE,
) -> str:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)
    if out_dir is None:
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        out_dir = os.path.join(os.getcwd(), "wav_views", stem)
    os.makedirs(out_dir, exist_ok=True)

    sr, audio_raw, audio, dtype, channels, frames, channel_label, channel_stats = read_wav(
        wav_path, channel_mode=channel_mode
    )
    original_sr = float(sr)
    decimate_k = max(1, int(decimate_k))
    center_duration_s = max(0.0, float(center_duration_s))
    window_size_s = max(0.0, float(window_size_s))
    hop_size_s = max(0.0, float(hop_size_s))

    audio_raw, analysis_sr, decimate_k_actual = decimate_audio(audio_raw, original_sr, decimate_k)
    audio, analysis_sr_norm, _ = decimate_audio(audio, original_sr, decimate_k)
    if abs(analysis_sr_norm - analysis_sr) > 1e-6:
        analysis_sr = analysis_sr_norm
    decimate_ratio = 1.0 / float(decimate_k_actual)

    audio_raw, audio, trim_info = center_crop_audio(audio_raw, audio, analysis_sr, center_duration_s)
    t = np.arange(audio.size, dtype=np.float64) / float(analysis_sr) if analysis_sr > 0 else np.array([])

    split_enabled = bool(export_slices)
    segment_samples = int(round(window_size_s * analysis_sr)) if split_enabled else 0
    segment_count = 0
    if split_enabled:
        hop_samples = int(round(hop_size_s * analysis_sr))
        if segment_samples <= 0 or hop_samples <= 0 or audio.size < segment_samples:
            split_enabled = False
        else:
            segment_count = 1 + (audio.size - segment_samples) // hop_samples

    t_amp, amp, inst_freq = compute_amp_freq_series(audio, analysis_sr, apply_bandpass=apply_bandpass)
    freqs, mag_db = compute_fft(audio, analysis_sr, apply_bandpass=apply_bandpass)
    peaks_list = extract_fft_peaks(freqs, mag_db, top_n=max(1, peaks))
    t_env, envelope_raw = compute_envelope_series(
        audio_raw, analysis_sr, apply_bandpass=apply_bandpass, env_cutoff=env_cutoff
    )
    detrended_raw, env_trend_raw, detrend_applied = detrend_envelope(
        envelope_raw, analysis_sr, env_detrend_cutoff
    )
    dtype_scale = _dtype_scale(audio_raw, dtype)
    envelope, env_trend, detrended_env, env_norm_scale = normalize_envelope_series(
        envelope_raw,
        env_trend_raw,
        detrended_raw,
        env_norm_mode,
        dtype_scale,
    )
    env_freqs = np.array([])
    env_mag_db = np.array([])
    env_peaks: List[Tuple[float, float]] = []
    if env_max_freq and env_max_freq > 0:
        env_freqs, env_mag_db = compute_envelope_fft(detrended_env, analysis_sr, max_freq=env_max_freq)
        env_peaks = extract_envelope_peaks(env_freqs, env_mag_db, top_n=max(1, peaks))

    label = os.path.splitext(os.path.basename(wav_path))[0]
    summary_rows = build_summary_rows(
        wav_path,
        original_sr,
        analysis_sr,
        decimate_k_actual,
        decimate_ratio,
        dtype,
        channels,
        frames,
        label,
        center_duration_s,
        split_enabled,
        window_size_s,
        hop_size_s,
        segment_count,
        segment_samples,
        trim_info["analysis_start_s"],
        trim_info["analysis_end_s"],
        trim_info["analysis_frames"],
        trim_info["analysis_duration_s"],
        audio,
        freqs,
        mag_db,
        inst_freq,
        peaks_list,
        env_peaks,
        apply_bandpass,
        channel_mode,
        channel_label,
        channel_stats,
        env_cutoff,
        env_max_freq,
        env_detrend_cutoff,
        detrend_applied,
        env_norm_mode,
        env_norm_scale,
    )
    write_summary_csv(os.path.join(out_dir, "summary.csv"), summary_rows)
    write_fft_peaks_csv(os.path.join(out_dir, "fft_peaks.csv"), peaks_list)
    if env_max_freq and env_max_freq > 0:
        write_fft_peaks_csv(os.path.join(out_dir, "envelope_peaks.csv"), env_peaks)

    full_dir = os.path.join(out_dir, "full_csv")
    os.makedirs(full_dir, exist_ok=True)
    write_series_csv(os.path.join(full_dir, "raw.csv"), t, audio_raw, "raw")
    write_envelope_csv(os.path.join(full_dir, "envelope.csv"), t_env, envelope)
    write_envelope_detrended_csv(os.path.join(full_dir, "envelope_detrended.csv"), t_env, detrended_env)
    if detrend_applied:
        write_envelope_trend_csv(os.path.join(full_dir, "envelope_trend.csv"), t_env, env_trend)

    if split_enabled:
        split_time_series_to_folder(
            out_dir,
            "raw.csv",
            label,
            [t, audio_raw],
            ["time_s", "raw"],
            [lambda v: f"{v:.9f}", lambda v: f"{v:.6f}"],
            window_size_s,
            hop_size_s,
            analysis_sr,
        )
        split_time_series_to_folder(
            out_dir,
            "envelope.csv",
            label,
            [t_env, envelope],
            ["time_s", "envelope"],
            [lambda v: f"{v:.9f}", lambda v: f"{v:.6f}"],
            window_size_s,
            hop_size_s,
            analysis_sr,
        )
        split_time_series_to_folder(
            out_dir,
            "envelope_detrended.csv",
            label,
            [t_env, detrended_env],
            ["time_s", "envelope_detrended"],
            [lambda v: f"{v:.9f}", lambda v: f"{v:.6f}"],
            window_size_s,
            hop_size_s,
            analysis_sr,
        )

    return out_dir


def main() -> int:
    args = parse_args()
    apply_bandpass = not args.no_bandpass
    try:
        out_dir = process_wav(
            args.wav_path,
            out_dir=args.out_dir,
            apply_bandpass=apply_bandpass,
            peaks=args.peaks,
            channel_mode=args.channel,
            env_cutoff=args.env_cutoff,
            env_max_freq=args.env_max_freq,
            env_detrend_cutoff=args.env_detrend_cutoff,
            env_norm_mode=args.env_norm,
            center_duration_s=args.center_duration,
            decimate_k=args.decimate_k,
            export_slices=args.export_slices,
            window_size_s=args.window_size,
            hop_size_s=args.hop_size,
        )
    except FileNotFoundError:
        print(f"File not found: {args.wav_path}", file=sys.stderr)
        return 1
    print(f"Saved outputs to: {out_dir}")
    print("Generated: summary.csv, fft_peaks.csv")
    if args.env_max_freq and args.env_max_freq > 0:
        print("Generated: envelope_peaks.csv")
    print("Generated: full_csv/raw.csv, full_csv/envelope.csv, full_csv/envelope_detrended.csv")
    if args.env_detrend_cutoff and args.env_detrend_cutoff > 0:
        print("Generated (if detrend applied): full_csv/envelope_trend.csv")
    if args.export_slices:
        print("Generated: raw.csv/, envelope.csv/, envelope_detrended.csv/ (sliced CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
