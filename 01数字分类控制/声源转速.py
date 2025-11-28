import sys
import os
import time
import wave
import struct
import csv
import collections
import serial
import serial.tools.list_ports
import numpy as np
from scipy.signal import hilbert, butter, filtfilt, spectrogram, find_peaks
from typing import Dict, Tuple, List, Optional

# --- PyTorch Imports ---

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox, QSlider,
    QMessageBox, QDialog, QDialogButtonBox, QSpinBox,
    QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, pyqtSlot
 

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from matplotlib import font_manager

matplotlib.use('QtAgg')

_font_candidates = [
    'SimHei',
    'Microsoft YaHei',
    'WenQuanYi Zen Hei',
    'Noto Sans CJK SC',
    'Source Han Sans CN',
    'PingFang SC',
    'Heiti SC',
    'Songti SC',
    'Arial Unicode MS',
    'DejaVu Sans',
]


def _normalize_font_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _candidate_keys(font_name: str) -> List[str]:
    normalized = _normalize_font_name(font_name)
    keys = [normalized]
    if "cjk" in normalized:
        idx = normalized.index("cjk") + 3
        keys.append(normalized[:idx])
    for suffix in ("sc", "tc", "jp", "kr", "cn"):
        if normalized.endswith(suffix):
            keys.append(normalized[: -len(suffix)])
    return [key for key in keys if key]


def _build_font_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    normalized_to_name: Dict[str, str] = {}
    name_to_path: Dict[str, str] = {}
    try:
        system_fonts = font_manager.findSystemFonts(fontext="ttf")
    except Exception:
        system_fonts = []
    for font_path in system_fonts:
        try:
            prop = font_manager.FontProperties(fname=font_path)
            fam_name = prop.get_name()
        except Exception:
            continue
        if not fam_name:
            continue
        norm = _normalize_font_name(fam_name)
        normalized_to_name.setdefault(norm, fam_name)
        name_to_path.setdefault(fam_name, font_path)
    for font_item in getattr(font_manager.fontManager, "ttflist", []):
        fam_name = getattr(font_item, "name", None)
        font_path = getattr(font_item, "fname", None)
        if fam_name:
            normalized_to_name.setdefault(_normalize_font_name(fam_name), fam_name)
            if font_path:
                name_to_path.setdefault(fam_name, font_path)
    return normalized_to_name, name_to_path


def _find_matching_font(
    candidate_keys: List[str], normalized_lookup: Dict[str, str]
) -> Optional[str]:
    for key in candidate_keys:
        actual_name = normalized_lookup.get(key)
        if actual_name:
            return actual_name
    for key in candidate_keys:
        if not key:
            continue
        require_cjk = "cjk" in key
        for normalized, actual_name in normalized_lookup.items():
            if require_cjk and "cjk" not in normalized:
                continue
            if normalized.startswith(key) or key.startswith(normalized):
                return actual_name
    return None


def _pick_available_font() -> str:
    normalized_lookup, path_lookup = _build_font_lookup()
    for font_name in _font_candidates:
        candidate_keys = _candidate_keys(font_name)
        actual_name = _find_matching_font(candidate_keys, normalized_lookup)
        if actual_name:
            font_path = path_lookup.get(actual_name)
            if font_path:
                try:
                    font_manager.fontManager.addfont(font_path)
                except Exception:
                    pass
            return actual_name
    # 延伸匹配：寻找名称中包含常见关键字的字体
    keywords = ("notosanscjk", "sourcehansans", "wenquanyi", "heiti", "pingfang", "ukai", "uming", "yahei")
    for normalized, actual_name in normalized_lookup.items():
        if any(keyword in normalized for keyword in keywords):
            font_path = path_lookup.get(actual_name)
            if font_path:
                try:
                    font_manager.fontManager.addfont(font_path)
                except Exception:
                    pass
            return actual_name
    return 'DejaVu Sans'


_resolved_font = _pick_available_font()
matplotlib.rcParams['font.family'] = _resolved_font
matplotlib.rcParams['font.sans-serif'] = [_resolved_font, 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 配置参数
# ==============================================================================
BAUD_RATE = 921600
SAMPLING_RATE = 16000
BANDPASS_LOWCUT = 2500
BANDPASS_HIGHCUT = 8000
BANDPASS_ORDER = 4


# ==============================================================================
# 2.（已移除：CNN 模型定义）
# ==============================================================================


# ==============================================================================
# 3. 预处理工具函数
# ==============================================================================
 

def lowpass_filter(audio_np, sr, cutoff=50, order=2):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    if normal_cutoff <= 0 or normal_cutoff >= 1.0:
        return audio_np
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio_np)

def bandpass_filter(audio_np, sr, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT, order=BANDPASS_ORDER):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    if low >= high or low <= 0 or high >= 1.0:
        return audio_np
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, audio_np)

 

 



def _compute_envelope(audio_np, sr):
    """
    Bandpass + Hilbert to obtain a smoothed amplitude envelope.
    Returns envelope (float32) with same length as input.
    """
    x = np.asarray(audio_np, dtype=np.float32)
    x_bp = bandpass_filter(x, sr)
    analytic = hilbert(x_bp)
    env = np.abs(analytic).astype(np.float32)
    try:
        env = lowpass_filter(env, sr, cutoff=10, order=2)
    except Exception:
        pass
    return env


def _estimate_period_seconds_from_env(env, sr, min_s=0.2, max_s=8.0):
    """
    Estimate dominant period (seconds) from envelope autocorrelation peak.
    """
    if len(env) < 4:
        return None
    x = env - float(np.mean(env))
    ac = np.correlate(x, x, mode='full')
    ac = ac[len(ac)//2 + 1:]
    L = int(max(1, min_s * sr))
    R = int(min(len(ac) - 1, max_s * sr))
    if R <= L:
        return None
    lag = int(np.argmax(ac[L:R]) + L)
    if lag <= 0:
        return None
    return lag / float(sr)


def segment_cycles(audio_int16, sr, expected_period_s=None):
    """
    Segment audio into cycles using envelope peaks.
    Returns list of (start_idx, end_idx) sample indices in the original signal.
    """
    if audio_int16 is None or len(audio_int16) < max(10, int(0.2 * sr)):
        return []

    x = np.asarray(audio_int16, dtype=np.float32) / 32768.0
    env = _compute_envelope(x, sr)

    if expected_period_s is None or expected_period_s <= 0:
        period_s = _estimate_period_seconds_from_env(env, sr)
    else:
        period_s = expected_period_s

    if period_s is None:
        return []

    period_samples = max(1, int(period_s * sr))
    prom = float(np.percentile(env, 90) - np.percentile(env, 10)) * 0.25
    prom = max(prom, 1e-6)
    peaks, _ = find_peaks(env, distance=max(1, int(period_samples * 0.6)), prominence=prom)

    segs = []
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            s, e = int(peaks[i]), int(peaks[i + 1])
            dur = e - s
            if dur <= 0:
                continue
            if 0.5 * period_samples <= dur <= 1.5 * period_samples:
                segs.append((s, e))

    if not segs:
        start = 0
        while start + period_samples <= len(x):
            segs.append((start, start + period_samples))
            start += period_samples

    return segs

def compute_amp_freq_series(audio_int16, sr):
    """
    Compute time, amplitude (envelope), and instantaneous frequency (Hz)
    from an int16 PCM array using bandpass + Hilbert transform.
    Returns (t, amplitude, freq_hz) with length N.
    """
    if audio_int16 is None or len(audio_int16) == 0:
        return np.array([]), np.array([]), np.array([])

    # Normalize to [-1, 1]
    audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0

    # Band-limit to reduce noise before Hilbert
    audio_bp = bandpass_filter(audio, sr)

    # Analytic signal for envelope and phase
    analytic = hilbert(audio_bp)
    amplitude = np.abs(analytic)

    # Instantaneous frequency from unwrapped phase derivative
    phase = np.unwrap(np.angle(analytic))
    dphase = np.gradient(phase)
    inst_freq = (sr / (2.0 * np.pi)) * dphase

    # Mild smoothing for readability (lowpass). Falls back if cutoff invalid.
    try:
        amplitude = lowpass_filter(amplitude, sr, cutoff=50, order=2)
        inst_freq = lowpass_filter(inst_freq, sr, cutoff=50, order=2)
    except Exception:
        pass

    # Ensure non-negative frequency
    inst_freq = np.maximum(inst_freq, 0.0)

    # Time vector at sample precision
    t = np.arange(len(audio), dtype=np.float64) / float(sr)

    return t, amplitude.astype(np.float32), inst_freq.astype(np.float32)


def export_amp_freq_csv(audio_int16, sr, csv_path):
    """
    Export CSV with columns: time_s, amplitude, freq_hz
    """
    t, amp, freq = compute_amp_freq_series(audio_int16, sr)
    if t.size == 0:
        raise ValueError("No audio samples to export")

    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "amplitude", "freq_hz"])
        for i in range(len(t)):
            writer.writerow([f"{t[i]:.9f}", f"{amp[i]:.6f}", f"{freq[i]:.3f}"])


def _resample_series(t_src, y_src, n_points):
    """
    Resample a series y(t) to n_points using linear interpolation.
    Returns (t_resampled, y_resampled), where t_resampled spans [t_src[0], t_src[-1]].
    """
    if len(t_src) == 0 or len(y_src) == 0 or n_points <= 1:
        return np.array([]), np.array([])
    t0 = float(t_src[0])
    t1 = float(t_src[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        # fallback to index space
        t_src = np.arange(len(y_src), dtype=np.float64)
        t0, t1 = t_src[0], t_src[-1]
    t_new = np.linspace(t0, t1, int(n_points), dtype=np.float64)
    y_new = np.interp(t_new, t_src, y_src)
    return t_new, y_new.astype(np.float32)


def save_cycle_csvs(audio_int16, sr, out_dir, target_points_per_cycle=None):
    """
    Detect cycles and export each as a CSV with exactly target_points_per_cycle rows.
    Columns: time_s, amplitude, freq_hz (computed via Hilbert on the cycle and resampled).
    """
    os.makedirs(out_dir, exist_ok=True)
    segs = segment_cycles(audio_int16, sr)
    if not segs:
        return 0

    total_saved = 0
    for i, (s, e) in enumerate(segs):
        if e - s <= 10:
            continue
        segment = np.asarray(audio_int16[s:e], dtype=np.int16)
        # Compute series at native resolution
        t, amp, freq = compute_amp_freq_series(segment, sr)
        if t.size == 0:
            continue
        # Optionally resample if target_points_per_cycle is set (>1); otherwise keep original points
        if target_points_per_cycle and target_points_per_cycle > 1:
            t_out, amp_out = _resample_series(t, amp, target_points_per_cycle)
            _, freq_out = _resample_series(t, freq, target_points_per_cycle)
        else:
            t_out, amp_out, freq_out = t, amp, freq
        # Write CSV
        csv_path = os.path.join(out_dir, f"cycle_{i:02d}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "amplitude", "freq_hz"]) 
            for k in range(len(t_out)):
                writer.writerow([f"{t_out[k]:.9f}", f"{amp_out[k]:.6f}", f"{freq_out[k]:.3f}"])
        total_saved += 1
    return total_saved


def save_data_and_fft_plots(audio_int16, sr, out_dir):
    """
    Save two plots into out_dir:
      - data_plots.png: amplitude (envelope) vs time and instantaneous frequency vs time
      - fft_plot.png: magnitude spectrum of the (bandpass-filtered) audio
    """
    os.makedirs(out_dir, exist_ok=True)

    audio = np.asarray(audio_int16, dtype=np.float32) / 32768.0
    # Bandpass for stability/denoising
    audio_bp = bandpass_filter(audio, sr)

    # Time, amplitude envelope and inst. frequency
    t, amp, inst_f = compute_amp_freq_series(audio_int16, sr)

    # 1) Data plots (two subplots)
    fig1 = Figure(figsize=(10, 6), dpi=120)
    ax1, ax2 = fig1.subplots(2, 1, sharex=True)
    ax1.plot(t, amp, color='tab:blue', lw=0.8)
    ax1.set_ylabel('Amplitude (env)')
    ax1.grid(True, ls='--', alpha=0.4)
    ax2.plot(t, inst_f, color='tab:red', lw=0.8)
    ax2.set_ylabel('Freq (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, ls='--', alpha=0.4)

    # Detect cycles and annotate on plots
    try:
        segs = segment_cycles(audio_int16, sr)
    except Exception:
        segs = []
    if segs:
        ylim1 = ax1.get_ylim()
        y_text = ylim1[0] + 0.9 * (ylim1[1] - ylim1[0])
        for idx, (s, e) in enumerate(segs, start=1):
            ts = s / float(sr)
            te = e / float(sr)
            ax1.axvline(ts, color='tab:purple', lw=0.7, ls='--', alpha=0.7)
            ax2.axvline(ts, color='tab:purple', lw=0.7, ls='--', alpha=0.7)
            tc = 0.5 * (ts + te)
            try:
                ax1.text(tc, y_text, f"周期{idx}", color='tab:purple', fontsize=8,
                         ha='center', va='bottom', rotation=0, alpha=0.9)
            except Exception:
                pass

    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'data_plots.png'))

    # 2) FFT magnitude spectrum
    n = len(audio_bp)
    if n > 0:
        # Hann window to reduce leakage
        window = np.hanning(n).astype(np.float32)
        y = audio_bp * window
        # rFFT
        Y = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, d=1.0/sr)
        mag = np.abs(Y) / np.maximum(np.sum(window), 1e-8)
        # dB for readability
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        fig2 = Figure(figsize=(10, 4), dpi=120)
        ax = fig2.subplots(1, 1)
        ax.plot(freqs, mag_db, color='tab:green', lw=0.8)
        ax.set_xlim(0, sr/2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, ls='--', alpha=0.4)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, 'fft_plot.png'))

    # 3) Spectrogram (time-frequency)
    try:
        f, tt, S = spectrogram(
            audio_bp,
            fs=sr,
            window='hann',
            nperseg=1024,
            noverlap=768,
            detrend=False,
            scaling='spectrum',
            mode='magnitude'
        )
        S = np.maximum(S, 1e-12)
        S_db = 20.0 * np.log10(S)

        fig3 = Figure(figsize=(10, 4), dpi=120)
        ax3 = fig3.subplots(1, 1)
        im = ax3.imshow(
            S_db,
            origin='lower',
            aspect='auto',
            extent=[tt.min() if tt.size else 0.0,
                    tt.max() if tt.size else (n / float(sr)),
                    f.min() if f.size else 0.0,
                    f.max() if f.size else (sr / 2.0)],
            cmap='magma'
        )
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Spectrogram (dB)')
        fig3.colorbar(im, ax=ax3, label='dB')
        fig3.tight_layout()
        fig3.savefig(os.path.join(out_dir, 'spectrogram.png'))
    except Exception:
        pass


# ==============================================================================
# 4. PyQt6 GUI 组件
# ==============================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#e6e6e6')
        super(MplCanvas, self).__init__(self.fig)

class WaveformWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super(WaveformWidget, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(self)
        layout.addWidget(self.canvas)
        self.plot_window_samples = SAMPLING_RATE * 2
        self.data_buffer = collections.deque([0] * self.plot_window_samples, 
                                           maxlen=self.plot_window_samples)
        self.x_data = np.arange(self.plot_window_samples)
        self.canvas.axes.set_ylim(-32768, 32768)
        self.canvas.axes.set_xlim(0, self.plot_window_samples)
        self.canvas.axes.set_title("实时声音波形")
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6)
        self.line, = self.canvas.axes.plot(self.x_data, list(self.data_buffer), lw=1, color='c')
        self.canvas.fig.tight_layout()
        self.plot_timer = QTimer()
        self.plot_timer.setInterval(40)  # ~25 FPS
        self.plot_timer.timeout.connect(self._refresh_plot)
    
    def add_data(self, new_samples):
        self.data_buffer.extend(new_samples)
    
    def _refresh_plot(self):
        current_data_list = list(self.data_buffer)
        min_val, max_val = -32768, 32768
        if current_data_list and np.max(np.abs(current_data_list)) > 0:
            min_val_data, max_val_data = np.min(current_data_list), np.max(current_data_list)
            padding = (max_val_data - min_val_data) * 0.1 + 100
            min_val, max_val = min_val_data - padding, max_val_data + padding
        self.canvas.axes.set_ylim(min_val, max_val)
        self.line.set_ydata(current_data_list)
        self.canvas.draw()
    
    def start_updates(self):
        if not self.plot_timer.isActive():
            self.plot_timer.start()
    
    def stop_updates(self):
        if self.plot_timer.isActive():
            self.plot_timer.stop()
        self.line.set_ydata(np.zeros(self.plot_window_samples))
        self.canvas.draw()


# ==============================================================================
# 5. 串口工作线程
# ==============================================================================
class SerialWorker(QThread):
    text_received = pyqtSignal(str)
    binary_data_received = pyqtSignal(list)
    recording_finished = pyqtSignal(list)

    def __init__(self, serial_instance):
        super().__init__()
        self.serial = serial_instance
        self._is_running = True
        self._is_streaming_mode = False
        self._is_recording = False
        self._recorded_samples = []
        self._samples_to_record = 0
        

    def run(self):
        packet_accumulator = bytearray()

        while self._is_running and self.serial and self.serial.is_open:
            try:
                if self._is_streaming_mode:
                    if self.serial.in_waiting > 0:
                        data_chunk = self.serial.read(self.serial.in_waiting)
                        if data_chunk:
                            packet_accumulator.extend(data_chunk)
                            while len(packet_accumulator) >= 128:
                                raw_data_packet = packet_accumulator[:128]
                                del packet_accumulator[:128]
                                samples = list(struct.unpack('<64h', raw_data_packet))
                                self.binary_data_received.emit(samples)
                                # 录制逻辑：在采集中并且开启录制时，将样本写入缓冲并在达到目标长度后自动结束
                                if self._is_recording:
                                    remaining_needed = self._samples_to_record - len(self._recorded_samples)
                                    if remaining_needed > 0:
                                        take_count = min(remaining_needed, len(samples))
                                        self._recorded_samples.extend(samples[:take_count])
                                    if len(self._recorded_samples) >= self._samples_to_record:
                                        # 达到目标长度，自动停止并发出数据
                                        self.stop_recording(emit_signal=True)

                    
                else:
                    if self.serial.in_waiting > 0:
                        line_bytes = self.serial.readline()
                        if line_bytes:
                            text = line_bytes.decode('ascii', errors='ignore').strip()
                            if text:
                                self.text_received.emit(text)
                
                time.sleep(0.01)
            except serial.SerialException:
                break
        print("后台工作线程结束。")

    

    def stop(self):
        self._is_running = False
    
    def set_streaming_mode(self, enabled: bool):
        self._is_streaming_mode = enabled
        
    
    def start_recording(self, duration_seconds: float, sample_rate: int):
        if not self._is_streaming_mode:
            return
        # 将秒数转换为采样点并取整，确保为正整数
        try:
            total_samples = int(max(1, round(float(duration_seconds) * float(sample_rate))))
        except Exception:
            total_samples = int(max(1, sample_rate))
        self._samples_to_record = total_samples
        self._recorded_samples = []
        self._is_recording = True
    
    def stop_recording(self, emit_signal=False):
        if not self._is_recording:
            return
        self._is_recording = False
        if emit_signal:
            self.recording_finished.emit(list(self._recorded_samples))
        self._recorded_samples = []


# ==============================================================================
# 6. LED控制窗口
# ==============================================================================
 


# ==============================================================================
# 7. 主窗口
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("集成式声学数字分类与控制面板")
        self.setGeometry(100, 100, 1200, 800)
        self.serial_port = None
        self.serial_worker = None
        self.is_recording = False
        self.controls_to_disable = []
        
        
        # 初始化UI
        self.initUI()
        self.toggle_all_controls(False)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 控制面板
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        
        # 显示面板
        display_panel = QWidget()
        display_layout = QVBoxLayout(display_panel)
        
        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(display_panel, 2)

        # 顶部布局
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._create_connection_group())
        top_layout.addWidget(self._create_status_group())
        
        # 网格布局
        grid_controls = QGridLayout()
        grid_controls.addWidget(self._create_playback_group(), 0, 0)
        grid_controls.addWidget(self._create_audio_capture_group(), 0, 1)
        grid_controls.addWidget(self._create_motor_group(), 1, 0)
        
        
        controls_layout.addLayout(top_layout)
        controls_layout.addLayout(grid_controls)
        controls_layout.addStretch()
        
        # 波形图显示
        self.waveform_plot = WaveformWidget()
        display_layout.addWidget(self.waveform_plot, 3)

    

    def _create_connection_group(self):
        group = QGroupBox("连接设置")
        layout = QGridLayout(group)
        
        self.port_input = QComboBox()
        self.refresh_ports()
        
        self.baud_input = QLineEdit("921600")
        self.baud_input.setReadOnly(True)
        
        self.connect_btn = QPushButton("打开连接")
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_ports)
        
        layout.addWidget(QLabel("串口:"), 0, 0)
        layout.addWidget(self.port_input, 0, 1)
        layout.addWidget(refresh_btn, 0, 2)
        layout.addWidget(QLabel("波特率:"), 1, 0)
        layout.addWidget(self.baud_input, 1, 1)
        layout.addWidget(self.connect_btn, 2, 0, 1, 3)
        
        return group
    
    def _create_status_group(self):
        group = QGroupBox("设备状态")
        layout = QGridLayout(group)
        
        self.player_status = QLabel("⚪ 未连接")
        self.led_motor_status = QLabel("⚪ 未连接")
        self.mic_status = QLabel("⚪ 未连接")
        self.rec_status_label = QLabel("⚪ 空闲")
        
        layout.addWidget(QLabel("播放器:"), 0, 0)
        layout.addWidget(self.player_status, 0, 1)
        layout.addWidget(QLabel("LED/马达:"), 1, 0)
        layout.addWidget(self.led_motor_status, 1, 1)
        layout.addWidget(QLabel("麦克风:"), 2, 0)
        layout.addWidget(self.mic_status, 2, 1)
        layout.addWidget(QLabel("录制状态:"), 3, 0)
        layout.addWidget(self.rec_status_label, 3, 1)
        
        return group

    def _create_playback_group(self):
        group = QGroupBox("播放控制")
        layout = QGridLayout(group)
        
        play_btn = QPushButton("▶️ 开始播放")
        play_btn.clicked.connect(lambda: self.send_command("P1"))
        
        pause_btn = QPushButton("⏸️ 暂停播放")
        pause_btn.clicked.connect(lambda: self.send_command("P0"))
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(1, 30)
        self.volume_slider.setValue(20)
        
        self.volume_label = QLabel(f"音量: {self.volume_slider.value()}")
        self.volume_slider.valueChanged.connect(self.update_volume)
        
        self.freq_combo = QComboBox()
        self.freq_map = {
            "100 Hz": "F1", "200 Hz": "F2", "300 Hz": "F3", 
            "500 Hz": "F4", "750 Hz": "F5", "1000 Hz": "F6", 
            "1500 Hz": "F7", "2000 Hz": "F8", "2500 Hz": "F9", 
            "3000 Hz": "F10", "4000 Hz": "F11", "5000 Hz": "F12"
        }
        self.freq_combo.addItems(self.freq_map.keys())
        # 默认设置为 750 Hz
        self.freq_combo.setCurrentText("750 Hz")
        self.freq_combo.currentTextChanged.connect(self.send_frequency)
        
        layout.addWidget(play_btn, 0, 0)
        layout.addWidget(pause_btn, 0, 1)
        layout.addWidget(self.volume_label, 1, 0)
        layout.addWidget(self.volume_slider, 1, 1)
        layout.addWidget(QLabel("频率:"), 2, 0)
        layout.addWidget(self.freq_combo, 2, 1)
        
        self.controls_to_disable.extend([play_btn, pause_btn, self.volume_slider, self.freq_combo])
        return group

    def _create_audio_capture_group(self):
        group = QGroupBox("音频采集与录制")
        layout = QGridLayout(group)
        
        self.start_capture_btn = QPushButton("🎤 开始采集")
        self.start_capture_btn.clicked.connect(self.start_audio_capture)
        
        self.stop_capture_btn = QPushButton("⏹️ 停止采集")
        self.stop_capture_btn.clicked.connect(self.stop_audio_capture)
        self.stop_capture_btn.setEnabled(False)
        
        self.record_btn = QPushButton("⏺️ 开始录制")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setDecimals(2)
        self.duration_spinbox.setSingleStep(0.10)
        self.duration_spinbox.setRange(1.00, 3600.00)
        self.duration_spinbox.setValue(10.00)
        self.duration_spinbox.setSuffix(" s")
        
        layout.addWidget(self.start_capture_btn, 0, 0)
        layout.addWidget(self.stop_capture_btn, 0, 1)
        layout.addWidget(QLabel("录制时长:"), 1, 0)
        layout.addWidget(self.duration_spinbox, 1, 1)
        layout.addWidget(self.record_btn, 2, 0, 1, 2)
        
        self.controls_to_disable.extend([
            self.start_capture_btn, 
            self.stop_capture_btn, 
            self.duration_spinbox, 
            self.record_btn
        ])
        return group

    def _create_motor_group(self):
        """马达控制组 - 使用数字输入框，精确到2位小数"""
        group = QGroupBox("马达控制")
        layout = QGridLayout(group)
        
        # 速度输入框
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(-10.00, 10.00)  # 速度范围：-10.00到10.00圈/秒
        self.speed_input.setSingleStep(0.01)      # 步长0.01
        self.speed_input.setDecimals(2)          # 小数点后2位
        self.speed_input.setValue(0.00)          # 初始值0.00
        self.speed_input.setSuffix(" r/s")       # 单位：圈/秒
        
        # 确认按钮
        self.speed_confirm_btn = QPushButton("设置速度")
        self.speed_confirm_btn.clicked.connect(self.send_speed_command)
        
        layout.addWidget(QLabel("速度:"), 0, 0)
        layout.addWidget(self.speed_input, 0, 1)
        layout.addWidget(self.speed_confirm_btn, 1, 0, 1, 2)
        
        self.controls_to_disable.extend([self.speed_input, self.speed_confirm_btn])
        return group

    
        
    def refresh_ports(self):
        self.port_input.clear()
        self.port_input.addItems([p.device for p in serial.tools.list_ports.comports()])
    
    def toggle_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.disconnect_serial()
        else:
            self.connect_serial()
    
    def connect_serial(self):
        port = self.port_input.currentText()
        if not port:
            QMessageBox.critical(self, "错误", "未选择串口。")
            return
        
        try:
            baud = int(self.baud_input.text())
            self.serial_port = serial.Serial(port, baud, timeout=1)
            self.connect_btn.setText("断开连接")
            self.port_input.setEnabled(False)
            self.baud_input.setEnabled(False)
            
            self.serial_worker = SerialWorker(self.serial_port)
            self.serial_worker.text_received.connect(self.handle_status_message)
            self.serial_worker.binary_data_received.connect(self.handle_audio_data)
            self.serial_worker.recording_finished.connect(self.handle_recording_finished)
            self.serial_worker.start()
            
            self.toggle_all_controls(True)

            # 确保连接后设置默认频率（当前选项）
            try:
                self.send_frequency(self.freq_combo.currentText())
            except Exception:
                pass
        except serial.SerialException as e:
            QMessageBox.critical(self, "连接错误", f"无法打开串口 {port}.\n{e}")
        except ValueError:
            QMessageBox.critical(self, "错误", "波特率必须为数字。")
    
    def disconnect_serial(self):
        self.stop_audio_capture()
        
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.serial_port = None
        self.serial_worker = None
        
        self.connect_btn.setText("打开连接")
        self.port_input.setEnabled(True)
        self.baud_input.setEnabled(True)
        
        self.player_status.setText("⚪ 未连接")
        self.led_motor_status.setText("⚪ 未连接")
        self.mic_status.setText("⚪ 未连接")
        self.rec_status_label.setText("⚪ 空闲")
        
        self.toggle_all_controls(False)
    
    def send_command(self, cmd):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write((cmd + '\r\n').encode('ascii'))
            except serial.SerialException:
                self.disconnect_serial()
    
    def send_speed_command(self):
        """发送速度命令到单片机，精确到2位小数"""
        speed = self.speed_input.value()
        # 格式化为2位小数，保留符号
        speed_str = f"{speed:.2f}"
        
        # 发送命令
        self.send_command(f"S{speed_str}")
    
    def handle_status_message(self, data: str):
        if data == "M1":
            self.player_status.setText("🟢 已连接")
        elif data == "M0":
            self.player_status.setText("🔴 未连接")
        elif data == "L1":
            self.led_motor_status.setText("🟢 已连接")
        elif data == "L0":
            self.led_motor_status.setText("🔴 未连接")
        elif data == "C1":
            self.mic_status.setText("🟢 已连接")
        elif data == "C0":
            self.mic_status.setText("🔴 未连接")
    
    def handle_audio_data(self, data: list):
        self.waveform_plot.add_data(data)
    
    def handle_recording_finished(self, recorded_data: list):
        # 先更新UI显示录音结束，再进行后续处理
        self.is_recording = False
        self.update_recording_ui()
        try:
            self.rec_status_label.setText("录音结束")
        except Exception:
            pass
        try:
            print("录音结束，开始处理...")
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        if not recorded_data:
            QMessageBox.warning(self, "录制警告", "录制数据为空。")
            return
        
        base_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(base_dir, timestamp)
        os.makedirs(folder, exist_ok=True)
        
        filename = os.path.join(folder, "audio.wav")
        csv_path = os.path.splitext(filename)[0] + ".csv"
        
        try:
            # Save WAV
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack(f'<{len(recorded_data)}h', *recorded_data))

            

            # Save plots（会在 data_plots.png 上标注识别到的周期）
            save_data_and_fft_plots(recorded_data, 16000, folder)

            # Save per-cycle CSVs (each cycle as-is, no resampling)
            csv_dir = os.path.join(folder, 'csv')
            cycles_saved = save_cycle_csvs(recorded_data, 16000, csv_dir)

            QMessageBox.information(
                self,
                "保存成功",
                f"文件已保存到文件夹:\n{folder}\n\n"
                f"- 音频: {os.path.basename(filename)}\n"
                f"- 数据: {os.path.basename(csv_path)}\n"
                f"- 图表: data_plots.png, fft_plot.png, spectrogram.png"
            )
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存录音/CSV失败:\n{e}")
    
    def update_volume(self, value):
        self.volume_label.setText(f"音量: {value}")
        self.send_command(f"V{value}")
    
    def send_frequency(self, text):
        command = self.freq_map.get(text)
        if command:
            self.send_command(command)
    
    
    
    def start_audio_capture(self):
        self.send_command("E1")
        
        if self.serial_worker:
            self.serial_worker.set_streaming_mode(True)
        
        self.waveform_plot.start_updates()
        self.toggle_capture_controls(is_capturing=True)
    
    def stop_audio_capture(self):
        if self.is_recording:
            if self.serial_worker:
                self.serial_worker.stop_recording(emit_signal=True)
        
        self.send_command("E0")
        
        if self.serial_worker:
            self.serial_worker.set_streaming_mode(False)
        
        self.waveform_plot.stop_updates()
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.reset_input_buffer()
        
        self.toggle_capture_controls(is_capturing=False)
    
    def toggle_recording(self):
        if not self.serial_worker:
            return
        
        if not self.is_recording:
            self.serial_worker.start_recording(self.duration_spinbox.value(), 16000)
            self.is_recording = True
        else:
            self.serial_worker.stop_recording(emit_signal=True)
            self.is_recording = False
        
        self.update_recording_ui()
    
    def update_recording_ui(self):
        if self.is_recording:
            self.record_btn.setText("⏹️ 停止录制")
            self.rec_status_label.setText("录制中...")
            self.duration_spinbox.setEnabled(False)
        else:
            self.record_btn.setText("⏺️ 开始录制")
            self.rec_status_label.setText("⚪ 空闲")
            self.duration_spinbox.setEnabled(True)
    
    def toggle_all_controls(self, enabled):
        for widget in self.controls_to_disable:
            widget.setEnabled(enabled)
        
        if not enabled:
            self.stop_capture_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
    
    def toggle_capture_controls(self, is_capturing):
        self.start_capture_btn.setEnabled(not is_capturing)
        self.stop_capture_btn.setEnabled(is_capturing)
        self.record_btn.setEnabled(is_capturing)
        
        if not is_capturing:
            self.rec_status_label.setText("⚪ 空闲")
    
    def closeEvent(self, event):
        self.disconnect_serial()
        event.accept()


# ==============================================================================
# 8. 主程序入口
# ==============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
