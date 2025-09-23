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
import librosa
from scipy.signal import find_peaks, hilbert, butter, filtfilt, spectrogram

# --- PyTorch Imports ---
import torch
import torch.nn as nn

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QLineEdit, QComboBox, QSlider,
    QMessageBox, QDialog, QDialogButtonBox, QColorDialog, QSpinBox,
    QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, pyqtSlot
from PyQt6.QtGui import QColor, QFont

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('QtAgg')
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. 配置参数
# ==============================================================================
BAUD_RATE = 921600
SAMPLING_RATE = 16000
CYCLE_DURATION_SECONDS = 4.0
BEEP_HIGH_PASS_FC = 4500
BEEP_THRESHOLD_RATIO = 0.3
BEEP_MIN_INTERVAL_SAMPLES = int(SAMPLING_RATE * (CYCLE_DURATION_SECONDS * 0.8))
BEEP_DETECTION_WINDOW_SECONDS = 5
BEEP_DETECTION_WINDOW_SAMPLES = int(BEEP_DETECTION_WINDOW_SECONDS * SAMPLING_RATE)
BANDPASS_LOWCUT = 1000
BANDPASS_HIGHCUT = 8000
BANDPASS_ORDER = 4
IMG_HEIGHT = 224
IMG_WIDTH = 287
EXPECTED_SAMPLES = SAMPLING_RATE * int(CYCLE_DURATION_SECONDS)
SAMPLES_TO_PAD = (IMG_HEIGHT * IMG_WIDTH) - EXPECTED_SAMPLES
CNN_MODEL_PATH = 'best_audio_cnn_model.pth'
LIVE_AUDIO_BUFFER_DURATION_SECONDS = 15
LIVE_AUDIO_BUFFER_MAXLEN = int(LIVE_AUDIO_BUFFER_DURATION_SECONDS * SAMPLING_RATE)


# ==============================================================================
# 2. CNN 模型定义
# ==============================================================================
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 35, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


# ==============================================================================
# 3. 预处理工具函数
# ==============================================================================
def highpass_filter(audio_np, sr, cutoff=BEEP_HIGH_PASS_FC, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    if normal_cutoff <= 0 or normal_cutoff >= 1.0:
        return audio_np
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, audio_np)

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

def detect_beeps_in_buffer(audio_chunk_np, sr):
    if len(audio_chunk_np) == 0:
        return np.array([])
    filtered_for_beep = highpass_filter(audio_chunk_np, sr)
    analytic_signal = hilbert(filtered_for_beep)
    envelope = np.abs(analytic_signal)
    smoothed_envelope = lowpass_filter(envelope, sr)
    if len(smoothed_envelope) == 0:
        return np.array([])
    threshold = BEEP_THRESHOLD_RATIO * np.max(smoothed_envelope)
    peaks, _ = find_peaks(smoothed_envelope, height=threshold*0.4, 
                         distance=BEEP_MIN_INTERVAL_SAMPLES, prominence=threshold*0.3)
    return peaks

def preprocess_slice_for_cnn(audio_slice_int16):
    try:
        audio = audio_slice_int16.astype(np.float32)
        audio_filtered = bandpass_filter(audio / 32768.0, SAMPLING_RATE)
        if len(audio_filtered) < EXPECTED_SAMPLES:
            audio_filtered = np.pad(audio_filtered, (0, EXPECTED_SAMPLES - len(audio_filtered)), 'constant')
        elif len(audio_filtered) > EXPECTED_SAMPLES:
            audio_filtered = audio_filtered[:EXPECTED_SAMPLES]
        audio_padded = np.pad(audio_filtered, (0, SAMPLES_TO_PAD), 'constant')
        matrix = audio_padded.reshape(IMG_HEIGHT, IMG_WIDTH)
        min_val, max_val = np.min(matrix), np.max(matrix)
        if (max_val - min_val) > 1e-6:
            matrix = (matrix - min_val) / (max_val - min_val)
        return torch.FloatTensor(matrix).unsqueeze(0)
    except Exception as e:
        print(f"Error in preprocess_slice_for_cnn: {e}")
        return None


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
    classification_result = pyqtSignal(str)


    def __init__(self, serial_instance, cnn_model, device):
        super().__init__()
        self.serial = serial_instance
        self.cnn_model = cnn_model
        self.device = device
        self._is_running = True
        self._is_streaming_mode = False
        self._is_recording = False
        self._recorded_samples = []
        self._samples_to_record = 0
        self.live_audio_buffer = collections.deque(maxlen=LIVE_AUDIO_BUFFER_MAXLEN)
        self.last_beep_global_sample_index = None
        self.global_sample_counter = 0

    def run(self):
        packet_accumulator = bytearray()
        last_processing_time = time.time()
        PROCESSING_INTERVAL = 0.5  # 每隔0.5秒进行一次分析和预测尝试

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
                                self.live_audio_buffer.extend(samples)
                                self.global_sample_counter += len(samples)
                                # 录制逻辑：在采集中并且开启录制时，将样本写入缓冲并在达到目标长度后自动结束
                                if self._is_recording:
                                    remaining_needed = self._samples_to_record - len(self._recorded_samples)
                                    if remaining_needed > 0:
                                        take_count = min(remaining_needed, len(samples))
                                        self._recorded_samples.extend(samples[:take_count])
                                    if len(self._recorded_samples) >= self._samples_to_record:
                                        # 达到目标长度，自动停止并发出数据
                                        self.stop_recording(emit_signal=True)

                    current_time = time.time()
                    if current_time - last_processing_time > PROCESSING_INTERVAL:
                        self._run_classification_cycle()
                        last_processing_time = current_time
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

    def _run_classification_cycle(self):
        if len(self.live_audio_buffer) < BEEP_DETECTION_WINDOW_SAMPLES or self.cnn_model is None:
            return
        
        current_samples = np.array(list(self.live_audio_buffer)[-BEEP_DETECTION_WINDOW_SAMPLES:]).astype(np.float32) / 32768.0
        peak_indices = detect_beeps_in_buffer(current_samples, SAMPLING_RATE)
        
        if len(peak_indices) > 0:
            window_start_idx = self.global_sample_counter - len(current_samples)
            for peak_idx in peak_indices:
                new_beep_idx = window_start_idx + peak_idx
                if (self.last_beep_global_sample_index is not None and 
                    (new_beep_idx - self.last_beep_global_sample_index) < BEEP_MIN_INTERVAL_SAMPLES):
                    continue
                
                if self.last_beep_global_sample_index is not None:
                    start_g_idx, end_g_idx = self.last_beep_global_sample_index, new_beep_idx
                    duration_s = (end_g_idx - start_g_idx) / SAMPLING_RATE
                    
                    if abs(duration_s - CYCLE_DURATION_SECONDS) < 0.5:
                        buffer_start_g_idx = self.global_sample_counter - len(self.live_audio_buffer)
                        start_in_buffer = int(start_g_idx - buffer_start_g_idx)
                        end_in_buffer = int(end_g_idx - buffer_start_g_idx)
                        
                        if 0 <= start_in_buffer < end_in_buffer <= len(self.live_audio_buffer):
                            audio_slice = np.array(list(self.live_audio_buffer)[start_in_buffer:end_in_buffer], dtype=np.int16)
                            tensor_input = preprocess_slice_for_cnn(audio_slice)
                            
                            if tensor_input is not None:
                                tensor_input = tensor_input.unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    outputs = self.cnn_model(tensor_input)
                                    probabilities = torch.softmax(outputs, dim=1)[0]
                                prob_0 = probabilities[0].item()
                                prob_1 = probabilities[1].item()
                                label_name = "0" if prob_0 > prob_1 else "1"
                                result_text = f"识别结果: 数字 {label_name}\n\nP(0)={prob_0:.2f}, P(1)={prob_1:.2f}"
                                self.classification_result.emit(result_text)
                
                self.last_beep_global_sample_index = new_beep_idx

    def stop(self):
        self._is_running = False
    
    def set_streaming_mode(self, enabled: bool):
        self._is_streaming_mode = enabled
        if enabled:
            self.live_audio_buffer.clear()
            self.last_beep_global_sample_index = None
            self.global_sample_counter = 0
    
    def start_recording(self, duration_seconds: int, sample_rate: int):
        if not self._is_streaming_mode:
            return
        self._samples_to_record = duration_seconds * sample_rate
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
class LedGridWindow(QDialog):
    led_command_generated = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置LED颜色")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("LED单独设置功能待集成..."))


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
        self.led_grid_window = LedGridWindow(self)
        self.led_grid_window.led_command_generated.connect(self.send_command)
        
        # 加载CNN模型
        self.cnn_model = None
        self.device = None
        self.load_model()
        
        # 初始化UI
        self.initUI()
        self.toggle_all_controls(False)

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        try:
            self.cnn_model = CNNClassifier().to(self.device)
            self.cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=self.device))
            self.cnn_model.eval()
            print("CNN模型加载成功。")
        except Exception as e:
            print(f"加载CNN模型失败: {e}")
            QMessageBox.critical(self, "模型加载失败", f"无法加载模型 '{CNN_MODEL_PATH}'.\n{e}")

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
        grid_controls.addWidget(self._create_led_group(), 1, 1)
        
        controls_layout.addLayout(top_layout)
        controls_layout.addLayout(grid_controls)
        controls_layout.addStretch()
        
        # 波形图和分类结果显示
        self.waveform_plot = WaveformWidget()
        self.classification_group = self._create_classification_group()
        
        display_layout.addWidget(self.waveform_plot, 3)
        display_layout.addWidget(self.classification_group, 2)

    def _create_classification_group(self):
        """创建带有透明度效果的数字分类显示区域"""
        group = QGroupBox("实时数字分类")
        layout = QGridLayout(group)
        
        # 数字0的显示
        self.digit0_container = QWidget()
        vbox0 = QVBoxLayout(self.digit0_container)
        vbox0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.digit0_label = QLabel("0")
        self.digit0_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.digit0_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        self.digit0_label.setStyleSheet("color: rgba(0, 0, 255, 0.3);")  # 初始半透明蓝色
        vbox0.addWidget(self.digit0_label)
        
        self.prob0_label = QLabel("P(0): 0.00")
        self.prob0_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prob0_label.setStyleSheet("color: blue; font-weight: bold;")
        vbox0.addWidget(self.prob0_label)
        
        # 数字1的显示
        self.digit1_container = QWidget()
        vbox1 = QVBoxLayout(self.digit1_container)
        vbox1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.digit1_label = QLabel("1")
        self.digit1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.digit1_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        self.digit1_label.setStyleSheet("color: rgba(255, 0, 0, 0.3);")  # 初始半透明红色
        vbox1.addWidget(self.digit1_label)
        
        self.prob1_label = QLabel("P(1): 0.00")
        self.prob1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prob1_label.setStyleSheet("color: red; font-weight: bold;")
        vbox1.addWidget(self.prob1_label)
        
        # 最终结果标签
        self.final_result_label = QLabel("等待识别...")
        self.final_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.final_result_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        
        # 添加到布局
        layout.addWidget(self.digit0_container, 0, 0)
        layout.addWidget(self.digit1_container, 0, 1)
        layout.addWidget(self.final_result_label, 1, 0, 1, 2)
        
        return group

    @pyqtSlot(str)
    def update_classification_display(self, result_text):
        """更新分类结果显示，使用透明度表示概率"""
        try:
            parts = result_text.split("\n\n")
            if len(parts) >= 2:
                prob_parts = parts[1].split(", ")
                p0 = float(prob_parts[0].split("=")[1])
                p1 = float(prob_parts[1].split("=")[1])
                
                # 更新数字透明度
                self.digit0_label.setStyleSheet(f"color: rgba(0, 0, 255, {p0});")
                self.digit1_label.setStyleSheet(f"color: rgba(255, 0, 0, {p1});")
                
                # 更新概率标签
                self.prob0_label.setText(f"P(0): {p0:.2f}")
                self.prob1_label.setText(f"P(1): {p1:.2f}")
                
                # 更新最终结果
                if p0 > p1:
                    self.final_result_label.setText(f"识别结果: 数字 0 (置信度: {p0 * 100:.1f}%)")
                    self.final_result_label.setStyleSheet("color: blue;")
                else:
                    self.final_result_label.setText(f"识别结果: 数字 1 (置信度: {p1 * 100:.1f}%)")
                    self.final_result_label.setStyleSheet("color: red;")
        except Exception as e:
            print(f"更新分类显示时出错: {e}")
            self.final_result_label.setText("识别结果解析错误")

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
        
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(1, 3600)
        self.duration_spinbox.setValue(10)
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

    def _create_led_group(self):
        group = QGroupBox("LED 灯光控制")
        layout = QGridLayout(group)
        
        unified_color_btn = QPushButton("统一颜色")
        unified_color_btn.clicked.connect(self.set_unified_color)
        
        individual_led_btn = QPushButton("分别设置")
        individual_led_btn.clicked.connect(self.led_grid_window.exec)
        
        layout.addWidget(unified_color_btn, 0, 0)
        layout.addWidget(individual_led_btn, 0, 1)
        
        self.controls_to_disable.extend([unified_color_btn, individual_led_btn])
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
            
            self.serial_worker = SerialWorker(self.serial_port, self.cnn_model, self.device)
            self.serial_worker.text_received.connect(self.handle_status_message)
            self.serial_worker.binary_data_received.connect(self.handle_audio_data)
            self.serial_worker.recording_finished.connect(self.handle_recording_finished)
            self.serial_worker.classification_result.connect(self.update_classification_display)
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
        
        self.final_result_label.setText("等待识别...")
        self.digit0_label.setStyleSheet("color: rgba(0, 0, 255, 0.3);")
        self.digit1_label.setStyleSheet("color: rgba(255, 0, 0, 0.3);")
        self.prob0_label.setText("P(0): 0.00")
        self.prob1_label.setText("P(1): 0.00")
        
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
        self.is_recording = False
        self.update_recording_ui()
        
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

            # Save CSV (amplitude & freq vs time)
            export_amp_freq_csv(recorded_data, 16000, csv_path)

            # Save plots
            save_data_and_fft_plots(recorded_data, 16000, folder)

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
    
    def set_unified_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.send_command(f"A{color.green():02x}{color.red():02x}{color.blue():02x}".upper())
    
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
        self.final_result_label.setText("采集已停止")
    
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
