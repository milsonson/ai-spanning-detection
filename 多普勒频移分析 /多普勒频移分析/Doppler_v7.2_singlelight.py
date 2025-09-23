"""
多普勒效应音频分析工具

本程序是一个集实时数据采集、信号处理、物理模型拟合与可视化于一体的多普勒效应分析工具。
主要功能包括：
- 通过串口或WAV文件加载音频信号。
- 实时进行信号处理，包括带通滤波、包络校正和匹配追踪频率提取。
- 在启动采集的最初2秒内，自动进行声强分析以估计初始相位，并对周期、声速等物理参数进行拟合。利用前两秒的频率数据，生成48Hz的理论频率指示器。
- 实时可视化：
  - 声音波形图。
  - 根据用户设定周期折叠的频率散点图。
  - 基于拟合参数的理论频率指示器，通过颜色变化直观展示当前理论频率位置。
- 停止采集时，可自动保存最近60秒的音频数据并进行一次完整的离线分析，显示拟合结果。
- 支持GPU加速（CUDA / Metal）以提高实时处理性能。

作者: CHEN Jingxu
日期: 2025-07-05
"""

import os
import logging
import platform
import numpy as np
import torch.backends.mps
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt, hilbert, stft
from scipy.optimize import minimize
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.ttk as ttk
import torch
from PIL import Image, ImageTk
from concurrent.futures import ProcessPoolExecutor
import struct
import threading
import collections
import matplotlib.animation as animation
import wave
import time
import queue
import torch.fft
import numpy as np


# ========== 全局配置与日志 ==========
def setup_matplotlib_fonts():
    """配置matplotlib字体以支持中文显示。"""
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置更大的默认字体大小
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 14
    })
setup_matplotlib_fonts()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 多进程辅助函数 ==========
def _process_single_frame_mp(args):
    """匹配追踪的单帧处理函数 (用于并行计算)。"""
    frame_signal, delta_freq_grid, f0_base, t_within_frame = args
    best_correlation = 0
    best_delta_freq = 0
    for delta_f in delta_freq_grid:
        atom = np.exp(1j * 2 * np.pi * (f0_base + delta_f) * t_within_frame)
        correlation = np.abs(np.vdot(frame_signal, atom))
        if correlation > best_correlation:
            best_correlation = correlation
            best_delta_freq = delta_f
    return best_delta_freq

class Config:
    """集中管理所有配置参数"""
    # ========== 串口/音频输入设置 ==========
    BAUD_RATE = 921600  # 串口波特率
    BYTES_PER_READ = 128  # 每次从串口读取的字节数
    SAMPLES_PER_READ = BYTES_PER_READ // 2  # 每次读取的样本数 (int16)
    SAMPLING_RATE = 16000  # 音频采样率 (Hz)
    INT16_TO_FLOAT_SCALE = 32768.0  # int16到float的转换系数

    # ========== 缓冲区和队列设置 ==========
    PLOT_WINDOW_SAMPLES = 4096  # 实时波形图显示的样本数
    REALTIME_FREQ_POINTS_MAXLEN = 20000  # 实时频率图显示的最大点数
    FRAME_QUEUE_MAXSIZE = 10  # STFT处理帧队列的最大容量
    LONG_TERM_BUFFER_SECONDS = 60  # 停止采集时用于离线分析的音频缓冲区时长 (秒)
    INITIAL_ANALYSIS_SECONDS = 2  # 开始采集时用于自动参数拟合的初始音频时长 (秒)

    # ========== 分析参数 ==========
    MATCHING_PURSUIT_GRID_POINTS = 4001  # 匹配追踪算法中频率偏移的搜索点数
    PHASE_BIN_SIZE_DEGREES = 7.5  # 相位分箱的大小 (度)
    INTENSITY_FILTER_LOW_HZ = 2000  # 声强分析的带通滤波器下限 (Hz)
    INTENSITY_FILTER_HIGH_HZ = 6000  # 声强分析的带通滤波器上限 (Hz)
    INTENSITY_RMS_WINDOW_S = 0.01  # 计算RMS声强的窗口大小 (秒)

    # ========== GUI和绘图设置 ==========
    INDICATOR_UPDATE_HZ = 48  # 理论频率指示器的更新频率 (Hz)
    INDICATOR_UPDATE_MS = 21  # 理论频率指示器的更新间隔 (毫秒), 1000ms / 48Hz ~= 20.83ms
    DEFAULT_IMAGE_PATH = r"软件示意图.png"  # 默认显示的示意图路径

    # ========== 默认设置 ==========
    DEFAULT_SIGNAL_MODE = "doppler"  # 默认信号模式
    # 多普勒效应仿真参数
    DOPPLER_SIM_PARAMS = {'vs': 347.1, 'R': 0.1, 'l': 0.4, 'T': 1.0, 'f0': 500.0, 'amp': 5000, 'noise_std_ratio': 0.1}
    # 实时STFT分析参数
    REALTIME_STFT_PARAMS = {'tint': 0.019, 'overlap_percent': 0.125, 'fs': SAMPLING_RATE, 'f0': 500.0, 'max_offset': 30.0, 'window_length': 16000}
    ENABLE_60S_AUDIO_FEATURE = True  # 是否启用停止采集时自动分析60秒音频的功能
    # GUI默认参数
    DEFAULT_GUI_PARAMS = {
        'audio_file': '测试用音频/江湾60s.wav', 'f0': '500.0',
        'cut_percent': '0.05', 'tint': '0.019', 'window_length': '16000', 'period': '1', 'max_offset': '20.0',
        'overlap_percent': '0.125', 'margin': '0.0005', 'n_points': '21', 'l': '0.4',
        'realtime_freq_ymin': '499', 'realtime_freq_ymax': '505',
        'serial_port': 'COM5' # 将串口号添加到GUI参数中
    }

class RealtimeDataManager:
    """负责所有实时数据的采集、缓冲、同步和管理"""
    def __init__(self, config: Config):
        self.config = config
        # ========== 数据缓冲区和队列 ==========
        self.waveform_data_buffer = collections.deque(maxlen=self.config.PLOT_WINDOW_SAMPLES)  # 原始波形数据，用于STFT处理
        self.display_waveform_buffer = collections.deque(maxlen=self.config.PLOT_WINDOW_SAMPLES)  # 用于GUI波形图显示的数据
        self.long_term_audio_buffer = collections.deque(maxlen=self.config.SAMPLING_RATE * self.config.LONG_TERM_BUFFER_SECONDS)  # 60秒长期音频数据缓冲区
        self.frame_queue = queue.Queue(maxsize=self.config.FRAME_QUEUE_MAXSIZE)  # STFT帧队列，用于线程间通信
        self.realtime_freq_points = collections.deque(maxlen=self.config.REALTIME_FREQ_POINTS_MAXLEN)  # 实时计算出的频率点 (时间, 频率)

        # ========== 线程同步锁 ==========
        self.global_sample_counter_lock = threading.Lock()  # 全局样本计数器锁
        self.realtime_freq_lock = threading.Lock()  # 实时频率点数据锁
        self.dropped_samples_lock = threading.Lock()  # 丢弃样本计数器锁

        # ========== 状态标志和计数器 ==========
        self.is_sampling = False  # 是否正在进行实时采样
        self.is_60s_recording_active = False  # 是否正在录制60秒音频
        self.global_sample_counter = 0  # 全局样本计数器，用于计算时间戳
        self.dropped_samples_counter = 0  # 因队列满而丢弃的样本数

        # ========== 实时分析参数 ==========
        self.realtime_period = 1.0  # 实时折叠图的周期
        self.realtime_stft_params = self.config.REALTIME_STFT_PARAMS.copy()  # 实时STFT参数

        # ========== 音频文件模拟源相关 ==========
        self.audio_file_path = None  # 当前加载的音频文件路径
        self.audio_samples = None  # 加载的音频文件样本数据
        self.audio_sample_index = 0  # 当前音频文件播放位置
        self.audio_sample_rate = None  # 加载的音频文件采样率
        self.audio_file_loaded = False  # 标记音频文件是否已加载

        # ========== 2秒初始分析相关 ==========
        self.two_second_analysis_triggered = False  # 是否已触发2秒初始分析
        self.two_second_buffer = collections.deque(maxlen=self.config.SAMPLING_RATE * self.config.INITIAL_ANALYSIS_SECONDS)  # 2秒初始分析数据缓冲区

        # ========== 理论频率模型参数 (由2秒分析拟合得到) ==========
        self.fitted_T = None  # 拟合的周期 T
        self.fitted_vs = None  # 拟合的声速 vs
        self.fitted_phi = None  # 拟合的相位 phi
        self.fitted_f0 = None  # 拟合的中心频率 f0
        self.fitted_l_dist = None  # 拟合的距离 l
        self.last_theoretical_freq_log_time = 0.0  # 上次记录理论频率的时间点
        self.min_theoretical_freq = None  # 2秒分析得出的理论最小频率
        self.max_theoretical_freq = None  # 2秒分析得出的理论最大频率
        self.current_theoretical_freq = None  # 当前计算的理论频率，用于指示器
        self.precalculated_theoretical_freqs = None  # 为指示器预计算的理论频率序列
        self.precalculated_freq_index = 0  # 预计算频率序列的当前索引

    def start_threads(self):
        """启动所有后台数据处理线程。"""
        # 线程1: 负责从串口或模拟源读取数据
        threading.Thread(target=self._communication_loop, name="Thread-WaveformComm", daemon=True).start()
        # 线程2: 负责将原始数据分帧并放入队列
        threading.Thread(target=self._stft_data_collection, name="Thread-DataCollection", daemon=True).start()
        # 线程3: 负责从队列中取出帧进行STFT和频率追踪
        threading.Thread(target=self._stft_data_processing, name="Thread-DataProcessing", daemon=True).start()
        logger.info("所有实时数据处理线程已启动。")

    def _communication_loop(self):
        """
        数据通信循环，尝试连接串口并读取数据。
        如果失败，则自动切换到使用音频文件模拟数据流。
        """
        try:
            import serial
            # 从配置中获取串口号，该配置可由GUI更新
            ser = serial.Serial(port=self.config.SERIAL_PORT, baudrate=self.config.BAUD_RATE, timeout=0.1)
            logger.info(f"串口 {self.config.SERIAL_PORT} 连接成功。")
            packet_accumulator = bytearray()  # 用于累积不完整的字节包
            while True:
                if self.is_sampling:
                    chunk = ser.read(1024)  # 尝试读取数据
                    if chunk:
                        packet_accumulator.extend(chunk)
                    # 当累积的数据足够一个数据包时，进行处理
                    while len(packet_accumulator) >= self.config.BYTES_PER_READ:
                        raw_data_packet = packet_accumulator[:self.config.BYTES_PER_READ]
                        del packet_accumulator[:self.config.BYTES_PER_READ]
                        # 将字节解包为int16样本
                        raw_samples = struct.unpack(f'<{self.config.SAMPLES_PER_READ}h', raw_data_packet)
                        self._add_samples_to_buffers(raw_samples)
                else:
                    time.sleep(0.01)  # 非采样状态下短暂休眠，避免CPU空转
        except Exception as e:
            logger.warning(f"无法连接到串口: {e}。将启动模拟数据源。")
            self._generate_simulated_data()

    def _load_audio_file(self, file_path: str) -> bool:
        """
        加载指定的WAV音频文件。

        Args:
            file_path (str): 音频文件路径。

        Returns:
            bool: 如果加载成功返回True，否则返回False。
        """
        try:
            rate, data = wav.read(file_path)
            self.audio_samples = data
            self.audio_sample_rate = rate
            self.audio_sample_index = 0
            self.audio_file_path = file_path  # 保存当前加载的文件路径
            logger.info(f"已加载音频文件: {file_path}, 采样率: {rate}Hz, 样本数: {len(data)}")
            return True
        except Exception as e:
            logger.error(f"加载音频文件失败: {e}")
            return False

    def _generate_simulated_data(self):
        """
        当串口不可用时，从此方法生成模拟数据。
        它会循环播放一个WAV文件，模拟实时数据流。
        """
        logger.info("启动音频文件模拟数据源")
        fs = self.config.SAMPLING_RATE
        sample_interval = self.config.SAMPLES_PER_READ / fs  # 每个数据包的理论时间间隔
        next_sample_time = time.perf_counter()
        total_samples = 0

        while True:
            if self.is_sampling:
                # 仅在开始采样时或文件未加载时加载音频
                if not self.audio_file_loaded:
                    # 优先使用GUI指定的路径，否则使用默认路径
                    file_path = self.audio_file_path if self.audio_file_path else self.config.DEFAULT_GUI_PARAMS['audio_file']
                    if not self._load_audio_file(file_path):
                        logger.error(f"无法加载用于模拟的音频文件: {file_path}")
                        time.sleep(1)  # 等待后重试
                        continue
                    self.audio_file_loaded = True
                    logger.info(f"成功加载模拟音频文件: {file_path}")

                # 通过休眠来模拟精确的采样率
                current_time = time.perf_counter()
                if current_time < next_sample_time:
                    time.sleep(max(0, (next_sample_time - current_time) * 0.95))
                    continue

                # 从已加载的音频文件中读取一块数据
                remaining_samples = len(self.audio_samples) - self.audio_sample_index
                if remaining_samples <= 0:
                    logger.info("音频文件播放完毕，将从头开始循环播放。")
                    self.audio_sample_index = 0
                    continue

                read_count = min(self.config.SAMPLES_PER_READ, remaining_samples)
                raw_samples = self.audio_samples[self.audio_sample_index:self.audio_sample_index + read_count].tolist()
                self.audio_sample_index += read_count
                total_samples += read_count

                # 如果文件末尾的样本不足一个数据包，则用零填充
                if len(raw_samples) < self.config.SAMPLES_PER_READ:
                    raw_samples.extend([0] * (self.config.SAMPLES_PER_READ - len(raw_samples)))

                self._add_samples_to_buffers(raw_samples)

                # 更新下一次采样的时间点
                next_sample_time += sample_interval

            else:
                next_sample_time = time.perf_counter()  # 重置计时器
                time.sleep(0.01)

    def _add_samples_to_buffers(self, raw_samples: list):
        """
        将新获取的原始样本添加到各个缓冲区中。

        Args:
            raw_samples (list): 新的int16样本列表。
        """
        with self.global_sample_counter_lock:
            start_index = self.global_sample_counter
            # 为每个样本附加全局索引和计算出的时间戳
            samples_with_ts = [(start_index + i, val, (start_index + i) / self.config.SAMPLING_RATE)
                             for i, val in enumerate(raw_samples)]
            self.global_sample_counter += len(raw_samples)

        # 添加到不同的缓冲区
        self.waveform_data_buffer.extend(samples_with_ts)  # 用于STFT处理
        self.display_waveform_buffer.extend(samples_with_ts)  # 用于GUI显示
        if self.config.ENABLE_60S_AUDIO_FEATURE and self.is_60s_recording_active:
            self.long_term_audio_buffer.extend(raw_samples)

        # 触发2秒初始分析
        if not self.two_second_analysis_triggered:
            self.two_second_buffer.extend(raw_samples)
            if len(self.two_second_buffer) >= self.config.SAMPLING_RATE * self.config.INITIAL_ANALYSIS_SECONDS:
                self.two_second_analysis_triggered = True
                # 在新线程中执行分析，避免阻塞数据采集
                threading.Thread(target=self._perform_initial_analysis, daemon=True).start()

    def _stft_data_collection(self):
        """
        数据收集线程：从 `waveform_data_buffer` 中取出数据，
        将其分段成帧，并放入 `frame_queue` 供处理线程使用。
        """
        fs = self.realtime_stft_params['fs']
        nperseg = int(self.realtime_stft_params.get('window_length', fs * self.realtime_stft_params['tint']))
        hop = int(nperseg * (1 - self.realtime_stft_params['overlap_percent']))
        temp_buffer = collections.deque(maxlen=nperseg)  # 临时缓冲区，用于构建一帧数据

        while True:
            if self.is_sampling and len(self.waveform_data_buffer) > 0:
                try:
                    # 从主缓冲区取一个样本放入临时帧缓冲区
                    temp_buffer.append(self.waveform_data_buffer.popleft())
                    if len(temp_buffer) >= nperseg:
                        # 当帧缓冲区满时，打包成一帧并放入队列
                        frame_start_index = temp_buffer[0][0]
                        frame_data = np.array([s[1] for s in temp_buffer])
                        try:
                            self.frame_queue.put((frame_start_index, frame_data), block=False)
                            # 移除hop长度的旧数据，为新数据腾出空间，实现重叠
                            for _ in range(hop): temp_buffer.popleft()
                        except queue.Full:
                            # 如果队列已满，记录丢弃的样本数
                            with self.dropped_samples_lock:
                                self.dropped_samples_counter += nperseg
                except IndexError:
                    # 在高并发下，popleft可能因缓冲区暂时为空而失败
                    time.sleep(0.001)
            else:
                time.sleep(0.01)

    def _stft_data_processing(self):
        """
        数据处理线程：从 `frame_queue` 中取出音频帧，执行带通滤波、
        匹配追踪算法来计算瞬时频率，并将结果存入 `realtime_freq_points`。
        同时，在初始分析完成后，会计算并更新理论频率。
        """
        params = self.realtime_stft_params
        fs, tint, f0, max_offset = params['fs'], params['tint'], params['f0'], params['max_offset']
        nperseg_mt = int(tint * fs)
        hop = int(nperseg_mt * (1 - params['overlap_percent']))
        delta_grid = np.linspace(-max_offset, max_offset, self.config.MATCHING_PURSUIT_GRID_POINTS)

        # 优先选择GPU加速 (MPS for Apple Silicon, CUDA for Nvidia)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warning("未检测到GPU加速支持，将使用CPU模式进行实时计算。")

        # 将常用数据预先加载到GPU
        delta_grid_torch = torch.from_numpy(delta_grid).float().to(device)
        t_frame_torch = torch.from_numpy(np.arange(nperseg_mt) / fs).float().to(device)
        window_torch = torch.from_numpy(np.hanning(nperseg_mt)).float().to(device)

        while True:
            try:
                frame_start_index, frame_data = self.frame_queue.get(timeout=1)
                # 应用带通滤波，减少噪声干扰
                filtered_frame = AudioProcessor.apply_bandpass_filter(frame_data, fs, f0 * 0.95, f0 * 1.05)

                # 在一帧内再次分段进行匹配追踪
                for i in range(0, len(filtered_frame) - nperseg_mt + 1, hop):
                    segment = filtered_frame[i:i + nperseg_mt]
                    t_frame_time = (frame_start_index + i) / fs

                    # --- 核心频率追踪逻辑 (GPU加速) ---
                    frame_torch = torch.from_numpy(segment.copy()).float().to(device)
                    frame_windowed = (frame_torch * window_torch).to(torch.complex64)
                    # 构建原子库并计算相关性
                    atoms = torch.exp(1j * 2 * torch.pi * (f0 + delta_grid_torch).unsqueeze(1) * t_frame_torch.unsqueeze(0))
                    corr = torch.abs(torch.matmul(atoms, frame_windowed))
                    # 找到相关性最大的原子对应的频率
                    best_freq = (f0 + delta_grid_torch[torch.argmax(corr)]).item()
                    with self.realtime_freq_lock:
                        # 将 (折叠时间, 频率) 存入结果队列
                        self.realtime_freq_points.append((np.mod(t_frame_time, self.realtime_period), best_freq))

                    # --- 理论频率计算与更新 ---
                    if self.two_second_analysis_triggered and self.fitted_T is not None:
                        # 按指示器更新频率的间隔来计算
                        if t_frame_time >= self.last_theoretical_freq_log_time + (1.0 / self.config.INDICATOR_UPDATE_HZ):
                            model_params = (self.fitted_vs, self.fitted_phi, self.fitted_f0)
                            consts = {'R': 0.1, 'l': self.fitted_l_dist, 'vs': self.fitted_vs}
                            # 将绝对时间转换为周期内的相对时间
                            relative_time_in_period = np.mod(t_frame_time, self.fitted_T)
                            theoretical_freq = AudioProcessor._frequency_model(
                                model_params, relative_time_in_period, self.fitted_T, consts, 'vs'
                            )
                            self.current_theoretical_freq = theoretical_freq  # 更新当前理论频率
                            self.last_theoretical_freq_log_time = t_frame_time  # 更新时间戳

                self.frame_queue.task_done()
            except queue.Empty:
                continue  # 队列为空是正常情况，继续等待
            except Exception as e:
                logger.error(f"处理数据时发生错误: {e}")

    def _perform_initial_analysis(self):
        """
        使用采集到的最初2秒数据执行一次完整的离线分析，
        以自动拟合理论频率模型的参数 (T, vs, phi, f0, l)。
        这些参数随后用于实时计算理论频率并更新指示器。
        """
        logger.info(f"开始{self.config.INITIAL_ANALYSIS_SECONDS}秒初始分析...")
        audio_data_2s = np.array(list(self.two_second_buffer), dtype=np.int16)
        audio_data_float = audio_data_2s.astype(np.float32) / self.config.INT16_TO_FLOAT_SCALE

        # 使用GUI中的默认参数进行分析
        params_dict = {k: float(self.config.DEFAULT_GUI_PARAMS[k]) for k in ['f0', 'cut_percent', 'tint', 'period', 'max_offset', 'overlap_percent', 'margin', 'n_points', 'l']}

        # 调用核心分析函数，但不生成绘图
        fitted_results = AudioProcessor.perform_phase_and_freq_shift_analysis(
            audio_data_float, self.config.SAMPLING_RATE, params_dict, plot_results=False
        )

        # 存储拟合得到的模型参数
        self.fitted_T = fitted_results['opt_T']
        self.fitted_vs = fitted_results['opt_vs']
        self.fitted_phi = fitted_results['opt_phi']
        self.fitted_f0 = fitted_results['opt_f0']
        self.fitted_l_dist = fitted_results['l_dist']

        # 如果成功拟合，预计算一个周期的理论频率，用于指示器平滑显示
        if self.fitted_T is not None:
            # 使用GUI设定的周期和更新率来生成时间点
            indicator_period = float(self.config.DEFAULT_GUI_PARAMS['period'])
            t_full_period_for_indicator = np.linspace(0, indicator_period, self.config.INDICATOR_UPDATE_HZ, endpoint=False)

            model_params = (self.fitted_vs, self.fitted_phi, self.fitted_f0)
            consts = {'R': 0.1, 'l': self.fitted_l_dist, 'vs': self.fitted_vs}

            # 计算这些时间点的理论频率
            theoretical_freqs_for_indicator = AudioProcessor._frequency_model(
                model_params, t_full_period_for_indicator, self.fitted_T, consts, 'vs'
            )

            # 存储理论频率的最大最小值和完整序列
            self.min_theoretical_freq = np.min(theoretical_freqs_for_indicator)
            self.max_theoretical_freq = np.max(theoretical_freqs_for_indicator)
            self.precalculated_theoretical_freqs = theoretical_freqs_for_indicator
            logger.info(f"{self.config.INITIAL_ANALYSIS_SECONDS}秒初始分析完成。理论频率范围: [{self.min_theoretical_freq:.4f}, {self.max_theoretical_freq:.4f}] Hz")
            logger.info(f"[预计算频率] 时间范围: 0.0-{self.fitted_T:.3f}s | 点数: {self.config.INDICATOR_UPDATE_HZ} | 频率范围: {self.min_theoretical_freq:.2f}-{self.max_theoretical_freq:.2f}Hz")

        self.last_theoretical_freq_log_time = self.config.INITIAL_ANALYSIS_SECONDS
        logger.info(f"{self.config.INITIAL_ANALYSIS_SECONDS}秒初始分析完成。已获取理论频率模型参数并预计算指示器序列。")

from typing import Tuple, List, Dict, Any, Optional

class AudioProcessor:
    """封装所有核心音频信号处理和分析算法"""
    @staticmethod
    def load_audio(file_path: str) -> Tuple[int, np.ndarray]:
        """
        从WAV文件加载音频数据。

        Args:
            file_path (str): WAV文件路径。

        Returns:
            Tuple[int, np.ndarray]: 采样率和音频数据（单声道）。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到音频文件: {file_path}")
        rate, data = wav.read(file_path)
        # 如果是多声道，只取第一个声道
        return rate, data[:, 0] if data.ndim > 1 else data

    @staticmethod
    def apply_bandpass_filter(data: np.ndarray, rate: int, low: float, high: float, order: int = 3) -> np.ndarray:
        """
        对信号应用带通滤波器。

        Args:
            data (np.ndarray): 输入信号。
            rate (int): 采样率。
            low (float): 通带下限频率。
            high (float): 通带上限频率。
            order (int): 滤波器阶数。

        Returns:
            np.ndarray: 滤波后的信号。
        """
        nyq = 0.5 * rate  # 奈奎斯特频率
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, data)  # 使用filtfilt进行零相位滤波

    @staticmethod
    def envelope_correction(data: np.ndarray, rate: int, cutoff: float = 40.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过希尔伯特变换计算包络并进行平滑，然后用原始信号除以包络以校正幅度变化。

        Args:
            data (np.ndarray): 输入信号。
            rate (int): 采样率。
            cutoff (float): 用于平滑包络的低通滤波器截止频率。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 校正后的信号和平滑后的包络。
        """
        env = np.abs(hilbert(data))  # 计算解析信号的模，得到包络
        # 使用低通滤波器平滑包络
        b, a = butter(2, cutoff / (0.5 * rate), btype='low')
        smooth_env = filtfilt(b, a, env)
        # 信号除以包络，加上一个很小的数避免除以零
        return data / (smooth_env + 1e-8), smooth_env

    @staticmethod
    def compute_stft(data: np.ndarray, fs: int, tint: float = 0.019, overlap: float = 0.125, use_gpu: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        计算短时傅里叶变换 (STFT)，优先使用GPU加速。

        Args:
            data (np.ndarray): 输入信号。
            fs (int): 采样率。
            tint (float): 积分时间（窗口长度）。
            overlap (float): 窗口重叠率。
            use_gpu (bool): 是否尝试使用GPU。

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, int]: 频率轴, 时间轴, STFT结果 (复数矩阵), 窗口长度。
        """
        win_len = int(tint * fs)
        hop_len = int(win_len * (1 - overlap))
        if use_gpu and (torch.backends.mps.is_available() or torch.cuda.is_available()):
            try:
                device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
                signal_tensor = torch.from_numpy(data.astype(np.float32)).to(device)
                window_tensor = torch.hann_window(win_len).to(device)
                Z = torch.stft(signal_tensor, n_fft=win_len, hop_length=hop_len, win_length=win_len, window=window_tensor, return_complex=True)
                f = np.fft.rfftfreq(win_len, d=1/fs)
                t = np.arange(Z.shape[1]) * hop_len / fs
                return f, t, Z.cpu().numpy(), win_len
            except Exception as e:
                logger.warning(f'GPU/Metal STFT 加速失败: {e}。回退到CPU模式。')
        # CPU fallback
        f, t, Z = stft(data, fs=fs, window='hann', nperseg=win_len, noverlap=win_len - hop_len)
        return f, t, Z, win_len

    @staticmethod
    def apply_intensity_filter(data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        为声强分析应用特定的带通滤波器，以隔离蜂鸣器的高频谐波。

        Args:
            data (np.ndarray): 输入信号。
            sample_rate (int): 采样率。

        Returns:
            np.ndarray: 滤波后的信号。
        """
        nyq = 0.5 * sample_rate
        low = Config.INTENSITY_FILTER_LOW_HZ / nyq
        high = Config.INTENSITY_FILTER_HIGH_HZ / nyq
        b, a = butter(3, [low, high], btype='band')
        return filtfilt(b, a, data)

    @staticmethod
    def calculate_rms_intensity(data: np.ndarray, sample_rate: int, window_s: float = Config.INTENSITY_RMS_WINDOW_S, overlap: float = 0.5) -> List[float]:
        """
        计算信号的均方根 (RMS) 声强。

        Args:
            data (np.ndarray): 输入信号。
            sample_rate (int): 采样率。
            window_s (float): RMS计算的窗口时长（秒）。
            overlap (float): 窗口重叠率。

        Returns:
            List[float]: 每个窗口的RMS值列表。
        """
        data = AudioProcessor.apply_intensity_filter(data, sample_rate)
        window_size = int(window_s * sample_rate)
        hop = int(window_size * (1 - overlap))
        frames = [data[i:i+window_size] for i in range(0, len(data)-window_size+1, hop)]
        return [np.sqrt(np.mean(frame**2)) for frame in frames]

    @staticmethod
    def calculate_folded_intensity(audio_data: np.ndarray, sample_rate: int, period: float) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
        """
        计算折叠后的声强分布，并找出峰值位置，用于估计初始相位。

        Args:
            audio_data (np.ndarray): 原始音频数据。
            sample_rate (int): 采样率。
            period (float): 用于折叠的周期。

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[float], Optional[float]]:
            折叠后的时间轴, 对应的强度值, 峰值时间, 峰值强度。
        """
        audio_data = np.asarray(audio_data, dtype=np.float32)
        # 首先应用带通滤波以隔离目标信号
        audio_data = AudioProcessor.apply_intensity_filter(audio_data, sample_rate)
        if audio_data.size == 0:
            return np.array([]), np.array([]), None, None

        time_axis = np.linspace(0, (len(audio_data) - 1) / sample_rate, len(audio_data))

        # 使用滑动窗口计算RMS声强
        window_size = int(sample_rate * Config.INTENSITY_RMS_WINDOW_S)
        hop_size = window_size // 2  # 50% 重叠
        intensities, real_time = [], []
        for i in range(0, len(audio_data) - window_size + 1, hop_size):
            frame = audio_data[i:i + window_size]
            real_time.append(time_axis[i + window_size // 2])
            intensities.append(np.sqrt(np.mean(frame**2)))

        # 将时间轴按周期折叠，并对时间和强度进行排序
        real_time_folded = np.mod(real_time, period)
        sort_idx = np.argsort(real_time_folded)

        # 找到排序后强度数组中的峰值
        sorted_intensities = np.array(intensities)[sort_idx]
        peak_idx = np.argmax(sorted_intensities)
        peak_time = np.array(real_time_folded)[sort_idx][peak_idx]
        peak_value = sorted_intensities[peak_idx]

        return np.array(real_time_folded)[sort_idx], sorted_intensities, peak_time, peak_value

    @staticmethod
    def plot_intensity_distribution(folded_time: np.ndarray, intensities: np.ndarray, peak_time: Optional[float] = None, peak_value: Optional[float] = None) -> plt.Figure:
        """
        生成声强分布图，并可选择性地标记峰值点。

        Args:
            folded_time (np.ndarray): 折叠后的时间轴。
            intensities (np.ndarray): 对应的声强值。
            peak_time (Optional[float]): 峰值时间点。
            peak_value (Optional[float]): 峰值强度。

        Returns:
            plt.Figure: 生成的matplotlib图形对象。
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(folded_time, intensities, label='声强')
        
        if peak_time is not None and peak_value is not None:
            plt.scatter(peak_time, peak_value, c='red', marker='x', s=100, 
                       label=f'蜂鸣器位置 (t={peak_time:.3f}s, I={peak_value:.2f})')
        
        plt.xlabel('折叠时间 (s)')
        plt.ylabel('声强 (RMS)')
        plt.title('声强分布与蜂鸣器位置标记')
        plt.legend()
        plt.grid(True)
        return fig

    @staticmethod
    def perform_matching_pursuit(Zxx: np.ndarray, nperseg: int, fs: int, f0: float, delta_grid: np.ndarray, use_gpu: bool = True) -> np.ndarray:
        """
        执行匹配追踪算法来提取每个时间帧的主频率。
        优先使用GPU加速，如果失败则回退到CPU并行计算。

        Args:
            Zxx (np.ndarray): STFT结果矩阵。
            nperseg (int): 每个窗口的长度。
            fs (int): 采样率。
            f0 (float): 中心频率。
            delta_grid (np.ndarray): 频率搜索网格。
            use_gpu (bool): 是否尝试使用GPU。

        Returns:
            np.ndarray: 每个时间帧提取出的频率脊线。
        """
        t_frame = np.arange(nperseg) / fs
        if use_gpu and (torch.backends.mps.is_available() or torch.cuda.is_available()):
            try:
                device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
                Z_torch = torch.from_numpy(Zxx.astype(np.complex64)).to(device)
                # 逆STFT得到时域信号帧
                frames_sig = torch.fft.irfft(Z_torch.T, n=nperseg, dim=1).T
                delta_grid_torch = torch.from_numpy(delta_grid.astype(np.float32)).to(device)
                t_frame_torch = torch.from_numpy(t_frame.astype(np.float32)).to(device)
                # 构建原子库 (一系列不同频率的复指数信号)
                atoms = torch.exp(1j * 2 * np.pi * (f0 + delta_grid_torch[:, None]) * t_frame_torch[None, :])
                # 计算信号与原子库的内积（相关性）
                corr = torch.abs(atoms @ frames_sig.to(torch.complex64))
                # 找到每个时间帧中相关性最大的原子，其频率即为所求
                return f0 + delta_grid[torch.argmax(corr, dim=0).cpu().numpy()]
            except Exception as e:
                logger.warning(f"GPU/Metal 匹配追踪失败: {e}。回退到CPU模式。")
        
        # CPU并行计算作为后备方案
        frames_sig = np.fft.irfft(Zxx, n=nperseg, axis=0).T
        args = [(frame, delta_grid, f0, t_frame) for frame in frames_sig]
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(_process_single_frame_mp, args))
        return f0 + np.array(results)

    @staticmethod
    def _frequency_model(params: Tuple[float, float, float], t: np.ndarray, T: float, consts: Dict[str, float], mode: str) -> np.ndarray:
        """
        多普勒效应的物理模型，用于计算给定参数下的理论频率。

        Args:
            params (Tuple): 待拟合的参数 (vs, phi, f0_corr) 或 (R, phi, f0_corr)。
            t (np.ndarray): 时间轴。
            T (float): 周期。
            consts (Dict): 模型中的常量 (R, l, vs)。
            mode (str): 拟合模式 ('vs' 或 'R')。

        Returns:
            np.ndarray: 计算出的理论频率。
        """
        if mode == 'vs':
            vs, phi, f0_corr = params
            R = consts['R']
        else:
            R, phi, f0_corr = params
            vs = consts['vs']
        l = consts['l']
        theta = 2 * np.pi * t / T + phi
        # 避免分母为零
        denominator_in_sqrt = (l * np.cos(theta) - R)**2 / ((l * np.sin(theta))**2 + 1e-12)
        vd = (2 * np.pi * R) / (T * np.sqrt(1 + denominator_in_sqrt))
        # 判断声源是朝向还是背向观察者
        pm = np.where(np.mod(t + (phi / (2 * np.pi)) * T, T) < (T / 2.0), -1.0, 1.0)
        return f0_corr * (vs + pm * vd) / vs if vs > 0 else np.inf

    @staticmethod
    def fit_parameters(t: np.ndarray, freq: np.ndarray, T: float, f0: float, consts: Dict[str, float], mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用最小二乘法拟合物理模型参数。

        Args:
            t (np.ndarray): 时间轴。
            freq (np.ndarray): 测得的频率数据。
            T (float): 优化的周期。
            f0 (float): 中心频率。
            consts (Dict): 模型常量。
            mode (str): 拟合模式。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 优化后的参数和根据这些参数生成的模型频率。
        """
        def error_func(params: Tuple[float, float, float]) -> float:
            """定义误差函数（预测与实际的均方误差）。"""
            pred_freq = AudioProcessor._frequency_model(params, t, T, consts, mode)
            return np.mean((pred_freq - freq)**2)

        if mode == 'vs':
            initial_guess = [343.0, 0.0, f0]  # 初始猜测值
            bounds = [(300.0, 400.0), (-np.pi, np.pi), (f0 * 0.99, f0 * 1.01)]  # 参数边界
        else:  # mode == 'R'
            initial_guess = [0.1, 0.0, f0]
            bounds = [(0.01, 0.5), (-np.pi, np.pi), (f0 * 0.99, f0 * 1.01)]

        result = minimize(error_func, initial_guess, method='L-BFGS-B', bounds=bounds)
        return result.x, AudioProcessor._frequency_model(result.x, t, T, consts, mode)

    @staticmethod
    def optimize_period(t: np.ndarray, freq: np.ndarray, T_guess: float, margin: float = 0.005, n_harmonics: int = 7, n_points: int = 201) -> float:
        """
        通过在给定范围内搜索，找到使傅里叶级数拟合误差最小的周期。

        Args:
            t (np.ndarray): 时间轴。
            freq (np.ndarray): 频率数据。
            T_guess (float): 初始猜测的周期。
            margin (float): 周期搜索范围的百分比。
            n_harmonics (int): 傅里叶级数的谐波数。
            n_points (int): 周期搜索的点数。

        Returns:
            float: 优化后的最佳周期。
        """
        periods = np.linspace(T_guess * (1 - margin), T_guess * (1 + margin), n_points)
        min_mse, best_T = np.inf, T_guess
        for T_cand in periods:
            folded_t = np.mod(t, T_cand)
            sorted_indices = np.argsort(folded_t)
            # 构建三角函数设计矩阵
            X = AudioProcessor.trig_design_matrix(folded_t[sorted_indices], n_harmonics, T_cand)
            # 最小二乘拟合，计算残差
            _, res, _, _ = np.linalg.lstsq(X, freq[sorted_indices], rcond=None)
            mse = res[0] / len(freq) if res.size > 0 else 0.0
            if mse < min_mse:
                min_mse, best_T = mse, T_cand
        return best_T

    @staticmethod
    def trig_design_matrix(x: np.ndarray, n_harmonics: int, T: float) -> np.ndarray:
        """
        为傅里叶级数拟合构建设计矩阵。

        Args:
            x (np.ndarray): 输入的时间数据。
            n_harmonics (int): 谐波数。
            T (float): 周期。

        Returns:
            np.ndarray: 设计矩阵X。
        """
        X = [np.ones_like(x)]  # 第一列是直流分量
        for n in range(1, n_harmonics + 1):
            angle = 2 * np.pi * n * x / T
            X.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(X)

    @staticmethod
    def perform_phase_and_freq_shift_analysis(audio_data: np.ndarray, sample_rate: int, params: Dict[str, Any], plot_results: bool = True) -> Dict[str, Any]:
        """
        执行完整的离线分析流程，包括声强分析、频率追踪、参数拟合和相位校正。

        Args:
            audio_data (np.ndarray): 输入的音频数据。
            sample_rate (int): 采样率。
            params (Dict[str, Any]): 分析所需的参数字典。
            plot_results (bool): 是否返回用于绘图的完整数据。

        Returns:
            Dict[str, Any]: 包含分析结果的字典。如果 plot_results 为 False，则只返回核心拟合参数。
        """
        f0, cut, tint, period, offset, overlap, margin, n_points, l_dist = \
            [params[k] for k in ['f0', 'cut_percent', 'tint', 'period', 'max_offset', 'overlap_percent', 'margin', 'n_points', 'l']]

        # 步骤1: 声强分析，找到声强峰值对应的时间点，用于估计初始相位
        _, _, peak_time, peak_value = \
            AudioProcessor.calculate_folded_intensity(audio_data, sample_rate, period)

        initial_phi_degrees = 0.0
        if peak_time is not None:
            initial_phi_degrees = (peak_time / period) * 360
            logger.info(f"分析结果 - 声强峰值出现在折叠时间 {peak_time:.4f} s，强度为 {peak_value:.4f}")
            logger.info(f"分析结果 - 计算得到初始相位 phi: {initial_phi_degrees:.2f} 度")
        else:
            logger.warning("分析结果 - 未能找到声强峰值，初始相位将使用默认值 0。")

        # 步骤2: 信号预处理和频率追踪
        filtered = AudioProcessor.apply_bandpass_filter(audio_data, sample_rate, f0 * (1 - cut), f0 * (1 + cut))
        corrected, _ = AudioProcessor.envelope_correction(filtered, sample_rate)
        
        f_stft, t_stft, Zxx, win_len = AudioProcessor.compute_stft(corrected, sample_rate, tint, overlap)
        delta_grid = np.linspace(-offset, offset, Config.MATCHING_PURSUIT_GRID_POINTS)
        ridge_freq = AudioProcessor.perform_matching_pursuit(Zxx, win_len, sample_rate, f0, delta_grid)
        
        # 步骤3: 周期优化和参数拟合
        opt_T = AudioProcessor.optimize_period(t_stft, ridge_freq, period, margin=margin, n_harmonics=int(n_points))
        folded_time = np.mod(t_stft, opt_T)
        
        consts = {'R': 0.1, 'l': l_dist}  # 固定R，拟合vs
        (opt_vs, opt_phi, opt_f0), model_freq = AudioProcessor.fit_parameters(folded_time, ridge_freq, opt_T, f0, consts, 'vs')

        # 步骤4: 相位校正和频移计算
        original_phase_degrees = (folded_time / opt_T) * 360
        # 使用声强分析得到的初始相位进行校正
        adjusted_phase_degrees = (original_phase_degrees - initial_phi_degrees) % 360
        adjusted_phase_degrees[adjusted_phase_degrees < 0] += 360

        # 计算实际频移并按校正后的相位进行分箱
        freq_shift = ridge_freq - opt_f0
        bin_size = Config.PHASE_BIN_SIZE_DEGREES
        num_bins = int(360 / bin_size)
        binned_shifts = [[] for _ in range(num_bins)]
        for i, phase in enumerate(adjusted_phase_degrees):
            bin_idx = int(phase / bin_size)
            if 0 <= bin_idx < num_bins:
                binned_shifts[bin_idx].append(freq_shift[i])

        # 步骤5: 整理并返回结果
        return_data = {
            'opt_T': opt_T,
            'opt_vs': opt_vs,
            'opt_phi': opt_phi,
            'opt_f0': opt_f0,
            'l_dist': l_dist,
            'folded_time': folded_time,
            'ridge_freq': ridge_freq,
            'model_freq': model_freq,
            'adjusted_phase_degrees': adjusted_phase_degrees,
            'peak_time': peak_time,
            'initial_phi_degrees': initial_phi_degrees
        }
        
        if plot_results:
            return return_data
        else:
            # 如果不需要绘图，只返回核心的拟合参数
            return {
                'opt_T': opt_T,
                'opt_vs': opt_vs,
                'opt_phi': opt_phi,
                'opt_f0': opt_f0,
                'l_dist': l_dist,
                'initial_phi_degrees': initial_phi_degrees
            }

class GUIApp:
    """管理整个图形用户界面（GUI）的类"""
    def __init__(self, root: tk.Tk, config: Config):
        """
        初始化GUI应用。

        Args:
            root (tk.Tk): Tkinter的根窗口。
            config (Config): 全局配置对象。
        """
        self.root, self.config = root, config
        self.processor = AudioProcessor()
        self.data_manager = RealtimeDataManager(config)
        # 使用StringVar将GUI组件与参数关联起来
        self.gui_params = {k: tk.StringVar(value=v) for k, v in self.config.DEFAULT_GUI_PARAMS.items()}

        self.colors = ['' for i in range(48)]
        
        # ========== GUI组件引用 ==========
        self.btn_start_sampling = None
        self.btn_stop_sampling = None
        self.indicator_label = None
        
        # ========== GUI状态变量 ==========
        self.show_left_panel = False  # 控制左侧参数面板的显示/隐藏
        self.is_indicator_running = False  # 控制理论频率指示器的更新循环
        self.theoretical_freq_var = tk.StringVar(value="理论频率: N/A")

        # ========== 初始化流程 ==========
        self._setup_gui()  # 构建GUI布局
        self._toggle_left_panel()  # 默认隐藏左侧面板
        self.data_manager.start_threads()  # 启动后台数据处理线程

    def _setup_gui(self):
        """初始化并构建整个GUI界面。"""
        self.root.title("多普勒效应音频分析工具 (重构版)")
        self.root.geometry("1600x900")  # 设置默认窗口大小
        self._setup_styles()  # 配置ttk控件样式

        # 创建并布局各个GUI模块
        self._create_toolbar()
        self.left_panel, right_panel = self._create_main_layout()
        self._create_plot_panels(right_panel)
        self._create_params_notebook(self.left_panel)
        
        self._embed_plots()  # 将matplotlib绘图嵌入到Tkinter窗口
        self._update_sampling_button_states()  # 根据初始状态设置按钮可用性

    def _setup_styles(self):
        """为ttk控件配置自定义样式，例如彩色按钮。"""
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 12, 'bold'), padding=10)
        style.map('TButton',
                  foreground=[('pressed', 'white'), ('active', 'black')],
                  background=[('pressed', '!focus', 'SystemButtonFace'), ('active', 'SystemButtonFace')])
        # 定义不同颜色的按钮样式
        style.configure('Green.TButton', background='lightgreen', foreground='black')
        style.map('Green.TButton', background=[('active', '#90EE90')])
        style.configure('Red.TButton', background='salmon', foreground='black')
        style.map('Red.TButton', background=[('active', '#FA8072')])
        style.configure('Coral.TButton', background='lightcoral', foreground='black')
        style.map('Coral.TButton', background=[('active', '#F08080')])
        style.configure('Blue.TButton', background='lightblue', foreground='black')
        style.map('Blue.TButton', background=[('active', '#ADD8E6')])

    def _create_toolbar(self) -> tk.Frame:
        """创建并返回顶部的工具栏Frame。"""
        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        control_frame = tk.Frame(toolbar_frame)
        control_frame.pack(side=tk.LEFT)
        
        # 创建控制按钮
        self.btn_start_sampling = ttk.Button(control_frame, text="开始采集", command=self._start_sampling, style='Green.TButton')
        self.btn_start_sampling.pack(side=tk.LEFT, padx=5)
        self.btn_stop_sampling = ttk.Button(control_frame, text="停止采集", command=self._stop_sampling, style='Red.TButton', state=tk.DISABLED)
        self.btn_stop_sampling.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="清除实时数据", command=self._clear_buffers, style='Coral.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="开始离线分析", command=self._run_offline_analysis, style='Blue.TButton').pack(side=tk.LEFT, padx=5)
        
        # 新增清除灯光按钮
        ttk.Button(control_frame, text="清除灯光数据", command=self._clear_leds, style='Coral.TButton').pack(side=tk.LEFT, padx=5)
        
        # 新增旋转控制按钮
        ttk.Button(control_frame, text="开始旋转", command=self._start_rotation, style='Green.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="停止旋转", command=self._stop_rotation, style='Red.TButton').pack(side=tk.LEFT, padx=5)
        
        # 创建理论频率指示器
        self.indicator_label = tk.Label(control_frame, textvariable=self.theoretical_freq_var, font=("Arial", 12, "bold"), bg="gray", fg="white", width=20, relief="ridge", bd=2)
        self.indicator_label.pack(side=tk.LEFT, padx=15)
        
        # 创建显示/隐藏参数面板的复选框
        self.show_panel_var = tk.BooleanVar(value=False)
        show_panel_check = tk.Checkbutton(toolbar_frame, text="显示参数面板", variable=self.show_panel_var, command=self._toggle_left_panel, font=("Arial", 12))
        show_panel_check.pack(side=tk.RIGHT)
        
        return toolbar_frame

    def _create_main_layout(self) -> Tuple[tk.Frame, tk.Frame]:
        """创建主窗口的左右布局。"""
        # 左侧面板（可隐藏），用于放置参数设置
        left_panel = tk.Frame(self.root, width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)  # 防止面板因内容而自动缩放
        
        # 右侧面板，用于放置四个绘图区域
        right_panel = tk.Frame(self.root)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 配置右侧面板的网格布局，使其可以随窗口缩放
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_columnconfigure(1, weight=1)
        
        return left_panel, right_panel

    def _create_plot_panels(self, parent: tk.Frame):
        """在右侧主面板中创建2x2的网格布局用于放置四个绘图。"""
        self.image_frame = tk.Frame(parent, bd=2, relief="groove")
        self.waveform_frame = tk.Frame(parent, bd=2, relief="groove")
        self.realtime_freq_frame = tk.Frame(parent, bd=2, relief="groove")
        self.analysis_frame = tk.Frame(parent, bd=2, relief="groove")

        # 使用grid布局管理器将Frame放置在2x2网格中
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.waveform_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.realtime_freq_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.analysis_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    def _create_params_notebook(self, parent: tk.Frame):
        """在左侧面板中创建参数设置的Notebook控件。"""
        left_notebook = ttk.Notebook(parent)
        left_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        params_tab_frame = tk.Frame(left_notebook)
        left_notebook.add(params_tab_frame, text="参数设置")
        self._create_params_panel(params_tab_frame)

    def _embed_plots(self):
        """将所有matplotlib绘图和图片嵌入到各自的Tkinter Frame中。"""
        self._embed_offline_analysis_plot(self.analysis_frame)
        self._embed_waveform_plot(self.waveform_frame)
        self._embed_realtime_freq_plot(self.realtime_freq_frame)
        self._embed_image_plot(self.image_frame, self.config.DEFAULT_IMAGE_PATH)

    def _create_params_panel(self, parent: tk.Frame):
        """创建包含所有分析参数输入框的面板。"""
        frame = tk.LabelFrame(parent, text="分析参数", padx=10, pady=10, font=("Arial", 14))
        frame.pack(pady=10, fill=tk.X)

        # 音频文件选择
        tk.Label(frame, text="音频文件:", font=("Arial", 14)).grid(row=0, column=0, sticky=tk.W, pady=2)
        tk.Entry(frame, textvariable=self.gui_params['audio_file'], width=30, font=("Arial", 14)).grid(row=0, column=1, sticky=tk.EW)
        tk.Button(frame, text="浏览...", font=("Arial", 14), command=self._select_file).grid(row=0, column=2, padx=5)

        # 串口号输入
        tk.Label(frame, text="串口号:", font=("Arial", 14)).grid(row=1, column=0, sticky=tk.W, pady=2)
        tk.Entry(frame, textvariable=self.gui_params['serial_port'], width=10, font=("Arial", 14)).grid(row=1, column=1, columnspan=2, sticky=tk.EW)

        # 新增旋转方向选择
        tk.Label(frame, text="旋转方向:", font=("Arial", 14)).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.rotation_var = tk.StringVar(value="逆时针")
        tk.Radiobutton(frame, text="逆时针", variable=self.rotation_var, value="逆时针", 
                    font=("Arial", 14)).grid(row=2, column=1, sticky=tk.W)
        tk.Radiobutton(frame, text="顺时针", variable=self.rotation_var, value="顺时针", 
                    font=("Arial", 14)).grid(row=2, column=2, sticky=tk.W)

        # 其他数值参数
        param_labels = {
            'f0': "f0 (Hz):", 'cut_percent': "截止频率 %:", 'tint': "积分时间 (s):", 
            'window_length': "窗口长度:", 'period': "周期 T (s):", 'max_offset': "最大偏移:",
            'overlap_percent': "重叠率 %:", 'margin': "边际 %:", 'n_points': "点数:", 
            'l': "距离 l (m):", 'realtime_freq_ymin': "Y轴最小:", 'realtime_freq_ymax': "Y轴最大:"
        }
        # 从第5行开始动态创建标签和输入框
        for i, (key, text) in enumerate(param_labels.items(), 5):
            tk.Label(frame, text=text, font=("Arial", 14)).grid(row=i, column=0, sticky=tk.W, pady=2)
            tk.Entry(frame, textvariable=self.gui_params[key], width=10, font=("Arial", 14)).grid(row=i, column=1, columnspan=2, sticky=tk.EW)
        
        frame.columnconfigure(1, weight=1) # 让输入框列可以缩放

    def _select_file(self):
        """打开文件对话框以选择WAV文件。"""
        f = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if f:
            self.gui_params['audio_file'].set(f)

    def _embed_offline_analysis_plot(self, parent: tk.Frame):
        """在指定的父Frame中嵌入一个空的matplotlib绘图，用于显示离线分析结果。"""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        self.offline_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.offline_canvas_widget = self.offline_canvas.get_tk_widget()
        self.offline_canvas_widget.pack(fill=tk.BOTH, expand=True)
        ax.set_title('离线分析结果')
        ax.grid(True)

    def _run_offline_analysis(self):
        """
        “开始离线分析”按钮的回调函数。
        它会读取GUI中的参数，加载音频文件，执行完整的分析流程，并在右下面板中显示结果图。
        """
        # 清除上一次的分析结果图
        for widget in self.analysis_frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()
        
        try:
            # 从GUI获取所有必要的参数
            params_dict = {k: float(self.gui_params[k].get()) for k in ['f0', 'cut_percent', 'tint', 'period', 'max_offset', 'overlap_percent', 'margin', 'n_points', 'l']}
            params_dict['audio_file'] = self.gui_params['audio_file'].get()

            rate, audio = AudioProcessor.load_audio(params_dict['audio_file'])
            audio_data_float = audio.astype(np.float32) / self.config.INT16_TO_FLOAT_SCALE

            # 调用核心分析函数
            analysis_results = AudioProcessor.perform_phase_and_freq_shift_analysis(
                audio_data_float, rate, params_dict, plot_results=True
            )
            
            # 从结果字典中解包数据用于绘图
            opt_T, ridge_freq, model_freq, opt_vs, opt_phi, opt_f0 = \
                [analysis_results[k] for k in ['opt_T', 'ridge_freq', 'model_freq', 'opt_vs', 'opt_phi', 'opt_f0']]
            current_peak_time = analysis_results['peak_time']
            folded_time_for_plot = analysis_results['folded_time']

            # --- 绘图 ---
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.scatter(folded_time_for_plot, ridge_freq, s=10, alpha=0.5, label='原始数据')
            
            # 为了正确绘制拟合曲线，需要对x轴（折叠时间）进行排序
            sort_indices = np.argsort(folded_time_for_plot)
            ax.plot(folded_time_for_plot[sort_indices], model_freq[sort_indices], 'r-', lw=3.5, label=f"拟合: vs={opt_vs:.2f}m/s, φ={np.degrees(opt_phi):.1f}°")
            
            ax.axhline(y=opt_f0, color='teal', ls='--', lw=2, label=f"修正后f0={opt_f0:.2f} Hz")
            
            # 标记最大和最小频率点
            max_idx, min_idx = np.argmax(model_freq), np.argmin(model_freq)
            ax.plot(folded_time_for_plot[max_idx], model_freq[max_idx], 'r*', ms=10, label='最高点')
            ax.plot(folded_time_for_plot[min_idx], model_freq[min_idx], 'rv', ms=10, label='最低点')
            
            # 标记声强峰值时间
            if current_peak_time is not None:
                ax.axvline(x=current_peak_time, color='m', linestyle='--', linewidth=2, label=f'声强峰值 (t={current_peak_time:.3f}s)')
            
            # 自动调整Y轴范围
            p5_ridge, p95_ridge = np.percentile(ridge_freq, 5), np.percentile(ridge_freq, 95)
            min_model, max_model = np.min(model_freq), np.max(model_freq)
            min_freq_final, max_freq_final = min(p5_ridge, min_model), max(p95_ridge, max_model)
            range_freq = max_freq_final - min_freq_final
            ax.set_ylim(min_freq_final - range_freq * 0.1, max_freq_final + range_freq * 0.1)
            
            ax.set_xlabel(f"折叠时间 (s, T={opt_T:.4f}s)")
            ax.set_ylabel('频率 (Hz)')
            ax.set_title('声速-相位拟合结果')
            ax.legend()
            ax.grid(True)
            ax.set_xlim(0, opt_T)
            # plt.savefig('freq-time.png', dpi=300)
            
            # 将新图嵌入到GUI中
            self.offline_canvas = FigureCanvasTkAgg(fig, master=self.analysis_frame)
            self.offline_canvas_widget = self.offline_canvas.get_tk_widget()
            self.offline_canvas_widget.pack(fill=tk.BOTH, expand=True)
            self.offline_canvas.draw()
            
            messagebox.showinfo("成功", "离线分析完成。")
        except Exception as e:
            logger.error(f"离线分析失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"分析失败: {e}")

    def _embed_waveform_plot(self, parent: tk.Frame):
        """嵌入实时波形图。"""
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.subplots_adjust(bottom=0.25)
        x_data = np.arange(0, self.config.PLOT_WINDOW_SAMPLES)
        line, = ax.plot(x_data, np.zeros(self.config.PLOT_WINDOW_SAMPLES), lw=1)
        ax.set_ylim(-500, 500) # (-32768, 32768)
        ax.set_xlim(0, self.config.PLOT_WINDOW_SAMPLES)
        ax.set_title("实时声音波形")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def update(frame):
            """动画更新函数，从数据管理器获取最新波形数据并更新绘图。"""
            data = np.array([s[1] for s in self.data_manager.display_waveform_buffer])
            # 如果数据不足，用0填充
            if len(data) < self.config.PLOT_WINDOW_SAMPLES:
                data = np.pad(data, (self.config.PLOT_WINDOW_SAMPLES - len(data), 0))
            line.set_ydata(data)
            return line,
        # 使用FuncAnimation实现动态更新
        parent._ani = animation.FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

    def _embed_realtime_freq_plot(self, parent: tk.Frame):
        """嵌入实时折叠频率图。"""
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter([], [], s=10, alpha=0.8)
        ax.set_title("实时折叠频率")
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def update(frame):
            """动画更新函数，获取最新频率数据并更新散点图。"""
            with self.data_manager.realtime_freq_lock:
                if not self.data_manager.realtime_freq_points:
                    return scatter,
                data = np.array(list(self.data_manager.realtime_freq_points))
            
            if data.ndim == 2 and data.shape[1] == 2:
                scatter.set_offsets(data)
                
                # 动态调整Y轴范围，或使用用户指定范围
                ymin_str, ymax_str = self.gui_params['realtime_freq_ymin'].get(), self.gui_params['realtime_freq_ymax'].get()
                ymin_user = float(ymin_str) if ymin_str else None
                ymax_user = float(ymax_str) if ymax_str else None

                if ymin_user is not None and ymax_user is not None:
                    ax.set_ylim(ymin_user, ymax_user)
                elif len(data) > 0:
                    y_min_data, y_max_data = np.min(data[:, 1]), np.max(data[:, 1])
                    padding = (y_max_data - y_min_data) * 0.15 if (y_max_data - y_min_data) > 0 else 10
                    ax.set_ylim(y_min_data - padding, y_max_data + padding)
                
                ax.yaxis.set_major_locator(plt.MaxNLocator(prune='both'))
                ax.set_xlim(0, self.data_manager.realtime_period)
                ax.set_xlabel(f"折叠时间 (T={self.data_manager.realtime_period:.3f}s)")
            return scatter,
        parent._ani = animation.FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    def _embed_image_plot(self, parent: tk.Frame, image_path: str):
        """嵌入并自适应显示示意图。"""
        try:
            img = Image.open(image_path)
            
            def configure_image(event=None):
                """当窗口大小改变时，重新计算图片大小并更新。"""
                if event and event.widget != parent: return
                    
                parent_width, parent_height = parent.winfo_width(), parent.winfo_height()
                if parent_width <= 10 or parent_height <= 10: return
                    
                available_width, available_height = parent_width - 10, parent_height - 10
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                
                # 根据父容器和图片的宽高比，计算最佳缩放尺寸
                if (available_width / available_height) > aspect_ratio:
                    new_height = available_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    new_width = available_width
                    new_height = int(new_width / aspect_ratio)
                
                resized_img = img.resize((max(1, new_width), max(1, new_height)), Image.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(resized_img)
                
                # 更新或创建图片标签
                if hasattr(parent, 'image_label'):
                    parent.image_label.config(image=self.tk_image)
                else:
                    parent.image_label = tk.Label(parent, image=self.tk_image)
                    parent.image_label.place(relx=0.5, rely=0.5, anchor='center')
                parent.image_label.image = self.tk_image  # 保持引用，防止被垃圾回收

            parent.after(100, configure_image)  # 延迟执行以确保父容器尺寸有效
            parent.bind("<Configure>", configure_image)  # 绑定窗口大小变化事件

        except FileNotFoundError:
            label = tk.Label(parent, text=f"图片未找到:\n{image_path}", fg="red")
            label.place(relx=0.5, rely=0.5, anchor='center')
            logger.error(f"示意图文件未找到: {image_path}")
        except Exception as e:
            label = tk.Label(parent, text=f"加载图片失败: {e}", fg="red")
            label.place(relx=0.5, rely=0.5, anchor='center')
            logger.error(f"加载示意图失败: {e}", exc_info=True)



    def _update_sampling_button_states(self):
        """根据当前是否正在采样，更新开始/停止按钮的状态。"""
        if self.btn_start_sampling and self.btn_stop_sampling:
            is_sampling = self.data_manager.is_sampling
            self.btn_start_sampling.config(state=tk.DISABLED if is_sampling else tk.NORMAL)
            self.btn_stop_sampling.config(state=tk.NORMAL if is_sampling else tk.DISABLED)

    def _start_sampling(self):
        """“开始采集”按钮的回调函数。"""
        # 重置所有分析状态，为新的采集做准备
        self.data_manager.two_second_analysis_triggered = False
        self.data_manager.two_second_buffer.clear()
        self.data_manager.fitted_T = None
        self.data_manager.precalculated_theoretical_freqs = None
        self.data_manager.audio_file_loaded = False

        # 加载音频文件用于模拟
        file_path = self.gui_params['audio_file'].get()
        if file_path and not self.data_manager._load_audio_file(file_path):
            messagebox.showerror("错误", f"无法加载音频文件: {file_path}")
            return
            
        # 设置状态标志并启动相关进程
        self.data_manager.is_sampling = True
        self.is_indicator_running = True
        self._update_indicator_light()  # 启动指示器更新循环
        if self.config.ENABLE_60S_AUDIO_FEATURE: 
            self.data_manager.long_term_audio_buffer.clear()
            self.data_manager.is_60s_recording_active = True
        self._update_sampling_button_states()

    def _stop_sampling(self):
        """“停止采集”按钮的回调函数。"""
        self.is_indicator_running = False
        self.data_manager.is_sampling = False
        
        # 如果启用了60秒录音功能，则保存录音并进行分析
        if self.config.ENABLE_60S_AUDIO_FEATURE and self.data_manager.is_60s_recording_active:
            self.data_manager.is_60s_recording_active = False
            if self.data_manager.long_term_audio_buffer:
                # 保存录制的音频到临时WAV文件
                audio_data_raw = np.array(list(self.data_manager.long_term_audio_buffer), dtype=np.int16)
                path = os.path.join(os.getcwd(), f"temp_60s_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav")
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.config.SAMPLING_RATE)
                    wf.writeframes(audio_data_raw.tobytes())
                
                # 使用保存的60秒音频文件路径进行离线分析
                original_path = self.gui_params['audio_file'].get()
                self.gui_params['audio_file'].set(path)
                self._run_offline_analysis()
                self.gui_params['audio_file'].set(original_path)  # 恢复原始文件路径

        self._update_sampling_button_states()

    def _toggle_left_panel(self):
        """切换左侧参数面板的显示和隐藏。"""
        self.show_left_panel = self.show_panel_var.get()
        if self.show_left_panel:
            self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        else:
            self.left_panel.pack_forget()
        self.show_panel_var.set(self.show_left_panel)

    def time_location(self, peak_time, colors, period=1):
        start_prop = (peak_time / period ) % 1
        start_light = 3  # int(48 * start_prop) 
        self.colors = colors[start_light:] + colors[:start_light]

    
    def _send_colors_to_serial(self, colors):
        """将颜色数据发送到串口"""
        try:
            import serial
            ser = serial.Serial(port=self.gui_params['serial_port'].get(), 
                            baudrate=self.config.BAUD_RATE, 
                            timeout=0.1)
            if ser:
                cmd = 'L' + ''.join(colors) + '\r\n'
                ser.write(cmd.encode('ascii'))
                ser.close()
                logger.info(f"已发送颜色数据到串口 {self.gui_params['serial_port'].get()}")
        except Exception as e:
            logger.error(f"发送颜色数据时出错: {e}")
            messagebox.showerror("错误", f"发送颜色数据失败: {e}")


    def _get_color_for_frequency(self, freq: float, min_freq: float, max_freq: float) -> str:
        """
        根据频率在最小/最大频率范围内的位置，计算一个从红到蓝的颜色值。
        """
        if min_freq is None or max_freq is None or min_freq == max_freq:
            return "#808080"  # 如果范围无效，返回灰色

        # 将频率归一化到0-1范围
        normalized_freq = (freq - min_freq) / (max_freq - min_freq)
        normalized_freq = np.clip(normalized_freq, 0, 1)

        # 在红色(255,0,0)和蓝色(0,0,255)之间进行线性插值
        r = int(60 * (1 - normalized_freq))
        g = 0
        b = int(60 * normalized_freq)
        
        return f"#{r:02x}{g:02x}{b:02x}" # f"#{r:02x}{g:02x}{b:02x}"

    def _update_indicator_light(self):
        """
        以48Hz的频率更新理论频率指示器的文本和背景色。
        这是一个自调度的循环函数。
        """
        if not self.is_indicator_running:
            self.theoretical_freq_var.set("理论频率: N/A")
            self.indicator_label.config(bg="gray")
            logger.info("指示器更新循环已停止。")
            return

        precalculated_freqs = self.data_manager.precalculated_theoretical_freqs
        if precalculated_freqs is not None and len(precalculated_freqs) > 0:
            # 从预计算序列中循环获取当前频率
            current_freq = precalculated_freqs[self.data_manager.precalculated_freq_index]
            
            self.theoretical_freq_var.set(f"理论频率: {current_freq:.2f} Hz")
            color = self._get_color_for_frequency(current_freq, self.data_manager.min_theoretical_freq, self.data_manager.max_theoretical_freq)
            self.indicator_label.config(bg=color)

            self.colors[self.data_manager.precalculated_freq_index] = color[3:5] + color[5:] + color[1:3]

            # # 根据旋转方向处理colors列表
            # if self.data_manager.precalculated_freq_index == 47:
            #     audio_data_2s = np.array(list(self.data_manager.two_second_buffer), dtype=np.int16)
            #     audio_data_float = audio_data_2s.astype(np.float32) / self.config.INT16_TO_FLOAT_SCALE
            #     _, _, peak_time, peak_value = \
            #     AudioProcessor.calculate_folded_intensity(audio_data_float, self.config.SAMPLING_RATE, 1)  # period=1
            #     self.time_location(peak_time, self.colors)
            #
            #     # 获取当前旋转方向
            #     rotation = self.rotation_var.get()
            #     colors_to_send = self.colors.copy()
            #     if rotation == "顺时针":
            #         colors_to_send = colors_to_send[::-1]  # 逆序列表

            colors_to_send = ['000000' for i in range(48)]
            colors_to_send[self.data_manager.precalculated_freq_index] = color[3:5] + color[5:] + color[1:3]
                
            self._send_colors_to_serial(colors_to_send)

            # 更新索引以备下次调用
            self.data_manager.precalculated_freq_index = (self.data_manager.precalculated_freq_index + 1) % len(precalculated_freqs)
        else:
            self.theoretical_freq_var.set("理论频率: N/A")
            self.indicator_label.config(bg="gray")

        # 调度下一次更新
        self.root.after(self.config.INDICATOR_UPDATE_MS, self._update_indicator_light)

    def _clear_buffers(self):
        """“清除实时数据”按钮的回调函数，重置所有实时数据和分析状态。"""
        self.data_manager.display_waveform_buffer.clear()
        with self.data_manager.realtime_freq_lock:
            self.data_manager.realtime_freq_points.clear()
        with self.data_manager.global_sample_counter_lock:
            self.data_manager.global_sample_counter = 0
        
        # 重置2秒分析相关的状态
        self.data_manager.two_second_analysis_triggered = False
        self.data_manager.two_second_buffer.clear()
        self.data_manager.fitted_T = None
        self.data_manager.audio_file_loaded = False
        
        # 从GUI同步最新的实时参数
        try:
            params = {k: v.get() for k, v in self.gui_params.items()}
            self.data_manager.realtime_period = float(params['period'])
            self.data_manager.realtime_stft_params.update({k: float(params[k]) for k in ['tint', 'overlap_percent', 'f0', 'max_offset']})
            self.data_manager.realtime_stft_params['window_length'] = int(params['window_length'])
            self.config.SERIAL_PORT = params['serial_port']
        except Exception as e:
            logger.warning(f"同步实时参数时出错: {e}")

    def _clear_leds(self):
        """清除灯光数据按钮的回调函数，发送'A000000\r\n'命令给单片机"""
        cmd = 'A000000\r\n'
        try:
            import serial
            ser = serial.Serial(port=self.gui_params['serial_port'].get(), 
                            baudrate=self.config.BAUD_RATE, 
                            timeout=0.1)
            if ser:
                ser.write(cmd.encode('ascii'))
                ser.close()
                logger.info(f"已发送清除灯光命令到串口 {self.gui_params['serial_port'].get()}")
        except Exception as e:
            logger.error(f"清除灯光数据时出错: {e}")
            messagebox.showerror("错误", f"清除灯光数据失败: {e}")

    def _start_rotation(self):
        """开始旋转按钮的回调函数，根据旋转方向发送相应指令"""
        try:
            import serial
            rotation_direction = self.rotation_var.get()
            command = "S1\r\n" if rotation_direction == "顺时针" else "S-1\r\n"
            
            ser = serial.Serial(port=self.gui_params['serial_port'].get(), 
                            baudrate=self.config.BAUD_RATE, 
                            timeout=0.1)
            if ser:
                ser.write(command.encode('ascii'))
                ser.close()
                logger.info(f"已发送开始旋转指令到串口 {self.gui_params['serial_port'].get()}: {command.strip()}")
        except Exception as e:
            logger.error(f"发送开始旋转指令时出错: {e}")
            messagebox.showerror("错误", f"发送开始旋转指令失败: {e}")

    def _stop_rotation(self):
        """停止旋转按钮的回调函数，发送停止指令"""
        try:
            import serial
            command = "S0\r\n"
            
            ser = serial.Serial(port=self.gui_params['serial_port'].get(), 
                            baudrate=self.config.BAUD_RATE, 
                            timeout=0.1)
            if ser:
                ser.write(command.encode('ascii'))
                ser.close()
                logger.info(f"已发送停止旋转指令到串口 {self.gui_params['serial_port'].get()}: {command.strip()}")
        except Exception as e:
            logger.error(f"发送停止旋转指令时出错: {e}")
            messagebox.showerror("错误", f"发送停止旋转指令失败: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = GUIApp(root, Config())
    root.mainloop()
