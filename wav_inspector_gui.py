#!/usr/bin/env python3
import os
import queue
import threading
import tkinter as tk
from typing import List
from tkinter import filedialog, messagebox, ttk

import wav_inspector


class WavInspectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WAV Inspector")
        self.geometry("860x700")
        self.minsize(780, 580)

        self._log_queue = queue.Queue()
        self._running = False

        self.input_path_var = tk.StringVar(value="")
        self.output_root_var = tk.StringVar(value="")
        self.bandpass_var = tk.BooleanVar(value=True)
        self.recursive_var = tk.BooleanVar(value=False)
        self.peaks_var = tk.IntVar(value=10)
        self.channel_var = tk.StringVar(value="auto")
        self.env_cutoff_var = tk.DoubleVar(value=wav_inspector.DEFAULT_ENV_CUTOFF)
        self.env_max_freq_var = tk.DoubleVar(value=wav_inspector.DEFAULT_ENV_MAX_FREQ)
        self.env_detrend_cutoff_var = tk.DoubleVar(value=wav_inspector.DEFAULT_ENV_TREND_CUTOFF)
        self.env_norm_var = tk.StringVar(value=wav_inspector.DEFAULT_ENV_NORM_MODE)
        self.center_duration_var = tk.DoubleVar(value=wav_inspector.DEFAULT_CENTER_DURATION)
        self.decimate_k_var = tk.IntVar(value=wav_inspector.DEFAULT_DECIMATE_K)
        self.export_slices_var = tk.BooleanVar(value=wav_inspector.DEFAULT_EXPORT_SLICES)
        self.window_size_var = tk.DoubleVar(value=wav_inspector.DEFAULT_WINDOW_SIZE)
        self.hop_size_var = tk.DoubleVar(value=wav_inspector.DEFAULT_HOP_SIZE)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.after(150, self._drain_log_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Input (file or folder)").grid(row=0, column=0, sticky="w")
        input_entry = ttk.Entry(main, textvariable=self.input_path_var)
        input_entry.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(main, text="Choose File", command=self._choose_file).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(main, text="Choose Folder", command=self._choose_folder).grid(row=0, column=3)

        ttk.Label(main, text="Output root (subfolder per file)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        output_entry = ttk.Entry(main, textvariable=self.output_root_var)
        output_entry.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(main, text="Choose Output", command=self._choose_output).grid(row=1, column=2, columnspan=2, pady=(8, 0), sticky="w")

        options = ttk.LabelFrame(main, text="Options", padding=10)
        options.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        for col in range(4):
            options.columnconfigure(col, weight=1)

        ttk.Checkbutton(options, text="Enable bandpass", variable=self.bandpass_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options, text="Include subfolders", variable=self.recursive_var).grid(row=0, column=1, sticky="w")

        ttk.Label(options, text="Channel").grid(row=1, column=0, sticky="w", pady=(8, 0))
        channel_combo = ttk.Combobox(
            options,
            textvariable=self.channel_var,
            values=list(wav_inspector.CHANNEL_MODES),
            state="readonly",
            width=10,
        )
        channel_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(options, text="FFT peaks").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(options, from_=1, to=100, increment=1, textvariable=self.peaks_var, width=8).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(options, text="Envelope cutoff (Hz)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.0,
            to=500.0,
            increment=1.0,
            textvariable=self.env_cutoff_var,
            width=10,
        ).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(options, text="Envelope max freq (Hz)").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.0,
            to=500.0,
            increment=1.0,
            textvariable=self.env_max_freq_var,
            width=10,
        ).grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Label(options, text="Envelope detrend cutoff (Hz)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.0,
            to=50.0,
            increment=0.1,
            textvariable=self.env_detrend_cutoff_var,
            width=10,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(options, text="Envelope normalize").grid(row=3, column=2, sticky="w", pady=(8, 0))
        env_norm_combo = ttk.Combobox(
            options,
            textvariable=self.env_norm_var,
            values=list(wav_inspector.ENV_NORM_MODES),
            state="readonly",
            width=14,
        )
        env_norm_combo.grid(row=3, column=3, sticky="w", pady=(8, 0))

        ttk.Label(options, text="Decimate k").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=1,
            to=1000,
            increment=1,
            textvariable=self.decimate_k_var,
            width=10,
        ).grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Label(options, text="Center duration (s)").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.0,
            to=120.0,
            increment=0.5,
            textvariable=self.center_duration_var,
            width=10,
        ).grid(row=4, column=3, sticky="w", pady=(8, 0))

        ttk.Checkbutton(options, text="Export slices", variable=self.export_slices_var).grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(options, text="Window size (s)").grid(row=5, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.1,
            to=10.0,
            increment=0.1,
            textvariable=self.window_size_var,
            width=10,
        ).grid(row=5, column=3, sticky="w", pady=(8, 0))

        ttk.Label(options, text="Hop size (s)").grid(row=6, column=2, sticky="w", pady=(8, 0))
        ttk.Spinbox(
            options,
            from_=0.05,
            to=10.0,
            increment=0.05,
            textvariable=self.hop_size_var,
            width=10,
        ).grid(row=6, column=3, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(main)
        action_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        action_frame.columnconfigure(0, weight=1)
        self.process_btn = ttk.Button(action_frame, text="Process", command=self._start_processing)
        self.process_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(action_frame, text="Clear Log", command=self._clear_log).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(main, text="Log").grid(row=4, column=0, sticky="w", pady=(12, 0))
        log_frame = ttk.Frame(main)
        log_frame.grid(row=5, column=0, columnspan=4, sticky="nsew")
        main.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        status_bar = ttk.Label(main, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(8, 0))

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self.input_path_var.set(path)

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.input_path_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_root_var.set(path)

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.process_btn.configure(state="disabled" if running else "normal")
        self.status_var.set("Processing..." if running else "Ready")

    def _start_processing(self) -> None:
        if self._running:
            return
        path = self.input_path_var.get().strip()
        if not path:
            messagebox.showwarning("Missing input", "Please choose a file or folder.")
            return
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"Path does not exist:\n{path}")
            return
        try:
            peaks = int(self.peaks_var.get())
            env_cutoff = float(self.env_cutoff_var.get())
            env_max_freq = float(self.env_max_freq_var.get())
            env_detrend_cutoff = float(self.env_detrend_cutoff_var.get())
            decimate_k = int(self.decimate_k_var.get())
            center_duration = float(self.center_duration_var.get())
            window_size = float(self.window_size_var.get())
            hop_size = float(self.hop_size_var.get())
        except Exception:
            messagebox.showerror("Invalid options", "Options must be numeric values.")
            return
        peaks = max(1, peaks)
        env_cutoff = max(0.0, env_cutoff)
        env_max_freq = max(0.0, env_max_freq)
        env_detrend_cutoff = max(0.0, env_detrend_cutoff)
        decimate_k = max(1, decimate_k)
        center_duration = max(0.0, center_duration)
        window_size = max(0.0, window_size)
        hop_size = max(0.0, hop_size)
        channel_mode = (self.channel_var.get() or "auto").lower()
        if channel_mode not in wav_inspector.CHANNEL_MODES:
            channel_mode = "auto"
        env_norm_mode = (self.env_norm_var.get() or wav_inspector.DEFAULT_ENV_NORM_MODE).lower()
        if env_norm_mode not in wav_inspector.ENV_NORM_MODES:
            env_norm_mode = wav_inspector.DEFAULT_ENV_NORM_MODE
        export_slices = bool(self.export_slices_var.get())

        output_root = self.output_root_var.get().strip()
        bandpass = bool(self.bandpass_var.get())
        recursive = bool(self.recursive_var.get())

        self._set_running(True)
        self._append_log("Starting processing...")

        worker = threading.Thread(
            target=self._run_processing,
            args=(
                path,
                output_root,
                peaks,
                bandpass,
                recursive,
                channel_mode,
                env_cutoff,
                env_max_freq,
                env_detrend_cutoff,
                env_norm_mode,
                decimate_k,
                center_duration,
                export_slices,
                window_size,
                hop_size,
            ),
            daemon=True,
        )
        worker.start()

    def _collect_wavs(self, folder: str, recursive: bool) -> List[str]:
        wavs: List[str] = []
        if recursive:
            for root, dirs, files in os.walk(folder):
                dirs.sort()
                for name in sorted(files):
                    if name.lower().endswith(".wav"):
                        wavs.append(os.path.join(root, name))
        else:
            for name in sorted(os.listdir(folder)):
                full = os.path.join(folder, name)
                if os.path.isfile(full) and name.lower().endswith(".wav"):
                    wavs.append(full)
        return wavs

    def _run_processing(
        self,
        input_path: str,
        output_root: str,
        peaks: int,
        bandpass: bool,
        recursive: bool,
        channel_mode: str,
        env_cutoff: float,
        env_max_freq: float,
        env_detrend_cutoff: float,
        env_norm_mode: str,
        decimate_k: int,
        center_duration: float,
        export_slices: bool,
        window_size: float,
        hop_size: float,
    ) -> None:
        success = 0
        failures = 0
        last_out_dir = ""

        try:
            if os.path.isfile(input_path):
                wav_files = [input_path]
                if not input_path.lower().endswith(".wav"):
                    self._log_queue.put(("log", f"Warning: {input_path} does not end with .wav"))
                root = output_root or ""
            else:
                wav_files = self._collect_wavs(input_path, recursive)
                if not wav_files:
                    self._log_queue.put(("log", "No .wav files found in the selected folder."))
                    self._log_queue.put(("done", {"success": 0, "failures": 0, "last_out_dir": ""}))
                    return
                if output_root:
                    root = output_root
                else:
                    folder_name = os.path.basename(os.path.abspath(input_path))
                    root = os.path.join(os.getcwd(), "wav_views", folder_name)

            if root:
                os.makedirs(root, exist_ok=True)

            total = len(wav_files)
            for idx, wav_path in enumerate(wav_files, start=1):
                stem = os.path.splitext(os.path.basename(wav_path))[0]
                out_dir = os.path.join(root, stem) if root else None
                self._log_queue.put(("status", f"Processing {idx}/{total}: {os.path.basename(wav_path)}"))
                try:
                    out_dir_actual = wav_inspector.process_wav(
                        wav_path,
                        out_dir=out_dir,
                        apply_bandpass=bandpass,
                        peaks=peaks,
                        channel_mode=channel_mode,
                        env_cutoff=env_cutoff,
                        env_max_freq=env_max_freq,
                        env_detrend_cutoff=env_detrend_cutoff,
                        env_norm_mode=env_norm_mode,
                        decimate_k=decimate_k,
                        center_duration_s=center_duration,
                        export_slices=export_slices,
                        window_size_s=window_size,
                        hop_size_s=hop_size,
                    )
                    last_out_dir = out_dir_actual
                    success += 1
                    self._log_queue.put(("log", f"[OK] {wav_path} -> {out_dir_actual}"))
                except Exception as exc:
                    failures += 1
                    self._log_queue.put(("log", f"[ERROR] {wav_path}: {exc}"))

            self._log_queue.put(("done", {"success": success, "failures": failures, "last_out_dir": last_out_dir}))
        except Exception as exc:
            self._log_queue.put(("log", f"[FATAL] {exc}"))
            self._log_queue.put(("done", {"success": success, "failures": failures, "last_out_dir": last_out_dir}))

    def _drain_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self._log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "status":
                self.status_var.set(str(payload))
            elif kind == "done":
                self._set_running(False)
                summary = payload if isinstance(payload, dict) else {}
                success = summary.get("success", 0)
                failures = summary.get("failures", 0)
                last_out_dir = summary.get("last_out_dir", "")
                msg = f"Done. Success: {success}, Failures: {failures}."
                if last_out_dir:
                    msg += f"\nLast output: {last_out_dir}"
                messagebox.showinfo("Completed", msg)
        self.after(150, self._drain_log_queue)


if __name__ == "__main__":
    app = WavInspectorApp()
    app.mainloop()
