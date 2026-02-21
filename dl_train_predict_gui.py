#!/usr/bin/env python3
import csv
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import dl_train_predict

LABEL_OPTIONS = ["shape", "speed", "material"]
SOURCE_OPTIONS = ["raw", "envelope", "envelope_detrended"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]


class ScrollablePanel(ttk.Frame):
    def __init__(self, parent, bg_color="#2b2b2b", *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=bg_color)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        
        self.content.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas.bind("<Configure>", self._on_resize)
        
        # Bind mousewheel to canvas when mouse enters the widget
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_resize(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)
        
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        
    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        
    def _on_mousewheel(self, event):
        if self.canvas.winfo_exists():
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")
            else:
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class TrainPredictApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Deep Learning - AI Rotation Analysis")
        self.geometry("1200x800")
        self.minsize(800, 600)

        self._init_styles()

        self._train_queue = queue.Queue()
        self._predict_queue = queue.Queue()
        self._training = False
        self._predicting = False

        cwd = os.getcwd()
        self.train_label_var = tk.StringVar(value="shape")
        self.train_source_var = tk.StringVar(value="envelope_detrended")
        self.train_use_slices_var = tk.BooleanVar(value=True)
        self.train_augment_var = tk.BooleanVar(value=True)
        self.train_amp_var = tk.BooleanVar(value=True)
        self.train_cache_var = tk.BooleanVar(value=False)
        self.train_epochs_var = tk.IntVar(value=80)
        self.train_batch_var = tk.IntVar(value=32)
        self.train_seq_len_var = tk.IntVar(value=2048)
        self.train_base_ch_var = tk.IntVar(value=32)
        self.train_blocks_var = tk.IntVar(value=4)
        self.train_kernel_var = tk.IntVar(value=7)
        self.train_dropout_var = tk.DoubleVar(value=0.1)
        self.train_lr_var = tk.DoubleVar(value=1e-3)
        self.train_weight_decay_var = tk.DoubleVar(value=1e-4)
        self.train_patience_var = tk.IntVar(value=10)
        self.train_scheduler_patience_var = tk.IntVar(value=3)
        self.train_grad_clip_var = tk.DoubleVar(value=1.0)
        self.train_device_var = tk.StringVar(value="auto")
        self.train_workers_var = tk.IntVar(value=2)
        self.train_seed_var = tk.IntVar(value=42)
        self.train_test_ratio_var = tk.DoubleVar(value=0.2)
        self.train_output_var = tk.StringVar(value=os.path.join(cwd, "train_models"))
        self.train_mode_var = tk.StringVar(value="explicit")
        self.train_status_var = tk.StringVar(value="Ready")
        self.train_progress_var = tk.DoubleVar(value=0.0)

        self.predict_label_var = tk.StringVar(value="shape")
        self.predict_model_var = tk.StringVar(value="")
        self.predict_output_var = tk.StringVar(value=os.path.join(cwd, "predict_outputs"))
        self.predict_status_var = tk.StringVar(value="Ready")
        self.predict_model_info_var = tk.StringVar(value="No model selected")
        self.predict_device_var = tk.StringVar(value="auto")
        self.predict_amp_var = tk.BooleanVar(value=True)
        self.predict_workers_var = tk.IntVar(value=2)
        self.predict_progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.after(150, self._drain_train_queue)
        self.after(150, self._drain_predict_queue)

    def _init_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        bg_main = "#2b2b2b"
        bg_panel = "#333333"
        fg_main = "#ffffff"
        fg_dim = "#aaaaaa"
        accent = "#3794ff"
        input_bg = "#1e1e1e"
        border_col = "#454545"
        select_bg = "#204a87"

        self.configure(bg=bg_main)

        style.configure(".", background=bg_main, foreground=fg_main, borderwidth=0, font=("Segoe UI", 10))
        style.configure("TFrame", background=bg_main)
        style.configure("TLabelframe", background=bg_main, bordercolor=border_col, borderwidth=1, lightcolor=border_col, darkcolor=border_col)
        style.configure("TLabelframe.Label", background=bg_main, foreground=accent, font=("Segoe UI", 10, "bold"))

        style.configure(
            "TButton",
            background=bg_panel,
            foreground=fg_main,
            bordercolor=border_col,
            lightcolor=bg_panel,
            darkcolor=bg_panel,
            padding=6,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TButton",
            background=[("pressed", select_bg), ("active", "#404040")],
            foreground=[("pressed", "white"), ("active", accent)],
            bordercolor=[("active", accent)],
        )

        style.configure("TEntry", fieldbackground=input_bg, foreground=fg_main, insertcolor=fg_main, bordercolor=border_col, lightcolor=border_col, darkcolor=border_col, padding=5)
        style.configure("TSpinbox", fieldbackground=input_bg, foreground=fg_main, insertcolor=fg_main, bordercolor=border_col, arrowcolor=accent)
        style.configure("TCombobox", fieldbackground=input_bg, foreground=fg_main, arrowcolor=accent, bordercolor=border_col, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", input_bg)], selectbackground=[("readonly", select_bg)])

        style.configure("TNotebook", background=bg_main, tabposition="n", borderwidth=0)
        style.configure("TNotebook.Tab", background=bg_panel, foreground=fg_dim, padding=[15, 8], font=("Segoe UI", 11), borderwidth=0)
        style.map(
            "TNotebook.Tab",
            background=[("selected", bg_main), ("active", "#383838")],
            foreground=[("selected", accent), ("active", fg_main)],
            expand=[("selected", [0, 2, 0, 0])],
        )

        style.configure(
            "Treeview",
            background=input_bg,
            foreground=fg_main,
            fieldbackground=input_bg,
            rowheight=32,
            font=("Segoe UI", 10),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=bg_panel,
            foreground=fg_main,
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padding=[5, 5],
        )
        style.map("Treeview", background=[("selected", select_bg)], foreground=[("selected", "white")])

        style.configure("Vertical.TScrollbar", troughcolor=bg_main, background=bg_panel, arrowcolor=fg_dim, bordercolor=bg_main, relief="flat")
        style.configure("Horizontal.TScrollbar", troughcolor=bg_main, background=bg_panel, arrowcolor=fg_dim, bordercolor=bg_main, relief="flat")
        style.map("Vertical.TScrollbar", background=[("active", "#505050")])
        style.map("Horizontal.TScrollbar", background=[("active", "#505050")])

        style.configure("TCheckbutton", background=bg_main, foreground=fg_main, indicatorbackground=input_bg, indicatorforeground=accent)
        style.map("TCheckbutton", indicatorbackground=[("selected", accent)])
        style.configure("TRadiobutton", background=bg_main, foreground=fg_main, indicatorbackground=input_bg, indicatorforeground=accent)
        style.map("TRadiobutton", indicatorbackground=[("selected", accent)])

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create scrollable wrappers
        train_wrap = ScrollablePanel(notebook, bg_color="#2b2b2b")
        predict_wrap = ScrollablePanel(notebook, bg_color="#2b2b2b")
        
        notebook.add(train_wrap, text="Train")
        notebook.add(predict_wrap, text="Predict")

        # Build content inside the inner frame of the scrollable panel
        self._build_train_tab(train_wrap.content)
        self._build_predict_tab(predict_wrap.content)

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        tips = ttk.Label(
            parent,
            text="Tip: select multiple files/folders; sample folders are detected by summary.csv.",
            font=("Segoe UI", 10, "italic"),
        )
        tips.grid(row=0, column=0, sticky="w", pady=(0, 10))

        basic = ttk.LabelFrame(parent, text="Configuration", padding=15)
        basic.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            basic.columnconfigure(col, weight=1)

        ttk.Label(basic, text="Target Label (speed=rad/s):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            basic,
            textvariable=self.train_label_var,
            values=LABEL_OPTIONS,
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(basic, text="Data Source:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            basic,
            textvariable=self.train_source_var,
            values=SOURCE_OPTIONS,
            state="readonly",
            width=28,
        ).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Checkbutton(basic, text="Use Slice Aggregation", variable=self.train_use_slices_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Checkbutton(basic, text="Enable Augmentation", variable=self.train_augment_var).grid(
            row=1, column=2, columnspan=2, sticky="w", pady=(10, 0)
        )

        ttk.Checkbutton(basic, text="Use AMP (mixed precision)", variable=self.train_amp_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Checkbutton(basic, text="Cache CSV in RAM", variable=self.train_cache_var).grid(
            row=2, column=2, columnspan=2, sticky="w", pady=(10, 0)
        )

        advanced = ttk.LabelFrame(parent, text="Hyperparameters", padding=15)
        advanced.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        for col in range(6):
            advanced.columnconfigure(col, weight=1)

        ttk.Label(advanced, text="Epochs:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(advanced, from_=1, to=9999, textvariable=self.train_epochs_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(5, 20)
        )

        ttk.Label(advanced, text="Batch Size:").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(advanced, from_=1, to=512, textvariable=self.train_batch_var, width=10).grid(
            row=0, column=3, sticky="w", padx=(5, 20)
        )

        ttk.Label(advanced, text="Seq Length:").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(advanced, from_=256, to=20000, textvariable=self.train_seq_len_var, width=10).grid(
            row=0, column=5, sticky="w", padx=(5, 0)
        )

        ttk.Label(advanced, text="Learning Rate:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(advanced, textvariable=self.train_lr_var, width=12).grid(row=1, column=1, sticky="w", padx=(5, 20), pady=(10, 0))

        ttk.Label(advanced, text="Weight Decay:").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(advanced, textvariable=self.train_weight_decay_var, width=12).grid(row=1, column=3, sticky="w", padx=(5, 20), pady=(10, 0))

        ttk.Label(advanced, text="Early Stop Patience:").grid(row=1, column=4, sticky="w", pady=(10, 0))
        ttk.Spinbox(advanced, from_=0, to=200, textvariable=self.train_patience_var, width=10).grid(
            row=1, column=5, sticky="w", padx=(5, 0), pady=(10, 0)
        )

        ttk.Label(advanced, text="Grad Clip:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(advanced, textvariable=self.train_grad_clip_var, width=12).grid(
            row=2, column=1, sticky="w", padx=(5, 20), pady=(10, 0)
        )

        ttk.Label(advanced, text="Device:").grid(row=2, column=2, sticky="w", pady=(10, 0))
        ttk.Combobox(advanced, textvariable=self.train_device_var, values=DEVICE_OPTIONS, state="readonly", width=10).grid(
            row=2, column=3, sticky="w", padx=(5, 20), pady=(10, 0)
        )

        ttk.Label(advanced, text="Workers:").grid(row=2, column=4, sticky="w", pady=(10, 0))
        ttk.Spinbox(advanced, from_=0, to=16, textvariable=self.train_workers_var, width=10).grid(
            row=2, column=5, sticky="w", padx=(5, 0), pady=(10, 0)
        )

        ttk.Label(advanced, text="Base Channels:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(advanced, from_=4, to=1024, textvariable=self.train_base_ch_var, width=10).grid(
            row=3, column=1, sticky="w", padx=(5, 20), pady=(10, 0)
        )

        ttk.Label(advanced, text="Blocks:").grid(row=3, column=2, sticky="w", pady=(10, 0))
        ttk.Spinbox(advanced, from_=1, to=20, textvariable=self.train_blocks_var, width=10).grid(
            row=3, column=3, sticky="w", padx=(5, 20), pady=(10, 0)
        )

        ttk.Label(advanced, text="Kernel Size:").grid(row=3, column=4, sticky="w", pady=(10, 0))
        ttk.Spinbox(advanced, from_=1, to=99, textvariable=self.train_kernel_var, width=10).grid(
            row=3, column=5, sticky="w", padx=(5, 0), pady=(10, 0)
        )

        ttk.Label(advanced, text="Dropout:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(advanced, textvariable=self.train_dropout_var, width=12).grid(
            row=4, column=1, sticky="w", padx=(5, 20), pady=(10, 0)
        )

        ttk.Label(advanced, text="LR Scheduler Patience:").grid(row=4, column=2, sticky="w", pady=(10, 0))
        ttk.Spinbox(
            advanced,
            from_=0,
            to=200,
            textvariable=self.train_scheduler_patience_var,
            width=10,
        ).grid(row=4, column=3, sticky="w", padx=(5, 20), pady=(10, 0))

        misc = ttk.LabelFrame(parent, text="Output & Seed", padding=15)
        misc.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            misc.columnconfigure(col, weight=1)

        ttk.Label(misc, text="Random Seed:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(misc, from_=0, to=999999, textvariable=self.train_seed_var, width=12).grid(
            row=0, column=1, sticky="w", padx=(5, 20)
        )

        ttk.Label(misc, text="Output Folder:").grid(row=0, column=2, sticky="w")
        out_entry = ttk.Entry(misc, textvariable=self.train_output_var)
        out_entry.grid(row=0, column=3, sticky="ew", padx=(5, 0))
        ttk.Button(misc, text="Browse...", command=self._choose_train_output).grid(
            row=1, column=3, sticky="e", pady=(5, 0)
        )

        mode_frame = ttk.LabelFrame(parent, text="Data Selection Mode", padding=15)
        mode_frame.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        ttk.Radiobutton(
            mode_frame,
            text="Manual Train/Test Split",
            variable=self.train_mode_var,
            value="explicit",
            command=self._toggle_train_mode,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame,
            text="Auto Split (All Samples)",
            variable=self.train_mode_var,
            value="auto",
            command=self._toggle_train_mode,
        ).grid(row=0, column=1, sticky="w", padx=(30, 0))

        self.explicit_frame = ttk.Frame(parent)
        self.explicit_frame.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        self.explicit_frame.columnconfigure(0, weight=1)
        self.explicit_frame.columnconfigure(1, weight=1)

        self.train_listbox = self._build_path_box(self.explicit_frame, "Train Paths", 0)
        self.test_listbox = self._build_path_box(self.explicit_frame, "Test Paths (Optional)", 1)

        self.auto_frame = ttk.Frame(parent)
        self.auto_frame.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        self.auto_frame.columnconfigure(0, weight=1)
        auto_opts = ttk.Frame(self.auto_frame)
        auto_opts.grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Label(auto_opts, text="Test Ratio (0.0-1.0):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            auto_opts,
            from_=0.0,
            to=0.9,
            increment=0.05,
            textvariable=self.train_test_ratio_var,
            width=12,
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.all_listbox = self._build_path_box(self.auto_frame, "All Sample Paths", 0, row=1, colspan=1)

        actions = ttk.Frame(parent)
        actions.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        self.train_btn = ttk.Button(actions, text="Start Training", command=self._start_training)
        self.train_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Clear Log", command=self._clear_train_log).grid(row=0, column=1, sticky="w", padx=(15, 0))

        log_frame = ttk.LabelFrame(parent, text="Training Log", padding=10)
        log_frame.grid(row=7, column=0, sticky="nsew")
        parent.rowconfigure(7, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.train_log = tk.Text(log_frame, height=10, bg="#1e1e1e", fg="#f0f0f0", relief="flat")
        self.train_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.train_log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.train_log.configure(yscrollcommand=log_scroll.set)

        progress = ttk.Progressbar(
            parent,
            variable=self.train_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        progress.grid(row=8, column=0, sticky="ew", pady=(6, 0))

        status = ttk.Label(parent, textvariable=self.train_status_var, relief="sunken", anchor="w", padding=5)
        status.grid(row=9, column=0, sticky="ew")

        self._toggle_train_mode()

    def _build_predict_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        basic = ttk.LabelFrame(parent, text="Configuration", padding=15)
        basic.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for col in range(4):
            basic.columnconfigure(col, weight=1)

        ttk.Label(basic, text="Target Label (speed=rad/s):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            basic,
            textvariable=self.predict_label_var,
            values=LABEL_OPTIONS,
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(basic, text="Device:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            basic,
            textvariable=self.predict_device_var,
            values=DEVICE_OPTIONS,
            state="readonly",
            width=15,
        ).grid(row=0, column=3, sticky="w", padx=(5, 0))

        ttk.Checkbutton(basic, text="Use AMP", variable=self.predict_amp_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Label(basic, text="Workers:").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Spinbox(basic, from_=0, to=16, textvariable=self.predict_workers_var, width=10).grid(
            row=1, column=3, sticky="w", padx=(5, 0), pady=(10, 0)
        )

        model_frame = ttk.LabelFrame(parent, text="Model", padding=15)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(model_frame, textvariable=self.predict_model_var).grid(row=0, column=1, sticky="ew", padx=(5, 10))
        ttk.Button(model_frame, text="Browse...", command=self._choose_predict_model).grid(row=0, column=2, sticky="e")
        ttk.Label(model_frame, textvariable=self.predict_model_info_var, foreground="#9cdcfe").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        data_frame = ttk.LabelFrame(parent, text="Prediction Data", padding=10)
        data_frame.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        data_frame.columnconfigure(0, weight=1)
        self.predict_listbox = self._build_path_box(data_frame, "Data Paths", 0, compact=True)

        out_frame = ttk.LabelFrame(parent, text="Output", padding=10)
        out_frame.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        out_frame.columnconfigure(1, weight=1)
        ttk.Label(out_frame, text="Output Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(out_frame, textvariable=self.predict_output_var).grid(row=0, column=1, sticky="ew", padx=(5, 10))
        ttk.Button(out_frame, text="Browse...", command=self._choose_predict_output).grid(row=0, column=2, sticky="e")

        actions = ttk.Frame(parent)
        actions.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.predict_btn = ttk.Button(actions, text="Start Prediction", command=self._start_predict)
        self.predict_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Clear Log", command=self._clear_predict_log).grid(row=0, column=1, sticky="w", padx=(15, 0))

        results_frame = ttk.LabelFrame(parent, text="Predictions", padding=10)
        results_frame.grid(row=5, column=0, sticky="nsew")
        parent.rowconfigure(5, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        columns = ("sample_id", "predicted", "actual")
        self.predict_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)
        for col in columns:
            self.predict_tree.heading(col, text=col)
            self.predict_tree.column(col, width=200, anchor="w")
        self.predict_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.predict_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.predict_tree.configure(yscrollcommand=scroll.set)

        log_frame = ttk.LabelFrame(parent, text="Prediction Log", padding=10)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(10, 0))
        parent.rowconfigure(6, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.predict_log = tk.Text(log_frame, height=8, bg="#1e1e1e", fg="#f0f0f0", relief="flat")
        self.predict_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.predict_log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.predict_log.configure(yscrollcommand=log_scroll.set)

        progress = ttk.Progressbar(
            parent,
            variable=self.predict_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        progress.grid(row=7, column=0, sticky="ew", pady=(6, 0))

        status = ttk.Label(parent, textvariable=self.predict_status_var, relief="sunken", anchor="w", padding=5)
        status.grid(row=8, column=0, sticky="ew")

    def _build_path_box(
        self,
        parent: ttk.Frame,
        title: str,
        column: int,
        row: int = 0,
        colspan: int = 1,
        compact: bool = False,
    ) -> tk.Listbox:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=row, column=column, columnspan=colspan, sticky="ew", padx=(0, 15), pady=(0, 5))
        frame.columnconfigure(0, weight=1)

        listbox = tk.Listbox(
            frame,
            height=4 if compact else 6,
            selectmode="extended",
            bg="#1e1e1e",
            fg="#f0f0f0",
            selectbackground="#204a87",
            selectforeground="white",
            relief="flat",
            font=("Segoe UI", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        listbox.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        scroll = ttk.Scrollbar(frame, orient="vertical", command=listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        listbox.configure(yscrollcommand=scroll.set)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(btn_frame, text="Add Files...", command=lambda: self._add_files(listbox)).grid(row=0, column=0, sticky="w")
        ttk.Button(btn_frame, text="Add Folder...", command=lambda: self._add_folder(listbox)).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ttk.Button(btn_frame, text="Remove Selected", command=lambda: self._remove_selected(listbox)).grid(row=0, column=2, sticky="w", padx=(10, 0))
        ttk.Button(btn_frame, text="Clear All", command=lambda: self._clear_listbox(listbox)).grid(row=0, column=3, sticky="w", padx=(10, 0))
        return listbox

    def _toggle_train_mode(self) -> None:
        if self.train_mode_var.get() == "auto":
            self.explicit_frame.grid_remove()
            self.auto_frame.grid()
        else:
            self.auto_frame.grid_remove()
            self.explicit_frame.grid()

    def _choose_train_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.train_output_var.set(path)

    def _choose_predict_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.predict_output_var.set(path)

    def _choose_predict_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt"), ("All files", "*.*")])
        if not path:
            return
        self.predict_model_var.set(path)
        self._load_model_info(path)

    def _load_model_info(self, path: str) -> None:
        try:
            checkpoint = dl_train_predict._load_checkpoint(path)
            meta = checkpoint.get("metadata", {})
            label = meta.get("label_display") or meta.get("label_target", "")
            source = meta.get("source", "")
            task = meta.get("task", "")
            seq_len = meta.get("seq_len", "")
            info = f"label={label or 'unknown'} | source={source or 'unknown'} | task={task or 'unknown'} | seq={seq_len or 'unknown'}"
        except Exception as exc:
            info = f"Failed to read model: {exc}"
        self.predict_model_info_var.set(info)

    def _add_files(self, listbox: tk.Listbox) -> None:
        paths = filedialog.askopenfilenames()
        if paths:
            self._insert_paths(listbox, paths)

    def _add_folder(self, listbox: tk.Listbox) -> None:
        path = filedialog.askdirectory()
        if path:
            self._insert_paths(listbox, [path])

    def _insert_paths(self, listbox: tk.Listbox, paths) -> None:
        existing = set(listbox.get(0, tk.END))
        for path in paths:
            path = path.strip()
            if path and path not in existing:
                listbox.insert(tk.END, path)
                existing.add(path)

    def _remove_selected(self, listbox: tk.Listbox) -> None:
        indices = listbox.curselection()
        for idx in reversed(indices):
            listbox.delete(idx)

    def _clear_listbox(self, listbox: tk.Listbox) -> None:
        listbox.delete(0, tk.END)

    def _listbox_items(self, listbox: tk.Listbox) -> list:
        return list(listbox.get(0, tk.END))

    def _start_training(self) -> None:
        if self._training:
            return
        label_target = self.train_label_var.get().strip()
        source = self.train_source_var.get().strip()
        if not label_target or not source:
            messagebox.showerror("Invalid settings", "Please choose a valid label and source.")
            return
        try:
            test_ratio = float(self.train_test_ratio_var.get())
        except Exception:
            messagebox.showerror("Invalid settings", "Test ratio must be a number.")
            return
        if test_ratio < 0 or test_ratio >= 1:
            messagebox.showerror("Invalid settings", "Test ratio must be in [0, 1).")
            return
        try:
            seed = int(self.train_seed_var.get())
        except Exception:
            messagebox.showerror("Invalid settings", "Seed must be an integer.")
            return
        out_dir = self.train_output_var.get().strip()
        if not out_dir:
            messagebox.showerror("Missing output", "Please choose an output folder.")
            return

        mode = self.train_mode_var.get()
        if mode == "auto":
            all_paths = self._listbox_items(self.all_listbox)
            if not all_paths:
                messagebox.showwarning("Missing data", "Please add all sample paths.")
                return
            train_paths = []
            test_paths = all_paths
        else:
            train_paths = self._listbox_items(self.train_listbox)
            test_paths = self._listbox_items(self.test_listbox)
            if not train_paths:
                messagebox.showwarning("Missing data", "Please add train paths.")
                return

        self._training = True
        self.train_btn.configure(state="disabled")
        self.train_status_var.set("Training...")
        self.train_progress_var.set(0.0)
        self._append_train_log("Training started. Resolving sample folders...")

        worker = threading.Thread(
            target=self._run_training,
            args=(
                train_paths,
                test_paths,
                label_target,
                source,
                bool(self.train_use_slices_var.get()),
                test_ratio,
                seed,
                out_dir,
            ),
            daemon=True,
        )
        worker.start()

    def _run_training(
        self,
        train_paths: list,
        test_paths: list,
        label_target: str,
        source: str,
        use_slices: bool,
        test_ratio: float,
        seed: int,
        out_dir: str,
    ) -> None:
        try:
            train_dirs = dl_train_predict.collect_sample_dirs(train_paths)
            test_dirs = dl_train_predict.collect_sample_dirs(test_paths)
            if not train_dirs and not test_dirs:
                self._train_queue.put(("error", "No valid sample folders found."))
                return
            self._train_queue.put(("log", f"Train folders: {len(train_dirs)} | Test folders: {len(test_dirs)}"))
            def _progress_callback(payload: dict) -> None:
                self._train_queue.put(("progress", payload))
            result = dl_train_predict.train_and_save(
                train_dirs,
                test_dirs,
                label_target=label_target,
                source=source,
                use_slices=use_slices,
                test_ratio=test_ratio,
                seed=seed,
                out_dir=out_dir,
                epochs=int(self.train_epochs_var.get()),
                batch_size=int(self.train_batch_var.get()),
                seq_len=int(self.train_seq_len_var.get()),
                base_ch=int(self.train_base_ch_var.get()),
                blocks=int(self.train_blocks_var.get()),
                kernel=int(self.train_kernel_var.get()),
                dropout=float(self.train_dropout_var.get()),
                lr=float(self.train_lr_var.get()),
                weight_decay=float(self.train_weight_decay_var.get()),
                patience=int(self.train_patience_var.get()),
                scheduler_patience=int(self.train_scheduler_patience_var.get()),
                device=str(self.train_device_var.get()),
                num_workers=int(self.train_workers_var.get()),
                augment=bool(self.train_augment_var.get()),
                amp=bool(self.train_amp_var.get()),
                grad_clip=float(self.train_grad_clip_var.get()),
                cache=bool(self.train_cache_var.get()),
                progress_callback=_progress_callback,
            )
            self._train_queue.put(("done", result))
        except Exception as exc:
            self._train_queue.put(("error", str(exc)))

    def _drain_train_queue(self) -> None:
        while True:
            try:
                kind, payload = self._train_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_train_log(str(payload))
            elif kind == "progress":
                info = payload or {}
                if info.get("type") == "start":
                    self.train_progress_var.set(0.0)
                elif info.get("type") == "batch":
                    # Smooth progress update including batch
                    epoch = int(info.get("epoch", 1))
                    total_epochs = int(info.get("epochs", 1))
                    batch = int(info.get("batch", 0))
                    total_batches = int(info.get("total_batches", 1))
                    
                    if total_epochs > 0 and total_batches > 0:
                        epoch_progress = (epoch - 1) / total_epochs
                        batch_progress = (batch / total_batches) / total_epochs
                        total_progress = (epoch_progress + batch_progress) * 100.0
                        self.train_progress_var.set(min(100.0, total_progress))
                        
                elif info.get("type") == "epoch":
                    epoch = int(info.get("epoch", 0))
                    total = int(info.get("epochs", 0))
                    # Ensure we hit the exact epoch mark
                    if total > 0:
                        self.train_progress_var.set(min(100.0, (epoch / total) * 100.0))
                    train_loss = float(info.get("train_loss", 0.0))
                    val_loss = float(info.get("val_loss", 0.0))
                    metric = float(info.get("metric", 0.0))
                    self.train_status_var.set(
                        f"Epoch {epoch}/{total} | loss={train_loss:.4f} | val={val_loss:.4f} | metric={metric:.4f}"
                    )
                elif info.get("type") == "done":
                    self.train_progress_var.set(100.0)
            elif kind == "error":
                self._append_train_log(f"[ERROR] {payload}")
                self._finish_training()
                messagebox.showerror("Training failed", str(payload))
            elif kind == "done":
                self._finish_training()
                train_count = len(payload.get("train_ids", []))
                test_count = len(payload.get("test_ids", []))
                best_model = payload.get("best_model", "")
                plots_dir = payload.get("plots_dir", "")
                report_md = payload.get("report_md", "")
                self._append_train_log(f"Training done. Best model: {best_model}")
                if plots_dir:
                    self._append_train_log(f"Plots folder: {plots_dir}")
                if report_md:
                    self._append_train_log(f"Report: {report_md}")
                msg = f"Training done.\nTrain samples: {train_count}  Test samples: {test_count}"
                messagebox.showinfo("Completed", msg)
        self.after(150, self._drain_train_queue)

    def _finish_training(self) -> None:
        self._training = False
        self.train_btn.configure(state="normal")
        self.train_status_var.set("Ready")

    def _append_train_log(self, text: str) -> None:
        self.train_log.configure(state="normal")
        self.train_log.insert(tk.END, text + "\n")
        self.train_log.see(tk.END)
        self.train_log.configure(state="disabled")

    def _clear_train_log(self) -> None:
        self.train_log.configure(state="normal")
        self.train_log.delete("1.0", tk.END)
        self.train_log.configure(state="disabled")

    def _start_predict(self) -> None:
        if self._predicting:
            return
        label_target = self.predict_label_var.get().strip()
        if not label_target:
            messagebox.showerror("Invalid settings", "Please choose a valid label.")
            return
        model_path = self.predict_model_var.get().strip()
        if not model_path:
            messagebox.showwarning("Missing model", "Please choose a model file.")
            return
        if not os.path.isfile(model_path):
            messagebox.showerror("Model not found", f"Model file not found:\n{model_path}")
            return
        data_paths = self._listbox_items(self.predict_listbox)
        if not data_paths:
            messagebox.showwarning("Missing data", "Please add prediction data paths.")
            return
        out_dir = self.predict_output_var.get().strip()
        if not out_dir:
            messagebox.showerror("Missing output", "Please choose an output folder.")
            return

        self._predicting = True
        self.predict_btn.configure(state="disabled")
        self.predict_status_var.set("Predicting...")
        self.predict_progress_var.set(0.0)
        self._append_predict_log("Prediction started. Resolving sample folders...")
        self._clear_predict_results()

        worker = threading.Thread(
            target=self._run_predict,
            args=(model_path, data_paths, out_dir, label_target),
            daemon=True,
        )
        worker.start()

    def _run_predict(self, model_path: str, data_paths: list, out_dir: str, label_target: str) -> None:
        try:
            checkpoint = dl_train_predict._load_checkpoint(model_path)
            meta = checkpoint.get("metadata", {})
            model_label = meta.get("label_target")
            if model_label and model_label != label_target:
                self._predict_queue.put(
                    ("error", f"Model label is {model_label}, but selected label is {label_target}.")
                )
                return
            data_dirs = dl_train_predict.collect_sample_dirs(data_paths)
            if not data_dirs:
                self._predict_queue.put(("error", "No valid sample folders found."))
                return
            self._predict_queue.put(("log", f"Prediction samples: {len(data_dirs)}"))
            
            def _progress_callback(payload: dict) -> None:
                self._predict_queue.put(("progress", payload))

            result = dl_train_predict.predict_with_model(
                model_path,
                data_dirs,
                out_dir,
                device=str(self.predict_device_var.get()),
                num_workers=int(self.predict_workers_var.get()),
                amp=bool(self.predict_amp_var.get()),
                progress_callback=_progress_callback,
            )
            self._predict_queue.put(("done", result))
        except Exception as exc:
            self._predict_queue.put(("error", str(exc)))

    def _drain_predict_queue(self) -> None:
        while True:
            try:
                kind, payload = self._predict_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_predict_log(str(payload))
            elif kind == "progress":
                info = payload or {}
                if info.get("type") == "start":
                    self.predict_progress_var.set(0.0)
                elif info.get("type") == "batch":
                    batch = int(info.get("batch", 0))
                    total = int(info.get("total_batches", 1))
                    if total > 0:
                        self.predict_progress_var.set(min(100.0, (batch / total) * 100.0))
                elif info.get("type") == "done":
                    self.predict_progress_var.set(100.0)
            elif kind == "error":
                self._append_predict_log(f"[ERROR] {payload}")
                self._finish_predict()
                messagebox.showerror("Prediction failed", str(payload))
            elif kind == "done":
                self._populate_predict_results(payload)
                self._finish_predict()
                metrics = payload.get("metrics", {})
                if metrics:
                    self._append_predict_log(f"Metrics: {metrics}")
                messagebox.showinfo("Completed", f"Prediction done. Rows: {payload.get('sample_count', 0)}")
        self.after(150, self._drain_predict_queue)

    def _finish_predict(self) -> None:
        self._predicting = False
        self.predict_btn.configure(state="normal")
        self.predict_status_var.set("Ready")

    def _populate_predict_results(self, payload: dict) -> None:
        pred_csv = payload.get("pred_csv")
        if not pred_csv or not os.path.isfile(pred_csv):
            self._append_predict_log("predictions.csv not found. Cannot display results.")
            return
        with open(pred_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                self.predict_tree.insert("", tk.END, values=(row[0], row[1], row[2]))

    def _clear_predict_results(self) -> None:
        for item in self.predict_tree.get_children():
            self.predict_tree.delete(item)

    def _append_predict_log(self, text: str) -> None:
        self.predict_log.configure(state="normal")
        self.predict_log.insert(tk.END, text + "\n")
        self.predict_log.see(tk.END)
        self.predict_log.configure(state="disabled")

    def _clear_predict_log(self) -> None:
        self.predict_log.configure(state="normal")
        self.predict_log.delete("1.0", tk.END)
        self.predict_log.configure(state="disabled")


if __name__ == "__main__":
    app = TrainPredictApp()
    app.mainloop()
