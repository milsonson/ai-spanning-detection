#!/usr/bin/env python3
import csv
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ml_train_predict

LABEL_OPTIONS = ["shape", "speed", "material"]
SOURCE_OPTIONS = ["raw", "envelope", "envelope_detrended"]


class TrainPredictApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Train & Predict - AI Rotation Analysis")
        self.geometry("1280x900")
        self.minsize(1000, 700)

        self._init_styles()

        self._train_queue = queue.Queue()
        self._predict_queue = queue.Queue()
        self._training = False
        self._predicting = False

        cwd = os.getcwd()
        self.train_label_var = tk.StringVar(value="shape")
        self.train_source_var = tk.StringVar(value="envelope_detrended")
        self.train_use_slices_var = tk.BooleanVar(value=True)
        self.train_seed_var = tk.IntVar(value=42)
        self.train_test_ratio_var = tk.DoubleVar(value=0.2)
        self.train_output_var = tk.StringVar(value=os.path.join(cwd, "train_models"))
        self.train_mode_var = tk.StringVar(value="explicit")
        self.train_status_var = tk.StringVar(value="Ready")

        self.predict_label_var = tk.StringVar(value="shape")
        self.predict_model_var = tk.StringVar(value="")
        self.predict_output_var = tk.StringVar(value=os.path.join(cwd, "predict_outputs"))
        self.predict_status_var = tk.StringVar(value="Ready")
        self.predict_model_info_var = tk.StringVar(value="No model selected")

        self._build_ui()
        self.after(150, self._drain_train_queue)
        self.after(150, self._drain_predict_queue)

    def _init_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        
        # --- Modern Professional Dark Palette ---
        bg_main = "#2b2b2b"       # Main Window Background
        bg_panel = "#333333"      # Secondary Background (frames)
        fg_main = "#ffffff"       # Main Text
        fg_dim = "#aaaaaa"        # Dimmed/Secondary Text
        accent = "#3794ff"        # Professional Blue Accent
        input_bg = "#1e1e1e"      # Dark Input Background
        border_col = "#454545"    # Subtle Borders
        select_bg = "#204a87"     # Selection Background
        
        self.configure(bg=bg_main)
        
        # --- Base Configuration ---
        style.configure(".", background=bg_main, foreground=fg_main, borderwidth=0, font=("Segoe UI", 10))
        
        # --- Frames & Labelframes ---
        style.configure("TFrame", background=bg_main)
        style.configure("TLabelframe", background=bg_main, bordercolor=border_col, borderwidth=1, lightcolor=border_col, darkcolor=border_col)
        style.configure("TLabelframe.Label", background=bg_main, foreground=accent, font=("Segoe UI", 10, "bold"))
        
        # --- Buttons ---
        # Flat, modern buttons with hover effects
        style.configure("TButton", 
            background=bg_panel, 
            foreground=fg_main, 
            bordercolor=border_col, 
            lightcolor=bg_panel, 
            darkcolor=bg_panel,
            padding=6, 
            relief="flat",
            font=("Segoe UI", 10, "bold")
        )
        style.map("TButton",
            background=[("pressed", select_bg), ("active", "#404040")],
            foreground=[("pressed", "white"), ("active", accent)],
            bordercolor=[("active", accent)]
        )
        
        # --- Inputs (Entry, Spinbox, Combobox) ---
        style.configure("TEntry", fieldbackground=input_bg, foreground=fg_main, insertcolor=fg_main, bordercolor=border_col, lightcolor=border_col, darkcolor=border_col, padding=5)
        style.configure("TSpinbox", fieldbackground=input_bg, foreground=fg_main, insertcolor=fg_main, bordercolor=border_col, arrowcolor=accent)
        
        style.configure("TCombobox", fieldbackground=input_bg, foreground=fg_main, arrowcolor=accent, bordercolor=border_col, padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", input_bg)], selectbackground=[("readonly", select_bg)])

        # --- Notebook (Tabs) ---
        style.configure("TNotebook", background=bg_main, tabposition="n", borderwidth=0)
        style.configure("TNotebook.Tab", background=bg_panel, foreground=fg_dim, padding=[15, 8], font=("Segoe UI", 11), borderwidth=0)
        style.map("TNotebook.Tab", 
            background=[("selected", bg_main), ("active", "#383838")], 
            foreground=[("selected", accent), ("active", fg_main)],
            expand=[("selected", [0, 2, 0, 0])] # Slight visual pop for selected tab
        )
        
        # --- Treeview (Tables) ---
        style.configure("Treeview", 
            background=input_bg, 
            foreground=fg_main, 
            fieldbackground=input_bg,
            rowheight=32,
            font=("Segoe UI", 10),
            borderwidth=0
        )
        style.configure("Treeview.Heading", 
            background=bg_panel, 
            foreground=fg_main, 
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padding=[5, 5]
        )
        style.map("Treeview", background=[("selected", select_bg)], foreground=[("selected", "white")])
        
        # --- Scrollbars ---
        # Minimalist scrollbars
        style.configure("Vertical.TScrollbar", troughcolor=bg_main, background=bg_panel, arrowcolor=fg_dim, bordercolor=bg_main, relief="flat")
        style.configure("Horizontal.TScrollbar", troughcolor=bg_main, background=bg_panel, arrowcolor=fg_dim, bordercolor=bg_main, relief="flat")
        style.map("Vertical.TScrollbar", background=[("active", "#505050")])
        style.map("Horizontal.TScrollbar", background=[("active", "#505050")])
        
        # --- Misc ---
        style.configure("TCheckbutton", background=bg_main, foreground=fg_main, indicatorbackground=input_bg, indicatorforeground=accent)
        style.map("TCheckbutton", indicatorbackground=[("selected", accent)])
        style.configure("TRadiobutton", background=bg_main, foreground=fg_main, indicatorbackground=input_bg, indicatorforeground=accent)
        style.map("TRadiobutton", indicatorbackground=[("selected", accent)])

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        train_tab = ttk.Frame(notebook, padding=10)
        predict_tab = ttk.Frame(notebook, padding=10)
        notebook.add(train_tab, text="Train")
        notebook.add(predict_tab, text="Predict")

        self._build_train_tab(train_tab)
        self._build_predict_tab(predict_tab)

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        tips = ttk.Label(
            parent,
            text="Tip: select multiple files/folders; sample folders are detected by summary.csv.",
            font=("Segoe UI", 10, "italic")
        )
        tips.grid(row=0, column=0, sticky="w", pady=(0, 10))

        basic = ttk.LabelFrame(parent, text="Configuration", padding=15)
        basic.grid(row=1, column=0, sticky="ew", pady=(0, 15))
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
            row=1, column=0, columnspan=2, sticky="w", pady=(15, 0)
        )

        ttk.Label(basic, text="Random Seed:").grid(row=1, column=2, sticky="w", pady=(15, 0))
        ttk.Spinbox(basic, from_=0, to=999999, textvariable=self.train_seed_var, width=12).grid(
            row=1, column=3, sticky="w", pady=(15, 0), padx=(5, 0)
        )

        ttk.Label(basic, text="Output Folder:").grid(row=2, column=2, sticky="w", pady=(15, 0))
        out_entry = ttk.Entry(basic, textvariable=self.train_output_var)
        out_entry.grid(row=2, column=3, sticky="ew", pady=(15, 0), padx=(5, 0))
        ttk.Button(basic, text="Browse...", command=self._choose_train_output).grid(
            row=3, column=3, sticky="e", pady=(5, 0)
        )

        mode_frame = ttk.LabelFrame(parent, text="Data Selection Mode", padding=15)
        mode_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
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
        self.explicit_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        self.explicit_frame.columnconfigure(0, weight=1)
        self.explicit_frame.columnconfigure(1, weight=1)

        self.train_listbox = self._build_path_box(self.explicit_frame, "Train Paths", 0)
        self.test_listbox = self._build_path_box(self.explicit_frame, "Test Paths (Optional)", 1)

        self.auto_frame = ttk.Frame(parent)
        self.auto_frame.grid(row=3, column=0, sticky="ew", pady=(0, 15))
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

        self._toggle_train_mode()

        action = ttk.Frame(parent)
        action.grid(row=4, column=0, sticky="ew", pady=(0, 15))
        action.columnconfigure(0, weight=1)
        self.train_btn = ttk.Button(action, text="START TRAINING", command=self._start_training, width=20)
        self.train_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(action, text="Clear Log", command=self._clear_train_log).grid(row=0, column=1, sticky="w", padx=(15, 0))

        log_frame = ttk.LabelFrame(parent, text="Activity Log", padding=10)
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        parent.rowconfigure(5, weight=2)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.train_log = tk.Text(log_frame, height=8, wrap="word", state="disabled", 
                                bg="#1e1e1e", fg="#f0f0f0", insertbackground="white", 
                                relief="flat", font=("Consolas", 10), padx=5, pady=5)
        self.train_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.train_log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.train_log.configure(yscrollcommand=log_scroll.set)

        status = ttk.Label(parent, textvariable=self.train_status_var, relief="sunken", anchor="w", padding=5)
        status.grid(row=6, column=0, sticky="ew")

    def _build_predict_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        tips = ttk.Label(
            parent,
            text="Tip: label mismatch is checked; results scroll in the table below.",
            font=("Segoe UI", 10, "italic")
        )
        tips.grid(row=0, column=0, sticky="w", pady=(0, 10))

        basic = ttk.LabelFrame(parent, text="Model & Output Configuration", padding=15)
        basic.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        for col in range(3):
            basic.columnconfigure(col, weight=1)

        ttk.Label(basic, text="Target Label (speed=rad/s):").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            basic,
            textvariable=self.predict_label_var,
            values=LABEL_OPTIONS,
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(basic, text="Model File:").grid(row=1, column=0, sticky="w", pady=(15, 0))
        model_entry = ttk.Entry(basic, textvariable=self.predict_model_var)
        model_entry.grid(row=1, column=1, sticky="ew", pady=(15, 0), padx=(0, 10))
        ttk.Button(basic, text="Browse...", command=self._choose_predict_model).grid(
            row=1, column=2, sticky="w", pady=(15, 0)
        )

        ttk.Label(basic, text="Model Info:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(basic, textvariable=self.predict_model_info_var, foreground="#3794ff").grid(row=2, column=1, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Label(basic, text="Output Folder:").grid(row=3, column=0, sticky="w", pady=(15, 0))
        out_entry = ttk.Entry(basic, textvariable=self.predict_output_var)
        out_entry.grid(row=3, column=1, sticky="ew", pady=(15, 0), padx=(0, 10))
        ttk.Button(basic, text="Browse...", command=self._choose_predict_output).grid(
            row=3, column=2, sticky="w", pady=(15, 0)
        )

        data_frame = ttk.LabelFrame(parent, text="Prediction Data Paths", padding=10)
        data_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        data_frame.columnconfigure(0, weight=1)
        self.predict_listbox = self._build_path_box(data_frame, "Sample Paths", 0, compact=True)

        action = ttk.Frame(parent)
        action.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        action.columnconfigure(0, weight=1)
        self.predict_btn = ttk.Button(action, text="START PREDICTION", command=self._start_predict, width=20)
        self.predict_btn.grid(row=0, column=0, sticky="w")
        ttk.Button(action, text="Clear Results", command=self._clear_predict_results).grid(row=0, column=1, sticky="w", padx=(15, 0))
        ttk.Button(action, text="Clear Log", command=self._clear_predict_log).grid(row=0, column=2, sticky="w", padx=(15, 0))

        result_frame = ttk.LabelFrame(parent, text="Prediction Results", padding=10)
        result_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        parent.rowconfigure(4, weight=3)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.predict_tree = ttk.Treeview(
            result_frame,
            columns=("sample_id", "predicted", "actual"),
            show="headings",
            height=12,
        )
        for col, text, width in [
            ("sample_id", "Sample ID", 350),
            ("predicted", "Predicted", 180),
            ("actual", "Actual", 180),
        ]:
            self.predict_tree.heading(col, text=text)
            self.predict_tree.column(col, width=width, anchor="w", stretch=True)
            
        pred_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=self.predict_tree.yview)
        pred_scroll_x = ttk.Scrollbar(result_frame, orient="horizontal", command=self.predict_tree.xview)
        self.predict_tree.configure(yscrollcommand=pred_scroll.set, xscrollcommand=pred_scroll_x.set)
        
        self.predict_tree.grid(row=0, column=0, sticky="nsew")
        pred_scroll.grid(row=0, column=1, sticky="ns")
        pred_scroll_x.grid(row=1, column=0, sticky="ew")

        log_frame = ttk.LabelFrame(parent, text="Activity Log", padding=10)
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        parent.rowconfigure(5, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.predict_log = tk.Text(log_frame, height=5, wrap="word", state="disabled",
                                  bg="#1e1e1e", fg="#f0f0f0", insertbackground="white", 
                                  relief="flat", font=("Consolas", 10), padx=5, pady=5)
        self.predict_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.predict_log.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.predict_log.configure(yscrollcommand=log_scroll.set)

        status = ttk.Label(parent, textvariable=self.predict_status_var, relief="sunken", anchor="w", padding=5)
        status.grid(row=6, column=0, sticky="ew")

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

        listbox = tk.Listbox(frame, height=4 if compact else 6, selectmode="extended",
                             bg="#1e1e1e", fg="#f0f0f0", selectbackground="#204a87", selectforeground="white",
                             relief="flat", font=("Segoe UI", 10), borderwidth=0, highlightthickness=0)
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
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        self.predict_model_var.set(path)
        self._load_model_info(path)

    def _load_model_info(self, path: str) -> None:
        try:
            bundle = ml_train_predict.load_model_bundle(path)
            meta = bundle.get("metadata", {})
            label = meta.get("label_display") or meta.get("label_target", "")
            source = meta.get("source", "")
            task = meta.get("task", "")
            info = f"label={label or 'unknown'} | source={source or 'unknown'} | task={task or 'unknown'}"
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
            train_dirs = ml_train_predict.collect_sample_dirs(train_paths) if train_paths else []
            test_dirs = ml_train_predict.collect_sample_dirs(test_paths) if test_paths else []
            if not train_dirs and not test_dirs:
                self._train_queue.put(("error", "No valid sample folders found."))
                return
            if train_dirs:
                self._train_queue.put(("log", f"Train samples: {len(train_dirs)}"))
            if test_dirs:
                self._train_queue.put(("log", f"Test samples: {len(test_dirs)}"))
            result = ml_train_predict.train_and_save(
                train_dirs,
                test_dirs,
                label_target=label_target,
                source=source,
                use_slices=use_slices,
                test_ratio=test_ratio,
                seed=seed,
                out_dir=out_dir,
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
            elif kind == "error":
                self._append_train_log(f"[ERROR] {payload}")
                self._finish_training()
                messagebox.showerror("Training failed", str(payload))
            elif kind == "done":
                self._finish_training()
                train_count = len(payload.get("train_ids", []))
                test_count = len(payload.get("test_ids", []))
                saved = payload.get("saved_models", [])
                models_dir = payload.get("models_dir", "")
                report_md = payload.get("report_md", "")
                self._append_train_log(f"Training done. Saved models: {len(saved)}")
                if models_dir:
                    self._append_train_log(f"Models folder: {models_dir}")
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
            bundle = ml_train_predict.load_model_bundle(model_path)
            meta = bundle.get("metadata", {})
            model_label = meta.get("label_target")
            if model_label and model_label != label_target:
                self._predict_queue.put(
                    ("error", f"Model label is {model_label}, but selected label is {label_target}.")
                )
                return
            data_dirs = ml_train_predict.collect_sample_dirs(data_paths)
            if not data_dirs:
                self._predict_queue.put(("error", "No valid sample folders found."))
                return
            self._predict_queue.put(("log", f"Prediction samples: {len(data_dirs)}"))
            result = ml_train_predict.predict_with_model(model_path, data_dirs, out_dir)
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
            elif kind == "error":
                self._append_predict_log(f"[ERROR] {payload}")
                self._finish_predict()
                messagebox.showerror("Prediction failed", str(payload))
            elif kind == "done":
                self._populate_predict_results(payload)
                self._finish_predict()
                metrics = payload.get("metrics", {})
                plot_dir = payload.get("plot_dir", "")
                if metrics:
                    self._append_predict_log(f"Metrics: {metrics}")
                if plot_dir:
                    self._append_predict_log(f"Plots folder: {plot_dir}")
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
