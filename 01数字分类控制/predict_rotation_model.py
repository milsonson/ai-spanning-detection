"""Predict rotation speed from per-cycle CSV files using a trained model.

Usage example (from project root):

  python predict_rotation_model.py \
    --model-path build/models/rotation_speed_model.pkl \
    --csv-dir recordings/20251104_161245/csv

This script reuses the same feature-extraction logic as `train_rotation_model.py`
so that inference is consistent with training.

本脚本提供两种使用方式：
  1) 命令行模式：指定模型文件与 CSV 目录，打印预测结果；
  2) GUI 模式：不带参数运行时，弹出 PyQt 窗口，可通过对话框选择模型与数据。
"""

from __future__ import annotations  # 允许在类型注解中使用前向引用

import sys  # 访问命令行参数，启动 Qt 应用
import argparse  # 命令行参数解析
import json  # 可选地将预测结果保存为 JSON
import pickle  # 加载训练好的模型（scikit-learn Pipeline）
from pathlib import Path  # 方便的路径处理工具
from typing import Dict, List, Optional, Tuple  # 类型注解

import numpy as np  # 数值运算
import pandas as pd  # 读写 CSV

from train_rotation_model import compute_features, extract_label  # 复用训练时的特征提取与标签解析逻辑


def parse_args() -> argparse.Namespace:
    """解析命令行参数（仅在 CLI 模式下使用）。"""
    parser = argparse.ArgumentParser(
        description="Predict rotation speed from per-cycle CSV files."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("build") / "models" / "rotation_speed_model.pkl",
        help="Path to the trained model (.pkl) saved by train_rotation_model.py.",  # 模型文件路径
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory that contains per-cycle CSV files (e.g. recordings/20251104_161245/csv).",  # 周期 CSV 所在目录
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save all predictions as a JSON report.",  # 可选：保存预测结果为 JSON
    )
    return parser.parse_args()


def load_features_from_csv(csv_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """从单个周期 CSV 中提取特征，并返回 (特征向量, 可选标签)。"""
    if not csv_path.is_file():
        # 选定路径若不是文件，则报错
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()  # 读取第一行，可能是标签

    # 尝试从第一行解析标签（支持 rad/s 或 rpm）
    label = extract_label(first_line)
    skiprows = 1 if label is not None else 0  # 若存在标签行，则读取数据时跳过

    df = pd.read_csv(csv_path, skiprows=skiprows)  # 载入剩余的时序数据
    expected_cols = {"time_s", "amplitude", "freq_hz"}  # 期望的列名集合
    if not expected_cols.issubset(df.columns):
        # 如缺少必要列，说明 CSV 结构不符合要求
        raise ValueError(
            f"{csv_path} missing required columns. Found {list(df.columns)}"
        )

    # 复用训练脚本中的特征提取逻辑，保证推理与训练一致
    features = compute_features(df)
    return np.asarray(features, dtype=np.float32), label  # 返回特征与（可能存在的）真值标签


def predict_from_files(
    model_path: Path, csv_files: List[Path]
) -> Tuple[Dict[str, object], List[str]]:
    """对一组 CSV 文件执行预测，返回 (汇总结果字典, 被跳过的文件列表)。"""
    if not model_path.is_file():
        # 模型文件不存在，提示用户先完成训练
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please train the model first via train_rotation_model.py."
        )
    if not csv_files:
        # 没有任何输入 CSV，无法预测
        raise RuntimeError("No CSV files provided for prediction.")

    # 加载训练好的 pipeline（通常是 StandardScaler + 某个回归模型）
    with model_path.open("rb") as f:
        pipeline = pickle.load(f)

    feature_rows: List[np.ndarray] = []  # 收集所有样本的特征向量
    labels: List[Optional[float]] = []  # 存储 CSV 中的真实标签（若存在）
    names: List[str] = []  # 对应的文件路径字符串
    skipped: List[str] = []  # 记录因错误被跳过的文件及原因

    # 逐个 CSV 文件提取特征
    for csv_path in csv_files:
        try:
            feats, label = load_features_from_csv(csv_path)  # 提取特征与标签
        except Exception as exc:  # noqa: BLE001
            # 若解析失败，则记录错误并跳过该文件
            skipped.append(f"{csv_path}: {exc}")
            continue
        feature_rows.append(feats)
        labels.append(label)
        names.append(str(csv_path))

    if not feature_rows:
        # 如果所有文件都失败，则直接报错
        raise RuntimeError("No valid CSV samples after loading features.")

    # 将特征列表堆叠成特征矩阵 X
    X = np.vstack(feature_rows).astype(np.float32)
    # 使用已加载的模型进行批量预测
    y_pred = pipeline.predict(X)

    # 按顺序汇总每个样本的预测结果
    results: List[Dict[str, Optional[float]]] = []
    for name, pred, true_label in zip(names, y_pred, labels):
        results.append(
            {
                "file": name,  # 文件路径
                "pred_rad_s": float(pred),  # 预测转速（rad/s）
                "true_rad_s": float(true_label) if true_label is not None else None,  # 若有标签则一起返回
            }
        )

    mean_pred = float(np.mean(y_pred))  # 所有样本预测值的平均数
    report: Dict[str, object] = {
        "model_path": str(model_path),  # 使用的模型路径
        "mean_pred_rad_s": mean_pred,  # 平均预测转速
        "samples": results,  # 逐样本的详细结果
    }
    return report, skipped


def main_cli() -> None:
    """命令行入口：保持原有 CLI 用法不变。"""
    args = parse_args()  # 解析命令行参数

    model_path: Path = args.model_path  # 模型文件路径
    csv_dir: Path = args.csv_dir  # 周期 CSV 所在目录

    if not csv_dir.is_dir():
        # 若给定路径不是目录，则抛出异常
        raise NotADirectoryError(f"CSV directory not found: {csv_dir}")

    # 仅使用该目录下的 *.csv 文件（不递归）
    csv_files: List[Path] = sorted(p for p in csv_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        raise RuntimeError(f"No CSV files found under {csv_dir}")

    # 调用通用预测函数
    report, skipped = predict_from_files(model_path, csv_files)
    # 额外记录目录信息，方便人类阅读 / JSON 输出
    report["csv_dir"] = str(csv_dir)

    if skipped:
        print("Skipped CSV files:")
        for line in skipped:
            print(f"  - {line}")

    samples = report["samples"]  # type: ignore[assignment]  # 逐样本结果
    mean_pred = report["mean_pred_rad_s"]  # type: ignore[assignment]  # 平均预测值

    # 在终端打印摘要
    print(f"已加载模型: {report['model_path']}")
    print(f"共读取 {len(samples)} 个周期 CSV，目录: {csv_dir}")
    print()
    print("逐周期预测结果 (单位 rad/s):")
    for item in samples:  # type: ignore[assignment]
        file_name = item["file"]
        pred_val = item["pred_rad_s"]
        true_val = item["true_rad_s"]
        if true_val is None:
            print(f"- {file_name}: 预测 {pred_val:.4f} rad/s")
        else:
            print(
                f"- {file_name}: 预测 {pred_val:.4f} rad/s, 标签 {true_val:.4f} rad/s"
            )
    print()
    print(f"平均预测转速: {mean_pred:.4f} rad/s")


def run_gui() -> None:
    """启动一个简单的 PyQt GUI，用于选择模型和 CSV 进行预测。"""
    from PyQt6.QtCore import Qt  # noqa: F401  # 仅用于类型与常量
    from PyQt6.QtWidgets import (
        QApplication,  # Qt 应用对象
        QFileDialog,  # 文件 / 文件夹选择对话框
        QGridLayout,  # 网格布局
        QGroupBox,  # 分组框
        QHBoxLayout,  # 水平布局
        QLabel,  # 文本标签
        QLineEdit,  # 单行文本框
        QMessageBox,  # 消息提示框
        QPushButton,  # 按钮
        QTextEdit,  # 多行文本显示
        QVBoxLayout,  # 垂直布局
        QWidget,  # 基础窗口部件
    )

    class PredictWindow(QWidget):
        def __init__(self) -> None:
            super().__init__()  # 初始化 QWidget
            self.setWindowTitle("转速预测工具")  # 设置窗口标题
            self.model_path: Optional[Path] = None  # 当前选中的模型路径
            self.csv_dir: Optional[Path] = None  # 当前选择的 CSV 根目录（文件夹模式）
            self.csv_files: List[Path] = []  # 当前参与预测的 CSV 文件列表（文件或文件夹展开后的结果）
            self._build_ui()  # 构建界面布局与控件

        def _build_ui(self) -> None:
            """构建 GUI 的控件与布局。"""
            main_layout = QVBoxLayout(self)  # 整个窗口的垂直布局

            # 选择区域：模型 + CSV 路径
            select_group = QGroupBox("模型与数据选择")  # 分组框
            grid = QGridLayout(select_group)  # 网格布局

            # 模型路径显示与选择按钮
            self.model_edit = QLineEdit()  # 显示当前选中的模型路径
            self.model_edit.setReadOnly(True)
            btn_model = QPushButton("选择模型")  # 打开文件对话框选择模型
            btn_model.clicked.connect(self.choose_model)

            # CSV 文件夹路径显示与选择按钮（文件夹模式）
            self.csv_dir_edit = QLineEdit()
            self.csv_dir_edit.setReadOnly(True)
            btn_dir = QPushButton("选择CSV文件夹")
            btn_dir.clicked.connect(self.choose_csv_dir)

            # CSV 文件列表显示与选择按钮（文件模式）
            self.csv_files_edit = QLineEdit()
            self.csv_files_edit.setReadOnly(True)
            btn_files = QPushButton("选择CSV文件")
            btn_files.clicked.connect(self.choose_csv_files)

            # 第一行：模型
            grid.addWidget(QLabel("模型文件:"), 0, 0)
            grid.addWidget(self.model_edit, 0, 1)
            grid.addWidget(btn_model, 0, 2)

            # 第二行：CSV 文件夹
            grid.addWidget(QLabel("CSV文件夹:"), 1, 0)
            grid.addWidget(self.csv_dir_edit, 1, 1)
            grid.addWidget(btn_dir, 1, 2)

            # 第三行：CSV 文件
            grid.addWidget(QLabel("CSV文件:"), 2, 0)
            grid.addWidget(self.csv_files_edit, 2, 1)
            grid.addWidget(btn_files, 2, 2)

            main_layout.addWidget(select_group)

            # 底部的“开始预测”按钮条
            btn_predict = QPushButton("开始预测")
            btn_predict.clicked.connect(self.run_prediction)
            btn_bar = QHBoxLayout()
            btn_bar.addStretch(1)  # 左侧留空
            btn_bar.addWidget(btn_predict)  # 按钮靠右
            main_layout.addLayout(btn_bar)

            # 结果展示区域：多行文本框
            result_group = QGroupBox("预测结果")
            result_layout = QVBoxLayout(result_group)
            self.result_text = QTextEdit()
            self.result_text.setReadOnly(True)  # 只读
            self.result_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)  # 不自动换行，便于对齐
            result_layout.addWidget(self.result_text)

            main_layout.addWidget(result_group, 1)  # 结果区域占据剩余空间

        def choose_model(self) -> None:
            """弹出文件选择框，选择模型 .pkl 文件。"""
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择模型文件",
                "",
                "模型文件 (*.pkl);;所有文件 (*)",
            )
            if not file_path:
                return  # 用户取消选择
            self.model_path = Path(file_path)  # 记录模型路径
            self.model_edit.setText(file_path)  # 在文本框中显示

        def choose_csv_dir(self) -> None:
            """弹出目录选择框，选择包含周期 CSV 的文件夹。"""
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "选择包含周期CSV的文件夹",
                "",
            )
            if not dir_path:
                return  # 用户取消
            csv_dir = Path(dir_path)  # 转为 Path 对象
            # 记录所选根目录，并按训练脚本的约定递归查找所有 ".../csv/*.csv"
            self.csv_dir = csv_dir
            self.csv_files = sorted(
                p
                for p in csv_dir.rglob("*.csv")
                if p.is_file() and p.parent.name == "csv"
            )
            self.csv_dir_edit.setText(dir_path)
            if self.csv_files:
                self.csv_files_edit.setText(
                    f"{dir_path} 下共找到 {len(self.csv_files)} 个周期 CSV"
                )
            else:
                self.csv_files_edit.setText("所选文件夹中没有 CSV 文件")

        def choose_csv_files(self) -> None:
            """弹出多选对话框，直接选择一个或多个 CSV 文件。"""
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "选择一个或多个周期CSV文件",
                "",
                "CSV文件 (*.csv);;所有文件 (*)",
            )
            if not files:
                return  # 用户取消
            # 按文件选择时，清除“文件夹模式”
            self.csv_dir = None
            self.csv_files = [Path(f) for f in files]
            if len(files) == 1:
                self.csv_files_edit.setText(files[0])
            else:
                self.csv_files_edit.setText(
                    f"已选择 {len(files)} 个 CSV 文件（显示第一个：{files[0]}）"
                )
            # 当按文件选择时，清空文件夹字段避免误解
            self.csv_dir_edit.clear()

        def run_prediction(self) -> None:
            """根据当前选择的模型与 CSV，执行预测并在界面显示结果。"""
            if self.model_path is None:
                QMessageBox.warning(self, "缺少模型", "请先选择模型文件（.pkl）。")
                return
            # 根据用户选择的模式决定数据来源：
            # 1) 若选了文件夹，则优先使用文件夹（递归搜索 .../csv/*.csv）
            # 2) 否则使用“选择CSV文件”得到的显式文件列表
            if self.csv_dir is not None:
                csv_files = sorted(
                    p
                    for p in self.csv_dir.rglob("*.csv")
                    if p.is_file() and p.parent.name == "csv"
                )
            else:
                csv_files = list(self.csv_files)

            if not csv_files:
                QMessageBox.warning(
                    self,
                    "缺少数据",
                    "请通过“选择CSV文件夹”或“选择CSV文件”添加至少一个周期CSV。",
                )
                return

            try:
                report, skipped = predict_from_files(self.model_path, csv_files)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "预测出错", str(exc))
                return

            # 将结果渲染到多行文本框中
            self.result_text.clear()
            lines: List[str] = []
            lines.append(f"模型文件: {report['model_path']}")
            lines.append(f"样本数量: {len(report['samples'])}")  # type: ignore[index]
            lines.append(
                f"平均预测转速: {report['mean_pred_rad_s']:.4f} rad/s"  # type: ignore[index]
            )
            lines.append("")
            lines.append("逐周期预测结果 (单位 rad/s):")
            for item in report["samples"]:  # type: ignore[index]
                file_name = item["file"]
                pred_val = item["pred_rad_s"]
                true_val = item["true_rad_s"]
                if true_val is None:
                    lines.append(f"- {file_name}: 预测 {pred_val:.4f} rad/s")
                else:
                    lines.append(
                        f"- {file_name}: 预测 {pred_val:.4f} rad/s, 标签 {true_val:.4f} rad/s"
                    )
            if skipped:
                lines.append("")
                lines.append("以下 CSV 因格式/内容问题被跳过：")
                for msg in skipped:
                    lines.append(f"- {msg}")

            self.result_text.setPlainText("\n".join(lines))

    # 创建并启动 Qt 应用主循环
    app = QApplication(sys.argv)
    win = PredictWindow()
    win.resize(900, 600)  # 设置窗口初始大小
    win.show()
    app.exec()


if __name__ == "__main__":
    # 若带有命令行参数（如 --model-path/--csv-dir），保持 CLI 行为；
    # 若不带参数，默认启动图形界面，方便通过 GUI 选择模型和数据。
    if len(sys.argv) > 1:
        main_cli()
    else:
        run_gui()
