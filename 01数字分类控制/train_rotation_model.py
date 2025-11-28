"""Train a supervised model that maps processed acoustic cycles to rotation speed.

Each CSV file inside `recordings/**/csv/*.csv` is treated as one sample.

Two ways to provide the target rotation speed (label, internally in rad/s):

1) Put the label in the *first line* of each CSV, e.g.:
      3rad/s
      time_s,amplitude,freq_hz
      ...
   Supported units: `rad/s` or `rpm` (will be converted to rad/s).

2) Encode the speed in the *session folder name*, e.g.:
      recordings/s1_80_2/csv/cycle_00.csv
      recordings/s1_180_5/csv/cycle_01.csv
   For names matching the pattern `s<idx>_<speed>_<trial>`, the `<speed>` part
   is interpreted as RPM and automatically converted to rad/s.

本脚本的核心流程：
  - 遍历 recordings 目录里的周期 CSV 文件；
  - 解析或推断每个样本的真值转速（rad/s）；
  - 计算统计特征，构成特征矩阵 X 和标签向量 y；
  - 在同一份 train/test 划分上训练多种回归模型，并分别保存。
"""

from __future__ import annotations  # 允许在类型注解中使用前向引用（Python 3.7+）

import argparse  # 命令行参数解析
import json  # 保存训练指标到 JSON 文件
import math  # 数学运算（这里只在注释中说明，实际主逻辑用不到）
import re  # 正则表达式，用于解析标签与文件夹名
from pathlib import Path  # 更方便、跨平台地处理路径
from typing import Dict, List, Optional, Tuple  # 类型注解用

import pickle  # 序列化 / 反序列化模型
import numpy as np  # 数值计算，向量 / 矩阵运算
import pandas as pd  # 读取 CSV 数据
from sklearn.ensemble import (  # 各种集成回归模型
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归族
from sklearn.metrics import mean_absolute_error, r2_score  # 回归评估指标
from sklearn.model_selection import train_test_split  # 划分训练集 / 测试集
from sklearn.neighbors import KNeighborsRegressor  # KNN 回归
from sklearn.neural_network import MLPRegressor  # 简单前馈神经网络回归
from sklearn.pipeline import Pipeline  # 串联预处理与模型
from sklearn.preprocessing import StandardScaler  # 数据标准化（零均值单位方差）
from sklearn.svm import SVR  # 支持向量机回归


# 匹配标签行，例如 "3rad/s" 或 "180rpm"
LABEL_PATTERN = re.compile(r"([-+]?\d*\.?\d+)\s*(rad/s|rpm)?", re.IGNORECASE)
# 匹配会话文件夹名，例如 "s1_80_2" 或 "s2-180-5"
SESSION_NAME_PATTERN = re.compile(r"s\d+[_-](\d+)[_-]\d+", re.IGNORECASE)

# 离散档位代码与对应机械周期（秒）的映射。
# 真实角速度 ω（rad/s）由公式 ω = 2 * π / T 得到。
#
# 例如：
#   - 80  档 -> 周期 2.0832 s
#   - 180 档 -> 周期 0.45137 s
SPEED_CODE_TO_PERIOD_S: Dict[float, float] = {
    80.0: 2.0832,
    180.0: 0.45137,
    100.0: 1.25656,
    120.0: 0.85412,
    140.0: 0.65272,
    160.0: 0.53112,
    60.0: 5.37714,
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a rotation speed regressor.")  # 创建解析器并设置描述
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("recordings"),
        help="Root directory that contains recording sessions.",  # 数据根目录（录制会话所在目录）
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build") / "models",
        help="Where to store the trained model and metrics.",  # 模型与评估指标输出目录
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of samples used for testing.",  # 测试集占全部样本的比例
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",  # 随机种子，保证可复现
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=4,
        help="Require at least this many labelled samples before training.",  # 最少需要多少条带标签样本
    )
    return parser.parse_args()  # 解析并返回参数对象


def discover_csv_files(root: Path) -> List[Path]:
    """在数据根目录下递归查找所有周期 CSV 文件。"""
    if not root.exists():
        # 若根目录不存在，直接抛出异常提示
        raise FileNotFoundError(f"Data root {root} does not exist.")
    # 只使用位于 ".../csv/*.csv" 下的文件，
    # 这与采集程序（01数字实时分类_v3.py）的输出布局保持一致，
    # 并且避免把顶层的 audio.csv 误当作样本。
    return sorted(
        p for p in root.rglob("*.csv") if p.is_file() and p.parent.name == "csv"
    )


def extract_label(line: str) -> Optional[float]:
    """从 CSV 第一行中解析标签（若存在），并统一换算为 rad/s。"""
    if not line:
        # 空行直接返回 None，表示没有标签
        return None
    match = LABEL_PATTERN.search(line)  # 使用正则匹配数字与单位
    if not match:
        return None  # 匹配失败则视为无标签
    value = float(match.group(1))  # 解析数值部分
    unit = match.group(2)  # 解析单位（可能为 None）
    if unit and unit.lower() == "rpm":
        # 若单位为 rpm，则换算为 rad/s
        value = value * (2 * math.pi / 60.0)
    return value  # 返回统一为 rad/s 的标签值


def infer_label_from_path(csv_path: Path) -> Optional[float]:
    """Infer label (in rad/s) from the recording session folder name.

    Expected session folder format (one of the path components):
        s<idx>_<speed_code>_<trial>
    For example:
        recordings/s1_80_2/csv/cycle_00.csv
        recordings/s1-180-5/csv/cycle_03.csv

    Here `<speed_code>` is NOT directly in rad/s or rpm; instead, it is a
    discrete code that maps to a measured mechanical cycle period via
    `SPEED_CODE_TO_PERIOD_S`. The physical angular velocity (rad/s) is then
    computed as:

        omega = 2 * pi / period
    """
    # 遍历路径的所有组成部分，查找形如 "s1_80_2" 的会话文件夹名
    for part in csv_path.parts:
        match = SESSION_NAME_PATTERN.fullmatch(part)  # 检查当前部分是否匹配模式
        if not match:
            continue  # 不匹配则继续检查下一个部分
        speed_code = float(match.group(1))  # 提取中间的档位数字，例如 80 / 180
        period_s = SPEED_CODE_TO_PERIOD_S.get(speed_code)  # 查表得到对应周期
        if period_s is None:
            # 未在映射表中找到对应周期，则无法从路径推断标签
            # 调用方会退回到其它机制（例如显式标签）或将样本视为无效。
            return None
        # 根据周期计算角速度：ω = 2 * π / T
        return 2 * math.pi / period_s
    # 未发现任何匹配的会话文件夹名
    return None


def load_sample(csv_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    """读取单个 CSV 文件，返回特征向量以及元信息（包含标签）。"""
    with csv_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()  # 先读取第一行，可能是标签

    # 1) 优先尝试从 CSV 第一行解析标签
    label = extract_label(first_line)
    skiprows = 1 if label is not None else 0  # 若第一行是标签，则读取数据时跳过这一行
    df = pd.read_csv(csv_path, skiprows=skiprows)  # 载入剩余数据

    expected_cols = {"time_s", "amplitude", "freq_hz"}  # 期望的列名集合
    if not expected_cols.issubset(df.columns):
        # 如果缺少所需列，则说明 CSV 格式不正确
        raise ValueError(
            f"{csv_path} missing required columns. Found {list(df.columns)}"
        )

    features = compute_features(df)  # 从时序数据中计算特征向量

    # 2) 若没有显式标签，则尝试从会话文件夹名推断
    if label is None:
        label = infer_label_from_path(csv_path)
        if label is None:
            # 既没有标签行，也无法从路径推断转速，则认为这个样本无标签
            raise ValueError(
                f"{csv_path} is missing the label line and no valid speed "
                "could be inferred from its parent folder name."
            )

    # 返回特征向量（float32）以及包含标签和路径信息的字典
    return np.array(features, dtype=np.float32), {"label": label, "path": str(csv_path)}


def compute_features(df: pd.DataFrame) -> List[float]:
    """将 time/amplitude/freq 三列序列压缩成一个固定长度的特征向量。"""
    amp = df["amplitude"].to_numpy(dtype=np.float64)  # 幅值序列
    freq = df["freq_hz"].to_numpy(dtype=np.float64)  # 频率序列
    time = df["time_s"].to_numpy(dtype=np.float64)  # 时间轴
    amp = np.nan_to_num(amp)  # 将 NaN/inf 替换为 0，避免数值问题
    freq = np.nan_to_num(freq)

    feats: List[float] = []  # 收集各类特征
    # 幅值的统计量：均值、标准差、最小值、最大值、分位数等
    feats.extend(summary_stats(amp))
    # 频率的统计量
    feats.extend(summary_stats(freq))
    # 幅值绝对值的积分（包络能量近似）
    feats.append(np.trapz(np.abs(amp), x=time))  # envelope energy
    # 频率随时间的积分（累计频率变化）
    feats.append(np.trapz(freq, x=time))  # accumulated frequency
    # 幅值梯度的平均绝对值，反映波形变化快慢
    feats.append(float(np.mean(np.abs(np.gradient(amp, time, edge_order=1)))))
    # 频率梯度的平均绝对值
    feats.append(float(np.mean(np.abs(np.gradient(freq, time, edge_order=1)))))

    return feats  # 返回一维特征向量


def summary_stats(arr: np.ndarray) -> List[float]:
    """计算一维数组的一组统计特征。"""
    percentiles = np.percentile(arr, [5, 25, 50, 75, 95])  # 分位点：5,25,50,75,95%
    return [
        float(np.mean(arr)),  # 均值
        float(np.std(arr)),  # 标准差
        float(np.min(arr)),  # 最小值
        float(np.max(arr)),  # 最大值
        *(float(p) for p in percentiles),  # 各个分位数
    ]


def ensure_output_dir(path: Path) -> None:
    """确保输出目录存在，如不存在则递归创建。"""
    path.mkdir(parents=True, exist_ok=True)


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[Dict[str, Pipeline], Dict[str, object]]:
    """在同一 train/test 划分上训练多种回归模型。

    当前训练的模型包括：
      - 'rf':     RandomForestRegressor  随机森林回归
      - 'et':     ExtraTreesRegressor   极端随机森林
      - 'gbr':    GradientBoostingRegressor 梯度提升回归
      - 'knn':    KNeighborsRegressor   K 最近邻回归
      - 'svr':    SVR (RBF kernel)      支持向量机回归（RBF 核）
      - 'lin':    LinearRegression      普通线性回归
      - 'ridge':  Ridge                 岭回归
      - 'lasso':  Lasso                 L1 正则回归
      - 'nn':     MLPRegressor          简单前馈神经网络
    """
    # 使用相同的 train/test 划分，确保不同模型的指标可直接比较
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 定义需要训练的模型配置。
    # 所有模型共享同一组特征与 train/test 划分，因此指标具有可比性。
    model_specs = {
        "rf": RandomForestRegressor(n_estimators=300, random_state=random_state),
        "et": ExtraTreesRegressor(n_estimators=300, random_state=random_state),
        "gbr": GradientBoostingRegressor(random_state=random_state),
        "knn": KNeighborsRegressor(n_neighbors=5),
        "svr": SVR(kernel="rbf", C=10.0, epsilon=0.01),
        "lin": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.0005, max_iter=5000, random_state=random_state),
        "nn": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=500,
            random_state=random_state,
        ),
    }

    pipelines: Dict[str, Pipeline] = {}  # 保存训练好的 pipeline
    metrics_by_model: Dict[str, Dict[str, float]] = {}  # 保存每个模型的指标

    # 逐个模型训练并评估
    for name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),  # 先做标准化
                ("model", estimator),  # 再喂给回归模型
            ]
        )
        pipeline.fit(X_train, y_train)  # 拟合当前模型
        y_pred = pipeline.predict(X_test)  # 在测试集上做预测
        metrics_by_model[name] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),  # 平均绝对误差
            "r2": float(r2_score(y_test, y_pred)),  # 拟合优度
        }
        pipelines[name] = pipeline  # 保存训练好的 pipeline

    # 汇总整体指标信息：训练/测试样本数 + 各模型指标
    global_metrics: Dict[str, object] = {
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "models": metrics_by_model,
    }
    return pipelines, global_metrics


def main() -> None:
    """主入口：加载数据、训练多个模型并保存结果。"""
    args = parse_args()  # 解析命令行参数
    csv_files = discover_csv_files(args.data_root)  # 找到所有周期 CSV 文件
    if not csv_files:
        # 若没有找到任何 CSV，则直接报错
        raise RuntimeError(f"No CSV files found under {args.data_root}")

    features: List[np.ndarray] = []  # 存放每个样本的特征向量
    labels: List[float] = []  # 存放每个样本的标签（rad/s）
    failed: List[str] = []  # 记录加载失败的样本及原因

    # 遍历每一个 CSV 文件
    for csv_file in csv_files:
        try:
            feat_vec, meta = load_sample(csv_file)  # 加载样本并提取特征与标签
        except Exception as exc:  # noqa: BLE001
            # 若出错，则记录并跳过该样本
            failed.append(f"{csv_file}: {exc}")
            continue
        features.append(feat_vec)
        labels.append(meta["label"])

    # 如有被跳过的样本，在控制台打印出来方便排查
    if failed:
        print("Skipped samples:")
        for line in failed:
            print(f"  - {line}")

    # 检查样本数是否足够
    if len(features) < args.min_samples:
        raise RuntimeError(
            f"Need at least {args.min_samples} labelled samples, got {len(features)}."
        )

    # 需要至少两个不同的转速值才能做监督学习回归
    if len(set(labels)) < 2:
        raise RuntimeError(
            "Need at least two distinct rotation speeds to train a supervised model."
        )

    # 将特征列表堆叠成二维矩阵 X，标签列表转成一维向量 y
    X = np.vstack(features)
    y = np.array(labels, dtype=np.float32)

    # 在同一 train/test 划分上训练多种模型
    pipelines, metrics = train_models(X, y, args.test_size, args.random_state)
    ensure_output_dir(args.output_dir)  # 确保输出目录存在

    # 分别保存每一种模型，例如 rotation_speed_model_rf.pkl
    for name, pipeline in pipelines.items():
        model_path = args.output_dir / f"rotation_speed_model_{name}.pkl"
        with model_path.open("wb") as f:
            pickle.dump(pipeline, f)
        print(f"Saved model '{name}' to {model_path}")

    # 为了兼容旧版本，同时将随机森林模型保存为 rotation_speed_model.pkl
    if "rf" in pipelines:
        legacy_path = args.output_dir / "rotation_speed_model.pkl"
        with legacy_path.open("wb") as f:
            pickle.dump(pipelines["rf"], f)
        print(
            f"Legacy default model (rf) also saved to {legacy_path}"
        )

    # 将所有模型的评估指标写入 metrics.json
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
