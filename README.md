# AI Rotation Analysis Toolkit 🤖🔊

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-f7931e.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个完整的音频信号处理与机器学习分析工具包，用于从WAV音频文件中提取特征并训练模型进行多标签预测（形状、转速、材料）。

## ✨ 功能特性

- **🔊 音频特征提取**：从WAV文件中提取时域/频域特征，支持包络线分析、频谱峰值检测
- **🤖 传统机器学习**：基于scikit-learn的分类/回归（随机森林、SVM、梯度提升等）
- **🧠 深度学习**：基于PyTorch的1D CNN模型，支持GPU加速
- **🖥️ 图形界面**：提供友好的GUI工具，无需编写代码即可使用
- **📊 可视化输出**：训练历史、预测结果、混淆矩阵等自动绘图

## 📁 项目结构

```
.
├── wav_inspector.py              # 核心音频特征提取模块
├── wav_inspector_gui.py          # 音频特征提取GUI
├── ml_train_predict.py           # 传统ML训练/预测CLI
├── ml_train_predict_gui.py       # 传统ML训练/预测GUI
├── dl_train_predict.py           # 深度学习训练/预测CLI
├── dl_train_predict_gui.py       # 深度学习训练/预测GUI
├── output/                       # 提取的特征数据（示例）
├── train_models/                 # 训练好的模型保存目录
└── recordings/                   # 原始WAV录音文件
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ai-rotation-analysis.git
cd ai-rotation-analysis

# 安装依赖
pip install numpy scipy pandas matplotlib seaborn scikit-learn joblib torch
```

### 图形界面使用（推荐）

**1. 特征提取**
```bash
python3 wav_inspector_gui.py
```
选择WAV文件或文件夹，提取的特征将保存为CSV格式。

**2. 训练模型（传统ML）**
```bash
python3 ml_train_predict_gui.py
```
选择特征数据文件夹，设置标签类型（shape/speed/material），开始训练。

**3. 训练模型（深度学习）**
```bash
python3 dl_train_predict_gui.py
```
支持GPU加速，适合大规模数据集。

## 📊 数据格式说明

### 样本文件夹命名规则
```
{shape}_{direction}_{speed}_{material}
```
例如：`10_c_100_p` 表示：
- 形状编号：10
- 方向：c（顺时针）
- 转速：100（对应0.3536 rad/s）
- 材料：p（塑料）

### 转速映射表

| 档位 | 周期(s) | 角速度(rad/s) |
|------|---------|---------------|
| 80   | 2.522   | 2.49          |
| 100  | 1.433   | 4.38          |
| 120  | 0.983   | 6.39          |
| 140  | 0.733   | 8.57          |
| 160  | 0.600   | 10.47         |
| 180  | 0.495   | 12.70         |
| 200  | 0.4326  | 14.52         |
| 220  | 0.397   | 15.82         |
| 240  | 0.3536  | 17.77         |

## 🛠️ CLI命令行使用

### 特征提取
```bash
python3 wav_inspector.py \
  --input /path/to/audio.wav \
  --output /path/to/output \
  --bandpass --channel auto --peaks 10
```

### 传统ML训练
```bash
python3 ml_train_predict.py train \
  --label shape \
  --source envelope_detrended \
  --train "/path/train" \
  --test "/path/test" \
  --out /path/to/models
```

### 传统ML预测
```bash
python3 ml_train_predict.py predict \
  --model /path/to/model.joblib \
  --data "/path/test" \
  --out /path/to/predictions
```

### 深度学习训练
```bash
python3 dl_train_predict.py train \
  --label speed \
  --source envelope_detrended \
  --test "/path/data" \
  --out /path/to/models \
  --epochs 80 --batch-size 32 --device cuda
```

### 深度学习预测
```bash
python3 dl_train_predict.py predict \
  --model /path/to/best_model.pt \
  --data "/path/test" \
  --out /path/to/predictions \
  --device cuda
```

## 📁 输出目录结构

### 训练输出
```
models/                 # 保存的模型文件
├── model1.joblib
├── model2.joblib
└── ...
train_report.csv        # 原始评估指标
train_report.md         # 可读的训练报告
train_config.json       # 训练配置
plots/                  # 训练曲线（仅深度学习）
├── train_history.png
└── validation_curves.png
```

### 预测输出
```
predictions.csv         # 预测结果
test_report.md          # 测试报告
plots/                  # 可视化结果
├── predict_plot.png
└── confusion_matrix.png
```

## 🎯 支持的标签类型

| 标签类型 | 任务类型 | 说明 |
|----------|----------|------|
| `shape` | 分类 | 物体形状编号 |
| `speed` | 回归 | 转速（rad/s）|
| `material` | 分类 | 材料类型 |

## 🧪 高级配置

### 深度学习模型调参
```bash
python3 dl_train_predict.py train \
  --label shape \
  --base-ch 64 \
  --blocks 4 \
  --kernel 7 \
  --dropout 0.2 \
  --scheduler-patience 5
```

### 特征提取参数
- **带通滤波**：50-8000 Hz
- **包络截止频率**：可配置（默认20Hz）
- **FFT峰值数**：可配置（默认10个）
- **数据分片**：支持滑动窗口分片

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [SciPy](https://scipy.org/) - 科学计算库

---

**Star 🌟 本项目如果它对您有帮助！**
