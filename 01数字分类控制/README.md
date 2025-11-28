# AI 转速识别 – 使用说明（训练 & 预测）

本目录实现了一个基于声学信号的转速估计流程：  
从后声场采集到的音频中切分出若干 **周期 CSV**，再用监督学习模型（随机森林回归）预测对应的 **角速度（rad/s）**。

即使没有接触过这个项目，只要按照本文操作，也可以完成：

- 准备 / 组织数据
- 训练转速预测模型
- 使用命令行或图形界面对新数据做预测

---

## 1. 目录结构概览

在当前目录（`01数字分类控制`）下，核心文件有：

- `01数字实时分类_v3.py`  
  实时声学采集和保存工具，负责：
  - 连接串口采集音频
  - 将录音切分成多个周期，并写入 `recordings/**/csv/cycle_xx.csv`
  - 保存波形图、频谱图等

- `train_rotation_model.py`  
  训练脚本：从 `recordings/**/csv/*.csv` 提取特征并训练一个 **转速回归模型**。

- `predict_rotation_model.py`  
  预测脚本/工具：
  - 带参数运行：命令行批量预测
  - 不带参数运行：启动一个小型 **GUI 工具**，用对话框选择模型和 CSV 进行预测

- `TRAINING_USAGE.md`  
  原始训练脚本使用说明（与本文内容一致、略更简短）。

- `build/models/rotation_speed_model.pkl`  
  训练完成后保存的模型文件（`pickle` 序列化），供预测脚本加载。

- `recordings/`  
  录制数据目录，结构类似：
  - `recordings/s1_80_2/audio.wav`
  - `recordings/s1_80_2/audio.csv`
  - `recordings/s1_80_2/csv/cycle_00.csv` … `cycle_xx.csv`
  - `recordings/s1_80_2/data_plots.png`, `fft_plot.png`, `spectrogram.png`

---

## 2. 环境准备

项目默认使用一个独立的 Python 虚拟环境（路径示例为作者本机路径，可按需替换成你自己的环境）：

```bash
# 激活虚拟环境
source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
```

### 2.1 安装依赖（首次使用时）

建议在虚拟环境中安装以下依赖：

```bash
pip install numpy pandas scikit-learn pyqt6 matplotlib scipy
```

说明：

- `numpy` / `pandas` / `scikit-learn`：训练和预测所需；
- `pyqt6`：预测 GUI 使用；
- `matplotlib` / `scipy`：主 GUI（`01数字实时分类_v3.py`）用来绘图和信号处理。

---

## 3. 数据结构与标注规则

### 3.1 一次录制的数据结构

通过 `01数字实时分类_v3.py` 录制一次音频后，将在 `recordings/` 下生成类似结构：

```text
recordings/
  s1_80_2/
    audio.wav
    audio.csv
    data_plots.png
    fft_plot.png
    spectrogram.png
    csv/
      cycle_00.csv
      cycle_01.csv
      ...
```

其中：

- `csv/cycle_xx.csv`：**每个周期一个 CSV 文件**，是训练和预测的基本样本单位；
- `audio.csv`：整段音频的时序数据（训练脚本不会直接使用）。

> 训练脚本只会使用 `recordings/**/csv/*.csv` 中的文件。

### 3.2 每个 CSV 的格式

每个周期 CSV（`cycle_xx.csv`）内部结构：

```text
time_s,amplitude,freq_hz
0.000000000,0.018503,4065.956
0.000062500,0.018804,4051.235
...
```

- `time_s`：时间轴（秒）
- `amplitude`：包络幅值
- `freq_hz`：瞬时频率（Hz）

### 3.3 标签（真实转速）从哪里来？

训练模型需要知道每个周期的真实转速（角速度，rad/s）。本项目支持两种标注方式：

#### 方式 A：在 CSV 第一行写标签（显式标签）

适合手动标注或实验用例。

示例：

```text
3rad/s
time_s,amplitude,freq_hz
...
```

或：

```text
180rpm
time_s,amplitude,freq_hz
...
```

说明：

- 支持单位：`rad/s` 或 `rpm`；
- 训练脚本会自动转换成 **rad/s**；
- 若有第一行标签，则优先使用这一行作为真值。

#### 方式 B：从会话文件夹名推断（推荐，用于批量采集）

对于批量采集数据，本项目使用 **“档位代码 → 周期 → 角速度”** 的标定方式：

- 录制会话文件夹名形如：

  ```text
  recordings/s1_80_2/...
  recordings/s1_180_5/...
  ```

  符合模式：`s<idx>_<speed_code>_<trial>`

- 其中 `<speed_code>` 是 **档位代码**（不是直接的 rpm），当前映射关系为：

  ```text
  speed_code 80  -> 周期 T_80  = 2.0832 s
  speed_code 180 -> 周期 T_180 = 0.45137 s
  ```

- 真实角速度（训练标签）按公式计算：

  ```text
  ω = 2 * π / T   [rad/s]
  ```

  因此：

  - `ω_80  = 2π / 2.0832   rad/s`
  - `ω_180 = 2π / 0.45137  rad/s`

- 对于同一个会话文件夹（比如 `s1_80_2`），其 `csv/` 目录下 **所有 `cycle_xx.csv` 周期样本** 会共享同一个角速度标签。

> 若 CSV 内没有第一行标签，训练脚本会尝试用这种“文件夹名 + 映射表”来推断标签。  
> 若两种方式都无法得到标签，该样本会被跳过并打印在 “Skipped samples:” 列表中。

---

## 4. 训练转速预测模型

训练脚本：`train_rotation_model.py`

作用：

- 自动遍历 `recordings/**/csv/*.csv`；
- 对每个周期 CSV 计算一组统计特征：
  - 包络与频率的均值、方差、分位数、最小值、最大值等；
  - 振幅的积分能量、频率积分；
  - 振幅和频率的梯度平均绝对值；
- **在同一批特征上训练多种回归模型**，当前包括（约 9 种）：
  - 随机森林回归（`RandomForestRegressor`，模型名 `rf`）
  - 极端随机森林回归（`ExtraTreesRegressor`，模型名 `et`）
  - 梯度提升回归（`GradientBoostingRegressor`，模型名 `gbr`）
  - K 最近邻回归（`KNeighborsRegressor`，模型名 `knn`）
  - 支持向量机回归（`SVR`，RBF 核，模型名 `svr`）
  - 线性回归（`LinearRegression`，模型名 `lin`）
  - 岭回归（`Ridge`，模型名 `ridge`）
  - Lasso 回归（`Lasso`，模型名 `lasso`）
  - 简单前馈神经网络（`MLPRegressor`，模型名 `nn`，可理解为一小层多层感知机）
- 按指定比例将数据随机划分为训练 / 测试集；
- 为每个模型分别保存权重文件和评估指标。

### 4.1 快速开始：70% 训练 + 30% 测试

在虚拟环境中，进入项目目录：

```bash
source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
cd /home/milsonson/PycharmProjects/JupyterProject/AI转速/01数字分类控制

python train_rotation_model.py \
  --data-root recordings \
  --output-dir build/models \
  --test-size 0.3 \
  --random-state 42
```

参数说明：

- `--data-root`：录制数据根目录，默认为 `recordings`；
- `--output-dir`：模型和评估指标的输出目录，默认为 `build/models`；
- `--test-size`：测试集占比，例如 `0.3` 表示 30% 样本用于测试，70% 用于训练；
- `--random-state`：随机种子，使训练/测试划分可复现。

> 划分方式由 `sklearn.model_selection.train_test_split` 完成，会对全部样本随机打乱再按照比例划分。  
> 每个周期 CSV 都是独立样本，被分到训练集或测试集的概率相同。

### 4.2 训练输出结果

训练成功后，会看到类似输出：

```text
Saved model 'rf' to build/models/rotation_speed_model_rf.pkl
Saved model 'nn' to build/models/rotation_speed_model_nn.pkl
Legacy default model (rf) also saved to build/models/rotation_speed_model.pkl
Metrics: { ... }
```

并生成多个模型文件（至少包括以下几个）：  

- `build/models/rotation_speed_model_rf.pkl`  
  - 随机森林回归模型（`StandardScaler + RandomForestRegressor`）
- `build/models/rotation_speed_model_et.pkl`  
  - 极端随机森林回归模型（`StandardScaler + ExtraTreesRegressor`）
- `build/models/rotation_speed_model_gbr.pkl`  
  - 梯度提升回归模型（`StandardScaler + GradientBoostingRegressor`）
- `build/models/rotation_speed_model_knn.pkl`  
  - K 最近邻回归模型（`StandardScaler + KNeighborsRegressor`）
- `build/models/rotation_speed_model_svr.pkl`  
  - 支持向量机回归模型（`StandardScaler + SVR`）
- `build/models/rotation_speed_model_lin.pkl`  
  - 线性回归模型（`StandardScaler + LinearRegression`）
- `build/models/rotation_speed_model_ridge.pkl`  
  - 岭回归模型（`StandardScaler + Ridge`）
- `build/models/rotation_speed_model_lasso.pkl`  
  - Lasso 回归模型（`StandardScaler + Lasso`）
- `build/models/rotation_speed_model_nn.pkl`  
  - 简单前馈神经网络模型（`StandardScaler + MLPRegressor`）
- `build/models/rotation_speed_model.pkl`  
  - 为兼容旧版本，始终保存为 **随机森林模型的别名**（你可以像以前一样使用这个默认路径）

- `build/models/metrics.json`：结构示例  
  测试集上的评估指标：
  ```json
  {
    "test_size": 50,
    "train_size": 147,
    "models": {
      "rf": {
        "mae": 0.0043,
        "r2": 0.9999
      },
      "nn": {
        "mae": 0.0061,
        "r2": 0.9997
      }
    }
  }
  ```
  - `test_size` / `train_size`：样本数量；
  - `"models"` 下记录了每种模型的 `mae`（平均绝对误差，rad/s）和 `r2`（拟合优度）。

---

## 5. 使用模型做预测

预测脚本：`predict_rotation_model.py`

提供两种使用方式：

1. **命令行预测**（适合批量评估 / 脚本调用）
2. **独立 GUI 工具**（适合交互式选择模型和数据）

### 5.1 命令行方式

典型用法：

```bash
source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
cd /home/milsonson/PycharmProjects/JupyterProject/AI转速/01数字分类控制

python predict_rotation_model.py \
  --model-path build/models/rotation_speed_model.pkl \
  --csv-dir recordings/s1_80_5/csv
```

参数说明：

- `--model-path`：训练好的模型路径（默认为 `build/models/rotation_speed_model.pkl`）；
- `--csv-dir`：包含周期 CSV 的目录（例如 `recordings/s1_80_5/csv`）；
- `--output-json`（可选）：若提供，则将所有预测结果保存为 JSON 报告。

脚本会：

- 使用与训练阶段完全一致的 `compute_features` 提取特征；
- 对 `csv-dir` 下所有 `*.csv` 做预测；
- 若 CSV 的第一行带有标签，会一起输出“预测 vs 标签”的对比；
- 打印每个周期的预测转速（rad/s），以及所有周期的平均预测值。

### 5.2 GUI 方式（推荐用于手动选择文件夹 / 文件）

如果不带任何命令行参数运行 `predict_rotation_model.py`，会自动启动一个小型图形界面：

```bash
source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
cd /home/milsonson/PycharmProjects/JupyterProject/AI转速/01数字分类控制

python predict_rotation_model.py
```

弹出的“转速预测工具”窗口中包含三部分：

1. **模型与数据选择**

   - `模型文件` + “选择模型”  
     - 选择训练好的 `rotation_speed_model.pkl` 文件（例如 `build/models/rotation_speed_model.pkl`）。

   - `CSV文件夹` + “选择CSV文件夹”  
     - 选择一个根目录（例如 `recordings` 或具体某个会话目录，如 `recordings/s1_80_5`）。  
     - 程序会从该目录起 **递归搜索所有 `.../csv/*.csv`**，即：
       - `recordings/s1_80_5/csv/cycle_00.csv` 等周期 CSV；
       - 保证与训练脚本的数据选择逻辑一致。

   - `CSV文件` + “选择CSV文件”  
     - 若只想对 **一个或几个具体周期 CSV 文件** 做预测（不按整个会话），可以在这里多选 `cycle_xx.csv`；  
     - 当使用“选择CSV文件”时，会清空“CSV文件夹”字段，表示当前处于“按文件预测”模式。

2. **开始预测**

   - 点击“开始预测”按钮时：
     - 如果选择了 **CSV 文件夹**，则 **优先按文件夹模式**，递归使用其中所有 `.../csv/*.csv`；
     - 否则，使用通过“选择CSV文件”得到的文件列表。

3. **预测结果显示**

   - 下方“预测结果”区域会显示：
     - 模型文件路径；
     - 样本数量；
     - 平均预测转速（rad/s）；
     - 每个周期 CSV 的预测值（若存在标签也会显示对比）；
     - 若有文件因格式问题被跳过，则在末尾列出原因。

---

## 6. 在自有代码中调用模型（可选）

若你希望在自己的脚本或应用中直接使用训练好的模型，可以按如下方式加载：

```python
from pathlib import Path
import pickle
import pandas as pd
from train_rotation_model import compute_features

# 1. 加载模型
model_path = Path("build/models/rotation_speed_model.pkl")
with model_path.open("rb") as f:
    pipeline = pickle.load(f)

# 2. 加载一个周期 CSV（与训练使用的格式一致）
csv_path = Path("recordings/s1_80_5/csv/cycle_00.csv")
df = pd.read_csv(csv_path)  # 若第一行有标签则使用 skiprows=1

# 3. 提取特征并预测
features = compute_features(df)
X = [features]  # 形状 (1, n_features)
pred_rad_s = pipeline.predict(X)[0]
print("预测角速度:", pred_rad_s, "rad/s")
```

---

## 7. 常见问题与注意事项

- **为什么测试指标（R²）看起来非常高？**  
  当前数据集中，同一会话文件夹内的周期样本较为相似，如果训练/测试划分在样本级（而不是会话级），
  测试集可能包含与训练集非常接近的样本，因此 R² 可能接近 1。  
  若你需要更加严格的评估，可考虑改为“按会话文件夹划分训练/测试”。

- **能不能用其它档位（不是 80/180）？**  
  可以，只需在 `train_rotation_model.py` 中的 `SPEED_CODE_TO_PERIOD_S` 字典里加入新的档位码及其对应周期。

- **能不能用卷积神经网络（CNN）？**  
  当前训练脚本内置的是传统机器学习模型（随机森林）和一个简单的前馈神经网络（`MLPRegressor`），它们基于周期的统计特征做回归，计算体验轻量、依赖少。  
  如果你希望尝试真正的 CNN（例如基于时序波形或时频图），建议单独使用 PyTorch / TensorFlow 之类的框架，从原始波形或谱图出发设计网络结构，再复用这里的特征工程或数据组织方式。

- **预测阶段会不会“偷看文件名”来作弊？**  
  不会。预测阶段只使用 CSV 内容（`time_s`, `amplitude`, `freq_hz`）提特征进行推理。  
  文件/文件夹名仅用于 **训练阶段** 的标签推断（从档位码 → 周期 → 角速度）。

如果你在训练或预测时遇到报错，可以先查看终端输出的 “Skipped samples” 或异常信息，通常是：

- CSV 缺失列（不是 `time_s, amplitude, freq_hz`）；
- CSV 第一行标签格式无法解析；
- 会话文件夹的命名不符合 `s<idx>_<speed_code>_<trial>` 规范且没有显式标签。

如需在这些基础上扩展功能（更多档位、更复杂的模型、与主 GUI 深度集成），可以在现有脚本上继续迭代。欢迎在此基础上二次开发。  
