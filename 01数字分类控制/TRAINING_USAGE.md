"AI转速识别"训练脚本使用说明
================================

本说明介绍如何基于 `recordings/**/csv/*.csv` 中的样本训练一个监督学习模型，从后声场特征推断转速（单位：rad/s）。

环境准备
--------

1. 启动 PyCharm 虚拟环境：
   ```bash
   source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
   ```
2. 安装依赖（仅需一次，若环境中已存在可跳过）：
   ```bash
   pip install numpy pandas scikit-learn
   ```

数据标注
--------

* 每个 `recordings/**/csv/cycle_xx.csv` 代表一个学习样本。
* 提供标签有两种方式（二选一即可）：
  1. **在 CSV 第一行写入标签**（适合手动标注）  
     例如：
     ```text
     3rad/s
     time_s,amplitude,freq_hz
     ...
     ```
     支持单位：`rad/s` 或 `rpm`（会自动转换为 rad/s）。
  2. **在录制会话文件夹名中编码“转速档位”**（适合批量采集）  
     例如：
     * `recordings/s1_80_2/csv/cycle_00.csv`
     * `recordings/s1_180_5/csv/cycle_01.csv`  
     脚本会自动解析形如 `s<idx>_<speed_code>_<trial>` 的文件夹名，将中间的 `80/180` 等视为
     **离散档位代码**，再根据预先标定好的“档位 ↔ 周期（秒）”关系换算成实际转速 rad/s。  
     例如当前配置：
     * 档位 80  ⇒ 周期 2.0832 s ⇒ `omega_80 = 2π / 2.0832` rad/s
     * 档位 180 ⇒ 周期 0.45137 s ⇒ `omega_180 = 2π / 0.45137` rad/s  
     同一会话文件夹（例如 `s1_80_2`）下所有 `csv/` 内的周期样本都会共享同一个角速度标签。

训练步骤
--------

```bash
cd /home/milsonson/PycharmProjects/JupyterProject/AI转速/01数字分类控制
python train_rotation_model.py \
  --data-root recordings \
  --output-dir build/models \
  --test-size 0.25 \
  --random-state 42
```

脚本功能
--------

* 自动遍历 `recordings` 下所有 CSV。
* 对每个样本提取统计特征（包络均值/方差/分位数、频率统计量、梯度平均值、积分能量等）。
* 使用 `train_test_split` 构造训练/测试集，基于 `RandomForestRegressor` 拟合。
* 输出：
  * `build/models/rotation_speed_model.pkl`：已训练模型（`pickle`）。
  * `build/models/metrics.json`：测试 MAE、R² 以及样本数。

预测脚本（使用已训练模型）
--------------------------

当 `build/models/rotation_speed_model.pkl` 已存在时，可使用 `predict_rotation_model.py`
对新的录制数据进行转速预测。假设某次录制后的周期 CSV 位于
`recordings/20251104_161245/csv`，可执行：

```bash
source /home/milsonson/虚拟环境pycharm/ai转速.venv/bin/activate
cd /home/milsonson/PycharmProjects/JupyterProject/AI转速/01数字分类控制
python predict_rotation_model.py \
  --model-path build/models/rotation_speed_model.pkl \
  --csv-dir recordings/20251104_161245/csv
```

脚本会输出每个周期的预测转速（单位 rad/s），以及所有周期的平均预测值。

常见问题
--------

* **样本太少**：默认需要 ≥4 个带标签的样本，且至少包含 2 个不同转速。可通过 `--min-samples` 调整。
* **新增数据**：将新录制的数据放入与现有 `recordings/s1_80_2/` 类似的目录结构，
  并通过“第一行标签”或“会话文件夹名编码转速”任意一种方式提供标签，重新运行训练脚本即可。
* **模型推理**：当前目录已提供 `predict_rotation_model.py`，会自动加载 `rotation_speed_model.pkl`，
  并复用 `compute_features` 对新的 CSV/cycle 进行特征提取和预测。
