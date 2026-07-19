import os
import scipy.io
import pandas as pd
import numpy as np

# ==================== 自动路径（不用改） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
OUT_DIR = BASE_DIR

# ==================== 固定取前 10 个通道（解决报错！） ====================
FIXED_CHANNELS = 10

all_features = []
all_labels = []

for filename in os.listdir(RAW_DIR):
    if filename.endswith(".mat"):
        print("正在处理：", filename)
        file_path = os.path.join(RAW_DIR, filename)
        mat_data = scipy.io.loadmat(file_path)

        emg = mat_data["emg"]
        stimulus = mat_data["stimulus"].flatten()

        # ==================== 关键修复：自动对齐到 10 通道 ====================
        if emg.shape[1] > FIXED_CHANNELS:
            emg = emg[:, :FIXED_CHANNELS]  # 多的截断

        # 特征和标签长度必须一样
        min_len = min(len(emg), len(stimulus))
        emg = emg[:min_len]
        stimulus = stimulus[:min_len]

        all_features.append(emg)
        all_labels.append(stimulus)

# ==================== 现在可以安全拼接 ====================
all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

# ==================== 保存 ====================
pd.DataFrame(all_features).to_csv(os.path.join(OUT_DIR, "动作特征数据.csv"), index=False, header=False)
pd.DataFrame(all_labels).to_csv(os.path.join(OUT_DIR, "动作标签.csv"), index=False, header=False)

print("✅ 全部成功生成！")
print("特征形状：", all_features.shape)
print("标签长度：", len(all_labels))