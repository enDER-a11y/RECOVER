import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# 自动从 .mat 提取动作标签
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
raw_path = os.path.join(BASE_DIR, "data", "raw")
label_save_path = os.path.join(BASE_DIR, "data", "processed", "动作标签.csv")

WINDOW = 3000
STEP = 200

all_labels = []

for filename in os.listdir(raw_path):
    if not filename.endswith(".mat"):
        continue

    full_path = os.path.join(raw_path, filename)
    mat = loadmat(full_path)

    label = mat["stimulus"].flatten()
    emg = mat["emg"]

    max_start = len(emg) - WINDOW
    if max_start <= 0:
        continue

    for i in range(0, max_start, STEP):
        # 取窗口中间的标签作为该段特征的标签
        mid_idx = i + WINDOW // 2
        all_labels.append(label[mid_idx])

pd.DataFrame(all_labels).to_csv(label_save_path, header=False, index=False)
print(f"标签生成总数：{len(all_labels)}")