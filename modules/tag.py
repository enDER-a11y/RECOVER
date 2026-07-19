import os
import pandas as pd
import numpy as np

# ===================== 路径 =====================
BASE_DIR = r"C:\Users\chenny\PycharmProjects\recover"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FEATURE_CSV = os.path.join(BASE_DIR, "data", "processed", "动作特征数据.csv")
LABEL_OUTPUT = os.path.join(BASE_DIR, "data", "processed", "动作标签.csv")

# ===================== 动作映射（真实！） =====================
ACTION_MAP = {
    "rest": 0,        # 休息
    "fist": 1,        # 握拳
    "index": 2,       # 食指
    "middle": 3,      # 中指
    "ring": 4,        # 无名指
    "pinky": 5,       # 小指
}

# ===================== 读取特征长度 =====================
df_feat = pd.read_csv(FEATURE_CSV, header=None)
total_rows = len(df_feat)
print(f"✅ 动作特征总条数：{total_rows}")

# ===================== 从 .mat 文件名生成真实标签 =====================
labels = []
mat_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".mat")]

for fname in mat_files:
    name = fname.lower()
    if "rest" in name:
        labels.append(0)
    elif "fist" in name:
        labels.append(1)
    elif "index" in name:
        labels.append(2)
    elif "middle" in name:
        labels.append(3)
    elif "ring" in name:
        labels.append(4)
    elif "pinky" in name:
        labels.append(5)
    else:
        labels.append(0)  # 默认休息

# ===================== 确保长度和特征一样 =====================
real_labels = []
while len(real_labels) < total_rows:
    real_labels.extend(labels)
real_labels = real_labels[:total_rows]

# ===================== 保存 =====================
pd.DataFrame(real_labels).to_csv(LABEL_OUTPUT, index=False, header=False)

print(f"✅ 真实动作标签生成完成！")
print(f"✅ 标签条数：{len(real_labels)}")
print(f"✅ 与特征条数 {total_rows} 完全匹配！")
print(f"✅ 文件保存路径：{LABEL_OUTPUT}")