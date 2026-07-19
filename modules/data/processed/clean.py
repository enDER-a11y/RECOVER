import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feat_path = os.path.join(BASE_DIR, "data/processed/动作特征数据.csv")
label_path = os.path.join(BASE_DIR, "data/processed/动作标签.csv")

# 读取原始数据
feats = pd.read_csv(feat_path, header=None).values
labels = pd.read_csv(label_path, header=None).values.ravel()

# 1. 统计每类动作的数量
unique_labels, counts = np.unique(labels, return_counts=True)
print("原始标签分布：")
for lbl, cnt in zip(unique_labels, counts):
    print(f"  动作{lbl}：{cnt}条")

# 2. 找到最少的动作数量（作为平衡目标）
min_count = min(counts)
print(f"\n平衡目标：每类动作 {min_count} 条")

# 3. 每类动作随机选择 min_count 条
balanced_feats = []
balanced_labels = []

for lbl in unique_labels:
    # 找到该动作的所有索引
    lbl_indices = np.where(labels == lbl)[0]
    # 随机选择 min_count 条
    selected_indices = np.random.choice(lbl_indices, min_count, replace=False)
    # 添加到平衡数据中
    balanced_feats.append(feats[selected_indices])
    balanced_labels.append(labels[selected_indices])

# 4. 合并并打乱
balanced_feats = np.vstack(balanced_feats)
balanced_labels = np.hstack(balanced_labels)

# 打乱顺序
shuffle_idx = np.random.permutation(len(balanced_labels))
balanced_feats = balanced_feats[shuffle_idx]
balanced_labels = balanced_labels[shuffle_idx]

# 5. 保存平衡后的数据
pd.DataFrame(balanced_feats).to_csv(feat_path, header=False, index=False)
pd.DataFrame(balanced_labels).to_csv(label_path, header=False, index=False)

# 6. 打印结果
print(f"\n平衡完成！")
print(f"平衡后总数据量：{len(balanced_labels)} 条")
balanced_counts = np.unique(balanced_labels, return_counts=True)
for lbl, cnt in zip(balanced_counts[0], balanced_counts[1]):
    print(f" 动作{lbl}：{cnt}条")