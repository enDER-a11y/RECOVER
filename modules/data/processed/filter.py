import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
feat_path = os.path.join(BASE_DIR, "data/processed/动作特征数据.csv")
lab_path = os.path.join(BASE_DIR, "data/processed/动作标签.csv")

feats = pd.read_csv(feat_path, header=None).values
labels = pd.read_csv(lab_path, header=None).values.ravel()

keep_label = [0, 1, 2, 7, 14, 15]
mask = np.isin(labels, keep_label)
feats_filter = feats[mask]
labels_filter = labels[mask]

label_map = {0:0, 1:1, 2:2, 7:3, 14:4, 15:5}
new_labels = np.array([label_map[x] for x in labels_filter])

pd.DataFrame(feats_filter).to_csv(feat_path, index=False, header=False)
pd.DataFrame(new_labels).to_csv(lab_path, index=False, header=False)

print("筛选完成，剩余样本数：", len(new_labels))