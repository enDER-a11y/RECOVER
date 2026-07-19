import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

# 1. 从你的scaler_mean/std.npy重建StandardScaler
scaler_mean = np.load("models/scaler_mean.npy")
scaler_std = np.load("models/scaler_std.npy")

scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_std
scaler.var_ = scaler_std ** 2

# 保存scaler.pkl
joblib.dump(scaler, "models/scaler.pkl")
print("✅ 已生成 scaler.pkl")

# 2. 生成label_encoder.pkl（从你的动作标签.csv里提取）
y = pd.read_csv("data/动作标签.csv", header=None).values.flatten()
le = LabelEncoder()
le.fit(y)
joblib.dump(le, "models/label_encoder.pkl")
print("✅ 已生成 label_encoder.pkl")