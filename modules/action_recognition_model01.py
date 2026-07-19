import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FEATURE_CSV = os.path.join(BASE_DIR, "data/processed/动作特征数据.csv")
LABEL_CSV = os.path.join(BASE_DIR, "data/processed/动作标签.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
ACTION_MODEL_PATH = os.path.join(BASE_DIR, "models/cnn_lstm_action_model.h5")
FATIGUE_MODEL_PATH = os.path.join(BASE_DIR, "models/mlp_fatigue_model.h5")

# ====================== 动作映射 ======================
action_map = {0: "休息", 1: "握拳", 2: "食指伸", 3: "中指伸", 4: "无名指伸", 5: "小指伸"}

# ====================== 数据加载与标准化 ======================
X = pd.read_csv(RAW_FEATURE_CSV, header=None).values.astype(np.float32)
y = pd.read_csv(LABEL_CSV, header=None).values.ravel().astype(int)

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)

# reshape 为 CNN-LSTM 输入格式
X_norm = X_norm.reshape((X_norm.shape[0], 1, X_norm.shape[1]))
y_cat = to_categorical(y, num_classes=6)

# ====================== CNN-LSTM 模型定义 ======================
input_layer = Input(shape=(X_norm.shape[1], X_norm.shape[2]))
x = Conv1D(64, kernel_size=1, activation='relu')(input_layer)
x = Dropout(0.3)(x)
x = LSTM(128, return_sequences=False)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(6, activation='softmax')(x)

cnn_lstm_model = Model(inputs=input_layer, outputs=output_layer)
cnn_lstm_model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ====================== 模型训练 ======================
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_norm, y_cat, test_size=0.2, random_state=42, stratify=y)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
cnn_lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)
cnn_lstm_model.save(ACTION_MODEL_PATH)
print("✅ CNN-LSTM 动作识别模型训练完成")

# ====================== 中医疲劳评估 MLP ======================
n_samples = len(X)
tcm_data = np.zeros((n_samples, 9), dtype=np.float32)
np.random.seed(42)
tcm_data[:, 0] = np.random.uniform(60, 130, n_samples)
tcm_data[:, 1] = np.random.uniform(0.8, 2.0, n_samples)
tcm_data[:, 2] = np.random.uniform(0.5, 1.5, n_samples)
tcm_data[:, 3] = np.random.uniform(0.05, 0.4, n_samples)
tcm_data[:, 4] = np.random.uniform(100, 200, n_samples)
tcm_data[:, 5] = np.random.uniform(100, 200, n_samples)
tcm_data[:, 6] = np.random.uniform(100, 200, n_samples)
tcm_data[:, 7] = np.random.uniform(80, 200, n_samples)
tcm_data[:, 8] = np.random.uniform(0.1, 0.5, n_samples)

# 构造疲劳标签
fatigue_labels = []
for row in tcm_data:
    pr, pa, pw, pv, r, g, b, br, ct = row
    score = 0.0
    if pr > 105: score += 0.4
    if pv > 0.22: score += 0.3
    if br < 150: score += 0.2
    if ct > 0.3: score += 0.1
    fatigue_labels.append(min(score, 1.0))
fatigue_labels = np.array(fatigue_labels, dtype=np.float32)

# MLP模型
from tensorflow.keras.models import Sequential
mlp_model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
mlp_model.compile(optimizer=Adam(0.001), loss='mse')

fatigue_features = tcm_data[:, [0,3,1,7,8]]
mlp_model.fit(fatigue_features, fatigue_labels, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
mlp_model.save(FATIGUE_MODEL_PATH)
print("✅ MLP 疲劳评估模型训练完成")
