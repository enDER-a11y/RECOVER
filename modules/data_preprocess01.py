import os
import pandas as pd
import numpy as np
from scipy import signal
import pywt
from scipy.io import loadmat

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_dir = os.path.join(BASE_DIR, "data", "raw")
output_path = os.path.join(BASE_DIR, "data", "processed")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 滑窗参数：窗口3秒，步长0.2秒（数据量×5，同时保留动作连续性）
WINDOW_SIZE = 3000
STEP_SIZE = 200


# 工具函数
def sync_signals(emg_data, angle_data):
    ratio = 10
    angle_len = len(angle_data)
    x_old = np.linspace(0, angle_len, angle_len)
    x_new = np.linspace(0, angle_len, angle_len * ratio)
    angle_resampled = np.zeros((len(x_new), angle_data.shape[1]))
    for i in range(angle_data.shape[1]):
        angle_resampled[:, i] = np.interp(x_new, x_old, angle_data[:, i])
    final_len = min(len(emg_data), len(angle_resampled))
    return emg_data[:final_len], angle_resampled[:final_len]


def emg_denoise(emg):
    b, a = signal.iirnotch(50, 30, fs=1000)
    emg = signal.filtfilt(b, a, emg, axis=0)
    b, a = signal.butter(4, [10, 450], btype="bandpass", fs=1000)
    emg = signal.filtfilt(b, a, emg, axis=0)
    coeffs = pywt.wavedec(emg, "db4", level=5, axis=0)
    threshold = np.std(coeffs[1]) * 0.6745
    coeffs[1:] = [pywt.threshold(c, threshold, "soft") for c in coeffs[1:]]
    emg_denoised = pywt.waverec(coeffs, "db4", axis=0)
    return emg_denoised[:len(emg)]


def angle_denoise(angle):
    b, a = signal.butter(4, 5, "lowpass", fs=1000)
    angle = signal.filtfilt(b, a, angle, axis=0)
    return signal.medfilt(angle, (5, 1))


def extract_features(emg, angle):
    features = []
    # EMG特征（16通道×4）
    for ch in range(emg.shape[1]):
        s = emg[:, ch]
        iemg = np.sum(np.abs(s))
        rms = np.sqrt(np.mean(s ** 2)) if len(s) > 0 else 0
        f, pxx = signal.periodogram(s, fs=1000)
        s_sum = np.sum(pxx)
        mpf = np.sum(f * pxx) / s_sum if s_sum != 0 else 0
        pf = f[np.argmax(pxx)] if len(pxx) > 0 else 0
        features.extend([iemg, rms, mpf, pf])
    # 角度特征（22通道×4）
    for ch in range(angle.shape[1]):
        s = angle[:, ch]
        max_a = np.max(s)
        d1 = np.diff(s) if len(s) > 1 else [0]
        rate = np.mean(np.abs(d1))
        d2 = np.diff(s, n=2) if len(s) > 2 else [0]
        curv = np.mean(np.abs(d2))
        std = np.std(s)
        features.extend([max_a, rate, curv, std])
    return np.nan_to_num(features, nan=0, posinf=0, neginf=0)


# 自动遍历所有.mat文件
all_features = []
all_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".mat")]

for filename in all_files:
    file_path = os.path.join(raw_data_dir, filename)
    print(f"正在处理：{filename}")

    try:
        mat = loadmat(file_path)
        emg = mat["emg"].astype(np.float64)
        angle = mat["glove"].astype(np.float64)
        print(f"数据维度：EMG{emg.shape}，角度{angle.shape}")

        # 同步+去噪+滑窗提取
        emg_s, angle_s = sync_signals(emg, angle)
        max_start = len(emg_s) - WINDOW_SIZE
        if max_start <= 0:
            print(f"数据长度不足，跳过")
            continue

        for i in range(0, max_start, STEP_SIZE):
            e_win = emg_s[i:i + WINDOW_SIZE]
            a_win = angle_s[i:i + WINDOW_SIZE]
            e_den = emg_denoise(e_win)
            a_den = angle_denoise(a_win)
            feat = extract_features(e_den, a_den)
            all_features.append(feat)
    except Exception as e:
        print(f"处理失败：{e}")
        continue

# 保存结果
save_path = os.path.join(output_path, "动作特征数据.csv")
pd.DataFrame(all_features).to_csv(save_path, index=False, header=False)
