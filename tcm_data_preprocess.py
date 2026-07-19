# 中医生理信号预处理（模拟数据版，技术路线模块1/4）
import pandas as pd
import numpy as np
from scipy import signal

output_path = "./data/processed"

# 1. 生成模拟脉诊数据（4通道MEMS，500Hz采样）
def generate_simulate_pulse_data():
    fs = 500
    t = np.linspace(0, 3, 3*fs)
    pulse_samples = []
    for i in range(50):
        # 模拟不同脉率（60-130次/分）和幅值波动
        pulse_rate = np.random.uniform(60, 130)
        pulse_amp = np.random.uniform(0.5, 2.0)
        pulse = pulse_amp * np.sin(2*np.pi*(pulse_rate/60)*t) + 0.1 * np.random.randn(len(t))
        # 0.5-20Hz带通滤波
        b, a = signal.iirfilter(4, [0.5, 20], btype="bandpass", fs=fs)
        pulse_denoised = signal.filtfilt(b, a, pulse)
        pulse_samples.append(pulse_denoised)
    return np.array(pulse_samples)

# 2. 脉诊特征提取
def extract_pulse_feat(pulse):
    peaks, _ = signal.find_peaks(pulse, height=0.1)
    pulse_rate = len(peaks) * 20  # 3秒→每分钟脉率
    peak_amp = np.mean(pulse[peaks]) if len(peaks) > 0 else 0
    pulse_width = np.mean(np.diff(peaks)) / 500 * 1000 if len(peaks) > 1 else 0
    cv_amp = np.std(pulse[peaks]) / np.mean(pulse[peaks]) if len(peaks) > 0 else 0
    return [pulse_rate, peak_amp, pulse_width, cv_amp]

# 3. 生成模拟舌诊特征（RGB、亮度、苔色占比）
def generate_simulate_tongue_feat():
    tongue_feats = []
    for i in range(50):
        # 舌色R值：140=气血不足，200=正常
        r_mean = np.random.uniform(140, 200)
        g_mean = np.random.uniform(130, 190)
        b_mean = np.random.uniform(120, 180)
        brightness = (r_mean + g_mean + b_mean) / 3
        # 苔色占比：0.1-0.3
        coating_ratio = np.random.uniform(0.1, 0.3)
        tongue_feats.append([r_mean, g_mean, b_mean, brightness, coating_ratio])
    return np.array(tongue_feats)

# 主流程：生成模拟数据→提取特征
pulse_samples = generate_simulate_pulse_data()
tongue_feats = generate_simulate_tongue_feat()

# 提取50条中医特征样本
tcm_feat_list = []
for i in range(50):
    pulse_feat = extract_pulse_feat(pulse_samples[i])
    tcm_feat = pulse_feat + tongue_feats[i].tolist()
    tcm_feat_list.append(tcm_feat)

# 保存中医特征数据
pd.DataFrame(tcm_feat_list).to_csv(output_path + "中医特征数据.csv", index=False, header=False)
print("模块1：模拟中医数据预处理完成！")