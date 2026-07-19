import os
import numpy as np
import pandas as pd
from scipy import signal
import pywt
from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 多受试者配置（对齐参考代码的SUBJECTS_TO_PROCESS）
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw")
SUBJECTS_TO_PROCESS = [1, 2, 3, 4, 5]  # 可配置的受试者列表
EXERCISES_TO_PROCESS = [1, 2, 3]  # 对应动作类型的练习集
WINDOW_SIZE = 200  # 替换原有3000硬编码，对齐参考代码的200帧窗口
STEP = 50  # 步长50，平衡计算量和特征覆盖


# 1. 多受试者数据加载函数（复用参考代码逻辑）
def load_multisubject_ninapro_data(base_path, subjects, exercises):
    all_emg = []
    all_angle = []
    for subject_id in subjects:
        subject_emg = None
        subject_angle = None
        for exercise in exercises:
            file_path = os.path.join(base_path, f'S{subject_id}_A1_E{exercise}.mat')
            if not os.path.exists(file_path):
                print(f"警告：未找到文件 {file_path}，跳过该练习")
                continue
            mat_data = loadmat(file_path)
            emg = mat_data['emg'].astype(np.float64)  # 16通道EMG
            angle = mat_data['glove'].astype(np.float64)  # 22维角度

            # 合并同一受试者的多练习数据
            if subject_emg is None:
                subject_emg = emg
                subject_angle = angle
            else:
                subject_emg = np.vstack((subject_emg, emg))
                subject_angle = np.vstack((subject_angle, angle))

        if subject_emg is not None:
            all_emg.append(subject_emg)
            all_angle.append(subject_angle)
            print(f"受试者{subject_id}加载完成：EMG{subject_emg.shape}，角度{subject_angle.shape}")

    return all_emg, all_angle


# 2. 改造窗口生成逻辑（复用参考代码的create_windows）
def create_feature_windows(emg_sync, angle_sync, window_size=WINDOW_SIZE, step=STEP):
    """
    对齐参考代码的窗口化逻辑，仅保留有效动作区间（排除静止帧）
    """
    # 模拟动作标签（若有restimulus字段可替换，无则用EMG幅值判定有效区间）
    emg_amplitude = np.mean(np.abs(emg_sync), axis=1)
    active_indices = np.where(emg_amplitude > 0.1)[0]  # 排除静止段

    features_list = []
    for i in range(0, len(active_indices) - window_size, step):
        window_indices = active_indices[i: i + window_size]
        # 确保窗口连续
        if window_indices[-1] - window_indices[0] != window_size - 1:
            continue

        # 提取窗口内数据
        emg_window = emg_sync[window_indices]
        angle_window = angle_sync[window_indices]

        # 参考代码的标准化逻辑（替换原有固定scaler）
        mean = np.mean(emg_window, axis=0)
        std = np.std(emg_window, axis=0)
        emg_window = (emg_window - mean) / (std + 1e-8)  # 防除零

        # 去噪+特征提取
        emg_denoised = emg_denoise(emg_window)
        angle_denoised = angle_denoise(angle_window)
        feat = extract_features(emg_denoised, angle_denoised)
        features_list.append(feat)

    return np.array(features_list)


# 多受试者批量处理
if __name__ == "__main__":
    try:
        # 加载多受试者数据
        all_emg, all_angle = load_multisubject_ninapro_data(DATASET_PATH, SUBJECTS_TO_PROCESS, EXERCISES_TO_PROCESS)
        all_features = []

        # 逐受试者处理
        for idx in range(len(all_emg)):
            emg_data = all_emg[idx]
            angle_data = all_angle[idx]

            # 信号同步
            emg_sync, angle_sync = sync_signals(emg_data, angle_data)
            # 生成窗口化特征
            subject_features = create_feature_windows(emg_sync, angle_sync)
            all_features.append(subject_features)

        # 合并所有受试者特征
        final_features = np.concatenate(all_features, axis=0)
        # 保存特征（供动作识别模型训练/推理）
        save_path = os.path.join(BASE_DIR, "data/processed/动作特征数据.csv")
        pd.DataFrame(final_features).to_csv(save_path, index=False, header=False)

        print(f"多受试者预处理完成：共{final_features.shape[0]}条特征，维度{final_features.shape[1]}")
    except Exception as e:
        print(f"报错：{e}")
        import traceback

        traceback.print_exc()