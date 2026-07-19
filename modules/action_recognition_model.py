import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "动作特征数据.csv")
LABEL_PATH = os.path.join(BASE_DIR, "data", "processed", "动作标签.csv")
TCM_FEAT_SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "tcm_features.csv")

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "action_model.pkl")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FATIGUE_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "fatigue_model.pkl")

# 随机种子
RANDOM_SEED = 42
# 模型超参数
RF_N_ESTIMATORS = 150
RF_MAX_DEPTH = 12
KNN_NEIGHBORS = 5

#工具函数
def create_dir_if_not_exists(file_path):
    """创建文件所在目录"""
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_data(feat_path, label_path):
    """加载并校验动作特征和标签数据"""
    # 加载数据
    try:
        X = pd.read_csv(feat_path, header=None).values.astype(np.float32)
        y = pd.read_csv(label_path, header=None).values.ravel().astype(int)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"数据文件未找到：{e.filename}")
    except Exception as e:
        raise RuntimeError(f"数据加载失败：{str(e)}")

    # 数据校验
    if len(X) != len(y):
        raise ValueError(f"特征数据（{len(X)}条）与标签数据（{len(y)}条）长度不匹配")
    if X.size == 0:
        raise ValueError("特征数据为空，请检查数据文件")
    return X, y

def generate_tcm_data(n_samples, seed=42, save_to_file=True):
    """
    生成模拟中医特征数据（9维度）
    :param n_samples: 样本数量
    :param seed: 随机种子
    :param save_to_file: 是否保存到CSV文件
    :return: 中医特征数组 (n_samples, 9)
    """
    np.random.seed(seed)
    # 维度含义：
    # 0: 脉搏率(PR) 60-130 | 1: 脉搏振幅(PA) 0.8-2.0 | 2: 脉搏宽度(PW) 0.5-1.5
    # 3: 脉搏变异度(PV) 0.05-0.4 | 4: 红光值(R) 100-200 | 5: 绿光值(G) 100-200
    # 6: 蓝光值(B) 100-200 | 7: 血压值(BR) 80-200 | 8: 体温(CT) 0.1-0.5
    tcm_data = np.column_stack([
        np.random.uniform(60, 130, n_samples),   # PR
        np.random.uniform(0.8, 2.0, n_samples),  # PA
        np.random.uniform(0.5, 1.5, n_samples),  # PW
        np.random.uniform(0.05, 0.4, n_samples), # PV
        np.random.uniform(100, 200, n_samples),  # R
        np.random.uniform(100, 200, n_samples),  # G
        np.random.uniform(100, 200, n_samples),  # B
        np.random.uniform(80, 200, n_samples),   # BR
        np.random.uniform(0.1, 0.5, n_samples)   # CT
    ]).astype(np.float32)

    if save_to_file:
        create_dir_if_not_exists(TCM_FEAT_SAVE_PATH)
        pd.DataFrame(tcm_data).to_csv(TCM_FEAT_SAVE_PATH, index=False, header=False)
    return tcm_data

def calculate_fatigue_score(tcm_data):
    """
    基于中医特征计算疲劳评分（0-1）
    :param tcm_data: 中医特征数组 (n_samples, 9)
    :return: 疲劳评分数组 (n_samples,)
    """
    fatigue_labels = []
    for row in tcm_data:
        pr, pa, pw, pv, r, g, b, br, ct = row
        score = 0.0
        # 疲劳评分规则
        score += 0.4 if pr > 105 else 0.0    # 脉搏率高 → 疲劳（权重0.4）
        score += 0.3 if pv > 0.22 else 0.0   # 脉搏变异度大 → 疲劳（权重0.3）
        score += 0.2 if br < 150 else 0.0    # 血压低 → 疲劳（权重0.2）
        score += 0.1 if ct > 0.3 else 0.0    # 体温高 → 疲劳（权重0.1）
        fatigue_labels.append(min(score, 1.0))  # 限制最大评分1.0
    return np.array(fatigue_labels, dtype=np.float32)

if __name__ == "__main__":
    try:
        # 1. 初始化目录（确保模型/数据目录存在）
        create_dir_if_not_exists(MODEL_SAVE_PATH)
        create_dir_if_not_exists(TCM_FEAT_SAVE_PATH)

        # 2. 加载动作数据
        print("="*60)
        X, y = load_data(FEAT_PATH, LABEL_PATH)
        print(f"数据加载完成，共{len(X)}条样本，特征维度：{X.shape[1]}")

        # 3. 训练动作识别模型（随机森林）
        print("\n训练动作识别模型（随机森林）...")
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)

        # 划分训练/验证集（分层抽样，保证类别分布一致）
        X_train, X_val, y_train, y_val = train_test_split(
            X_norm, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )

        # 训练模型
        action_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        action_model.fit(X_train, y_train)

        # 验证动作模型
        y_val_pred = action_model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"动作模型验证集准确率：{val_acc:.2%}")

        # 4. 生成中医数据并训练疲劳评估模型（KNN回归）
        print("\n生成中医特征数据并训练疲劳模型（KNN回归）...")
        tcm_data = generate_tcm_data(len(X), seed=RANDOM_SEED)
        fatigue_labels = calculate_fatigue_score(tcm_data)

        # 划分中医特征的训练/验证集
        tcm_features = tcm_data[:, [0, 3, 1, 7, 8]]  # 选择关键特征
        tcm_train, tcm_val, fatigue_train, fatigue_val = train_test_split(
            tcm_features, fatigue_labels, test_size=0.2, random_state=RANDOM_SEED
        )

        # 训练疲劳模型
        fatigue_model = KNeighborsRegressor(
            n_neighbors=KNN_NEIGHBORS,
            weights="distance",  # 距离加权，近邻权重更高
            n_jobs=-1
        )
        fatigue_model.fit(tcm_train, fatigue_train)

        # 验证疲劳模型（使用验证集，而非全量数据）
        fatigue_pred = fatigue_model.predict(tcm_val)
        mse = mean_squared_error(fatigue_val, fatigue_pred)
        r2 = r2_score(fatigue_val, fatigue_pred)
        print(f"疲劳模型验证集 - MSE（均方误差）：{mse:.4f} | R²（决定系数）：{r2:.4f}")

        joblib.dump(action_model, MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        joblib.dump(fatigue_model, FATIGUE_MODEL_SAVE_PATH)


    except Exception as e:
        print(f"\n训练过程出错：{str(e)}")
        raise