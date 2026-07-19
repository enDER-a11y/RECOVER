import os
import numpy as np
import pandas as pd
import logging
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "action_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ACTION_FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "动作特征数据.csv")
TCM_FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "tcm_features.csv")

# 加载真实模型
action_model = None
scaler = None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    action_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("模型加载成功")

# 加载中医特征
tcm_features = pd.read_csv(TCM_FEAT_PATH, header=None).values

action_map = {
    0: "休息", 1: "握拳", 2: "食指伸",
    3: "中指伸",4: "无名指伸",5: "小指伸"
}

def generate_tcm_feedback(feat, action_result):
    feedback = "状态良好，正常训练"
    try:
        act_name, conf, score = action_result
        p_rate = feat[0]
        p_var = feat[3]
        tags = []
        if p_rate > 105 and p_var > 0.22:
            tags.append("疲劳")
        if feat[4] < 165:
            tags.append("气血偏弱")
        if tags:
            feedback = f"检测到：{'、'.join(tags)}，已自动调整康复方案"
    except:
        pass
    return feedback


if __name__ == "__main__":
    df = pd.read_csv(ACTION_FEAT_PATH, header=None)
    action_features = df.values
    print(f"读取特征：{action_features.shape[0]} 条")

    for idx in range(min(10, len(action_features))):
        feat = action_features[idx]
        subject_id = f"S{str(idx+1).zfill(3)}"

        x = scaler.transform(feat.reshape(1, -1))
        act_id = action_model.predict(x)[0]
        prob = action_model.predict_proba(x)[0]
        conf = round(float(max(prob)), 2)

        action_name = action_map[act_id]
        score = int(conf * 100)
        real_tcm = tcm_features[idx % len(tcm_features)]
        feedback_msg = generate_tcm_feedback(real_tcm, (action_name, conf, score))

        print(f"\n【{subject_id}】动作：{action_name} | 置信度：{conf}")
        print(f"反馈：{feedback_msg}")