import os
import numpy as np
import pandas as pd
import joblib

BASE_DIR = r"C:\Users\chenny\PycharmProjects\recover"

ACTION_FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "动作特征数据.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "action_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FATIGUE_MODEL_PATH = os.path.join(BASE_DIR, "models", "fatigue_model.pkl")
TCM_FEAT_PATH = os.path.join(BASE_DIR, "data", "processed", "tcm_features.csv")

#动作映射
action_map = {
    0: "休息", 1: "握拳", 2: "食指伸",
    3: "中指伸", 4: "无名指伸", 5: "小指伸"
}

#加载模型
action_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
fatigue_model = joblib.load(FATIGUE_MODEL_PATH)

#加载真实中医数据
tcm_features = pd.read_csv(TCM_FEAT_PATH).values[1:].astype(np.float32)  # <-- 修复在这里！

# VR绑定
def bind_action_to_vr_scene(action_id):
    bind_table = {
        0: {"scene": "休息康复区", "prop": "放松垫", "task": "静息恢复"},
        1: {"scene": "力量训练室", "prop": "握力球", "task": "握拳训练"},
        2: {"scene": "精细操作室", "prop": "按键按钮", "task": "食指伸展"},
        3: {"scene": "伸展训练室", "prop": "触碰标记", "task": "中指伸展"},
        4: {"scene": "灵活训练室", "prop": "环形道具", "task": "无名指伸展"},
        5: {"scene": "康复花园", "prop": "轻握花朵", "task": "小指伸展"}
    }
    return bind_table[action_id]

#疲劳评分（KNN）
def get_fatigue_score(tcm_feat):
    feat = tcm_feat[[0, 3, 1, 7, 8]].reshape(1, -1)
    score = float(fatigue_model.predict(feat)[0])
    return round(np.clip(score, 0.0, 1.0), 2)

# 自动调整训练参数
def adjust_params_by_tcm(fatigue_score):
    base = {
        "speed": 1.0,
        "amplitude": 1.0,
        "difficulty": "中等",
        "rest_time": 2
    }
    if fatigue_score >= 0.66:
        base["speed"] = 0.5
        base["amplitude"] = 0.6
        base["difficulty"] = "简单"
        base["rest_time"] = 5
    elif fatigue_score >= 0.33:
        base["speed"] = 0.75
        base["amplitude"] = 0.85
        base["difficulty"] = "中等偏易"
        base["rest_time"] = 3
    return base

# 生成康复报告
def generate_rehab_report(action_name, vr_info, params, fatigue, confidence):
    return {
        "识别动作": action_name,
        "VR场景": vr_info["scene"],
        "VR道具": vr_info["prop"],
        "训练任务": vr_info["task"],
        "疲劳指数": fatigue,
        "识别置信度": round(confidence, 2),
        "训练速度": params["speed"],
        "动作幅度": params["amplitude"],
        "难度": params["difficulty"],
        "休息时长": params["rest_time"],
        "中医状态": "疲劳" if fatigue >= 0.5 else "正常",
        "结论": "已根据中医状态智能优化训练方案"
    }

if __name__ == "__main__":
    print("=" * 80)
    print("多模态智能康复训练系统（真实数据版）")
    print("=" * 80)

    # 加载动作特征
    act_feats = pd.read_csv(ACTION_FEAT_PATH, header=None).values.astype(np.float32)
    print(f"成功加载动作数据：{len(act_feats)} 条")
    print(f"成功加载中医特征：{len(tcm_features)} 条")
    print(f"动作识别模型、疲劳模型、标准化器 加载完成\n")

    # 逐条推理
    for i in range(min(10, len(act_feats))):
        feat = act_feats[i]
        tcm_feat = tcm_features[i]

        # 动作识别
        x = scaler.transform(feat.reshape(1, -1))
        act_id = int(action_model.predict(x)[0])
        confidence = max(action_model.predict_proba(x)[0])

        # VR绑定
        vr_info = bind_action_to_vr_scene(act_id)

        # 疲劳评估
        fatigue = get_fatigue_score(tcm_feat)

        # 调整参数
        params = adjust_params_by_tcm(fatigue)

        # 生成报告
        report = generate_rehab_report(
            action_map[act_id], vr_info, params, fatigue, confidence
        )

        # 输出
        print(f"【第 {i+1} 条康复记录】")
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("-" * 60)

    print("\n系统运行完成！")