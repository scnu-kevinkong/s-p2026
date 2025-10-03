import os
import pickle
import json
from .autonomous_vehicle.TracePreprocess import Trace, raw_to_lawbreaker_API
from .build_model import *
from .autonomous_vehicle.abstraction import AVAbstraction, traffic_rules, parse_law


def av_build_model(abs, logs, model_path = "default_av.dtmc"): 
    model = build_model(logs, abs)
    store_model(model, model_path, abs)
    
# 覆盖本仓库 samples 中出现的所有法则 -> 对应 traffic_rules 的键
rules_map = {
    "Law38_0_2": "rule38_1",   # 绿灯通行子法则
    "Law38_1_1": "rule38_1",
    "Law38_2_1": "rule38_2",   # 黄灯相关子法则
    "Law38_2_2": "rule38_2",
    "Law44_2": "rule44",
    "Law53_0_1" : "rule53"
}

LOG_F = 'safereach/autonomous_vehicle/Law38_0_2_record_1.00000.20250618131620.record.pickle'
SAMPLES = "safereach/autonomous_vehicle/samples/"
# 确保输出根目录存在
MODEL_BASE = "safereach/autonomous_vehicle/dtmcs/"
os.makedirs(MODEL_BASE, exist_ok=True)

# 针对每个样本法则构建专属 DTMC/PRISM 模型
for s in os.listdir(SAMPLES):
    SCENEARIO = f"{SAMPLES}{s}/" 
    if s not in rules_map:
        print(f"[av_build] skip {s}: no rule mapping")
        continue

    law = traffic_rules[rules_map[s]]
    predicates = parse_law(law)
    abs = AVAbstraction(predicates)
    logs = []
    for f in os.listdir(SCENEARIO):
        if not f.endswith("pickle"):
            continue
        with open(f"{SCENEARIO}{f}", 'rb') as pic:
            trace = pickle.load(pic)["trace"] 
            time_seq = sorted(list(trace.keys()))
            if len(time_seq) == 0:
                continue
            initial_timestamp = time_seq[0]
            # 按时间顺序构造 observation 序列
            logs.append([raw_to_lawbreaker_API(trace[t], initial_timestamp) for t in time_seq])

    # 输出目录需以斜杠结尾，避免路径拼接错误
    model_dir = os.path.join(MODEL_BASE, s) + "/"
    av_build_model(abs, logs, model_path=model_dir)

    