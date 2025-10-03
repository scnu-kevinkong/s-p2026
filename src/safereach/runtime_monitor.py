import os
import json
import subprocess
import re
import time
import math
from fractions import Fraction
from typing import Dict, List
from pathlib import Path
from agentspec.agent import Action
from .abstraction import Abstraction
from .embodied.abstraction import EmbodiedAbstraction
# 移除对AVAbstraction的直接导入以避免在Embodied评测中引入rtamt依赖
# from .autonomous_vehicle.abstraction import AVAbstraction

# 计算从当前状态最终到达不安全集合的概率（使用model.json，避免依赖PRISM本地库）
def _reach_prob_vector_from_model(model_json_path: str, unsafe_states: set, use_ucb: bool = False, ucb_coeff: float = 0.0) -> List[float]:
    with open(model_json_path, 'r') as f:
        m = json.load(f)
    states = m['states']
    state_index = m['state_index']
    trans = m['transition_probs']
    counts = m.get('transition_counts', None)

    n = len(states)
    # 将字符串分数转换为浮点数，并按索引构建转移矩阵的稀疏表示
    index_to_key = {idx: key for key, idx in state_index.items()}
    finish_idx = state_index.get('finish', None)

    # 预构建每个状态的出边
    outgoing: Dict[int, Dict[int, float]] = {}
    out_deg: Dict[int, int] = {}
    total_counts: Dict[int, int] = {}
    for key, edges in trans.items():
        i = state_index[key]
        row: Dict[int, float] = {}
        for k2, w in edges.items():
            j = state_index[k2]
            try:
                val = float(Fraction(w))
            except Exception:
                if isinstance(w, str) and '/' in w:
                    num, den = w.split('/')
                    val = float(num) / float(den)
                else:
                    val = float(w)
            row[j] = val
        outgoing[i] = row
        out_deg[i] = len(row)
        # 统计该状态的总观测转移计数（若提供）
        if counts is not None and key in counts:
            try:
                total_counts[i] = sum(int(v) for v in counts[key].values())
            except Exception:
                total_counts[i] = 0
        else:
            total_counts[i] = 0

    # 初始化边界条件向量
    p = [0.0] * n
    for s in unsafe_states:
        if 0 <= s < n:
            p[s] = 1.0
    if finish_idx is not None:
        p[finish_idx] = 0.0

    # 迭代求解 p = T p （边界条件在p中固定）
    max_iters = 2000
    tol = 1e-10
    for _ in range(max_iters):
        changed = 0.0
        p_new = p[:]
        for i in range(n):
            # 边界状态保持不变
            if i in unsafe_states:
                p_new[i] = 1.0
                continue
            if finish_idx is not None and i == finish_idx:
                p_new[i] = 0.0
                continue
            ssum = 0.0
            for j, prob in outgoing.get(i, {}).items():
                ssum += prob * p[j]
            p_new[i] = ssum
            changed = max(changed, abs(p_new[i] - p[i]))
        p = p_new
        if changed < tol:
            break

    # UCB后处理：对非边界状态给予不确定性乐观上界
    if use_ucb and ucb_coeff > 0.0:
        p_ucb = p[:]
        for i in range(n):
            if i in unsafe_states or (finish_idx is not None and i == finish_idx):
                continue
            # 若提供了计数，则基于样本量进行不确定性项缩放；否则回退出度缩放
            Ni = total_counts.get(i, 0)
            if Ni > 0:
                epsilon_i = ucb_coeff / math.sqrt(Ni + 1.0)
            else:
                deg = max(1, out_deg.get(i, 1))
                epsilon_i = ucb_coeff / (deg ** 0.5)
            p_ucb[i] = min(1.0, p[i] + epsilon_i)
        p = p_ucb

    return p


def _reach_prob_from_model(model_json_path: str, current_state_idx: int, unsafe_states: set, use_ucb: bool = False, ucb_coeff: float = 0.0) -> float:
    p = _reach_prob_vector_from_model(model_json_path, unsafe_states, use_ucb=use_ucb, ucb_coeff=ucb_coeff)
    return p[current_state_idx]

 
#returns true if intervention needed.  
def runtime_monitor(observation, dtmc_path, abs: Abstraction, unsafe_states, cache={}):  
    # print(cache)
    state_idx = abs.get_state_idx()

    # Step 1: Abstraction
    # t = time.time()
    current_state = abs.encode(observation)
    current_state = state_idx[current_state]
    # t = time.time() - t
    # print(f"abstract: {t * 1000:.2f} ms")

    # Step 2: Cache check
    # print(cache)
    cache_key = (current_state, str(unsafe_states))
    cache_hit = cache_key in cache
    if cache_hit:
        return cache[cache_key]

    # Step 2.1: 离线预计算所有状态的到达概率（可选）
    precompute = os.environ.get("PRECOMPUTE_VALUES", "off").lower() == "on"
    if precompute:
        base_dir = Path(__file__).resolve().parent
        model_json_path = os.path.join(os.path.dirname(dtmc_path), 'model.json')
        if not os.path.exists(model_json_path):
            default_model_json = str(base_dir.parent / 'default_av.dtmcmodel.json')
            if os.path.exists(default_model_json):
                model_json_path = default_model_json
        try:
            use_ucb = (os.environ.get("SAFE_REACH_UCB", "off").lower() == "on") or (os.environ.get("AV_UCB", "off").lower() == "on")
            ucb_coeff = float(os.environ.get("UCB_COEFF", "0.0"))
            p_vec = _reach_prob_vector_from_model(model_json_path, set(unsafe_states), use_ucb=use_ucb, ucb_coeff=ucb_coeff)
            # 写入缓存
            # 获取索引数量
            with open(model_json_path, 'r') as f:
                m = json.load(f)
            n = len(m['states'])
            for s in range(n):
                cache[(s, str(unsafe_states))] = p_vec[s]
            # 预计算后直接返回当前状态概率
            return cache[cache_key]
        except Exception:
            # 预计算失败则继续常规流程
            pass
    
    # Step 3: Rewrite DTMC init state
    with open(dtmc_path, 'r') as f:
        model_txt = f.read()

    updated_model = re.sub(
        r's\s*:\s*\[\d+\.\.\d+\]\s+init\s+\d+;',
        lambda m: re.sub(r'init\s+\d+', f'init {current_state}', m.group(0)),
        model_txt
    )

    with open(dtmc_path, 'w') as f:
        f.write(updated_model)
    # Step 4: Run PRISM to check the prob reaching unsafe state
    states = ""
    if len(unsafe_states)==1:
        state = f"s={list(unsafe_states)[0]}"
    else :
        state = "(" +  "|".join([f"s={s}" for s in unsafe_states]) + ")"

    pctl_formula = f"P=? [ F {state} ]"

    # 优先使用环境变量 PRISM_HOME 指向的Linux版PRISM；若不存在则回退为本地prism目录并直接调用Java
    prism_home = os.environ.get('PRISM_HOME')
    if prism_home is None:
        prism_home = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'prism'))

    prism_bin = os.path.join(prism_home, 'bin', 'prism')
    use_script = False  # Force Java fallback due to shell incompatibility of prism script on this environment

    stdout = ''
    stderr = ''
    prob = None
    if use_script:
        # 通过PRISM脚本调用（自动处理类路径和本地库）
        cmd = [prism_bin, dtmc_path, '-pf', pctl_formula]
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f"{os.path.join(prism_home, 'lib')}:{env.get('LD_LIBRARY_PATH','')}"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        stdout = result.stdout
        stderr = result.stderr
    else:
        # 回退：直接以Java方式调用（需要库路径与类路径）
        prism_dir = prism_home
        classpath = ":".join([
            os.path.join(prism_dir, 'lib', 'prism.jar'),
            os.path.join(prism_dir, 'classes'),
            prism_dir,
            os.path.join(prism_dir, 'lib', 'pepa.zip'),
            os.path.join(prism_dir, 'lib', '*'),
        ])

        java_cmd = [
            'java',
            '-Xmx1g',
            '-Xss4m',
            '-Djava.awt.headless=true',
            f"-Djava.library.path={os.path.join(prism_dir, 'lib')}",
            '-classpath',
            classpath,
            'prism.PrismCL',
            dtmc_path,
            '-pf',
            pctl_formula,
        ]
        result = subprocess.run(java_cmd, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr

    match = re.search(r"Result:\s*([0-9.]+)", stdout)
    if match:
        prob = float(match.group(1))

    # 如果PRISM不可用或解析失败，回退到纯Python基于model.json计算
    if prob is None:
        try:
            # 首先尝试与dtmc同目录的model.json
            model_json_path = os.path.join(os.path.dirname(dtmc_path), 'model.json')
            if not os.path.exists(model_json_path):
                # 回退到src目录下的默认模型文件 default_av.dtmcmodel.json
                base_dir = Path(__file__).resolve().parent  # safereach目录
                default_model_json = base_dir.parent / 'default_av.dtmcmodel.json'
                if default_model_json.exists():
                    model_json_path = str(default_model_json)
            use_ucb = (os.environ.get("SAFE_REACH_UCB", "off").lower() == "on") or (os.environ.get("AV_UCB", "off").lower() == "on")
            ucb_coeff = float(os.environ.get("UCB_COEFF", "0.0"))
            prob = _reach_prob_from_model(model_json_path, current_state, unsafe_states, use_ucb=use_ucb, ucb_coeff=ucb_coeff)
        except Exception as e:
            # 打印便于诊断
            print(stdout)
            print(stderr)
            print(f"Model JSON fallback failed: {e}")
            raise RuntimeError("Could not parse probability from PRISM output.")

    cache[(current_state, str(unsafe_states))] = prob  # Save to cache
    return prob
    # if prob > threshold:
    #     return True
