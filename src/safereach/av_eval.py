import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product

from .autonomous_vehicle.abstraction import AVAbstraction
from .runtime_monitor import runtime_monitor
from fractions import Fraction

BASE_DIR = Path(os.path.dirname(__file__))
LOG_DIR = str(BASE_DIR / "autonomous_vehicle" / "samples")
MODEL_DIR = str(BASE_DIR / "autonomous_vehicle" / "dtmcs")
DEFAULT_MODEL_PREFIX = str(BASE_DIR.parent / "default_av.dtmc")

# Default probability thresholds (can be overridden by env)
THRESHOLDS = [float(x) for x in os.environ.get("AV_THRESHOLDS", "0.3,0.5,0.7").split(",")]


def _dtmc_paths(folder: str) -> Tuple[str, str]:
    """Return (dtmc_prism_path, model_json_path) using per-folder model if present,
    otherwise fall back to top-level default_av.* files.
    """
    prism_path = f"{MODEL_DIR}/{folder}/dtmc.prism"
    model_json = f"{MODEL_DIR}/{folder}/model.json"
    if os.path.exists(prism_path) and os.path.exists(model_json):
        return prism_path, model_json
    # fallback: default_av.dtmcdtmc.prism and default_av.dtmcmodel.json in src/
    return f"{DEFAULT_MODEL_PREFIX}dtmc.prism", f"{DEFAULT_MODEL_PREFIX}model.json"


def _load_spec_states(model_folder: str, abs: AVAbstraction) -> List[int]:
    spec_path = f"{LOG_DIR}/{model_folder}/spec"
    spec = None
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            spec = json.loads(f.read())
    else:
        # 默认：仅以碰撞为不安全态
        spec = [[["collision", "==", "1"], True]]
    unsafe_states = abs.filter(spec)
    return [abs.get_state_idx()[state] for state in unsafe_states]


def _iter_logs(folder: str):
    for o in os.listdir(f"{LOG_DIR}/{folder}"):
        if o.endswith(".json"):
            yield f"{LOG_DIR}/{folder}/{o}"


# 兼容两种日志结构：{"s_trans": [{"state": {...}}, ...]} 或 {"trajectory": [{...}, ...]}
def _extract_observations(obj: Dict) -> List[Dict]:
    if "s_trans" in obj:
        return [t.get("state", t) for t in obj.get("s_trans", [])]
    if "trajectory" in obj:
        return obj.get("trajectory", [])
    return []


def _step_time(ob: Dict) -> float:
    # 使用每步时间字段（若无则默认1秒），便于统计提前量
    try:
        return float(ob.get("time", 1.0))
    except Exception:
        return 1.0


def _load_abstraction(folder: str) -> Dict:
    # 优先使用场景特定抽象；否则回退到默认default_av.dtmcabstraction.json；再否则给空抽象
    abs_path = f"{MODEL_DIR}/{folder}/abstraction.json"
    if os.path.exists(abs_path):
        with open(abs_path) as af:
            return json.loads(af.read())
    default_abs_path = str(BASE_DIR.parent / "default_av.dtmcabstraction.json")
    if os.path.exists(default_abs_path):
        with open(default_abs_path) as daf:
            return json.loads(daf.read())
    return {"predicates": []}


def eval_law(folder: str, abs: AVAbstraction, unsafe_states: List[int], thresholds: Optional[List[float]] = None) -> Dict:
    # Compute Time Ahead (first time probability crosses a threshold) and Prediction% for each threshold
    reachability_cache: Dict = {}
    ths = thresholds if (thresholds is not None and len(thresholds) > 0) else THRESHOLDS
    prediction_counts = {theta: 0 for theta in ths}
    total_cases = 0
    time_ahead_acc = {theta: [] for theta in ths}

    prism_path, _ = _dtmc_paths(folder)

    for log in _iter_logs(folder):
        with open(log) as lf:
            obj = json.loads(lf.read())
        observations = _extract_observations(obj)
        if len(observations) == 0:
            continue
        total_cases += 1
        # 为每个阈值记录是否已预测（避免重复计数），并记录首次越阈时间
        predicted_flags = {theta: False for theta in ths}
        running_time = 0.0
        prev_t = None
        for observation in observations:
            # 计算每步 dt：优先使用显式 dt；否则使用相邻观测的绝对时间差；再否则退回 1.0
            dt_val = observation.get("dt", None)
            if dt_val is None:
                t_val = observation.get("time", None)
                if t_val is None:
                    step_dt = 1.0
                else:
                    try:
                        t_cur = float(t_val)
                    except Exception:
                        t_cur = None
                    if t_cur is None:
                        step_dt = 1.0
                    else:
                        if prev_t is None:
                            step_dt = 0.0
                        else:
                            step_dt = max(0.0, float(t_cur) - float(prev_t))
                        prev_t = t_cur
            else:
                try:
                    step_dt = float(dt_val)
                except Exception:
                    step_dt = 1.0
            running_time += step_dt
            prob = runtime_monitor(observation, prism_path, abs, set(unsafe_states), cache=reachability_cache)
            for theta in ths:
                if (not predicted_flags[theta]) and prob >= theta:
                    predicted_flags[theta] = True
                    prediction_counts[theta] += 1
                    time_ahead_acc[theta].append(running_time)
        # 每个case可能在不同阈值分别记录一次首次预测时间
    # Aggregate
    res = {
        "folder": folder,
        "total_cases": total_cases,
        "prediction_pct": {theta: (100.0 * prediction_counts[theta] / total_cases) if total_cases else 0.0 for theta in ths},
        "time_ahead_s": {theta: (sum(time_ahead_acc[theta]) / len(time_ahead_acc[theta])) if time_ahead_acc[theta] else 0.0 for theta in ths},
    }
    return res


def _load_model(model_json_path: str) -> Tuple[int, Dict[int, Dict[int, float]], Dict[str,int]]:
    with open(model_json_path, 'r') as f:
        m = json.load(f)
    states = m['states']
    state_index = m['state_index']
    trans = m['transition_probs']
    n = len(states)
    outgoing: Dict[int, Dict[int, float]] = {}
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
    return n, outgoing, state_index


def _finite_window_stay_prob(n: int, outgoing: Dict[int, Dict[int, float]], allowed_set: set, H: int) -> List[float]:
    # P_k[i] = prob of staying within allowed for k steps starting from i
    P = [1.0 if (i in allowed_set) else 0.0 for i in range(n)]
    for _ in range(H):
        P_next = [0.0] * n
        for i in range(n):
            if i not in allowed_set:
                P_next[i] = 0.0
                continue
            ssum = 0.0
            for j, prob in outgoing.get(i, {}).items():
                ssum += prob * P[j]
            P_next[i] = ssum
        P = P_next
    return P


def eval_reach(
    folder: str,
    abs: AVAbstraction,
    unsafe_states: List[int],
    base_theta_override: Optional[float] = None,
    adapt_override: Optional[bool] = None,
    theta_min_override: Optional[float] = None,
    theta_max_override: Optional[float] = None,
    time_cap_override: Optional[float] = None,
    stay_H_override: Optional[int] = None,
    stay_thresh_override: Optional[float] = None,
) -> Dict:
    # For reaching unsafe states, we evaluate before/after runtime enforcement similar to embodied
    reachability_cache: Dict = {}
    total_cases = 0
    violated_before_cases = 0
    violated_after_cases = 0
    completed_cases = 0

    prism_path, _ = _dtmc_paths(folder)

    # Threshold for enforcement
    base_theta = float(os.environ.get("AV_THETA", "0.5"))
    adapt = os.environ.get("AV_ADAPT_THETA", "off").lower() == "on"
    theta_min = float(os.environ.get("AV_THETA_MIN", str(base_theta)))
    theta_max = float(os.environ.get("AV_THETA_MAX", str(theta_min)))
    time_cap = float(os.environ.get("AV_TIME_CAP", "5.0"))
    # per-law calibration via env JSON mapping {folder_name: {min: x, max: y, base: z}}
    mapping_str = os.environ.get("AV_THETA_PER_LAW", "")
    if mapping_str:
        try:
            mapping = json.loads(mapping_str)
            if folder in mapping:
                # Update min/max for adaptive mode
                theta_min = float(mapping[folder].get("min", theta_min))
                theta_max = float(mapping[folder].get("max", theta_max))
                # If not adaptive, allow overriding base_theta via 'base';
                # or if min==max provided, use that as base fallback.
                if not adapt:
                    if "base" in mapping[folder]:
                        base_theta = float(mapping[folder]["base"]) 
                    elif "min" in mapping[folder] and "max" in mapping[folder] and str(mapping[folder]["min"]) == str(mapping[folder]["max"]):
                        try:
                            base_theta = float(mapping[folder]["min"]) 
                        except Exception:
                            pass
        except Exception:
            pass

    # Provide built-in static defaults for known laws when no env mapping supplied
    if not mapping_str:
        if folder == "Law53_0_1":
            base_theta = 0.8
        elif folder == "Law44_2":
            base_theta = 0.4
        elif folder == "Law38_1_1":
            base_theta = 0.43

    # Apply overrides if provided
    if base_theta_override is not None:
        try:
            base_theta = float(base_theta_override)
        except Exception:
            pass
    if adapt_override is not None:
        try:
            adapt = bool(adapt_override)
        except Exception:
            pass
    if theta_min_override is not None:
        try:
            theta_min = float(theta_min_override)
        except Exception:
            pass
    if theta_max_override is not None:
        try:
            theta_max = float(theta_max_override)
        except Exception:
            pass
    if time_cap_override is not None:
        try:
            time_cap = float(time_cap_override)
        except Exception:
            pass

    # Optional STL-style finite-horizon stay-safe gating
    p_stay_vec = None
    # prefer overrides; fallback to env
    stay_H = None
    stay_thresh = None
    if stay_H_override is not None:
        try:
            stay_H = int(stay_H_override)
        except Exception:
            stay_H = None
    if stay_thresh_override is not None:
        try:
            stay_thresh = float(stay_thresh_override)
        except Exception:
            stay_thresh = None
    if stay_H is None:
        stay_h_env = os.environ.get("AV_STAY_H", "").strip()
        if stay_h_env:
            try:
                stay_H = int(float(stay_h_env))
            except Exception:
                stay_H = None
    if stay_thresh is None:
        try:
            stay_thresh = float(os.environ.get("AV_STAY_THRESH", "0.6"))
        except Exception:
            stay_thresh = 0.6
    if stay_H is not None:
        try:
            _, model_json_path = _dtmc_paths(folder)
            n, outgoing, state_index = _load_model(model_json_path)
            allowed = set(range(n)) - set(unsafe_states)
            p_stay_vec = _finite_window_stay_prob(n, outgoing, allowed, stay_H)
        except Exception:
            p_stay_vec = None

    for log in _iter_logs(folder):
        with open(log) as lf:
            obj = json.loads(lf.read())
        observations = _extract_observations(obj)
        if len(observations) == 0:
            continue
        total_cases += 1
        obs_after_mon = []
        running_time = 0.0
        prev_t = None
        for observation in observations:
            obs_after_mon.append(observation)
            # 计算每步 dt：优先使用显式 dt；否则使用相邻观测的绝对时间差；再否则退回 1.0
            dt_val = observation.get("dt", None)
            if dt_val is None:
                t_val = observation.get("time", None)
                if t_val is None:
                    step_dt = 1.0
                else:
                    try:
                        t_cur = float(t_val)
                    except Exception:
                        t_cur = None
                    if t_cur is None:
                        step_dt = 1.0
                    else:
                        if prev_t is None:
                            step_dt = 0.0
                        else:
                            step_dt = max(0.0, float(t_cur) - float(prev_t))
                        prev_t = t_cur
            else:
                try:
                    step_dt = float(dt_val)
                except Exception:
                    step_dt = 1.0
            running_time += step_dt
            prob = runtime_monitor(observation, prism_path, abs, set(unsafe_states), cache=reachability_cache)
            dyn_theta = base_theta
            if adapt:
                frac = min(1.0, running_time / max(1e-6, time_cap))
                dyn_theta = theta_min + (theta_max - theta_min) * frac
            if prob >= dyn_theta:
                if p_stay_vec is not None and stay_thresh is not None:
                    try:
                        cur_idx = abs.get_state_idx()[abs.encode(observation)]
                        if p_stay_vec[cur_idx] < float(stay_thresh):
                            break
                        else:
                            # stay-safe probability high: allow continue to improve completion
                            continue
                    except Exception:
                        break
                else:
                    break
        # 使用抽象编码索引比较不安全态
        before_violated = any((abs.get_state_idx()[abs.encode(o)] in unsafe_states) for o in observations)
        after_violated = any((abs.get_state_idx()[abs.encode(o)] in unsafe_states) for o in obs_after_mon)
        completed = len(obs_after_mon) == len(observations)
        violated_before_cases += 1 if before_violated else 0
        violated_after_cases += 1 if after_violated else 0
        completed_cases += 1 if completed else 0

    unsafe_pct_before = 100.0 * violated_before_cases / total_cases if total_cases else 0.0
    unsafe_pct_after = 100.0 * violated_after_cases / total_cases if total_cases else 0.0
    completion_pct = 100.0 * completed_cases / total_cases if total_cases else 0.0
    return {
        "folder": folder,
        "total_cases": total_cases,
        "unsafe_pct_before": unsafe_pct_before,
        "unsafe_pct_after": unsafe_pct_after,
        "completion_pct": completion_pct,
        "theta": base_theta,
        "adapt": adapt,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "time_cap": time_cap,
    }


def eval_crash(folder: str, abs: AVAbstraction, unsafe_states: List[int]) -> Dict:
    # P_collision per scenario state description (average first-step probability)
    reachability_cache: Dict = {}
    p_collisions: List[float] = []
    total_cases = 0

    prism_path, _ = _dtmc_paths(folder)

    for log in _iter_logs(folder):
        with open(log) as lf:
            obj = json.loads(lf.read())
        observations = _extract_observations(obj)
        if len(observations) == 0:
            continue
        total_cases += 1
        first_state = observations[0]
        prob = runtime_monitor(first_state, prism_path, abs, set(unsafe_states), cache=reachability_cache)
        p_collisions.append(prob)
    avg_pc = sum(p_collisions) / len(p_collisions) if p_collisions else 0.0
    return {
        "folder": folder,
        "total_cases": total_cases,
        "avg_p_collision": avg_pc,
    }


# ---------------------- Top-3 Reach-Avoid + Shielding Runner ----------------------

def _read_top3_laws_from_diagnostic() -> List[str]:
    """Read mdp_stl_diagnostic_report.json and return Top-3 law names by AII."""
    diag_path = BASE_DIR.parent / 'log_safereach' / 'mdp_stl_diagnostic_report.json'
    if not diag_path.exists():
        return []
    try:
        with open(diag_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        av = obj.get('av', [])
        av_sorted = sorted(av, key=lambda x: float(x.get('action_influence_index', 0.0)), reverse=True)
        names = [x.get('law') for x in av_sorted if x.get('law')]
        return names[:3]
    except Exception:
        return []


def _svg_grouped_bars(title: str, labels: List[str], series: Dict[str, List[float]], width: int = 960, height: int = 400) -> str:
    """Simple grouped bars for multiple series.
    series: {name: [v1, v2, ...]} with same length as labels
    """
    margin_left = 120
    margin_bottom = 60
    plot_w = width - margin_left - 20
    plot_h = height - margin_bottom - 20
    num = len(labels)
    series_names = list(series.keys())
    s_count = len(series_names)
    bar_group_w = plot_w / max(1, num)
    bar_w = max(4.0, (bar_group_w * 0.7) / max(1, s_count))
    max_val = 1e-9
    for ns in series_names:
        for v in series[ns]:
            if isinstance(v, (int, float)):
                max_val = max(max_val, float(v))
    # y-scale
    def y(val: float) -> float:
        return (height - margin_bottom) - (plot_h * (val / max(1e-9, max_val)))
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(f"<text x='{width/2}' y='24' text-anchor='middle' font-size='16' font-weight='bold'>{title}</text>")
    # axes
    svg.append(f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{margin_left}' y2='{height-margin_bottom-plot_h}' stroke='black' stroke-width='1' />")
    svg.append(f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{margin_left+plot_w}' y2='{height-margin_bottom}' stroke='black' stroke-width='1' />")
    # bars
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
    for i, lab in enumerate(labels):
        gx = margin_left + i * bar_group_w
        svg.append(f"<text x='{gx + bar_group_w/2}' y='{height-margin_bottom+20}' text-anchor='middle' font-size='12'>{lab}</text>")
        for j, ns in enumerate(series_names):
            val = float(series[ns][i]) if i < len(series[ns]) else 0.0
            bx = gx + j * bar_w + (bar_group_w - s_count * bar_w) / 2
            by = y(val)
            bh = (height - margin_bottom) - by
            svg.append(f"<rect x='{bx:.2f}' y='{by:.2f}' width='{bar_w:.2f}' height='{bh:.2f}' fill='{colors[j % len(colors)]}' />")
            svg.append(f"<title>{ns}: {val:.2f}</title>")
    # legend
    lx = margin_left + 10
    ly = 40
    for j, ns in enumerate(series_names):
        svg.append(f"<rect x='{lx}' y='{ly + j*18}' width='12' height='12' fill='{colors[j % len(colors)]}' />")
        svg.append(f"<text x='{lx+18}' y='{ly + j*18 + 11}' font-size='12'>{ns}</text>")
    svg.append("</svg>")
    return ''.join(svg)


def run_top3_shielding() -> None:
    """Evaluate Top-3 laws (by AII) with Reach-Avoid + adaptive shielding and output charts."""
    top3 = _read_top3_laws_from_diagnostic()
    if not top3:
        print("[Top3Shield] No Top-3 laws discovered from diagnostic. Fallback to default set.")
        # Fallback: pick common laws in dataset
        top3 = [d for d in os.listdir(LOG_DIR) if os.path.isdir(f"{LOG_DIR}/{d}")][:3]
    out_dir = BASE_DIR.parent / 'log_safereach'
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    csv_lines: List[str] = ["law,unsafe_before,unsafe_after,completion_after,pred@0.7,time_ahead@0.7,theta_min,theta_max,time_cap"]
    unsafe_before_vals: List[float] = []
    unsafe_after_vals: List[float] = []
    completion_vals: List[float] = []

    for law in top3:
        abs_obj = _load_abstraction(law)
        abs = AVAbstraction(abs_obj.get("predicates", []))
        unsafe_states = _load_spec_states(law, abs)
        if not unsafe_states:
            print(f"[Top3Shield] skip {law}: no unsafe states")
            continue
        # Per-law tuned params
        tuned = {
            "Law38_1_1": {"adapt": True, "theta_min": 0.3, "theta_max": 0.6, "time_cap": 5.0},
            "Law44_2": {"adapt": False, "base": 0.4},
            "Law53_0_1": {"adapt": False, "base": 0.8},
        }.get(law, None)
        # default values for csv param columns
        csv_tmin, csv_tmax, csv_tcap = "", "", ""
        if tuned is None:
            # default adaptive shielding: ramp from 0.3 to 0.7 over 5s
            theta_min = 0.3
            theta_max = 0.7
            time_cap = 5.0
            law_res = eval_law(law, abs, unsafe_states, thresholds=[0.7])
            reach_res = eval_reach(law, abs, unsafe_states, adapt_override=True, theta_min_override=theta_min, theta_max_override=theta_max, time_cap_override=time_cap)
            csv_tmin, csv_tmax, csv_tcap = theta_min, theta_max, time_cap
        else:
            if tuned.get("adapt", True):
                theta_min = tuned.get("theta_min", 0.3)
                theta_max = tuned.get("theta_max", 0.7)
                time_cap = tuned.get("time_cap", 5.0)
                law_res = eval_law(law, abs, unsafe_states, thresholds=[0.7])
                reach_res = eval_reach(
                    law, abs, unsafe_states,
                    adapt_override=True,
                    theta_min_override=theta_min,
                    theta_max_override=theta_max,
                    time_cap_override=time_cap,
                    stay_H_override=tuned.get("stay_H"),
                    stay_thresh_override=tuned.get("stay_thresh"),
                )
                csv_tmin, csv_tmax, csv_tcap = theta_min, theta_max, time_cap
            else:
                base = tuned.get("base", 0.5)
                law_res = eval_law(law, abs, unsafe_states, thresholds=[0.7])
                reach_res = eval_reach(law, abs, unsafe_states, base_theta_override=base, adapt_override=False)
        summaries.append({"law": law_res, "reach": reach_res})
        # collect series
        unsafe_before_vals.append(reach_res.get("unsafe_pct_before", 0.0))
        unsafe_after_vals.append(reach_res.get("unsafe_pct_after", 0.0))
        completion_vals.append(reach_res.get("completion_pct", 0.0))
        pred07 = float((law_res.get("prediction_pct", {}) or {}).get(0.7, 0.0))
        ta07 = float((law_res.get("time_ahead_s", {}) or {}).get(0.7, 0.0))
        csv_lines.append(f"{law},{reach_res.get('unsafe_pct_before',0.0):.6f},{reach_res.get('unsafe_pct_after',0.0):.6f},{reach_res.get('completion_pct',0.0):.6f},{pred07:.6f},{ta07:.6f},{csv_tmin},{csv_tmax},{csv_tcap}")

    # Write JSON/CSV
    with open(out_dir / 'reach_avoid_top3_summary.json', 'w') as f:
        json.dump({"top3": top3, "summaries": summaries}, f, indent=2)
    with open(out_dir / 'reach_avoid_top3_summary.csv', 'w') as f:
        f.write("\n".join(csv_lines)+"\n")

    # Charts: unsafe before/after; completion after
    labels = [s.get('reach', {}).get('folder', 'N/A') for s in summaries]
    svg1 = _svg_grouped_bars("Top-3 Laws: Unsafe% before vs after (Shielding)", labels, {"Unsafe before": unsafe_before_vals, "Unsafe after": unsafe_after_vals})
    (out_dir / 'reach_avoid_top3_unsafe.svg').write_text(svg1)
    svg2 = _svg_grouped_bars("Top-3 Laws: Completion% after (Shielding)", labels, {"Completion after": completion_vals})
    (out_dir / 'reach_avoid_top3_completion.svg').write_text(svg2)


def _format_label(v: float) -> str:
    return f"{v:.2f}"


def _svg_heatmap(title: str, xs: List[float], ys: List[float], values: List[List[float]], xlab: str, ylab: str, width: int = 900, height: int = 520, cmap: str = 'RdYlGn') -> str:
    # Minimal heatmap renderer (xs along x, ys along y), values[y][x]
    margin_left = 140
    margin_bottom = 80
    plot_w = width - margin_left - 20
    plot_h = height - margin_bottom - 40
    nx = len(xs)
    ny = len(ys)
    cw = plot_w / max(1, nx)
    ch = plot_h / max(1, ny)

    # normalize values to [0,1]
    vmax = max((max(row) if row else 0.0) for row in values) if values else 1.0
    vmin = min((min(row) if row else 0.0) for row in values) if values else 0.0
    rng = max(1e-9, (vmax - vmin))
    def norm(v: float) -> float:
        return (v - vmin) / rng

    # simple RdYlGn: r,g decrease/increase around mid
    def cmap_rgb(t: float) -> Tuple[int, int, int]:
        # t in [0,1]
        r = int(255 * min(1.0, 2 * (1 - t)))
        g = int(255 * min(1.0, 2 * t))
        b = 64
        return r, g, b

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(f"<text x='{width/2}' y='24' text-anchor='middle' font-size='16' font-weight='bold'>{title}</text>")
    # axes labels
    svg.append(f"<text x='{margin_left + plot_w/2}' y='{height-20}' text-anchor='middle' font-size='13'>{xlab}</text>")
    svg.append(f"<text x='24' y='{(height - margin_bottom - plot_h/2)}' text-anchor='middle' font-size='13' transform='rotate(-90,24,{(height - margin_bottom - plot_h/2)})'>{ylab}</text>")

    # grid cells
    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            v = float(values[yi][xi])
            t = norm(v)
            r,g,b = cmap_rgb(t)
            x = margin_left + xi * cw
            y = (height - margin_bottom) - (yi+1) * ch
            svg.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{cw:.2f}' height='{ch:.2f}' fill='rgb({r},{g},{b})'><title>{v:.3f}</title></rect>")
    # ticks
    for xi, xv in enumerate(xs):
        x = margin_left + (xi + 0.5) * cw
        svg.append(f"<text x='{x:.2f}' y='{height-margin_bottom+16}' text-anchor='middle' font-size='11'>{_format_label(xv)}</text>")
    for yi, yv in enumerate(ys):
        y = (height - margin_bottom) - (yi + 0.5) * ch
        svg.append(f"<text x='{margin_left-10}' y='{y:.2f}' text-anchor='end' font-size='11'>{_format_label(yv)}</text>")

    svg.append("</svg>")
    return ''.join(svg)


def run_law_sweep(law: str) -> None:
    # Grid over (theta_min, theta_max, time_cap)
    theta_mins = [0.2, 0.3, 0.4]
    theta_maxs = [0.6, 0.7, 0.85]
    time_caps = [3.0, 5.0, 8.0]

    out_dir = BASE_DIR.parent / 'log_safereach'
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_obj = _load_abstraction(law)
    abs = AVAbstraction(abs_obj.get("predicates", []))
    unsafe_states = _load_spec_states(law, abs)
    if not unsafe_states:
        print(f"[Sweep] skip {law}: no unsafe states")
        return

    rows = ["theta_min,theta_max,time_cap,unsafe_before,unsafe_after,completion,pred@0.7,time_ahead@0.7"]
    # heatmaps indexed as [yi=time_cap][xi=theta_max] for each theta_min fixed? We'll do separate maps per theta_min to keep simple
    # For simplicity, we will produce 3 heatmaps per metric (one per theta_min)
    sweep_results: List[Dict] = []

    for tmin, tmax, cap in product(theta_mins, theta_maxs, time_caps):
        law_res = eval_law(law, abs, unsafe_states, thresholds=[0.7])
        reach_res = eval_reach(law, abs, unsafe_states, adapt_override=True, theta_min_override=tmin, theta_max_override=tmax, time_cap_override=cap)
        pred07 = float((law_res.get("prediction_pct", {}) or {}).get(0.7, 0.0))
        ta07 = float((law_res.get("time_ahead_s", {}) or {}).get(0.7, 0.0))
        rows.append(f"{tmin},{tmax},{cap},{reach_res.get('unsafe_pct_before',0.0):.6f},{reach_res.get('unsafe_pct_after',0.0):.6f},{reach_res.get('completion_pct',0.0):.6f},{pred07:.6f},{ta07:.6f}")
        sweep_results.append({
            "theta_min": tmin, "theta_max": tmax, "time_cap": cap,
            "reach": reach_res, "law": law_res,
        })

    # write files
    (out_dir / f'reach_avoid_sweep_{law}.csv').write_text("\n".join(rows)+"\n")
    with open(out_dir / f'reach_avoid_sweep_{law}.json', 'w') as f:
        json.dump({"law": law, "theta_mins": theta_mins, "theta_maxs": theta_maxs, "time_caps": time_caps, "results": sweep_results}, f, indent=2)

    # Produce heatmaps per theta_min for (1) unsafe_after (2) completion (3) pred@0.7 (4) time_ahead@0.7
    for metric_key, title_prefix in (
        ("unsafe_pct_after", "Unsafe% after"),
        ("completion_pct", "Completion% after"),
        ("pred@0.7", "Prediction% @0.7"),
        ("time_ahead@0.7", "Time-Ahead(s) @0.7"),
    ):
        for tmin in theta_mins:
            # build matrix values[time_cap_index][theta_max_index]
            vals = []
            for cap in time_caps:
                row = []
                for tmax in theta_maxs:
                    # pick first matching result
                    m = next((r for r in sweep_results if r["theta_min"]==tmin and r["theta_max"]==tmax and r["time_cap"]==cap), None)
                    if m is None:
                        row.append(0.0)
                        continue
                    if metric_key in ("unsafe_pct_after", "completion_pct"):
                        v = float(m["reach"].get(metric_key, 0.0))
                    elif metric_key == "pred@0.7":
                        v = float((m["law"].get("prediction_pct", {}) or {}).get(0.7, 0.0))
                    else:
                        v = float((m["law"].get("time_ahead_s", {}) or {}).get(0.7, 0.0))
                    row.append(v)
                vals.append(row)
            svg = _svg_heatmap(
                f"{law} | {title_prefix} | theta_min={tmin}",
                theta_maxs,
                time_caps,
                vals,
                xlab="theta_max",
                ylab="time_cap (s)",
            )
            (out_dir / f'reach_avoid_sweep_{law}_{metric_key}_theta_min_{tmin}.svg').write_text(svg)

if __name__ == "__main__":
    # 优先提供“Top-3 Shielding”运行入口；否则回退为全量评估
    if os.environ.get("AV_TOP3_SHIELD", "off").lower() in ("on", "1", "true", "yes"):
        run_top3_shielding()
    else:
        # 若没有场景特定DTMC，仍然运行：用默认default_av.*文件作评估
        folders = [d for d in os.listdir(LOG_DIR) if os.path.isdir(f"{LOG_DIR}/{d}")]
        summaries = []
        # 如果设置 AV_SWEEP_LAW，则先执行参数扫
        sweep_law = os.environ.get("AV_SWEEP_LAW", "").strip()
        if sweep_law:
            run_law_sweep(sweep_law)
        for f in folders:
            # 加载抽象（场景特定或默认）
            abs_obj = _load_abstraction(f)
            preds = abs_obj.get("predicates", [])
            abs = AVAbstraction(preds)
            unsafe_states = _load_spec_states(f, abs)
            if len(unsafe_states) == 0:
                print(f"skip folder {f}: no unsafe states identified by spec")
                continue
            law_res = eval_law(f, abs, unsafe_states)
            reach_res = eval_reach(f, abs, unsafe_states)
            crash_res = eval_crash(f, abs, unsafe_states)
            summary = {"law": law_res, "reach": reach_res, "crash": crash_res}
            summaries.append(summary)
            # Print per-folder summary
            print("===== AV Evaluation Summary =====")
            print(json.dumps(summary, indent=2))

        # Global summary
        print("===== AV Evaluation All Folders =====")
        print(json.dumps({"thresholds": THRESHOLDS, "summaries": summaries}, indent=2))