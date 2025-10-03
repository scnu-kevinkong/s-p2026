import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from .autonomous_vehicle.abstraction import AVAbstraction
from .runtime_monitor import runtime_monitor

# Reuse AV dataset layout for the prototype
BASE_DIR = Path(os.path.dirname(__file__))
LOG_DIR = str(BASE_DIR / "autonomous_vehicle" / "samples")
MODEL_DIR = str(BASE_DIR / "autonomous_vehicle" / "dtmcs")
DEFAULT_MODEL_PREFIX = str(BASE_DIR.parent / "default_av.dtmc")


def _dtmc_paths(folder: str) -> Tuple[str, str]:
    prism_path = f"{MODEL_DIR}/{folder}/dtmc.prism"
    model_json = f"{MODEL_DIR}/{folder}/model.json"
    if os.path.exists(prism_path) and os.path.exists(model_json):
        return prism_path, model_json
    return f"{DEFAULT_MODEL_PREFIX}dtmc.prism", f"{DEFAULT_MODEL_PREFIX}model.json"


def _load_abstraction(folder: str) -> Dict:
    abs_path = f"{MODEL_DIR}/{folder}/abstraction.json"
    if os.path.exists(abs_path):
        with open(abs_path) as af:
            return json.loads(af.read())
    default_abs_path = str(BASE_DIR.parent / "default_av.dtmcabstraction.json")
    if os.path.exists(default_abs_path):
        with open(default_abs_path) as daf:
            return json.loads(daf.read())
    return {"predicates": []}


def _load_spec_states(model_folder: str, abs: AVAbstraction) -> List[int]:
    spec_path = f"{LOG_DIR}/{model_folder}/spec"
    if os.path.exists(spec_path):
        with open(spec_path) as f:
            spec = json.loads(f.read())
    else:
        spec = [[["collision", "==", "1"], True]]
    unsafe_states = abs.filter(spec)
    return [abs.get_state_idx()[state] for state in unsafe_states]


def _iter_logs(folder: str):
    for o in os.listdir(f"{LOG_DIR}/{folder}"):
        if o.endswith(".json"):
            yield f"{LOG_DIR}/{folder}/{o}"


def _extract_observations(obj: Dict) -> List[Dict]:
    if "s_trans" in obj:
        return [t.get("state", t) for t in obj.get("s_trans", [])]
    if "trajectory" in obj:
        return obj.get("trajectory", [])
    return []


# ------------------ DTMC helpers for offline statistics ------------------

def _load_model(model_json_path: str) -> Tuple[int, Dict[int, Dict[int, float]], Dict[str, int]]:
    with open(model_json_path, 'r') as f:
        m = json.load(f)
    states = m['states']
    idx = m['state_index']
    trans = m['transition_probs']
    n = len(states)
    outgoing: Dict[int, Dict[int, float]] = {}
    for sk, edges in trans.items():
        i = idx[sk]
        row: Dict[int, float] = {}
        for tk, w in edges.items():
            j = idx[tk]
            # 数字或分数字符串
            if isinstance(w, str) and '/' in w:
                num, den = w.split('/')
                val = float(num) / float(den)
            else:
                val = float(w)
            row[j] = val
        outgoing[i] = row
    return n, outgoing, idx


def _bounded_hit_prob_vectors(n: int, outgoing: Dict[int, Dict[int, float]], target: Set[int], max_k: int) -> List[List[float]]:
    """Return f[k][s] = P_s(hit target within ≤k steps). f[0] is base."""
    f: List[List[float]] = []
    base = [1.0 if s in target else 0.0 for s in range(n)]
    f.append(base)
    for _ in range(1, max_k + 1):
        prev = f[-1]
        cur = [0.0] * n
        for s in range(n):
            if s in target:
                cur[s] = 1.0
                continue
            ssum = 0.0
            for j, p in outgoing.get(s, {}).items():
                ssum += p * prev[j]
            cur[s] = ssum
        f.append(cur)
    return f  # length max_k+1


def _absorption_prob_to_target(n: int, outgoing: Dict[int, Dict[int, float]], target: Set[int], max_iters: int = 5000, tol: float = 1e-10) -> List[float]:
    """p[s] = P_s(eventually hit target)."""
    p = [1.0 if s in target else 0.0 for s in range(n)]
    for _ in range(max_iters):
        changed = 0.0
        p_new = p[:]
        for s in range(n):
            if s in target:
                p_new[s] = 1.0
                continue
            ssum = 0.0
            for j, w in outgoing.get(s, {}).items():
                ssum += w * p[j]
            changed = max(changed, abs(ssum - p[s]))
            p_new[s] = ssum
        p = p_new
        if changed < tol:
            break
    return p


@dataclass
class Mode:
    name: str
    prism_path: str
    model_json: str
    abstraction: AVAbstraction
    unsafe_states: List[int]


class MetaMDP:
    """
    Minimal meta-level skeleton treating each AV scenario folder as a "mode".
    This prototype only performs offline checks (no online control), ensuring
    each mode can be loaded and queried for reachability probabilities.
    """

    def __init__(self) -> None:
        self.modes: Dict[str, Mode] = {}

    def register_mode(self, folder: str) -> Optional[Mode]:
        try:
            abs_obj = _load_abstraction(folder)
            abs = AVAbstraction(abs_obj.get("predicates", []))
            unsafe_states = _load_spec_states(folder, abs)
            if len(unsafe_states) == 0:
                return None
            prism_path, model_json = _dtmc_paths(folder)
            mode = Mode(name=folder, prism_path=prism_path, model_json=model_json, abstraction=abs, unsafe_states=unsafe_states)
            self.modes[folder] = mode
            return mode
        except Exception:
            return None

    def load_all_modes(self) -> None:
        if not os.path.isdir(LOG_DIR):
            return
        for d in os.listdir(LOG_DIR):
            p = os.path.join(LOG_DIR, d)
            if os.path.isdir(p):
                self.register_mode(d)

    def estimate_first_step_risk(self, mode: Mode, trajectory: List[Dict]) -> float:
        if not trajectory:
            return 0.0
        first_obs = trajectory[0]
        cache: Dict = {}
        return runtime_monitor(first_obs, mode.prism_path, mode.abstraction, set(mode.unsafe_states), cache=cache)

    # --------- New: per-mode window risk and terminal distributions ---------
    def per_mode_offline_stats(self, mode: Mode, initial_states: List[int], ks: List[int]) -> Dict:
        if not os.path.exists(mode.model_json):
            return {"mode": mode.name, "error": "model_json_missing"}
        n, outgoing, idx = _load_model(mode.model_json)
        unsafe_set = set(mode.unsafe_states)
        # bounded hit probabilities for S=unsafe
        max_k = max(ks) if ks else 0
        fks = _bounded_hit_prob_vectors(n, outgoing, unsafe_set, max_k)
        # terminal absorption to unsafe (U) and finish (F)
        finish_idx = idx.get('finish', None)
        finish_set: Set[int] = {finish_idx} if finish_idx is not None else set()
        p_unsafe = _absorption_prob_to_target(n, outgoing, unsafe_set)
        p_finish = _absorption_prob_to_target(n, outgoing, finish_set) if finish_set else [0.0] * n
        # aggregate across provided initial states
        if not initial_states:
            initial_states = [0]  # fallback to 0
        pk_avg = {k: float(sum(fks[k][s] for s in initial_states) / len(initial_states)) for k in ks}
        unsafe_avg = float(sum(p_unsafe[s] for s in initial_states) / len(initial_states))
        finish_avg = float(sum(p_finish[s] for s in initial_states) / len(initial_states))
        timeout_avg = max(0.0, 1.0 - unsafe_avg - finish_avg)
        return {
            "mode": mode.name,
            "num_initials": len(initial_states),
            "pk": pk_avg,
            "terminal": {"unsafe": unsafe_avg, "safe_done": finish_avg, "timeout": timeout_avg},
        }


def _svg_bar_chart_multi(title: str, labels: List[str], values: List[float], width: int = 800, height: int = 360) -> str:
    pad = 30
    n = len(labels)
    if n == 0:
        n = 1
    gap = 10
    bar_w = max(10, int((width - 2 * pad - (n - 1) * gap) / n))
    max_val = max(values + [1.0])
    def bar_height(v: float) -> float:
        return (height - 100) * (v / max_val)
    base_y = height - 60
    title_el = f"<text x='{width/2}' y='30' text-anchor='middle' font-size='16' font-family='Arial'>{title}</text>"
    axis = f"<line x1='{pad}' y1='{base_y}' x2='{width-pad}' y2='{base_y}' stroke='#333'/>"
    svg_parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>", title_el, axis]
    x = pad
    for i, (lab, v) in enumerate(zip(labels, values)):
        h = bar_height(v)
        rect = f"<rect x='{x}' y='{base_y - h}' width='{bar_w}' height='{h}' fill='#ff7043'/>"
        txt = f"<text x='{x + bar_w/2}' y='{base_y + 18}' text-anchor='middle' font-size='11'>{lab}</text>"
        valtxt = f"<text x='{x + bar_w/2}' y='{base_y - h - 5}' text-anchor='middle' font-size='10'>{v:.3f}</text>"
        svg_parts.extend([rect, txt, valtxt])
        x += bar_w + gap
    svg_parts.append("</svg>")
    return ''.join(svg_parts)


def _load_mode_tags() -> Optional[Dict]:
    # Prefer dtmcs folder; fallback project root search
    candidate = Path(MODEL_DIR) / 'mode_tags.json'
    if candidate.exists():
        try:
            with open(candidate, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _validate_mode_tags(tags: Dict, mode_names: Set[str]) -> Dict:
    res = {
        'loaded': True,
        'num_modes': 0,
        'num_edges': 0,
        'invalid_modes': [],
        'invalid_edges': [],
        'modes_list': [],
    }
    try:
        modes_obj = tags.get('modes')
        modes_list: List[str] = []
        if isinstance(modes_obj, dict):
            modes_list = list(modes_obj.keys())
        elif isinstance(modes_obj, list):
            for m in modes_obj:
                if isinstance(m, dict) and 'name' in m:
                    modes_list.append(m['name'])
                elif isinstance(m, str):
                    modes_list.append(m)
        res['modes_list'] = modes_list
        res['num_modes'] = len(modes_list)
        for m in modes_list:
            if m not in mode_names:
                res['invalid_modes'].append(m)
        edges = tags.get('edges', [])
        res['num_edges'] = len(edges) if isinstance(edges, list) else 0
        for e in edges if isinstance(edges, list) else []:
            if isinstance(e, dict):
                u, v = e.get('src'), e.get('dst')
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                u, v = e[0], e[1]
            else:
                res['invalid_edges'].append(str(e))
                continue
            if u not in mode_names or v not in mode_names:
                res['invalid_edges'].append(f"{u}->{v}")
    except Exception:
        return {'loaded': False}
    return res


if __name__ == "__main__":
    # Offline run: load modes, compute window risk pk and terminal distributions
    # Aggregate across initial abstract states derived from first observation of each log
    ks_env = os.environ.get("META_KS", "5,10,20").strip()
    try:
        KS = [int(x) for x in ks_env.split(',') if x]
    except Exception:
        KS = [5, 10, 20]

    out_dir = Path(BASE_DIR.parent / 'log_safereach' / 'offline_meta')
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = MetaMDP()
    meta.load_all_modes()
    if not meta.modes:
        print("[MetaMDP] No modes discovered.")
        raise SystemExit(0)

    all_stats: List[Dict] = []
    unsafe_bars_labels: List[str] = []
    unsafe_bars_values: List[float] = []
    for name, mode in meta.modes.items():
        initials: List[int] = []
        # derive initial abstract states from logs
        for log_path in _iter_logs(name):
            try:
                with open(log_path) as lf:
                    obj = json.load(lf)
                traj = _extract_observations(obj)
                if not traj:
                    continue
                s0_name = mode.abstraction.encode(traj[0])
                s_idx = mode.abstraction.get_state_idx()[s0_name]
                initials.append(s_idx)
            except Exception:
                continue
        stats = meta.per_mode_offline_stats(mode, initials, KS)
        all_stats.append(stats)
        # write per-mode
        with open(out_dir / f"{name}_summary.json", 'w') as f:
            json.dump(stats, f, indent=2)
        # stdout brief
        pk_str = ", ".join([f"k={k}:{stats['pk'].get(k, 0):.4f}" for k in KS])
        term = stats.get('terminal', {})
        print(f"[MetaMDP] {name}: initials={len(initials)} | {pk_str} | term(unsafe={term.get('unsafe',0):.4f}, finish={term.get('safe_done',0):.4f}, timeout={term.get('timeout',0):.4f})")
        unsafe_bars_labels.append(name)
        unsafe_bars_values.append(float(term.get('unsafe', 0.0)))

    # combined summary
    summary_obj = {"ks": KS, "modes": all_stats}

    # mode_tags.json load & validate
    tags = _load_mode_tags()
    if tags is not None:
        val = _validate_mode_tags(tags, set(meta.modes.keys()))
        summary_obj["mode_tags"] = val
        with open(out_dir / 'mode_tags_validation.json', 'w') as f:
            json.dump(val, f, indent=2)

    with open(out_dir / 'offline_meta_summary.json', 'w') as f:
        json.dump(summary_obj, f, indent=2)

    # overall SVG: unsafe terminal probability per mode
    try:
        svg = _svg_bar_chart_multi("Terminal unsafe probability per mode", unsafe_bars_labels, unsafe_bars_values)
        with open(out_dir / 'terminal_unsafe_summary.svg', 'w') as f:
            f.write(svg)
    except Exception:
        pass