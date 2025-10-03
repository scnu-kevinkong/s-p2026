import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import random

from .autonomous_vehicle.abstraction import AVAbstraction
from .runtime_monitor import runtime_monitor
from .av_eval import eval_law, eval_reach

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


# -------------- DTMC utils for product-style probability --------------

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
            if isinstance(w, str) and '/' in w:
                num, den = w.split('/')
                val = float(num) / float(den)
            else:
                val = float(w)
            row[j] = val
        outgoing[i] = row
    return n, outgoing, idx


def _distribution_after_k(n: int, outgoing: Dict[int, Dict[int, float]], s0: int, k: int) -> List[float]:
    dist = [0.0] * n
    dist[s0] = 1.0
    for _ in range(k):
        nxt = [0.0] * n
        for i, row in outgoing.items():
            pi = dist[i]
            if pi == 0.0:
                continue
            for j, p in row.items():
                nxt[j] += pi * p
        dist = nxt
    return dist


def _finite_window_stay_prob(n: int, outgoing: Dict[int, Dict[int, float]], allowed_set: Set[int], H: int) -> List[float]:
    # DP for staying within allowed_set for exactly H steps, starting in state s (at window start)
    dp = [1.0 if (i in allowed_set) else 0.0 for i in range(n)]
    for _ in range(H):
        nxt = [0.0] * n
        for i in range(n):
            if i not in allowed_set:
                continue
            row = outgoing.get(i, {})
            acc = 0.0
            for j, p in row.items():
                if j in allowed_set:
                    acc += dp[j] * p
            nxt[i] = acc
        dp = nxt
    return dp


@dataclass
class STLSpec:
    kind: str  # "F" or "G"
    a: float
    b: float
    predicate: Dict  # e.g., {"collision": 1}
    theta: float = 0.5  # risk threshold to interpret predicate truth (baseline)


class STLEvaluator:
    """
    Two-channel evaluation:
    - Baseline (existing): per-log threshold check using runtime_monitor.
    - Optimized (new): DTMC product-style probability for windowed F/G.
    """

    def __init__(self, folder: str) -> None:
        abs_obj = _load_abstraction(folder)
        self.abs = AVAbstraction(abs_obj.get("predicates", []))
        self.folder = folder
        self.prism_path, self.model_json = _dtmc_paths(folder)

    def _unsafe_indices_from_predicate(self, predicate: Dict) -> Set[int]:
        # Interpret predicate as equality on decoded observation fields.
        # Works with AVAbstraction bitstring states; last two bits are reach/collision.
        # Example predicate: {"collision": 1}
        if not predicate:
            return set()
        # Single-key assumption for now
        k, v = next(iter(predicate.items()))
        try:
            target_val = float(v)
        except Exception:
            try:
                target_val = float(str(v))
            except Exception:
                target_val = None
        state_idx = self.abs.get_state_idx()
        unsafe: Set[int] = set()
        for name, i in state_idx.items():
            try:
                ob = self.abs.decode(name)
                if k in ob and target_val is not None and float(ob[k]) == target_val:
                    unsafe.add(i)
            except Exception:
                continue
        # fallback using abstraction.filter for special flags
        if not unsafe and k in ("collision", "reach_destination"):
            from .autonomous_vehicle.abstraction import COLLISION_PREDICATE, REACH_PREDICATE
            pred_tuple = COLLISION_PREDICATE if k == "collision" else REACH_PREDICATE
            want_true = (int(v) == 1) if isinstance(v, (int, float, str)) else True
            try:
                states = self.abs.filter(specs=[(pred_tuple, bool(want_true))])
                for s in states:
                    idx_val = state_idx.get(s)
                    if idx_val is not None:
                        unsafe.add(idx_val)
            except Exception:
                pass
        return unsafe

    def _step_time(self, ob: Dict) -> float:
        # Use dt if present else 1.0
        return float(ob.get("dt", 1.0))

    def _initial_indices_from_logs(self) -> List[int]:
        initials: List[int] = []
        for log in _iter_logs(self.folder):
            try:
                with open(log) as lf:
                    obj = json.loads(lf.read())
                observations = _extract_observations(obj)
                if not observations:
                    continue
                s0_name = self.abs.encode(observations[0])
                s_idx = self.abs.get_state_idx()[s0_name]
                initials.append(s_idx)
            except Exception:
                continue
        return initials

    def _monte_carlo_prob(self, n: int, outgoing: Dict[int, Dict[int, float]], initials: List[int], pred_set: Set[int], a_steps: int, b_steps: int, trials: int = 1000, kind: str = "F") -> float:
        if not initials:
            initials = [0]
        H = max(0, int(round(b_steps - a_steps + 1)))
        if H <= 0:
            return 0.0
        # Precompute cumulative distributions for sampling next state
        cdfs: Dict[int, List[Tuple[float, int]]] = {}
        for i, row in outgoing.items():
            acc = 0.0
            items: List[Tuple[float, int]] = []
            for j, p in row.items():
                acc += p
                items.append((acc, j))
            if items:
                last = items[-1][0]
                items = [(c/last if last > 0 else 1.0, j) for c, j in items]
            cdfs[i] = items
        def step_sample(state: int) -> int:
            row = cdfs.get(state)
            if not row:
                return state
            r = random.random()
            for c, j in row:
                if r <= c:
                    return j
            return row[-1][1]
        success = 0
        for _ in range(trials):
            s = random.choice(initials)
            # advance to a_steps
            for _k in range(a_steps):
                s = step_sample(s)
            if kind == "F":
                hit = (s in pred_set)
                for _h in range(max(0, H-1)):
                    if hit:
                        break
                    s = step_sample(s)
                    if s in pred_set:
                        hit = True
                        break
                if hit:
                    success += 1
            else:  # G
                hold = (s in pred_set)
                for _h in range(max(0, H-1)):
                    if not hold:
                        break
                    s = step_sample(s)
                    if s not in pred_set:
                        hold = False
                        break
                if hold:
                    success += 1
        return success / float(trials) if trials > 0 else 0.0

    # ---------------- Baseline channel ----------------
    def baseline_check(self, spec: STLSpec) -> Dict:
        sat_count = 0
        total = 0
        unsafe_indices = self._unsafe_indices_from_predicate(spec.predicate)
        for log in _iter_logs(self.folder):
            try:
                with open(log) as lf:
                    obj = json.loads(lf.read())
            except Exception:
                continue
            observations = _extract_observations(obj)
            if not observations:
                continue
            total += 1
            t = 0.0
            satisfied = False if spec.kind == "F" else True
            entered_window = False
            cache: Dict = {}
            for ob in observations:
                t += self._step_time(ob)
                if t < spec.a:
                    continue
                entered_window = True
                if t > spec.b:
                    break
                prob = runtime_monitor(ob, self.prism_path, self.abs, unsafe_indices, cache=cache)
                truth = prob >= spec.theta
                if spec.kind == "F":
                    if truth:
                        satisfied = True
                        break
                else:
                    if not truth:
                        satisfied = False
                        break
            if spec.kind == "G" and not entered_window:
                satisfied = False
            if satisfied:
                sat_count += 1
        return {
            "baseline_total_logs": total,
            "baseline_satisfied": sat_count,
            "baseline_pct": (sat_count / total * 100.0) if total else 0.0,
        }

    # ---------------- Optimized probability channel ----------------
    def prob_from_model(self, spec: STLSpec) -> Dict:
        if not os.path.exists(self.model_json):
            return {"error": "model_json_missing"}
        n, outgoing, idx = _load_model(self.model_json)
        pred_set = self._unsafe_indices_from_predicate(spec.predicate)
        not_pred_set = set(range(n)) - pred_set
        a_steps = int(round(spec.a))
        b_steps = int(round(spec.b))
        if b_steps < a_steps:
            a_steps, b_steps = 0, 0
        H = max(0, b_steps - a_steps + 1)
        initials = self._initial_indices_from_logs()
        if not initials:
            initials = [0]
        # DP survival arrays precomputed once per horizon H
        stay_not_pred = _finite_window_stay_prob(n, outgoing, not_pred_set, H)
        stay_pred = _finite_window_stay_prob(n, outgoing, pred_set, H)
        # aggregate over initial states
        # F[a,b] p: P = 1 - sum_s Q_a[s ∈ ¬p] * stay_not_pred_H(s)
        # G[a,b] p: P =     sum_s Q_a[s]        * stay_pred_H(s)
        total_prob = 0.0
        for s0 in initials:
            Qa = _distribution_after_k(n, outgoing, s0, a_steps)
            if spec.kind == "F":
                nohit = 0.0
                for s in range(n):
                    if s in not_pred_set:
                        nohit += Qa[s] * stay_not_pred[s]
                total_prob += (1.0 - nohit)
            else:  # G
                allhold = 0.0
                for s in range(n):
                    allhold += Qa[s] * stay_pred[s]
                total_prob += allhold
        avg_prob = total_prob / len(initials)
        # Monte Carlo cross-validation (optional)
        mc_on = os.environ.get("STL_MC", "on").lower() in ("on", "1", "true", "yes")
        mc_trials = int(os.environ.get("STL_MC_TRIALS", "1000"))
        mc_prob = None
        if mc_on and mc_trials > 0:
            mc_prob = self._monte_carlo_prob(n=n, outgoing=outgoing, initials=initials, pred_set=pred_set, a_steps=a_steps, b_steps=b_steps, trials=mc_trials, kind=spec.kind)
        res = {
            "optimized_prob": avg_prob,
            "initial_states": len(initials),
            "a_steps": a_steps,
            "b_steps": b_steps,
        }
        if mc_prob is not None:
            # 95% CI under binomial assumption
            import math as _math
            stderr = (_math.sqrt(mc_prob * (1.0 - mc_prob) / mc_trials) if mc_trials > 0 else 0.0)
            ci = 1.96 * stderr
            res.update({
                "mc_trials": mc_trials,
                "mc_prob": mc_prob,
                "mc_stderr": stderr,
                "mc_ci95_low": max(0.0, mc_prob - ci),
                "mc_ci95_high": min(1.0, mc_prob + ci),
                "mc_vs_optimized_delta": (mc_prob - avg_prob),
            })
        return res

    # ---------------- Report and visualization ----------------
    def save_report(self, spec: STLSpec, baseline: Dict, optimized: Dict, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Compute ProGuard-style offline metrics for comparison
        try:
            unsafe_indices_pg = list(self._unsafe_indices_from_predicate(spec.predicate))
            pg_law = eval_law(self.folder, self.abs, unsafe_indices_pg)
            pg_reach = eval_reach(self.folder, self.abs, unsafe_indices_pg)
            proguard_offline = {"law": pg_law, "reach": pg_reach}
        except Exception:
            proguard_offline = None
        report = {
            "folder": self.folder,
            "spec": {
                "kind": spec.kind,
                "a": spec.a,
                "b": spec.b,
                "theta": spec.theta,
                "predicate": spec.predicate,
            },
            "baseline": baseline,
            "optimized": optimized,
        }
        if proguard_offline is not None:
            report["proguard_offline"] = proguard_offline
        with open(out_dir / f"{self.folder}_stl_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        # CSV for quick comparison
        with open(out_dir / f"{self.folder}_stl_report.csv", 'w') as f:
            f.write("metric,value\n")
            f.write(f"baseline_pct,{baseline.get('baseline_pct', 0.0):.6f}\n")
            f.write(f"optimized_prob,{optimized.get('optimized_prob', 0.0):.6f}\n")
            if 'mc_prob' in optimized:
                f.write(f"mc_prob,{optimized.get('mc_prob', 0.0):.6f}\n")
            # Append ProGuard-style metrics if available
            if proguard_offline is not None:
                reach = proguard_offline.get("reach", {})
                # Unsafe% and Completion%
                f.write(f"unsafe_pct_before,{reach.get('unsafe_pct_before', 0.0):.6f}\n")
                f.write(f"unsafe_pct_after,{reach.get('unsafe_pct_after', 0.0):.6f}\n")
                f.write(f"completion_pct,{reach.get('completion_pct', 0.0):.6f}\n")
                # Prediction (%) and Time Ahead (s) per threshold
                law = proguard_offline.get("law", {})
                pred = law.get("prediction_pct", {}) or {}
                ta = law.get("time_ahead_s", {}) or {}
                for theta, val in sorted(pred.items(), key=lambda x: float(x[0])):
                    f.write(f"prediction_pct@{theta},{float(val):.6f}\n")
                for theta, val in sorted(ta.items(), key=lambda x: float(x[0])):
                    f.write(f"time_ahead_s@{theta},{float(val):.6f}\n")
        # SVG bar chart
        svg = _svg_bar_chart(
            title=f"{self.folder} {spec.kind}[{spec.a},{spec.b}] vs baseline",
            left_label="baseline %",
            left_value=baseline.get('baseline_pct', 0.0),
            right_label="optimized prob %",
            right_value=optimized.get('optimized_prob', 0.0) * 100.0,
            width=600,
            height=320,
        )
        with open(out_dir / f"{self.folder}_stl_report.svg", 'w') as f:
            f.write(svg)


def _svg_bar_chart(title: str, left_label: str, left_value: float, right_label: str, right_value: float, width: int = 600, height: int = 320) -> str:
    pad = 30
    bar_w = 120
    gap = 60
    max_val = max(left_value, right_value, 1.0)
    def h(v: float) -> float:
        return (height - 100) * (v / max_val)
    base_y = height - 60
    title_el = f"<text x='{width/2}' y='30' text-anchor='middle' font-size='16' font-family='Arial'>{title}</text>"
    axis = f"<line x1='{pad}' y1='{base_y}' x2='{width-pad}' y2='{base_y}' stroke='#333'/>"
    left_x = pad + 50
    right_x = left_x + bar_w + gap
    left_rect = f"<rect x='{left_x}' y='{base_y - h(left_value)}' width='{bar_w}' height='{h(left_value)}' fill='#42a5f5'/>"
    right_rect = f"<rect x='{right_x}' y='{base_y - h(right_value)}' width='{bar_w}' height='{h(right_value)}' fill='#66bb6a'/>"
    left_txt = f"<text x='{left_x + bar_w/2}' y='{base_y + 18}' text-anchor='middle' font-size='12'>{left_label}</text>"
    right_txt = f"<text x='{right_x + bar_w/2}' y='{base_y + 18}' text-anchor='middle' font-size='12'>{right_label}</text>"
    left_val = f"<text x='{left_x + bar_w/2}' y='{base_y - h(left_value) - 5}' text-anchor='middle' font-size='12'>{left_value:.2f}</text>"
    right_val = f"<text x='{right_x + bar_w/2}' y='{base_y - h(right_value) - 5}' text-anchor='middle' font-size='12'>{right_value:.2f}</text>"
    return ''.join([
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        title_el, axis, left_rect, right_rect, left_txt, right_txt, left_val, right_val,
        "</svg>"
    ])


# ---------------- Batch CLI helpers ----------------

def _svg_grouped_bars(title: str, labels: List[str], base_vals: List[float], opt_vals: List[float], width: int = 960, height: int = 400) -> str:
    pad = 40
    n = len(labels) if labels else 1
    group_gap = 16
    bar_gap = 6
    bar_w = max(10, int((width - 2 * pad - (n - 1) * group_gap) / n / 2 - bar_gap))
    max_val = max((base_vals + opt_vals + [1.0]))
    def bh(v: float) -> float:
        return (height - 120) * (v / max_val)
    base_y = height - 70
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
           f"<text x='{width/2}' y='28' text-anchor='middle' font-size='16' font-family='Arial'>{title}</text>",
           f"<line x1='{pad}' y1='{base_y}' x2='{width-pad}' y2='{base_y}' stroke='#333'/>",
           f"<rect x='{pad}' y='{20}' width='14' height='14' fill='#42a5f5'/>",
           f"<text x='{pad+20}' y='{32}' font-size='12'>baseline %</text>",
           f"<rect x='{pad+120}' y='{20}' width='14' height='14' fill='#66bb6a'/>",
           f"<text x='{pad+140}' y='{32}' font-size='12'>optimized %</text>"]
    x = pad
    for i, lab in enumerate(labels):
        b = base_vals[i]
        o = opt_vals[i]
        # bars in a group
        bx = x
        ox = x + bar_w + bar_gap
        bh_b = bh(b)
        bh_o = bh(o)
        svg.append(f"<rect x='{bx}' y='{base_y - bh_b}' width='{bar_w}' height='{bh_b}' fill='#42a5f5'/>")
        svg.append(f"<rect x='{ox}' y='{base_y - bh_o}' width='{bar_w}' height='{bh_o}' fill='#66bb6a'/>")
        svg.append(f"<text x='{bx + bar_w}' y='{base_y + 16}' text-anchor='middle' font-size='11'>{lab}</text>")
        svg.append(f"<text x='{bx + bar_w/2}' y='{base_y - bh_b - 5}' text-anchor='middle' font-size='10'>{b:.2f}</text>")
        svg.append(f"<text x='{ox + bar_w/2}' y='{base_y - bh_o - 5}' text-anchor='middle' font-size='10'>{o:.2f}</text>")
        x += 2 * bar_w + bar_gap + group_gap
    svg.append("</svg>")
    return ''.join(svg)


def _list_modes() -> List[str]:
    if not os.path.isdir(LOG_DIR):
        return []
    modes: List[str] = []
    for d in os.listdir(LOG_DIR):
        p = os.path.join(LOG_DIR, d)
        if os.path.isdir(p):
            # ensure there is at least one json
            has_json = any(f.endswith('.json') for f in os.listdir(p))
            if has_json:
                modes.append(d)
    modes.sort()
    return modes


def _load_mode_tags() -> Optional[Dict]:
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


def _spec_key(spec: STLSpec) -> str:
    # key used for folder naming and grouping
    # predicate is assumed single key-value
    if spec.predicate:
        k, v = next(iter(spec.predicate.items()))
        pv = f"{k}=={v}"
    else:
        pv = "true"
    return f"{spec.kind}[{spec.a},{spec.b}]_{pv}"


def _parse_specs_from_env_or_file() -> List[STLSpec]:
    # Try file first
    path = os.environ.get("STL_SPECS_FILE", "").strip()
    specs: List[STLSpec] = []
    if path and os.path.exists(path):
        try:
            arr = json.loads(Path(path).read_text())
            for it in arr:
                specs.append(STLSpec(kind=it.get('kind','F'), a=float(it.get('a',0)), b=float(it.get('b',0)), predicate=it.get('predicate',{}), theta=float(it.get('theta',0.5))))
        except Exception:
            pass
    if specs:
        return specs
    # Try env string: "F,0,2,collision,1;G,1,3,collision,1"
    s = os.environ.get("STL_SPECS", "").strip()
    if s:
        for seg in s.split(';'):
            seg = seg.strip()
            if not seg:
                continue
            parts = seg.split(',')
            if len(parts) >= 5:
                kind, a, b, pk, pv = parts[:5]
                theta = float(parts[5]) if len(parts) >= 6 else 0.5
                specs.append(STLSpec(kind=kind, a=float(a), b=float(b), predicate={pk: int(pv)}, theta=theta))
    if specs:
        return specs
    # default
    return [STLSpec(kind='F', a=0.0, b=2.0, predicate={'collision': 1}, theta=0.5)]


def batch_evaluate() -> None:
    modes_env = os.environ.get("STL_MODES", "").strip()
    if modes_env:
        modes = [m for m in modes_env.split(',') if m]
    else:
        modes = _list_modes()
    specs = _parse_specs_from_env_or_file()
    out_root = BASE_DIR.parent / 'log_safereach' / 'offline_stl'
    out_root.mkdir(parents=True, exist_ok=True)
    # mode_tags.json validation
    tags = _load_mode_tags()
    tags_validation = _validate_mode_tags(tags, set(modes)) if tags is not None else {"loaded": False}

    overall_entries: List[Dict] = []
    # Per-spec grouped comparison SVG
    for spec in specs:
        spec_dir = out_root / _spec_key(spec)
        spec_dir.mkdir(parents=True, exist_ok=True)
        labels: List[str] = []
        base_vals: List[float] = []
        opt_vals: List[float] = []
        mc_vals: List[float] = []
        for mode in modes:
            ev = STLEvaluator(mode)
            baseline = ev.baseline_check(spec)
            optimized = ev.prob_from_model(spec)
            ev.save_report(spec, baseline, optimized, spec_dir)
            labels.append(mode)
            base_vals.append(baseline.get('baseline_pct', 0.0))
            opt_vals.append(optimized.get('optimized_prob', 0.0) * 100.0)
            mc_vals.append(optimized.get('mc_prob', None) * 100.0 if 'mc_prob' in optimized else float('nan'))
            overall_entries.append({
                "mode": mode,
                "spec": {
                    "kind": spec.kind, "a": spec.a, "b": spec.b, "theta": spec.theta, "predicate": spec.predicate
                },
                "baseline_pct": baseline.get('baseline_pct', 0.0),
                "optimized_prob_pct": optimized.get('optimized_prob', 0.0) * 100.0,
                "mc_prob_pct": (optimized.get('mc_prob', 0.0) * 100.0) if 'mc_prob' in optimized else None,
                "improved": (optimized.get('optimized_prob', 0.0) * 100.0 - baseline.get('baseline_pct', 0.0))
            })
        # save per-spec grouped svg
        try:
            svg = _svg_grouped_bars(title=f"{spec.kind}[{spec.a},{spec.b}] {_spec_key(spec).split('_')[-1]}: baseline vs optimized (all modes)", labels=labels, base_vals=base_vals, opt_vals=opt_vals)
            (out_root / f"overall_{_spec_key(spec)}.svg").write_text(svg)
        except Exception:
            pass
        # save per-spec CSV
        try:
            with open(out_root / f"overall_{_spec_key(spec)}.csv", 'w') as f:
                f.write("mode,baseline_pct,optimized_prob_pct,mc_prob_pct,delta(opt-baseline),delta(opt-mc)\n")
                for i, m in enumerate(labels):
                    b = base_vals[i]
                    o = opt_vals[i]
                    mc = mc_vals[i]
                    delta_ob = (o - b) if (not (o!=o) and not (b!=b)) else float('nan')
                    delta_om = (o - mc) if (not (o!=o) and not (mc!=mc)) else float('nan')
                    f.write(f"{m},{b:.6f},{o:.6f},{(mc if mc==mc else float('nan'))},{delta_ob:.6f},{(delta_om if delta_om==delta_om else float('nan'))}\n")
        except Exception:
            pass

    # overall summary json
    avg_base = sum(e['baseline_pct'] for e in overall_entries) / len(overall_entries) if overall_entries else 0.0
    avg_opt = sum(e['optimized_prob_pct'] for e in overall_entries) / len(overall_entries) if overall_entries else 0.0
    mc_list = [e['mc_prob_pct'] for e in overall_entries if isinstance(e.get('mc_prob_pct', None), (int, float)) and e['mc_prob_pct'] is not None]
    avg_mc = (sum(mc_list) / len(mc_list)) if mc_list else None
    improved_cnt = sum(1 for e in overall_entries if e['optimized_prob_pct'] > e['baseline_pct'])
    summary = {
        "modes": modes,
        "num_modes": len(modes),
        "specs": [
            {"kind": s.kind, "a": s.a, "b": s.b, "theta": s.theta, "predicate": s.predicate, "key": _spec_key(s)} for s in specs
        ],
        "entries": overall_entries,
        "avg_baseline_pct": avg_base,
        "avg_optimized_prob_pct": avg_opt,
        "avg_mc_prob_pct": avg_mc,
        "improved_cases": improved_cnt,
        "total_cases": len(overall_entries),
        "mode_tags_validation": tags_validation,
    }
    with open(out_root / 'overall_stl_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    # Batch mode or single evaluation
    if os.environ.get("STL_BATCH", "off").lower() in ("on", "1", "true", "yes"):
        batch_evaluate()
    else:
        # Example offline evaluation and report for one folder
        folder = os.environ.get("STL_FOLDER", "Law53_0_1")
        kind = os.environ.get("STL_KIND", "F")
        a = float(os.environ.get("STL_A", "0.0"))
        b = float(os.environ.get("STL_B", "2.0"))
        theta = float(os.environ.get("STL_THETA", "0.5"))
        # predicate e.g. collision==1
        pred_key = os.environ.get("STL_PRED_KEY", "collision")
        pred_val = int(os.environ.get("STL_PRED_VAL", "1"))

        spec = STLSpec(kind=kind, a=a, b=b, predicate={pred_key: pred_val}, theta=theta)
        ev = STLEvaluator(folder)
        baseline = ev.baseline_check(spec)
        optimized = ev.prob_from_model(spec)

        out_dir = BASE_DIR.parent / 'log_safereach' / 'offline_stl'
        ev.save_report(spec, baseline, optimized, out_dir)

        # Print brief
        print(json.dumps({"baseline": baseline, "optimized": optimized}, indent=2))