import os, json, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

from .embodied.abstraction import EmbodiedAbstraction
from .runtime_monitor import runtime_monitor

BASE_DIR = Path(os.path.dirname(__file__))
LOG_DIR = str(BASE_DIR / "embodied" / "samples")
MODEL_DIR = str(BASE_DIR / "embodied" / "dtmcs")

# ---------- DTMC helpers ----------

def _load_model(model_json_path: str) -> Tuple[int, Dict[int, Dict[int, float]], Dict[str,int]]:
    with open(model_json_path,'r') as f: m = json.load(f)
    states, idx, trans = m['states'], m['state_index'], m['transition_probs']
    n = len(states)
    outgoing: Dict[int, Dict[int, float]] = {}
    for sk, edges in trans.items():
        i = idx[sk]; row: Dict[int,float] = {}
        for tk, w in edges.items():
            j = idx[tk]
            if isinstance(w, str) and '/' in w:
                a,b = w.split('/'); val = float(a)/float(b)
            else:
                val = float(w)
            row[j] = val
        outgoing[i] = row
    return n, outgoing, idx

def _distribution_after_k(n: int, outgoing: Dict[int, Dict[int,float]], s0: int, k: int) -> List[float]:
    dist = [0.0]*n; dist[s0] = 1.0
    for _ in range(k):
        nxt = [0.0]*n
        for i,row in outgoing.items():
            pi = dist[i]
            if pi==0.0: continue
            for j,p in row.items(): nxt[j] += pi*p
        dist = nxt
    return dist

def _finite_window_stay_prob(n: int, outgoing: Dict[int, Dict[int,float]], allowed: Set[int], H: int) -> List[float]:
    dp = [1.0 if (i in allowed) else 0.0 for i in range(n)]
    for _ in range(H):
        nxt = [0.0]*n
        for i in range(n):
            if i not in allowed: continue
            row = outgoing.get(i, {}); acc = 0.0
            for j,p in row.items():
                if j in allowed: acc += dp[j]*p
            nxt[i] = acc
        dp = nxt
    return dp

# ---------- Embodied dataset helpers ----------

def _list_modes() -> List[str]:
    if not os.path.isdir(LOG_DIR): return []
    modes: List[str] = []
    for d in os.listdir(LOG_DIR):
        p = os.path.join(LOG_DIR,d)
        if os.path.isdir(p) and os.path.exists(f"{MODEL_DIR}/{d}/dtmc.prism") and os.path.exists(f"{MODEL_DIR}/{d}/abstraction.json") and os.path.exists(f"{LOG_DIR}/{d}/spec"):
            if any(f.endswith('.json') for f in os.listdir(p)):
                modes.append(d)
    modes.sort(); return modes

@dataclass
class STLSpec:
    kind: str  # 'F' or 'G'
    a: float
    b: float
    predicate: Dict
    theta: float = 0.5

class EmbodiedSTL:
    def __init__(self, folder: str) -> None:
        self.folder = folder
        with open(f"{MODEL_DIR}/{folder}/abstraction.json") as af:
            abs_obj = json.loads(af.read())
        self.abs = EmbodiedAbstraction(abs_obj["objectTypes"], abs_obj["keys"], abs_obj["parentReceptacles"]) 
        self.prism_path = f"{MODEL_DIR}/{folder}/dtmc.prism"
        self.model_json = f"{MODEL_DIR}/{folder}/model.json"
        # unsafe set from spec file
        with open(f"{LOG_DIR}/{folder}/spec") as sf:
            spec = json.loads(sf.read())
        unsafe_states = self.abs.filter(spec)
        st_idx = self.abs.get_state_idx()
        self.unsafe_indices: Set[int] = set(st_idx[s] for s in unsafe_states if s in st_idx)

    def _iter_logs(self):
        for o in os.listdir(f"{LOG_DIR}/{self.folder}"):
            if o.endswith('.json'):
                yield f"{LOG_DIR}/{self.folder}/{o}"

    def _extract_observations(self, obj: Dict) -> List[Dict]:
        if "s_trans" in obj:
            return [t.get("state", t) for t in obj.get("s_trans", [])]
        if "trajectory" in obj:
            return obj.get("trajectory", [])
        return []

    def _step_time(self, ob: Dict) -> float:
        # Embodied logs typically provide state snapshots as lists without explicit dt.
        # Use a configurable constant step time for window accumulation.
        return float(os.environ.get("SAFE_REACH_STEP_TIME_S", "1.0"))

    def _initial_indices_from_logs(self) -> List[int]:
        initials: List[int] = []
        idx = self.abs.get_state_idx()
        for log in self._iter_logs():
            try:
                obj = json.loads(Path(log).read_text())
                obs = self._extract_observations(obj)
                if not obs: continue
                s0_name = self.abs.encode(obs[0])
                if s0_name in idx: initials.append(idx[s0_name])
            except Exception:
                continue
        return initials

    def _monte_carlo_prob(self, n:int, outgoing:Dict[int,Dict[int,float]], initials:List[int], pred_set:Set[int], a_steps:int, b_steps:int, trials:int=1000, kind:str='F') -> float:
        if not initials: initials=[0]
        H = max(0, int(round(b_steps - a_steps + 1)))
        if H<=0: return 0.0
        cdfs: Dict[int, List[Tuple[float,int]]] = {}
        for i,row in outgoing.items():
            acc=0.0; items=[]
            for j,p in row.items(): acc+=p; items.append((acc,j))
            if items:
                last=items[-1][0]; items=[(c/last if last>0 else 1.0, j) for c,j in items]
            cdfs[i]=items
        def step(s:int)->int:
            row=cdfs.get(s); 
            if not row: return s
            r=random.random()
            for c,j in row:
                if r<=c: return j
            return row[-1][1]
        success=0
        for _ in range(trials):
            s = random.choice(initials)
            for _k in range(a_steps): s = step(s)
            if kind=='F':
                hit = (s in pred_set)
                for _h in range(max(0,H-1)):
                    if hit: break
                    s = step(s)
                    if s in pred_set: hit=True; break
                if hit: success+=1
            else:
                hold = (s in pred_set)
                for _h in range(max(0,H-1)):
                    if not hold: break
                    s = step(s)
                    if s not in pred_set: hold=False; break
                if hold: success+=1
        return success/float(trials) if trials>0 else 0.0

    # ---------- Baseline ----------
    def baseline_check(self, spec: STLSpec) -> Dict:
        sat = 0; total=0
        for log in self._iter_logs():
            try:
                obj = json.loads(Path(log).read_text())
            except Exception:
                continue
            obs = self._extract_observations(obj)
            if not obs: continue
            total += 1
            t = 0.0
            satisfied = (False if spec.kind=='F' else True)
            entered=False
            cache: Dict = {}
            for ob in obs:
                t += self._step_time(ob)
                if t < spec.a: continue
                entered=True
                if t > spec.b: break
                prob = runtime_monitor(ob, self.prism_path, self.abs, self.unsafe_indices, cache=cache)
                truth = (prob >= spec.theta)
                if spec.kind=='F':
                    if truth: satisfied=True; break
                else:
                    if not truth: satisfied=False; break
            if spec.kind=='G' and not entered: satisfied=False
            if satisfied: sat += 1
        return {
            "baseline_total_logs": total,
            "baseline_satisfied": sat,
            "baseline_pct": (sat/total*100.0) if total else 0.0,
        }

    # ---------- Optimized (DTMC) ----------
    def prob_from_model(self, spec: STLSpec) -> Dict:
        if not os.path.exists(self.model_json): return {"error":"model_json_missing"}
        n, outgoing, idx = _load_model(self.model_json)
        pred_set = set(self.unsafe_indices)
        not_pred = set(range(n)) - pred_set
        a_steps = int(round(spec.a)); b_steps = int(round(spec.b))
        if b_steps < a_steps: a_steps, b_steps = 0, 0
        H = max(0, b_steps - a_steps + 1)
        initials = self._initial_indices_from_logs() or [0]
        stay_not = _finite_window_stay_prob(n, outgoing, not_pred, H)
        stay_yes = _finite_window_stay_prob(n, outgoing, pred_set, H)
        total_prob = 0.0
        for s0 in initials:
            Qa = _distribution_after_k(n, outgoing, s0, a_steps)
            if spec.kind=='F':
                nohit=0.0
                for s in range(n):
                    if s in not_pred: nohit += Qa[s]*stay_not[s]
                total_prob += (1.0 - nohit)
            else:
                allhold=0.0
                for s in range(n): allhold += Qa[s]*stay_yes[s]
                total_prob += allhold
        avg_prob = total_prob/len(initials)
        mc_on = os.environ.get("STL_MC","on").lower() in ("on","1","true","yes")
        mc_trials = int(os.environ.get("STL_MC_TRIALS","1000"))
        res = {"optimized_prob": avg_prob, "initial_states": len(initials), "a_steps": a_steps, "b_steps": b_steps}
        if mc_on and mc_trials>0:
            mc_p = self._monte_carlo_prob(n,outgoing,initials,pred_set,a_steps,b_steps,mc_trials,spec.kind)
            import math as _m
            stderr = (_m.sqrt(mc_p*(1.0-mc_p)/mc_trials) if mc_trials>0 else 0.0)
            ci = 1.96*stderr
            res.update({"mc_trials": mc_trials, "mc_prob": mc_p, "mc_stderr": stderr, "mc_ci95_low": max(0.0,mc_p-ci), "mc_ci95_high": min(1.0,mc_p+ci), "mc_vs_optimized_delta": (mc_p-avg_prob)})
        return res

    # ---------- ProGuard-style metrics (per-folder) ----------
    def _eval_pg_law_reach(self) -> Dict:
        THS = [float(x) for x in os.environ.get("AV_THRESHOLDS","0.3,0.5,0.7").split(",")]
        step_time_s = float(os.environ.get("SAFE_REACH_STEP_TIME_S","1.0"))
        pred_counts = {t:0 for t in THS}
        ta_acc = {t:[] for t in THS}
        total=0; before_v=0; after_v=0; completed=0
        cache: Dict = {}
        for log in self._iter_logs():
            try:
                obj = json.loads(Path(log).read_text())
            except Exception:
                continue
            s_trans = obj.get("s_trans", [])
            steps = obj.get("intermediate_steps", [])
            if len(s_trans)!=len(steps): continue
            obs = [t.get("state", t) for t in s_trans]
            # first pass for LAW
            running=0.0; predicted={t:False for t in THS}
            probs=[]
            for ob in obs:
                p = runtime_monitor(ob, self.prism_path, self.abs, self.unsafe_indices, cache=cache)
                probs.append(p); running += step_time_s
                for th in THS:
                    if (not predicted[th]) and p>=th:
                        predicted[th]=True; pred_counts[th]+=1; ta_acc[th].append(running)
            # enforcement: stop at first p>=THETA
            THETA = float(os.environ.get("SAFE_REACH_THETA","0.85"))
            stop_idx = None
            for i,p in enumerate(probs):
                if p>=THETA: stop_idx=i; break
            obs_after = obs[:(stop_idx+1) if stop_idx is not None else len(obs)]
            before_viol = any(runtime_monitor(ob, self.prism_path, self.abs, self.unsafe_indices, cache=cache)>=1.0 for ob in obs)
            after_viol = any(runtime_monitor(ob, self.prism_path, self.abs, self.unsafe_indices, cache=cache)>=1.0 for ob in obs_after)
            done = (not after_viol) and (stop_idx is not None)
            total+=1; before_v += 1 if before_viol else 0; after_v += 1 if after_viol else 0; completed += 1 if done else 0
        if total==0:
            return {"reach":{}, "law":{}}
        reach = {
            "unsafe_pct_before": 100.0*before_v/total,
            "unsafe_pct_after": 100.0*after_v/total,
            "completion_pct": 100.0*completed/total
        }
        law = {
            "prediction_pct": {str(t): 100.0*pred_counts[t]/total for t in THS},
            "time_ahead_s": {str(t): (sum(ta_acc[t])/len(ta_acc[t]) if len(ta_acc[t])>0 else 0.0) for t in THS}
        }
        return {"reach": reach, "law": law}

    # ---------- Report ----------
    def save_report(self, spec: STLSpec, baseline: Dict, optimized: Dict, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pg = self._eval_pg_law_reach()
        report = {
            "folder": self.folder,
            "spec": {"kind": spec.kind, "a": spec.a, "b": spec.b, "theta": spec.theta, "predicate": spec.predicate},
            "baseline": baseline,
            "optimized": optimized,
            "proguard_offline": pg
        }
        (out_dir / f"{self.folder}_stl_report.json").write_text(json.dumps(report, indent=2))
        # csv
        with open(out_dir / f"{self.folder}_stl_report.csv", 'w') as f:
            f.write("metric,value\n")
            f.write(f"baseline_pct,{baseline.get('baseline_pct',0.0):.6f}\n")
            f.write(f"optimized_prob,{optimized.get('optimized_prob',0.0):.6f}\n")
            if 'mc_prob' in optimized: f.write(f"mc_prob,{optimized.get('mc_prob',0.0):.6f}\n")
            reach = pg.get("reach", {})
            f.write(f"unsafe_pct_before,{reach.get('unsafe_pct_before',0.0):.6f}\n")
            f.write(f"unsafe_pct_after,{reach.get('unsafe_pct_after',0.0):.6f}\n")
            f.write(f"completion_pct,{reach.get('completion_pct',0.0):.6f}\n")
            law = pg.get("law", {})
            for k,v in sorted((law.get("prediction_pct",{}) or {}).items(), key=lambda x: float(x[0])):
                f.write(f"prediction_pct@{k},{float(v):.6f}\n")
            for k,v in sorted((law.get("time_ahead_s",{}) or {}).items(), key=lambda x: float(x[0])):
                f.write(f"time_ahead_s@{k},{float(v):.6f}\n")

# ---------- Spec parsing and batch ----------

def _spec_key(spec: STLSpec) -> str:
    if spec.predicate:
        k,v = next(iter(spec.predicate.items()))
        pv = f"{k}=={v}"
    else:
        pv = "unsafe"
    return f"{spec.kind}[{spec.a},{spec.b}]_{pv}"

def _parse_specs() -> List[STLSpec]:
    # support STL_SPECS env: "F,0,2,x,1;G,0,5,x,1"; predicate ignored for embodied
    s = os.environ.get("STL_SPECS", "").strip()
    specs: List[STLSpec] = []
    if s:
        for seg in s.split(';'):
            parts = [p.strip() for p in seg.split(',') if p.strip()]
            if len(parts) >= 5:
                kind,a,b,pk,pv = parts[:5]
                theta = float(parts[5]) if len(parts)>=6 else 0.5
                specs.append(STLSpec(kind=kind, a=float(a), b=float(b), predicate={pk: pv}, theta=theta))
    if not specs:
        # default 4 specs
        specs = [
            STLSpec('F',0.0,2.0,{"unsafe":1},0.5),
            STLSpec('F',0.0,5.0,{"unsafe":1},0.5),
            STLSpec('G',0.0,2.0,{"unsafe":1},0.5),
            STLSpec('G',0.0,5.0,{"unsafe":1},0.5),
        ]
    return specs


def batch_evaluate() -> None:
    modes_env = os.environ.get("STL_MODES", "").strip()
    modes = [m for m in modes_env.split(',') if m] if modes_env else _list_modes()
    specs = _parse_specs()
    out_root = BASE_DIR.parent / 'log_safereach' / 'offline_stl_embodied'
    out_root.mkdir(parents=True, exist_ok=True)
    overall_entries: List[Dict] = []
    for spec in specs:
        spec_dir = out_root / _spec_key(spec)
        spec_dir.mkdir(parents=True, exist_ok=True)
        for mode in modes:
            ev = EmbodiedSTL(mode)
            baseline = ev.baseline_check(spec)
            optimized = ev.prob_from_model(spec)
            ev.save_report(spec, baseline, optimized, spec_dir)
            overall_entries.append({
                "mode": mode,
                "spec_key": _spec_key(spec),
                "spec": {"kind": spec.kind, "a": spec.a, "b": spec.b, "theta": spec.theta, "predicate": spec.predicate},
                "baseline_pct": baseline.get('baseline_pct',0.0),
                "optimized_prob_pct": optimized.get('optimized_prob',0.0)*100.0,
                "mc_prob_pct": (optimized.get('mc_prob',0.0)*100.0) if 'mc_prob' in optimized else None,
            })
    avg_base = sum(e['baseline_pct'] for e in overall_entries)/len(overall_entries) if overall_entries else 0.0
    avg_opt = sum(e['optimized_prob_pct'] for e in overall_entries)/len(overall_entries) if overall_entries else 0.0
    mc_list = [e['mc_prob_pct'] for e in overall_entries if isinstance(e.get('mc_prob_pct',None),(int,float)) and e['mc_prob_pct'] is not None]
    avg_mc = (sum(mc_list)/len(mc_list)) if mc_list else None
    summary = {
        "modes": modes,
        "num_modes": len(modes),
        "specs": [{"kind": s.kind, "a": s.a, "b": s.b, "theta": s.theta, "predicate": s.predicate, "key": _spec_key(s)} for s in specs],
        "entries": overall_entries,
        "avg_baseline_pct": avg_base,
        "avg_optimized_prob_pct": avg_opt,
        "avg_mc_prob_pct": avg_mc,
    }
    (out_root / 'overall_stl_summary.json').write_text(json.dumps(summary, indent=2))
    # also CSV
    with open(out_root / 'overall_stl_summary.csv', 'w') as f:
        f.write('mode,spec_key,baseline_pct,optimized_prob_pct,mc_prob_pct\n')
        for e in overall_entries:
            mc = '' if e.get('mc_prob_pct') is None else f"{e['mc_prob_pct']:.6f}"
            f.write(f"{e['mode']},{e['spec_key']},{e['baseline_pct']:.6f},{e['optimized_prob_pct']:.6f},{mc}\n")

# ---------- Cross-platform charts (AV vs Embodied) ----------

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
           f"<text x='{pad+20}' y='{32}' font-size='12'>AV</text>",
           f"<rect x='{pad+120}' y='{20}' width='14' height='14' fill='#66bb6a'/>",
           f"<text x='{pad+140}' y='{32}' font-size='12'>Embodied</text>"]
    x = pad
    for i, lab in enumerate(labels):
        b = base_vals[i]
        o = opt_vals[i]
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


def make_cross_platform_charts() -> None:
    root = BASE_DIR.parent / 'log_safereach'
    paper = root / 'paper_metrics_summary.json'
    if not paper.exists():
        # fallback to separate summaries
        av = json.loads((root / 'av_summary.json').read_text()) if (root / 'av_summary.json').exists() else None
        emb = json.loads((root / 'embodied_summary.json').read_text()) if (root / 'embodied_summary.json').exists() else None
        if not av or not emb:
            return
        data = {"av": av, "embodied": emb}
    else:
        data = json.loads(paper.read_text())
    av = data.get('av', {})
    emb = data.get('embodied', {})
    # Reach metrics
    av_r = av.get('reach', av)
    emb_r = emb.get('reach', emb)
    reach_labels = ["Unsafe% (after)", "Completion%"]
    reach_av_vals = [float(av_r.get('unsafe_pct_after', 0.0)), float(av_r.get('completion_pct', 0.0))]
    reach_emb_vals = [float(emb_r.get('unsafe_pct_after', 0.0)), float(emb_r.get('completion_pct', 0.0))]
    charts_dir = root / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    reach_svg = _svg_grouped_bars("Reach Metrics: AV vs Embodied", reach_labels, reach_av_vals, reach_emb_vals)
    (charts_dir / 'cross_platform_reach.svg').write_text(reach_svg)
    # Law metrics at threshold (default 0.5)
    thr = os.environ.get('CHART_THRESHOLD', '0.5')
    av_law = av.get('law', {})
    emb_law = emb.get('law', {})
    pred_av = float((av_law.get('prediction_pct', {}) or {}).get(thr, 0.0))
    pred_emb = float((emb_law.get('prediction_pct', {}) or {}).get(thr, 0.0))
    ta_av = float((av_law.get('time_ahead_s', {}) or {}).get(thr, 0.0))
    ta_emb = float((emb_law.get('time_ahead_s', {}) or {}).get(thr, 0.0))
    law_labels = [f"Prediction% @{thr}", f"Time Ahead @{thr} (s)"]
    law_av_vals = [pred_av, ta_av]
    law_emb_vals = [pred_emb, ta_emb]
    law_svg = _svg_grouped_bars("Law Metrics: AV vs Embodied", law_labels, law_av_vals, law_emb_vals)
    (charts_dir / f'cross_platform_law_thr{thr}.svg').write_text(law_svg)

if __name__ == "__main__":
    if os.environ.get("STL_BATCH","off").lower() in ("on","1","true","yes"):
        batch_evaluate()
        if os.environ.get("MAKE_CHARTS","on").lower() in ("on","1","true","yes"):
            make_cross_platform_charts()
    else:
        folder = os.environ.get("STL_FOLDER", "log_raw_t20")
        kind = os.environ.get("STL_KIND","F")
        a = float(os.environ.get("STL_A","0.0"))
        b = float(os.environ.get("STL_B","2.0"))
        theta = float(os.environ.get("STL_THETA","0.5"))
        spec = STLSpec(kind=kind, a=a, b=b, predicate={"unsafe":1}, theta=theta)
        ev = EmbodiedSTL(folder)
        baseline = ev.baseline_check(spec)
        optimized = ev.prob_from_model(spec)
        out_dir = BASE_DIR.parent / 'log_safereach' / 'offline_stl_embodied'
        ev.save_report(spec, baseline, optimized, out_dir)
        print(json.dumps({"baseline":baseline, "optimized":optimized}, indent=2))