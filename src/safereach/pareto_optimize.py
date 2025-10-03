import os
import json
import csv
from typing import Dict, Any, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')
OUT_DIR = os.path.join(LOG_DIR, 'pareto')


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _extract_metrics(item: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # completion_pct, unsafe_pct_after, time_ahead_s@0.7, prediction_pct@0.7
    reach = item.get('reach', {}) or {}
    law = item.get('law', {}) or {}
    comp = _safe_float(reach.get('completion_pct'))
    unsafe = _safe_float(reach.get('unsafe_pct_after'))
    ta_map = law.get('time_ahead_s', {}) or {}
    pred_map = law.get('prediction_pct', {}) or {}
    ta = None
    pred = None
    # prefer 0.7
    if isinstance(ta_map, dict) and ta_map:
        ta = _safe_float(ta_map.get('0.7'))
        if ta is None:
            # fallback first value
            try:
                k, v = next(iter(ta_map.items()))
                ta = _safe_float(v)
            except Exception:
                ta = None
    if isinstance(pred_map, dict) and pred_map:
        pred = _safe_float(pred_map.get('0.7'))
        if pred is None:
            try:
                k, v = next(iter(pred_map.items()))
                pred = _safe_float(v)
            except Exception:
                pred = None
    return comp, unsafe, ta, pred


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    # Maximize completion, time_ahead; Minimize unsafe
    ca, ua, ta = a['completion'], a['unsafe'], a['time_ahead']
    cb, ub, tb = b['completion'], b['unsafe'], b['time_ahead']
    # handle None by treating as worst
    def val_or(v: Optional[float], default: float) -> float:
        return v if isinstance(v, (int, float)) else default
    ca = val_or(ca, -1.0)
    cb = val_or(cb, -1.0)
    ta = val_or(ta, -1.0)
    tb = val_or(tb, -1.0)
    ua = val_or(ua, 1e9)
    ub = val_or(ub, 1e9)
    ge_all = (ca >= cb) and (ta >= tb) and (ua <= ub)
    gt_any = (ca > cb) or (ta > tb) or (ua < ub)
    return ge_all and gt_any


def _normalize(vals: List[Optional[float]], invert: bool = False) -> List[float]:
    xs = [v for v in vals if isinstance(v, (int,float))]
    if not xs:
        return [0.0 for _ in vals]
    vmin, vmax = min(xs), max(xs)
    if abs(vmax - vmin) < 1e-12:
        return [1.0 if (isinstance(v,(int,float)) and ((not invert and v==vmax) or (invert and v==vmin))) else 0.0 for v in vals]
    def norm(v: Optional[float]) -> float:
        if not isinstance(v, (int, float)):
            return 0.0
        x = (v - vmin) / (vmax - vmin)
        return (1.0 - x) if invert else x
    return [norm(v) for v in vals]


def _score(points: List[Dict[str, Any]]) -> None:
    comps = [p['completion'] for p in points]
    unsafes = [p['unsafe'] for p in points]
    tas = [p['time_ahead'] for p in points]
    comp_n = _normalize(comps, invert=False)
    unsafe_n = _normalize(unsafes, invert=True)
    ta_n = _normalize(tas, invert=False)
    # weights from env (defaults: comp=0.5, unsafe=0.4, ta=0.1)
    try:
        w_comp = float(os.environ.get('PARETO_W_COMP', '0.5'))
    except Exception:
        w_comp = 0.5
    try:
        w_unsafe = float(os.environ.get('PARETO_W_UNSAFE', '0.4'))
    except Exception:
        w_unsafe = 0.4
    try:
        w_ta = float(os.environ.get('PARETO_W_TA', '0.1'))
    except Exception:
        w_ta = 0.1
    s = w_comp + w_unsafe + w_ta
    if s > 0:
        w_comp, w_unsafe, w_ta = (w_comp / s, w_unsafe / s, w_ta / s)
    for i, p in enumerate(points):
        p['score'] = w_comp * comp_n[i] + w_unsafe * unsafe_n[i] + w_ta * ta_n[i]


def _pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if _dominates(b, a):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def load_sweeps() -> Dict[str, List[Dict[str, Any]]]:
    data: Dict[str, List[Dict[str, Any]]] = {}
    for name in os.listdir(LOG_DIR):
        if name.startswith('reach_avoid_sweep_') and name.endswith('.json'):
            path = os.path.join(LOG_DIR, name)
            obj = _read_json(path)
            if not obj:
                continue
            law = obj.get('law')
            results = obj.get('results', [])
            if law and isinstance(results, list):
                pts: List[Dict[str, Any]] = []
                for r in results:
                    comp, unsafe, ta, pred = _extract_metrics(r)
                    pts.append({
                        'theta_min': r.get('theta_min'),
                        'theta_max': r.get('theta_max'),
                        'time_cap': r.get('time_cap'),
                        'completion': comp,
                        'unsafe': unsafe,
                        'time_ahead': ta,
                        'prediction': pred,
                    })
                data[law] = pts
    return data


def write_outputs(summary: Dict[str, Any], perlaw: Dict[str, Any]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # summary json
    with open(os.path.join(OUT_DIR, 'pareto_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    # per-law jsons
    for law, obj in perlaw.items():
        with open(os.path.join(OUT_DIR, f'pareto_{law}.json'), 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2)
    # combined CSV
    with open(os.path.join(OUT_DIR, 'pareto_all.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['law','theta_min','theta_max','time_cap','completion_pct','unsafe_pct_after','time_ahead_s@0.7','prediction_pct@0.7','is_pareto','score'])
        w.writeheader()
        for law, obj in perlaw.items():
            for p in obj.get('points', []):
                w.writerow({
                    'law': law,
                    'theta_min': p.get('theta_min'),
                    'theta_max': p.get('theta_max'),
                    'time_cap': p.get('time_cap'),
                    'completion_pct': p.get('completion'),
                    'unsafe_pct_after': p.get('unsafe'),
                    'time_ahead_s@0.7': p.get('time_ahead'),
                    'prediction_pct@0.7': p.get('prediction'),
                    'is_pareto': int(p.get('is_pareto', False)),
                    'score': p.get('score'),
                })


def _svg_scatter_comp_unsafe(points: List[Dict[str, Any]], title: str) -> str:
    # x: unsafe (lower better), y: completion (higher better), color: time_ahead
    width, height, pad = 800, 500, 60
    xs = [p['unsafe'] if isinstance(p.get('unsafe'), (int,float)) else 0.0 for p in points]
    ys = [p['completion'] if isinstance(p.get('completion'), (int,float)) else 0.0 for p in points]
    cs = [p['time_ahead'] if isinstance(p.get('time_ahead'), (int,float)) else 0.0 for p in points]
    x_min, x_max = (min(xs) if xs else 0.0), (max(xs) if xs else 1.0)
    y_min, y_max = (min(ys) if ys else 0.0), (max(ys) if ys else 1.0)
    c_min, c_max = (min(cs) if cs else 0.0), (max(cs) if cs else 1.0)
    def norm(v, vmin, vmax):
        if vmax - vmin < 1e-12:
            return 0.5
        return (v - vmin) / (vmax - vmin)
    def map_x(x):
        return pad + (width - 2*pad) * (x - x_min) / (x_max - x_min if x_max > x_min else 1.0)
    def map_y(y):
        return height - pad - (height - 2*pad) * (y - y_min) / (y_max - y_min if y_max > y_min else 1.0)
    def color(c):
        t = norm(c, c_min, c_max)
        r = int(255 * (0.2 + 0.6 * t))
        g = int(80)
        b = int(255 * (1.0 - 0.6 * t))
        return f'rgb({r},{g},{b})'
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    svg.append(f'<text x="{width/2}" y="30" font-size="16" text-anchor="middle">{title}</text>')
    # axes
    svg.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>')
    svg.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>')
    svg.append(f'<text x="{width/2}" y="{height-20}" font-size="12" text-anchor="middle">Unsafe % (lower better)</text>')
    svg.append(f'<text x="20" y="{height/2}" font-size="12" transform="rotate(-90 20,{height/2})" text-anchor="middle">Completion % (higher better)</text>')
    # points
    for p in points:
        x = map_x(p.get('unsafe') or 0.0)
        y = map_y(p.get('completion') or 0.0)
        col = color(p.get('time_ahead') or 0.0)
        r = 5 if not p.get('is_pareto', False) else 7
        stroke = 'red' if p.get('is_recommended', False) else 'none'
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{col}" stroke="{stroke}" stroke-width="2"/>')
    svg.append('</svg>')
    return '\n'.join(svg)


def main():
    data = load_sweeps()
    os.makedirs(OUT_DIR, exist_ok=True)
    perlaw: Dict[str, Any] = {}
    summary = {'laws': [], 'counts': {}}
    for law, pts in data.items():
        if not pts:
            continue
        # compute scores
        _score(pts)
        # pareto front
        front = _pareto_front(pts)
        # mark
        for p in pts:
            p['is_pareto'] = any((p is q) or (p.get('theta_min')==q.get('theta_min') and p.get('theta_max')==q.get('theta_max') and p.get('time_cap')==q.get('time_cap')) for q in front)
        # recommended: highest score among pareto points (fallback to global max)
        rec = None
        pareto_pts = [p for p in pts if p.get('is_pareto')]
        if pareto_pts:
            rec = max(pareto_pts, key=lambda x: (x.get('score') or 0.0))
        else:
            rec = max(pts, key=lambda x: (x.get('score') or 0.0))
        if rec:
            rec['is_recommended'] = True
        # write per-law objects
        perlaw[law] = {
            'points': pts,
            'pareto_front': front,
            'recommended': rec,
        }
        # scatter svg
        svg = _svg_scatter_comp_unsafe(pts, f'{law}: Completion vs Unsafe (color=time-ahead)')
        with open(os.path.join(OUT_DIR, f'pareto_{law}_comp_vs_unsafe.svg'), 'w', encoding='utf-8') as f:
            f.write(svg)
        summary['laws'].append(law)
        summary['counts'][law] = {'total': len(pts), 'pareto': len(front)}
    write_outputs(summary, perlaw)
    print('Pareto optimization completed. Outputs in', OUT_DIR)


if __name__ == '__main__':
    main()