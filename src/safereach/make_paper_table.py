import os
import json
import csv
from typing import Dict, Any, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')


def read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def select_stl_entry_for_law(overall: Dict[str, Any], law: str) -> Optional[Dict[str, Any]]:
    """
    Pick the offline_stl overall entry for given law and spec matching F[0,2] collision==1 theta≈0.7.
    """
    entries = overall.get('entries', []) if isinstance(overall, dict) else []
    for e in entries:
        if e.get('mode') != law:
            continue
        spec = e.get('spec', {})
        if not isinstance(spec, dict):
            continue
        kind = spec.get('kind')
        a = safe_float(spec.get('a'))
        b = safe_float(spec.get('b'))
        theta = safe_float(spec.get('theta'))
        pred = spec.get('predicate', {}) or {}
        # Match F, [0,2], collision==1, theta around 0.7
        if kind == 'F' and (a is not None and abs(a - 0.0) < 1e-6) and (b is not None and abs(b - 2.0) < 1e-6):
            if isinstance(pred, dict) and pred.get('collision') == 1:
                if theta is None or abs(theta - 0.7) < 1e-6:
                    return e
    return None


def build_table() -> List[Dict[str, Any]]:
    diag = read_json(os.path.join(LOG_DIR, 'mdp_stl_diagnostic_report.json')) or {}
    av_list: List[Dict[str, Any]] = diag.get('av', []) if isinstance(diag, dict) else []
    stl_overall = read_json(os.path.join(LOG_DIR, 'offline_stl', 'overall_stl_summary.json')) or {}
    best_params = read_json(os.path.join(LOG_DIR, 'reach_avoid_best_params.json')) or {}
    # Fallback: read diagnostic CSV for improvement intervals if JSON misses them
    csv_map: Dict[str, Dict[str, Any]] = {}
    csv_path = os.path.join(LOG_DIR, 'mdp_stl_diagnostic_report.csv')
    try:
        if os.path.isfile(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                cr = csv.DictReader(f)
                for row in cr:
                    law_name = row.get('law')
                    if law_name:
                        csv_map[law_name] = row
    except Exception:
        csv_map = {}

    rows: List[Dict[str, Any]] = []
    for item in av_list:
        law = item.get('law')
        if not law:
            continue
        baseline = item.get('baseline', {}) or {}
        exp = item.get('expected_improvements', {}) or {}
        aii = safe_float(item.get('action_influence_index'))

        # ProGuard baseline metrics
        # Baseline (from av_eval.log parsed into diagnostic JSON): flat keys
        pg_unsafe = safe_float(baseline.get('unsafe_pct_after'))
        pg_comp = safe_float(baseline.get('completion_pct'))
        pg_pred07 = safe_float((baseline.get('prediction_pct', {}) or {}).get('0.7'))
        pg_tahead07 = safe_float((baseline.get('time_ahead_s', {}) or {}).get('0.7'))

        # Best params per law
        bp = (best_params.get(law) or {}) if isinstance(best_params, dict) else {}
        theta_min = safe_float(bp.get('theta_min'))
        theta_max = safe_float(bp.get('theta_max'))
        time_cap = safe_float(bp.get('time_cap'))

        # MDP+STL offline_stl entry
        stl_entry = select_stl_entry_for_law(stl_overall, law)
        stl_base_pct = safe_float(stl_entry.get('baseline_pct')) if stl_entry else None
        stl_opt_pct = safe_float(stl_entry.get('optimized_prob_pct')) if stl_entry else None

        # Improvement intervals from diagnostic JSON; fallback to CSV if missing
        # Keys align with gen_charts and mdp_stl_diagnostic CSV columns
        pred_lo = safe_float(exp.get('pred0.7_lower'))
        pred_hi = safe_float(exp.get('pred0.7_upper'))
        ta_lo = safe_float(exp.get('ta_factor_lower'))
        ta_hi = safe_float(exp.get('ta_factor_upper'))
        comp_lo = safe_float(exp.get('completion_lower'))
        comp_hi = safe_float(exp.get('completion_upper'))
        if (pred_lo is None or pred_hi is None or ta_lo is None or ta_hi is None or comp_lo is None or comp_hi is None):
            c = csv_map.get(law, {})
            if pred_lo is None:
                pred_lo = safe_float(c.get('pred0.7_lower'))
            if pred_hi is None:
                pred_hi = safe_float(c.get('pred0.7_upper'))
            if ta_lo is None:
                ta_lo = safe_float(c.get('ta_factor_lower'))
            if ta_hi is None:
                ta_hi = safe_float(c.get('ta_factor_upper'))
            if comp_lo is None:
                comp_lo = safe_float(c.get('completion_lower'))
            if comp_hi is None:
                comp_hi = safe_float(c.get('completion_upper'))

        rows.append({
            'law': law,
            'aii': aii,
            'pg_unsafe_after': pg_unsafe,
            'pg_completion_pct': pg_comp,
            'pg_prediction_pct@0.7': pg_pred07,
            'pg_time_ahead_s@0.7': pg_tahead07,
            'pg_best_theta_min': theta_min,
            'pg_best_theta_max': theta_max,
            'pg_best_time_cap': time_cap,
            'stl_baseline_pct': stl_base_pct,
            'stl_optimized_prob_pct': stl_opt_pct,
            'stl_pred0.7_lower': pred_lo,
            'stl_pred0.7_upper': pred_hi,
            'stl_timeahead_factor_lower': ta_lo,
            'stl_timeahead_factor_upper': ta_hi,
            'stl_completion_lower': comp_lo,
            'stl_completion_upper': comp_hi,
        })

    # Sort by AII descending (None treated as -1 and placed last)
    rows.sort(key=lambda r: (r['aii'] if r.get('aii') is not None else -1.0), reverse=True)
    return rows


def write_outputs(rows: List[Dict[str, Any]]) -> None:
    out_csv = os.path.join(LOG_DIR, 'paper_perlaw_comparison.csv')
    out_json = os.path.join(LOG_DIR, 'paper_perlaw_comparison.json')
    os.makedirs(LOG_DIR, exist_ok=True)
    # CSV header order
    headers = [
        'law', 'aii',
        'pg_unsafe_after', 'pg_completion_pct', 'pg_prediction_pct@0.7', 'pg_time_ahead_s@0.7',
        'pg_best_theta_min', 'pg_best_theta_max', 'pg_best_time_cap',
        'stl_baseline_pct', 'stl_optimized_prob_pct',
        'stl_pred0.7_lower', 'stl_pred0.7_upper',
        'stl_timeahead_factor_lower', 'stl_timeahead_factor_upper',
        'stl_completion_lower', 'stl_completion_upper',
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in headers})
    write_json(out_json, {'rows': rows})


def main():
    rows = build_table()
    write_outputs(rows)


if __name__ == '__main__':
    main()