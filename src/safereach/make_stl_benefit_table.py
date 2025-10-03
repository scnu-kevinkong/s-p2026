import os
import json
import csv
from typing import Dict, Any, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')
STL_DIR = os.path.join(LOG_DIR, 'offline_stl')
OUT_DIR = os.path.join(LOG_DIR, 'pareto')


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def build_summary() -> Dict[str, Any]:
    overall = _read_json(os.path.join(STL_DIR, 'overall_stl_summary.json'))
    entries: List[Dict[str, Any]] = overall.get('entries', []) if isinstance(overall, dict) else []
    per_law: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for e in entries:
        mode = e.get('mode')
        spec = e.get('spec', {}) or {}
        base = e.get('baseline_pct', 0.0)
        opt = e.get('optimized_prob_pct', 0.0)
        mc = e.get('mc_prob_pct')
        skey = f"{spec.get('kind','')}[{spec.get('a',0)},{spec.get('b',0)}]"
        rows.append({
            'law': mode,
            'spec_key': skey,
            'theta': spec.get('theta'),
            'baseline_pct': base,
            'optimized_prob_pct': opt,
            'mc_prob_pct': mc if (mc is not None) else ''
        })
        if mode not in per_law:
            per_law[mode] = {'count': 0, 'sum_base': 0.0, 'sum_opt': 0.0, 'sum_mc': 0.0, 'mc_count': 0}
        per_law[mode]['count'] += 1
        per_law[mode]['sum_base'] += (base or 0.0)
        per_law[mode]['sum_opt'] += (opt or 0.0)
        if isinstance(mc, (int, float)):
            per_law[mode]['sum_mc'] += mc
            per_law[mode]['mc_count'] += 1

    # Aggregate per law
    agg_rows: List[Dict[str, Any]] = []
    for law, acc in per_law.items():
        n = max(1, acc['count'])
        avg_base = acc['sum_base'] / n
        avg_opt = acc['sum_opt'] / n
        avg_mc = (acc['sum_mc'] / acc['mc_count']) if acc['mc_count'] > 0 else ''
        agg_rows.append({
            'law': law,
            'avg_baseline_pct': avg_base,
            'avg_optimized_prob_pct': avg_opt,
            'avg_mc_prob_pct': avg_mc,
            'delta_opt_minus_base': (avg_opt - avg_base)
        })

    return {'per_spec_rows': rows, 'per_law_rows': agg_rows}


def write_outputs(data: Dict[str, Any]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # JSON
    with open(os.path.join(OUT_DIR, 'mdp_stl_benefit_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    # CSVs
    per_spec = data.get('per_spec_rows', [])
    per_law = data.get('per_law_rows', [])
    if per_spec:
        with open(os.path.join(OUT_DIR, 'mdp_stl_benefit_per_spec.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(per_spec[0].keys()))
            w.writeheader()
            for r in per_spec:
                w.writerow(r)
    if per_law:
        with open(os.path.join(OUT_DIR, 'mdp_stl_benefit_per_law.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(per_law[0].keys()))
            w.writeheader()
            for r in per_law:
                w.writerow(r)


def main():
    data = build_summary()
    write_outputs(data)
    print('Wrote STL benefit summaries to', OUT_DIR)


if __name__ == '__main__':
    main()