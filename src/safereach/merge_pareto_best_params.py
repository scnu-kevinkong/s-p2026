import os
import json
import csv
from typing import Dict, Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')
PARETO_DIR = os.path.join(LOG_DIR, 'pareto')
OUT_JSON = os.path.join(LOG_DIR, 'reach_avoid_best_params_pareto.json')
OUT_CSV = os.path.join(LOG_DIR, 'reach_avoid_best_params_pareto.csv')


def merge() -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(PARETO_DIR):
        return best
    for name in os.listdir(PARETO_DIR):
        if not name.startswith('pareto_'):
            continue
        if not name.endswith('.json'):
            continue
        if name in ('pareto_summary.json',):
            continue
        law = name[len('pareto_'):-len('.json')]
        path = os.path.join(PARETO_DIR, name)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue
        rec = obj.get('recommended', {}) or {}
        if not rec:
            continue
        theta_min = rec.get('theta_min')
        theta_max = rec.get('theta_max')
        time_cap = rec.get('time_cap')
        best[law] = {
            'theta_min': theta_min,
            'theta_max': theta_max,
            'time_cap': time_cap,
            'source': 'pareto'
        }
    return best


def write(best: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['law', 'theta_min', 'theta_max', 'time_cap', 'source'])
        w.writeheader()
        for law, v in best.items():
            w.writerow({
                'law': law,
                'theta_min': v.get('theta_min'),
                'theta_max': v.get('theta_max'),
                'time_cap': v.get('time_cap'),
                'source': v.get('source', 'pareto')
            })


def main():
    best = merge()
    write(best)
    print('Wrote pareto best params to:', OUT_JSON)


if __name__ == '__main__':
    main()