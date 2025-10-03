import os
import json
import csv
from typing import Dict, Any, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')
PAPER_JSON = os.path.join(LOG_DIR, 'paper_perlaw_comparison.json')
PARETO_DIR = os.path.join(LOG_DIR, 'pareto')
BENEFIT_PER_LAW_CSV = os.path.join(PARETO_DIR, 'mdp_stl_benefit_per_law.csv')

OUT_PLUS_JSON = os.path.join(LOG_DIR, 'paper_perlaw_comparison_plus.json')
OUT_PLUS_CSV = os.path.join(LOG_DIR, 'paper_perlaw_comparison_plus.csv')
OUT_PLUS_TEX = os.path.join(LOG_DIR, 'paper_perlaw_table_plus.tex')


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _read_benefit_per_law_csv(path: str) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                law = row.get('law')
                if not law:
                    continue
                try:
                    base = float(row.get('avg_baseline_pct', '0') or '0')
                except Exception:
                    base = 0.0
                try:
                    opt = float(row.get('avg_optimized_prob_pct', '0') or '0')
                except Exception:
                    opt = 0.0
                try:
                    delta = float(row.get('delta_opt_minus_base', '0') or '0')
                except Exception:
                    delta = (opt - base)
                res[law] = {
                    'avg_baseline_pct': base,
                    'avg_optimized_prob_pct': opt,
                    'delta_opt_minus_base': delta,
                }
    except Exception:
        pass
    return res


def build_plus() -> List[Dict[str, Any]]:
    paper = _read_json(PAPER_JSON) or {}
    rows: List[Dict[str, Any]] = paper.get('rows', []) if isinstance(paper, dict) else []
    benefit = _read_benefit_per_law_csv(BENEFIT_PER_LAW_CSV)
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        law = r.get('law')
        b = benefit.get(law, {})
        merged = dict(r)
        merged['stl_avg_baseline_pct'] = b.get('avg_baseline_pct')
        merged['stl_avg_optimized_prob_pct'] = b.get('avg_optimized_prob_pct')
        merged['stl_delta_opt_minus_base'] = b.get('delta_opt_minus_base')
        out_rows.append(merged)
    return out_rows


def write_plus(rows: List[Dict[str, Any]]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_PLUS_JSON, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows}, f, indent=2)
    if rows:
        with open(OUT_PLUS_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)


def _latex_escape(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    return s.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return ''
    try:
        f = float(v)
        return f"{f:.{digits}f}"
    except Exception:
        return _latex_escape(str(v))


HEADERS = [
    'law', 'aii',
    'pg_unsafe_after', 'pg_completion_pct', 'pg_prediction_pct@0.7', 'pg_time_ahead_s@0.7',
    'pg_best_theta_min', 'pg_best_theta_max', 'pg_best_time_cap',
    'stl_baseline_pct', 'stl_optimized_prob_pct',
    'stl_pred0.7_lower', 'stl_pred0.7_upper',
    'stl_timeahead_factor_lower', 'stl_timeahead_factor_upper',
    'stl_completion_lower', 'stl_completion_upper',
    'stl_avg_baseline_pct', 'stl_avg_optimized_prob_pct', 'stl_delta_opt_minus_base',
]


def write_plus_tex(rows: List[Dict[str, Any]]) -> None:
    cols = len(HEADERS)
    align = '|' + '|'.join(['l'] * cols) + '|'
    lines: List[str] = []
    lines.append('\\begin{table}[ht]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append(f'  \\begin{{tabular}}{{{align}}}')
    lines.append('    \\hline')
    hdr = ' & '.join(_latex_escape(h) for h in HEADERS)
    lines.append(f'    {hdr} \\ \\hline')
    for r in rows:
        vals = []
        for k in HEADERS:
            vals.append(_fmt(r.get(k)))
        lines.append('    ' + ' & '.join(vals) + ' \\')
    lines.append('    \\hline')
    lines.append('  \\end{tabular}')
    lines.append('  \\caption{Per-law comparison with STL benefit: ProGuard vs MDP+STL (AII-sorted).}')
    lines.append('  \\label{tab:perlaw-comparison-plus}')
    lines.append('\\end{table}')
    with open(OUT_PLUS_TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    rows = build_plus()
    write_plus(rows)
    write_plus_tex(rows)
    print('Wrote paper per-law comparison plus to:', OUT_PLUS_JSON, OUT_PLUS_CSV, OUT_PLUS_TEX)


if __name__ == '__main__':
    main()