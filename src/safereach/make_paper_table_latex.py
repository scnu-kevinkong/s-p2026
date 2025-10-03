import os
import json
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _latex_escape(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    # minimal escaping for LaTeX
    s = s.replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
    return s


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
]


def build_latex_table(rows: List[Dict[str, Any]]) -> str:
    # tabular with p{...} or l columns; keep compact
    cols = len(HEADERS)
    align = '|' + '|'.join(['l'] * cols) + '|'
    lines = []
    lines.append('\\begin{table}[ht]')
    lines.append('  \\centering')
    lines.append('  \\small')
    lines.append(f'  \\begin{{tabular}}{{{align}}}')
    lines.append('    \\hline')
    # header row
    hdr = ' & '.join(_latex_escape(h) for h in HEADERS)
    lines.append(f'    {hdr} \\ \\hline')
    # data rows
    for r in rows:
        vals = []
        for k in HEADERS:
            v = r.get(k)
            out = _fmt(v)
            vals.append(out)
        lines.append('    ' + ' & '.join(vals) + ' \\')
    lines.append('    \\hline')
    lines.append('  \\end{tabular}')
    lines.append('  \\caption{Per-law comparison: ProGuard vs MDP+STL (AII-sorted).}')
    lines.append('  \\label{tab:perlaw-comparison}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


def main():
    src = os.path.join(LOG_DIR, 'paper_perlaw_comparison.json')
    obj = _read_json(src) or {}
    rows = obj.get('rows', []) if isinstance(obj, dict) else []
    tex = build_latex_table(rows)
    out_path = os.path.join(LOG_DIR, 'paper_perlaw_table.tex')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(tex)
    print('written:', out_path)


if __name__ == '__main__':
    main()