import os
import json
import csv
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')


def _read_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _summarize_section_av(av):
    # Extract commonly referenced metrics for AV
    if not isinstance(av, dict):
        return {}
    reach = av.get('reach', {})
    law = av.get('law', {})
    pred07 = None
    ta07 = None
    try:
        pred07 = _safe_float((law.get('prediction_pct', {}) or {}).get('0.7'))
    except Exception:
        pred07 = None
    try:
        ta07 = _safe_float((law.get('time_ahead_s', {}) or {}).get('0.7'))
    except Exception:
        ta07 = None
    return {
        'unsafe_pct_after': _safe_float(reach.get('unsafe_pct_after')),
        'completion_pct': _safe_float(reach.get('completion_pct')),
        'prediction_pct@0.7': pred07,
        'time_ahead_s@0.7': ta07,
    }


def _summarize_section_emb(emb):
    # Extract commonly referenced metrics for Embodied
    if not isinstance(emb, dict):
        return {}
    reach = emb.get('reach', {})
    law = emb.get('law', {})
    pred07 = None
    ta07 = None
    try:
        pred07 = _safe_float((law.get('prediction_pct', {}) or {}).get('0.7'))
    except Exception:
        pred07 = None
    try:
        ta07 = _safe_float((law.get('time_ahead_s', {}) or {}).get('0.7'))
    except Exception:
        ta07 = None
    return {
        'unsafe_pct_after': _safe_float(reach.get('unsafe_pct_after')),
        'completion_pct': _safe_float(reach.get('completion_pct')),
        'prediction_pct@0.7': pred07,
        'time_ahead_s@0.7': ta07,
    }


def build_paper_metrics_summary():
    av_summary = _read_json(os.path.join(LOG_DIR, 'av_summary.json'))
    embodied_summary = _read_json(os.path.join(LOG_DIR, 'embodied_summary.json'))
    av_stl_overall = _read_json(os.path.join(LOG_DIR, 'offline_stl', 'overall_stl_summary.json'))
    emb_stl_overall = _read_json(os.path.join(LOG_DIR, 'offline_stl_embodied', 'overall_stl_summary.json'))
    mdp_diag_json = _read_json(os.path.join(LOG_DIR, 'mdp_stl_diagnostic_report.json'))

    out = {
        'notes': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'sources': {
                'av_summary.json': bool(av_summary),
                'embodied_summary.json': bool(embodied_summary),
                'offline_stl/overall_stl_summary.json': bool(av_stl_overall),
                'offline_stl_embodied/overall_stl_summary.json': bool(emb_stl_overall),
                'mdp_stl_diagnostic_report.json': bool(mdp_diag_json),
            }
        },
        'av': av_summary or {},
        'embodied': embodied_summary or {},
        'av_mdp_stl_batch': av_stl_overall or {},
        'embodied_mdp_stl_batch': emb_stl_overall or {},
        'mdp_stl_diagnostic': mdp_diag_json or {},
        'high_level': {
            'av': _summarize_section_av(av_summary or {}),
            'embodied': _summarize_section_emb(embodied_summary or {}),
        }
    }

    _write_json(os.path.join(LOG_DIR, 'paper_metrics_summary.json'), out)
    return out


def write_flat_csv(summary_obj):
    # Produce a compact CSV with key metrics for AV and Embodied
    rows = []
    hl_av = summary_obj.get('high_level', {}).get('av', {})
    hl_emb = summary_obj.get('high_level', {}).get('embodied', {})
    for k, v in hl_av.items():
        rows.append({'section': 'AV', 'metric': k, 'value': v if v is not None else ''})
    for k, v in hl_emb.items():
        rows.append({'section': 'Embodied', 'metric': k, 'value': v if v is not None else ''})

    out_csv = os.path.join(LOG_DIR, 'paper_full_comparison.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['section', 'metric', 'value'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    obj = build_paper_metrics_summary()
    write_flat_csv(obj)


if __name__ == '__main__':
    main()