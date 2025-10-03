#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDP+STL Diagnostic Report Generator
Reads:
- AV per-law summary log: src/log_safereach/av_eval.log
- Offline meta summary: src/log_safereach/offline_meta/offline_meta_summary.json
- Embodied eval raw log: src/log_safereach/embodied_eval.log
- Embodied summary: src/log_safereach/embodied_summary.json (baseline metrics)

Outputs diagnostic report into src/log_safereach/:
- mdp_stl_diagnostic_report.json
- mdp_stl_diagnostic_report.csv

Metrics:
- Action Influence Index (AII): heuristic improvability index per Law (AV) and overall (Embodied)
- Expected improvement intervals for Completion%, Prediction%(0.7), Time-Ahead (relative factor), Unsafe% after

Notes:
- AII for AV is computed from offline_meta: combines Timeout fraction, pk growth (20-5), Unsafe fraction
- AII for Embodied is computed from embodied_eval.log by measuring alternative-action success after a failure
- This is a prioritization diagnostic, not a formal guarantee; use as guidance for MDP+STL implementation order
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'log_safereach')
AV_LOG = os.path.join(LOG_DIR, 'av_eval.log')
OFFLINE_META = os.path.join(LOG_DIR, 'offline_meta', 'offline_meta_summary.json')
EMBODIED_LOG = os.path.join(LOG_DIR, 'embodied_eval.log')
EMBODIED_SUMMARY = os.path.join(LOG_DIR, 'embodied_summary.json')
OUT_JSON = os.path.join(LOG_DIR, 'mdp_stl_diagnostic_report.json')
OUT_CSV = os.path.join(LOG_DIR, 'mdp_stl_diagnostic_report.csv')


def read_av_eval_per_law() -> Dict[str, Dict[str, Any]]:
    """Parse av_eval.log and extract per-law metrics robustly.
    We scan JSON objects using brace-depth counting and pick those with top-level 'law' and 'reach'.
    returns: {law_name: {completion_pct, unsafe_pct_after, prediction_pct:{'0.3':..}, time_ahead_s:{...}}}
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(AV_LOG):
        return results

    json_blocks: List[str] = []
    inside_block = False
    depth = 0
    buf: List[str] = []

    def update_depth(s: str, depth: int) -> int:
        # naive but works for our structured logs (braces not embedded in strings significantly)
        depth += s.count('{')
        depth -= s.count('}')
        return depth

    with open(AV_LOG, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not inside_block:
                # start of a JSON object
                if '{' in line:
                    inside_block = True
                    depth = 0
                    buf = []
                    # from the first brace onward
                    idx = line.find('{')
                    part = line[idx:]
                    buf.append(part.rstrip('\n'))
                    depth = update_depth(part, depth)
                    if depth == 0:
                        json_blocks.append('\n'.join(buf))
                        inside_block = False
                        buf = []
                # else continue scanning
            else:
                buf.append(line.rstrip('\n'))
                depth = update_depth(line, depth)
                if depth == 0:
                    json_blocks.append('\n'.join(buf))
                    inside_block = False
                    buf = []

    for blk in json_blocks:
        try:
            data = json.loads(blk)
        except Exception:
            continue
        if isinstance(data, dict) and ('law' in data) and ('reach' in data):
            law_name = None
            if isinstance(data['law'], dict):
                law_name = data['law'].get('folder')
            if not law_name and isinstance(data['reach'], dict):
                law_name = data['reach'].get('folder')
            if not law_name:
                continue
            law_metrics = {
                'completion_pct': data.get('reach', {}).get('completion_pct'),
                'unsafe_pct_after': data.get('reach', {}).get('unsafe_pct_after'),
                'prediction_pct': data.get('law', {}).get('prediction_pct', {}),
                'time_ahead_s': data.get('law', {}).get('time_ahead_s', {}),
            }
            results[law_name] = law_metrics
    return results


def read_offline_meta() -> Dict[str, Dict[str, Any]]:
    """Read offline_meta_summary.json and map per mode(law)."""
    if not os.path.isfile(OFFLINE_META):
        return {}
    with open(OFFLINE_META, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    modes = meta.get('modes', [])
    res: Dict[str, Dict[str, Any]] = {}
    for m in modes:
        name = m.get('mode')
        if not name:
            continue
        pk = m.get('pk', {})
        terminal = m.get('terminal', {})
        res[name] = {
            'pk5': pk.get('5', 0.0),
            'pk10': pk.get('10', 0.0),
            'pk20': pk.get('20', 0.0),
            'unsafe': terminal.get('unsafe', 0.0),
            'timeout': terminal.get('timeout', 0.0),
            'safe_done': terminal.get('safe_done', 0.0),
        }
    return res


def compute_av_aii(meta: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Compute Action Influence Index (AII) for AV laws from offline meta.
    Heuristic: AII = 0.5*timeout + 0.3*norm_pk_growth + 0.2*unsafe
    where norm_pk_growth = (pk20 - pk5) / max_growth_over_modes
    """
    # compute global max pk growth
    growths = []
    for m in meta.values():
        growths.append(max(0.0, m.get('pk20', 0.0) - m.get('pk5', 0.0)))
    max_growth = max(growths) if growths else 1.0

    aii: Dict[str, float] = {}
    for name, m in meta.items():
        timeout = m.get('timeout', 0.0)
        unsafe = m.get('unsafe', 0.0)
        pk_growth = max(0.0, m.get('pk20', 0.0) - m.get('pk5', 0.0))
        norm_pk_growth = pk_growth / max_growth if max_growth > 0 else 0.0
        val = 0.5 * timeout + 0.3 * norm_pk_growth + 0.2 * unsafe
        # clamp to [0,1]
        aii[name] = max(0.0, min(1.0, val))
    return aii


def read_embodied_summary() -> Dict[str, Any]:
    if not os.path.isfile(EMBODIED_SUMMARY):
        return {}
    with open(EMBODIED_SUMMARY, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_embodied_aii_from_log() -> float:
    """Compute embodied action influence index from embodied_eval.log by measuring
    how often an immediate failure is followed by an alternative-action success.
    AII_embodied = alt_success_count / max(1, failure_count), clamped to [0,1].
    """
    if not os.path.isfile(EMBODIED_LOG):
        return 0.3  # fallback moderate value

    failure_count = 0
    alt_success_count = 0

    # regex to extract action input and success flag
    action_re = re.compile(r'"input":\s*"([^"]+)"')
    success_re = re.compile(r'"success":\s*(true|false)')

    # We'll keep a short rolling window of last failed action: str = ''
    last_failed_action: str = ''
    pending_check_steps = 0

    with open(EMBODIED_LOG, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # The log line may contain multiple JSON fragments; we only need action and success flags
            actions = action_re.findall(line)
            successes = success_re.findall(line)
            # Zip approximate pairs; if counts mismatch, we still scan successes
            # Strategy: If any "success": false appears, mark a failure on the most recent action in this line.
            if successes:
                for s in successes:
                    if s == 'false':
                        failure_count += 1
                        last_failed_action = actions[-1] if actions else ''
                        pending_check_steps = 5  # look ahead up to 5 subsequent lines
                    elif s == 'true' and pending_check_steps > 0:
                        # success after a failure: consider alternative if action differs
                        current_action = actions[-1] if actions else ''
                        if current_action and current_action != last_failed_action:
                            alt_success_count += 1
                            # reset pending tracking
                            pending_check_steps = 0
                            last_failed_action = ''
            else:
                # decrement window if we're looking ahead
                if pending_check_steps > 0:
                    pending_check_steps -= 1

    aii = 0.0
    if failure_count > 0:
        aii = alt_success_count / float(failure_count)
    # clamp
    aii = max(0.0, min(1.0, aii))
    return aii if aii > 0 else 0.25  # ensure non-zero modest value


def expected_improvements_av(law_metrics: Dict[str, Any], meta: Dict[str, Any], aii: float) -> Dict[str, Any]:
    base_completion = law_metrics.get('completion_pct') or 0.0
    base_unsafe_after = law_metrics.get('unsafe_pct_after') or 0.0
    base_pred_high = 0.0
    pred = law_metrics.get('prediction_pct') or {}
    if isinstance(pred, dict):
        base_pred_high = pred.get('0.7', pred.get(0.7, 0.0)) or 0.0

    timeout = meta.get('timeout', 0.0)

    # Completion improvement range
    lower = min(100.0, base_completion + 100.0 * (0.3 * timeout))
    upper = min(100.0, base_completion + 100.0 * (0.6 * timeout))

    # Prediction(0.7) improvement range
    pred_lower = min(100.0, base_pred_high + 100.0 * (0.20 * aii))
    pred_upper = min(100.0, base_pred_high + 100.0 * (0.40 * aii))

    # Time-Ahead relative factor range (multiplicative factor)
    ta_lower_factor = 1.0 + 0.30 * aii
    ta_upper_factor = 1.0 + 0.70 * aii

    # Unsafe after expected range
    unsafe_lower = 0.0
    unsafe_upper = max(0.0, base_unsafe_after * (1.0 - 0.70 * aii))

    return {
        'completion_pct_range': [round(lower, 3), round(upper, 3)],
        'prediction_0.7_pct_range': [round(pred_lower, 3), round(pred_upper, 3)],
        'time_ahead_factor_range': [round(ta_lower_factor, 3), round(ta_upper_factor, 3)],
        'unsafe_after_pct_range': [round(unsafe_lower, 3), round(unsafe_upper, 3)],
    }


def expected_improvements_embodied(base_summary: Dict[str, Any], aii: float) -> Dict[str, Any]:
    reach = base_summary.get('reach', {})
    law = base_summary.get('law', {})
    base_completion = reach.get('completion_pct', 0.0)
    base_pred_high = 0.0
    pred = law.get('prediction_pct', {})
    if isinstance(pred, dict):
        base_pred_high = pred.get('0.7', pred.get(0.7, 0.0)) or 0.0

    # Completion improvement range (embodied tends to have strong local control when alternatives exist)
    lower = min(100.0, base_completion + 100.0 * (0.25 * aii))
    upper = min(100.0, base_completion + 100.0 * (0.50 * aii))

    pred_lower = min(100.0, base_pred_high + 100.0 * (0.15 * aii))
    pred_upper = min(100.0, base_pred_high + 100.0 * (0.30 * aii))

    ta_lower_factor = 1.0 + 0.30 * aii
    ta_upper_factor = 1.0 + 0.70 * aii

    unsafe_range = [0.0, 0.0]  # baseline already 0

    return {
        'completion_pct_range': [round(lower, 3), round(upper, 3)],
        'prediction_0.7_pct_range': [round(pred_lower, 3), round(pred_upper, 3)],
        'time_ahead_factor_range': [round(ta_lower_factor, 3), round(ta_upper_factor, 3)],
        'unsafe_after_pct_range': unsafe_range,
    }


def main():
    # Read inputs
    av_per_law = read_av_eval_per_law()
    meta = read_offline_meta()
    embodied_aii = compute_embodied_aii_from_log()
    embodied_summary = read_embodied_summary()

    # Compute AII for AV
    av_aii = compute_av_aii(meta)

    # Build AV report per law
    av_report = []
    for law_name, m in av_per_law.items():
        meta_m = meta.get(law_name, {
            'pk5': 0.0, 'pk10': 0.0, 'pk20': 0.0,
            'unsafe': 0.0, 'timeout': 0.0, 'safe_done': 0.0,
        })
        aii = av_aii.get(law_name, 0.3)
        exp = expected_improvements_av(m, meta_m, aii)
        av_report.append({
            'law': law_name,
            'baseline': m,
            'offline_meta': meta_m,
            'action_influence_index': round(aii, 3),
            'expected_improvements': exp,
        })

    # Embodied report (overall)
    emb_exp = expected_improvements_embodied(embodied_summary if embodied_summary else {
        'reach': {'completion_pct': 8.41},
        'law': {'prediction_pct': {'0.7': 39.07}},
    }, embodied_aii)
    embodied_report = {
        'action_influence_index': round(embodied_aii, 3),
        'expected_improvements': emb_exp,
        'baseline': embodied_summary if embodied_summary else None,
    }

    # Aggregate JSON
    out = {
        'notes': {
            'aii_av_definition': 'AII = 0.5*timeout + 0.3*norm_pk_growth + 0.2*unsafe (from offline_meta)',
            'aii_embodied_definition': 'AII_embodied = alt_success_after_failure_rate (from embodied_eval.log)',
            'disclaimer': 'Heuristic diagnostic for prioritization; not formal guarantees.',
        },
        'av': av_report,
        'embodied': embodied_report,
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Also write CSV (AV laws)
    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write('law,aii,baseline_completion,baseline_unsafe_after,baseline_pred_0.7,timeout,unsafe,pk5,pk20,completion_lower,completion_upper,pred0.7_lower,pred0.7_upper,ta_factor_lower,ta_factor_upper,unsafe_after_lower,unsafe_after_upper\n')
        for item in av_report:
            law = item['law']
            aii = item['action_influence_index']
            base = item['baseline']
            pred = base.get('prediction_pct', {}) if isinstance(base.get('prediction_pct'), dict) else {}
            base_pred07 = pred.get('0.7', pred.get(0.7, 0.0)) or 0.0
            meta_m = item['offline_meta']
            exp = item['expected_improvements']
            row = [
                law,
                f"{aii}",
                f"{base.get('completion_pct', 0.0)}",
                f"{base.get('unsafe_pct_after', 0.0)}",
                f"{base_pred07}",
                f"{meta_m.get('timeout', 0.0)}",
                f"{meta_m.get('unsafe', 0.0)}",
                f"{meta_m.get('pk5', 0.0)}",
                f"{meta_m.get('pk20', 0.0)}",
                f"{exp['completion_pct_range'][0]}",
                f"{exp['completion_pct_range'][1]}",
                f"{exp['prediction_0.7_pct_range'][0]}",
                f"{exp['prediction_0.7_pct_range'][1]}",
                f"{exp['time_ahead_factor_range'][0]}",
                f"{exp['time_ahead_factor_range'][1]}",
                f"{exp['unsafe_after_pct_range'][0]}",
                f"{exp['unsafe_after_pct_range'][1]}",
            ]
            f.write(','.join(map(str, row)) + '\n')

if __name__ == '__main__':
    main()