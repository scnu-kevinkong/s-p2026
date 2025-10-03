#!/usr/bin/env python3
"""
Generate simple SVG charts from mdp_stl_diagnostic_report.csv:
- mdp_aii_ranking.svg: AII per law sorted descending
- mdp_pred07_improvement.svg: Expected Prediction(0.7) improvement range per law
- mdp_timeahead_factor_improvement.svg: Expected Time-Ahead factor range per law
- mdp_completion_improvement.svg: Expected Completion% improvement range per law

No external dependencies; renders basic horizontal bar/interval charts.
"""
import os
import csv
from typing import List, Dict, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'log_safereach', 'mdp_stl_diagnostic_report.csv')
OUT_DIR = os.path.join(ROOT, 'log_safereach', 'charts')

# SVG helpers
class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items: List[str] = []

    def add(self, s: str):
        self.items.append(s)

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = 'none'):
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" />')

    def text(self, x: float, y: float, txt: str, size: int = 12, anchor: str = 'start', weight: str = 'normal', fill: str = '#333'):
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{escape(txt)}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = '#555', width: float = 1.0):
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width:.2f}" />')

    def save(self, path: str, title: str = ''):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">'  # noqa
        title_tag = f'<title>{escape(title)}</title>' if title else ''
        content = '\n'.join([header, title_tag] + self.items + ['</svg>'])
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


def escape(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def read_csv() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.isfile(CSV_PATH):
        return rows
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parse_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def chart_aii_ranking(rows: List[Dict[str, str]]):
    # Sort by AII descending
    data = [(r['law'], parse_float(r['aii'])) for r in rows]
    data.sort(key=lambda x: x[1], reverse=True)

    width = 1200
    bar_h = 26
    pad = 80
    inner_w = width - pad * 2
    height = pad + len(data) * (bar_h + 8) + pad
    svg = Svg(width, height)

    # Title and axis labels
    svg.text(width / 2, 30, 'MDP+STL Prioritization: Action Influence Index (AII) by Law', size=18, anchor='middle', weight='bold')
    svg.text(pad, 55, 'AII (0-1)', size=12, anchor='start')

    # Axis baseline
    base_y = 70
    svg.line(pad, base_y, width - pad, base_y, stroke='#888', width=1)
    # axis ticks
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = pad + inner_w * t
        svg.line(x, base_y - 4, x, base_y + 4, stroke='#888', width=1)
        svg.text(x, base_y - 10, f'{t:.2f}', size=12, anchor='middle')

    y = base_y + 20
    for law, aii in data:
        bar_w = inner_w * max(0.0, min(aii, 1.0))
        svg.text(pad - 10, y + bar_h * 0.7, law, size=12, anchor='end')
        svg.rect(pad, y, bar_w, bar_h, fill='#4E79A7')
        svg.text(pad + bar_w + 6, y + bar_h * 0.7, f'{aii:.3f}', size=12, anchor='start')
        y += bar_h + 8

    out = os.path.join(OUT_DIR, 'mdp_aii_ranking.svg')
    svg.save(out, 'AII Ranking')


def chart_interval(rows: List[Dict[str, str]], lower_key: str, upper_key: str, title: str, unit: str, filename: str, scale: Tuple[float, float]):
    # Prepare data and scale
    vals = []
    for r in rows:
        law = r['law']
        lo = parse_float(r[lower_key])
        hi = parse_float(r[upper_key])
        vals.append((law, lo, hi))
    # sort by upper desc
    vals.sort(key=lambda x: x[2], reverse=True)

    min_v, max_v = scale
    width = 1200
    bar_h = 20
    pad = 100
    inner_w = width - pad * 2
    height = pad + len(vals) * (bar_h + 12) + pad
    svg = Svg(width, height)

    svg.text(width / 2, 30, title, size=18, anchor='middle', weight='bold')
    svg.text(pad, 55, unit, size=12, anchor='start')

    base_y = 70
    svg.line(pad, base_y, width - pad, base_y, stroke='#888', width=1)
    # ticks
    ticks = 6
    for i in range(ticks + 1):
        t = min_v + (max_v - min_v) * i / ticks
        x = pad + inner_w * (t - min_v) / (max_v - min_v if max_v > min_v else 1.0)
        svg.line(x, base_y - 4, x, base_y + 4, stroke='#888', width=1)
        svg.text(x, base_y - 10, f'{t:.2f}', size=12, anchor='middle')

    y = base_y + 24
    for law, lo, hi in vals:
        x_lo = pad + inner_w * (lo - min_v) / (max_v - min_v if max_v > min_v else 1.0)
        x_hi = pad + inner_w * (hi - min_v) / (max_v - min_v if max_v > min_v else 1.0)
        # law label
        svg.text(pad - 10, y + bar_h * 0.7, law, size=12, anchor='end')
        # interval line
        svg.line(x_lo, y + bar_h * 0.5, x_hi, y + bar_h * 0.5, stroke='#59A14F', width=4)
        # end caps
        svg.line(x_lo, y + bar_h * 0.3, x_lo, y + bar_h * 0.7, stroke='#59A14F', width=2)
        svg.line(x_hi, y + bar_h * 0.3, x_hi, y + bar_h * 0.7, stroke='#59A14F', width=2)
        # value text
        mid = (lo + hi) / 2.0
        x_mid = pad + inner_w * (mid - min_v) / (max_v - min_v if max_v > min_v else 1.0)
        svg.text(x_mid, y + bar_h * 0.3, f'{mid:.2f}', size=12, anchor='middle', fill='#2F4B7C')
        y += bar_h + 12

    out = os.path.join(OUT_DIR, filename)
    svg.save(out, title)


def main():
    rows = read_csv()
    if not rows:
        print('No CSV rows found at', CSV_PATH)
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    chart_aii_ranking(rows)
    chart_interval(rows, 'pred0.7_lower', 'pred0.7_upper',
                   'Expected Improvement: Prediction(0.7) % range by Law', '%', 'mdp_pred07_improvement.svg', (0.0, max(30.0, max(parse_float(r['pred0.7_upper']) for r in rows))))
    chart_interval(rows, 'ta_factor_lower', 'ta_factor_upper',
                   'Expected Improvement: Time-Ahead factor range by Law', 'factor', 'mdp_timeahead_factor_improvement.svg', (1.0, max(1.3, max(parse_float(r['ta_factor_upper']) for r in rows))))
    chart_interval(rows, 'completion_lower', 'completion_upper',
                   'Expected Improvement: Completion % range by Law', '%', 'mdp_completion_improvement.svg', (0.0, 100.0))


if __name__ == '__main__':
    main()