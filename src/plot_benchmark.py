"""
Plot benchmark results from results_benchmark.csv.
Generates charts (PNG) for:
- Unknown vs False-known counts
- Persons detected vs Faces found
- Histogram of face distances

Usage:
  python src/plot_benchmark.py --csv results_benchmark.csv --outdir reports
"""
import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        return list(r)


def to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description='Plot charts from benchmark CSV')
    p.add_argument('--csv', default='results_benchmark.csv', help='Path to CSV produced by evaluate_benchmark.py')
    p.add_argument('--outdir', default='reports', help='Folder to save charts')
    args = p.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path)
    if not rows:
        print('No data in CSV'); return

    persons = sum(int(r['persons_detected']) for r in rows if r.get('persons_detected'))
    faces = sum(int(r['faces_found']) for r in rows if r.get('faces_found'))
    unknown = sum(int(r['unknown_count']) for r in rows if r.get('unknown_count'))
    false_known = sum(int(r['false_known']) for r in rows if r.get('false_known'))
    dists = [to_float(r['avg_dist']) for r in rows if r.get('avg_dist')]
    dists = [d for d in dists if d is not None]

    # 1) Unknown vs False-known
    plt.figure(figsize=(6,4))
    plt.bar(['Unknown', 'False-known'], [unknown, false_known], color=['tab:blue','tab:red'])
    plt.title('Unknown vs False-known (total)')
    for i, v in enumerate([unknown, false_known]):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    p1 = outdir / 'chart_unknown_falseknown.png'
    plt.savefig(p1)
    plt.close()

    # 2) Persons vs Faces
    plt.figure(figsize=(6,4))
    plt.bar(['Persons', 'Faces'], [persons, faces], color=['tab:green','tab:purple'])
    plt.title('Persons detected vs Faces found (total)')
    for i, v in enumerate([persons, faces]):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    p2 = outdir / 'chart_persons_faces.png'
    plt.savefig(p2)
    plt.close()

    # 3) Histogram distances
    if dists:
        plt.figure(figsize=(6,4))
        plt.hist(dists, bins=30, color='tab:orange', edgecolor='black')
        plt.title('Histogram of average face distances')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.tight_layout()
        p3 = outdir / 'chart_dist_hist.png'
        plt.savefig(p3)
        plt.close()
    else:
        p3 = None

    print('Saved charts:')
    print(p1)
    print(p2)
    if p3:
        print(p3)


if __name__ == '__main__':
    main()
