"""Run a threshold sweep for open-set face recognition metrics.

Example:
  python src/threshold_sweep.py --mixed-dir mixed_faces --encodings models/known_faces.pkl
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np

from mixed_benchmark import is_image, largest_box, load_known


DEFAULT_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70]
DEFAULT_ALIASES = {
    "pozeeu": "Eu",
    "pozemama": "Mama",
}


def parse_aliases(values: Iterable[str] | None) -> Dict[str, str]:
    aliases = dict(DEFAULT_ALIASES)
    if not values:
        return aliases

    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid alias '{item}'. Use folder_label=known_label.")
        source, target = item.split("=", 1)
        aliases[source.strip()] = target.strip()
    return aliases


def normalize_label(label: str, aliases: Dict[str, str]) -> str:
    if label.casefold() == "necunoscut":
        return "necunoscut"

    for source, target in aliases.items():
        if label.casefold() == source.casefold():
            return target
    return label


def best_match(encodings: list, names: list, face_enc: np.ndarray) -> tuple[str, float]:
    if len(encodings) == 0:
        return "necunoscut", float("inf")

    distances = face_recognition.face_distance(encodings, face_enc)
    best_index = int(np.argmin(distances))
    return str(names[best_index]), float(distances[best_index])


def collect_best_matches(
    mixed_dir: Path,
    encodings_path: Path,
    detector: str,
    aliases: Dict[str, str],
) -> List[dict]:
    encodings, names = load_known(encodings_path)
    images = [p for p in sorted(mixed_dir.rglob("*")) if is_image(p)]
    rows: List[dict] = []

    for img_path in images:
        raw_label = img_path.parent.name
        label = normalize_label(raw_label, aliases)
        best_name = "necunoscut"
        best_dist = float("inf")

        img = cv2.imread(str(img_path))
        if img is not None:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model=detector)
            box = largest_box(boxes)
            if box is not None:
                face_encs = face_recognition.face_encodings(rgb, [box])
                if face_encs:
                    best_name, best_dist = best_match(encodings, names, face_encs[0])

        rows.append(
            {
                "file": str(img_path),
                "raw_label": raw_label,
                "label": label,
                "best_name": best_name,
                "best_dist": best_dist,
            }
        )

    return rows


def predict_for_threshold(row: dict, threshold: float) -> str:
    if float(row["best_dist"]) < threshold:
        return str(row["best_name"])
    return "necunoscut"


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(rows: List[dict], threshold: float) -> dict:
    tp = fp = fn = 0

    for row in rows:
        label = str(row["label"])
        pred = predict_for_threshold(row, threshold)

        if label != "necunoscut" and pred == label:
            tp += 1
        elif pred != "necunoscut" and pred != label:
            fp += 1
        elif label != "necunoscut" and pred == "necunoscut":
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def write_metrics_csv(metrics: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def plot_metrics(metrics: List[dict], out_plot: Path, selected_threshold: float) -> None:
    thresholds = [m["threshold"] for m in metrics]
    precision = [m["precision"] * 100 for m in metrics]
    recall = [m["recall"] * 100 for m in metrics]
    f1_score = [m["f1_score"] * 100 for m in metrics]

    selected_f1 = next(
        (m["f1_score"] * 100 for m in metrics if abs(m["threshold"] - selected_threshold) < 1e-9),
        max(f1_score) if f1_score else 0,
    )
    best_metric = max(metrics, key=lambda item: item["f1_score"])
    selected_is_best = abs(best_metric["threshold"] - selected_threshold) < 1e-9
    selected_label = (
        f"Prag Optim ({selected_threshold:.2f})"
        if selected_is_best
        else f"Prag ales ({selected_threshold:.2f})\nF1={selected_f1:.1f}%"
    )

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, marker="o", linewidth=2.2, color="#1f77b4", label="Precision")
    plt.plot(thresholds, recall, marker="s", linewidth=2.2, color="#2ca02c", label="Recall")
    plt.plot(thresholds, f1_score, marker="^", linewidth=2.4, color="#d62728", label="F1 Score")

    plt.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.6)
    plt.annotate(
        selected_label,
        xy=(selected_threshold, selected_f1),
        xytext=(selected_threshold + 0.012, min(100, selected_f1 + 8)),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )

    plt.title("Threshold Sweep pentru recunoaștere facială open-set", fontsize=13, fontweight="bold")
    plt.xlabel("Prag distanță facială")
    plt.ylabel("Valoare metrică (%)")
    plt.xticks(thresholds, [f"{thr:.2f}" for thr in thresholds])
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate precision/recall/F1 over multiple distance thresholds.")
    parser.add_argument("--mixed-dir", default="mixed_faces", help="Folder with mixed known/unknown images")
    parser.add_argument("--encodings", default=str(Path("models") / "known_faces.pkl"), help="Path to known_faces.pkl")
    parser.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--selected-threshold", type=float, default=0.58)
    parser.add_argument("--out-csv", default=str(Path("reports") / "threshold_sweep_metrics.csv"))
    parser.add_argument("--out-plot", default=str(Path("reports") / "threshold_sweep_metrics.pdf"))
    parser.add_argument("--alias", nargs="*", default=None, help="Optional label aliases, e.g. pozeeu=Eu pozemama=Mama")
    args = parser.parse_args()

    aliases = parse_aliases(args.alias)
    rows = collect_best_matches(
        mixed_dir=Path(args.mixed_dir),
        encodings_path=Path(args.encodings),
        detector=args.detector,
        aliases=aliases,
    )

    if not rows:
        raise SystemExit(f"No images found in {args.mixed_dir}")

    metrics = [compute_metrics(rows, threshold) for threshold in args.thresholds]
    write_metrics_csv(metrics, Path(args.out_csv))
    plot_metrics(metrics, Path(args.out_plot), selected_threshold=args.selected_threshold)

    best = max(metrics, key=lambda item: item["f1_score"])
    print("--- Threshold sweep summary ---")
    for item in metrics:
        print(
            f"thr={item['threshold']:.2f} "
            f"TP={item['tp']} FP={item['fp']} FN={item['fn']} "
            f"P={item['precision']:.3f} R={item['recall']:.3f} F1={item['f1_score']:.3f}"
        )
    print(f"Best F1 threshold: {best['threshold']:.2f} (F1={best['f1_score']:.3f})")
    print(f"CSV saved to: {args.out_csv}")
    print(f"Plot saved to: {args.out_plot}")


if __name__ == "__main__":
    main()
