"""Build a mixed set and evaluate recognition accuracy."""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pickle
import face_recognition
import matplotlib.pyplot as plt


IMG_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
DEFAULT_ALIASES = {
    "pozeeu": "Eu",
    "pozemama": "Mama",
}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower().lstrip(".") in IMG_EXTS


def collect_known_images(dataset_dir: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = person_dir.name
        for img in sorted(person_dir.rglob("*")):
            if is_image(img):
                items.append((label, img))
    return items


def collect_unknown_images(unknown_dir: Path) -> List[Path]:
    return [p for p in sorted(unknown_dir.rglob("*")) if is_image(p)]


def safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    stem = dst.stem
    suffix = dst.suffix
    for i in range(1, 10000):
        cand = dst.with_name(f"{stem}_{i:04d}{suffix}")
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
    raise RuntimeError(f"Cannot create unique filename for {dst}")


def load_known(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])


def identify(encodings, names, face_enc, thr=0.50) -> Tuple[str, float]:
    if len(encodings) == 0:
        return "necunoscut", 1.0
    d = face_recognition.face_distance(encodings, face_enc)
    i = int(np.argmin(d))
    best_dist = float(d[i])
    return (names[i], best_dist) if best_dist < thr else ("necunoscut", best_dist)


def largest_box(boxes):
    if not boxes:
        return None

    def area(b):
        top, right, bottom, left = b
        return max(0, bottom - top) * max(0, right - left)

    return max(boxes, key=area)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def parse_aliases(values: List[str] | None) -> Dict[str, str]:
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


def plot_metrics_bar(metrics: Dict[str, float], out_plot: Path) -> None:
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Precision", "Recall", "F1 Score"]
    values = [
        metrics["precision"] * 100,
        metrics["recall"] * 100,
        metrics["f1_score"] * 100,
    ]
    colors = ["#2c7fb8", "#41ab5d", "#d95f02"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.62)
    plt.title("Performanța Sistemului (Metrici de Evaluare)", fontsize=13, fontweight="bold")
    plt.ylabel("Valoare metrică (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2, 99),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()


def build_mixed_set(
    known_items: List[Tuple[str, Path]],
    unknown_items: List[Path],
    out_dir: Path,
    total: int,
    known_ratio: float,
    seed: int | None,
) -> Dict[str, int]:
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    rng.shuffle(known_items)
    rng.shuffle(unknown_items)

    total = max(1, int(total))
    known_count = int(round(total * known_ratio))
    known_count = max(0, min(known_count, len(known_items)))
    unknown_count = max(0, min(total - known_count, len(unknown_items)))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (label, img) in enumerate(known_items[:known_count]):
        dst = out_dir / label / img.name
        safe_copy(img, dst)

    for idx, img in enumerate(unknown_items[:unknown_count]):
        dst = out_dir / "necunoscut" / img.name
        safe_copy(img, dst)

    return {"known": known_count, "unknown": unknown_count}


def benchmark_mixed(
    mixed_dir: Path,
    encodings_path: Path,
    out_csv: Path,
    out_plot: Path,
    out_metrics_plot: Path,
    detector: str,
    thr: float,
    aliases: Dict[str, str] | None = None,
) -> Dict[str, float]:
    encs, names = load_known(encodings_path)
    aliases = aliases or dict(DEFAULT_ALIASES)

    rows = []
    totals = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "known_total": 0,
        "known_correct": 0,
        "unknown_total": 0,
        "unknown_correct": 0,
    }

    images = [p for p in mixed_dir.rglob("*") if is_image(p)]

    for img_path in images:
        raw_label = img_path.parent.name
        label = normalize_label(raw_label, aliases)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=detector)
        box = largest_box(boxes)
        pred = "no_face"
        dist = ""

        if box is not None:
            encs_face = face_recognition.face_encodings(rgb, [box])
            if encs_face:
                pred, dist = identify(encs, names, encs_face[0], thr=thr)

        is_unknown = label.lower() == "necunoscut"
        is_correct = (pred == "necunoscut") if is_unknown else (pred == label)

        totals["total"] += 1
        if is_correct:
            totals["correct"] += 1
        else:
            totals["incorrect"] += 1

        if is_unknown:
            totals["unknown_total"] += 1
            if is_correct:
                totals["unknown_correct"] += 1
        else:
            totals["known_total"] += 1
            if is_correct:
                totals["known_correct"] += 1

        rows.append(
            {
                "file": str(img_path),
                "label": label,
                "raw_label": raw_label,
                "pred": pred,
                "dist": dist,
                "ok": int(is_correct),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "raw_label", "pred", "dist", "ok"])
        w.writeheader()
        w.writerows(rows)

    # Plot summary
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    known_wrong = totals["known_total"] - totals["known_correct"]
    unknown_wrong = totals["unknown_total"] - totals["unknown_correct"]

    labels = ["Known OK", "Known Wrong", "Unknown OK", "Unknown Wrong"]
    values = [totals["known_correct"], known_wrong, totals["unknown_correct"], unknown_wrong]
    colors = ["#2c7fb8", "#f03b20", "#41ab5d", "#fdae6b"]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color=colors)
    plt.title("Mixed Benchmark Results")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    plt.close()

    # Scientific metrics for open-set evaluation:
    # TP = Known OK, FP = Unknown Wrong, FN = Known Wrong.
    tp = totals["known_correct"]
    fp = unknown_wrong
    fn = known_wrong
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)

    metrics = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
    plot_metrics_bar(metrics, out_metrics_plot)

    acc = (totals["correct"] / totals["total"]) if totals["total"] else 0.0
    known_acc = (totals["known_correct"] / totals["known_total"]) if totals["known_total"] else 0.0
    unknown_acc = (totals["unknown_correct"] / totals["unknown_total"]) if totals["unknown_total"] else 0.0

    return {
        "total": totals["total"],
        "correct": totals["correct"],
        "incorrect": totals["incorrect"],
        "acc": acc,
        "known_acc": known_acc,
        "unknown_acc": unknown_acc,
        "known_total": totals["known_total"],
        "unknown_total": totals["unknown_total"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create mixed set and benchmark recognition")
    p.add_argument("--known-dataset", default="dataset", help="Folder with known people (subfolders per person)")
    p.add_argument("--unknown-dataset", default=str(Path("faces") / "Real Images"), help="Folder with unknown faces")
    p.add_argument("--out-dir", default="mixed_faces", help="Output folder for mixed set")
    p.add_argument("--total", type=int, default=200, help="Total images in mixed set")
    p.add_argument("--known-ratio", type=float, default=0.5, help="Known ratio in mixed set")
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling (omit for random each run)")
    p.add_argument("--encodings", default=str(Path("models") / "known_faces.pkl"))
    p.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    p.add_argument("--thr", type=float, default=0.58, help="Distance threshold for recognition")
    p.add_argument("--out-csv", default=str(Path("reports") / "mixed_benchmark.csv"))
    p.add_argument("--out-plot", default=str(Path("reports") / "mixed_benchmark.png"))
    p.add_argument("--out-metrics-plot", default=str(Path("reports") / "metrics_benchmark.pdf"))
    p.add_argument("--alias", nargs="*", default=None, help="Optional label aliases, e.g. pozeeu=Eu pozemama=Mama")
    args = p.parse_args()

    known_dir = Path(args.known_dataset)
    unknown_dir = Path(args.unknown_dataset)
    out_dir = Path(args.out_dir)

    known_items = collect_known_images(known_dir)
    unknown_items = collect_unknown_images(unknown_dir)

    if len(known_items) == 0:
        raise SystemExit(f"No known images found in {known_dir}")
    if len(unknown_items) == 0:
        raise SystemExit(f"No unknown images found in {unknown_dir}")

    counts = build_mixed_set(
        known_items=known_items,
        unknown_items=unknown_items,
        out_dir=out_dir,
        total=args.total,
        known_ratio=args.known_ratio,
        seed=args.seed,
    )

    print(f"Mixed set created in {out_dir} (known={counts['known']}, unknown={counts['unknown']})")
    aliases = parse_aliases(args.alias)

    summary = benchmark_mixed(
        mixed_dir=out_dir,
        encodings_path=Path(args.encodings),
        out_csv=Path(args.out_csv),
        out_plot=Path(args.out_plot),
        out_metrics_plot=Path(args.out_metrics_plot),
        detector=args.detector,
        thr=args.thr,
        aliases=aliases,
    )

    print("--- Mixed benchmark summary ---")
    print(f"Total: {summary['total']}")
    print(f"Correct: {summary['correct']}")
    print(f"Incorrect: {summary['incorrect']}")
    print(f"Accuracy: {summary['acc']:.3f}")
    print(f"Known accuracy: {summary['known_acc']:.3f} (n={summary['known_total']})")
    print(f"Unknown accuracy: {summary['unknown_acc']:.3f} (n={summary['unknown_total']})")
    print(f"TP={summary['tp']} FP={summary['fp']} FN={summary['fn']}")
    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall: {summary['recall']:.3f}")
    print(f"F1 Score: {summary['f1_score']:.3f}")
    print(f"CSV: {args.out_csv}")
    print(f"Plot: {args.out_plot}")
    print(f"Metrics plot: {args.out_metrics_plot}")


if __name__ == "__main__":
    main()
