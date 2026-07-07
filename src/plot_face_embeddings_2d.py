"""Visualize known face embeddings in 2D using PCA and LDA for a selected subset.

Example:
  python src/plot_face_embeddings_2d.py --encodings models/known_faces.pkl --out reports/embedding_pca_lda_5_persoane.pdf
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TARGETS = ["eu", "mama", "tata", "vali", "AMi"]


def _is_embedding_matrix(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 1


def _is_label_vector(value: object, expected_len: int) -> bool:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        return False
    return all(isinstance(item, (str, int, np.integer)) for item in value)


def find_embeddings_and_labels(data: object) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Find the embedding matrix and its associated labels from common pickle layouts."""
    if not isinstance(data, dict):
        raise ValueError("Expected a dictionary in the encodings pickle file.")

    matrix_candidates: list[tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if _is_embedding_matrix(value):
            matrix_candidates.append((str(key), np.asarray(value, dtype=float)))

    if not matrix_candidates:
        raise ValueError("No 2D numeric embedding matrix was found in the pickle file.")

    preferred_matrix_keys = ("encodings", "embeddings", "known_encodings", "face_encodings")
    matrix_key, embeddings = sorted(
        matrix_candidates,
        key=lambda item: (
            preferred_matrix_keys.index(item[0]) if item[0] in preferred_matrix_keys else len(preferred_matrix_keys),
            -item[1].shape[0],
        ),
    )[0]

    label_candidates: list[tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if _is_label_vector(value, expected_len=len(embeddings)):
            label_candidates.append((str(key), np.asarray(value, dtype=str)))

    if not label_candidates:
        raise ValueError("No label vector with the same length as the embeddings was found.")

    preferred_label_keys = ("names", "labels", "known_names", "ids", "targets")
    label_key, labels = sorted(
        label_candidates,
        key=lambda item: preferred_label_keys.index(item[0]) if item[0] in preferred_label_keys else len(preferred_label_keys),
    )[0]

    return embeddings, labels, matrix_key, label_key


def resolve_target_labels(labels: Sequence[str], requested: Iterable[str]) -> list[str]:
    """Map requested names to their exact spelling in the data, preserving request order."""
    exact_by_casefold: dict[str, str] = {}
    for label in labels:
        exact_by_casefold.setdefault(str(label).casefold(), str(label))

    resolved: list[str] = []
    missing: list[str] = []
    for name in requested:
        exact = exact_by_casefold.get(str(name).casefold())
        if exact is None:
            missing.append(str(name))
        elif exact not in resolved:
            resolved.append(exact)

    if missing:
        available = ", ".join(sorted(set(str(label) for label in labels)))
        raise ValueError(f"Missing requested labels: {missing}. Available labels: {available}")
    return resolved


def filter_subset(embeddings: np.ndarray, labels: np.ndarray, target_labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return copies containing only the requested people."""
    mask = np.isin(labels, np.asarray(target_labels, dtype=str))
    subset_embeddings = embeddings[mask].copy()
    subset_labels = labels[mask].copy()
    if subset_embeddings.size == 0:
        raise ValueError("The selected labels produced an empty subset.")
    return subset_embeddings, subset_labels


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def lda_2d(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 3:
        raise ValueError("LDA needs at least 3 classes to produce a 2D projection.")

    overall_mean = embeddings.mean(axis=0)
    n_features = embeddings.shape[1]
    sw = np.zeros((n_features, n_features), dtype=float)
    sb = np.zeros((n_features, n_features), dtype=float)

    for label in unique_labels:
        class_rows = embeddings[labels == label]
        class_mean = class_rows.mean(axis=0)
        centered_class = class_rows - class_mean
        sw += centered_class.T @ centered_class

        mean_diff = (class_mean - overall_mean).reshape(-1, 1)
        sb += class_rows.shape[0] * (mean_diff @ mean_diff.T)

    eig_matrix = np.linalg.pinv(sw) @ sb
    eig_values, eig_vectors = np.linalg.eig(eig_matrix)
    order = np.argsort(np.real(eig_values))[::-1]
    components = np.real(eig_vectors[:, order[:2]])
    return embeddings @ components


def plot_projection(
    ax: plt.Axes,
    projection: np.ndarray,
    labels: np.ndarray,
    target_labels: Sequence[str],
    title: str,
) -> None:
    colors = plt.get_cmap("tab10")
    for idx, label in enumerate(target_labels):
        rows = labels == label
        ax.scatter(
            projection[rows, 0],
            projection[rows, 1],
            s=45,
            alpha=0.85,
            color=colors(idx),
            edgecolors="white",
            linewidths=0.5,
            label=label,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Componenta 1")
    ax.set_ylabel("Componenta 2")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(title="Persoane", loc="best", frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PCA and LDA projections for selected face embeddings.")
    parser.add_argument("--encodings", default=str(Path("models") / "known_faces.pkl"), help="Path to known_faces.pkl")
    parser.add_argument("--out", default=str(Path("reports") / "embedding_pca_lda_5_persoane.pdf"), help="Output PDF path")
    parser.add_argument("--people", nargs="+", default=DEFAULT_TARGETS, help="People to keep before PCA/LDA")
    args = parser.parse_args()

    with Path(args.encodings).open("rb") as f:
        data = pickle.load(f)

    embeddings, labels, embeddings_key, labels_key = find_embeddings_and_labels(data)
    target_labels = resolve_target_labels(labels, args.people)
    subset_embeddings, subset_labels = filter_subset(embeddings, labels, target_labels)

    pca_projection = pca_2d(subset_embeddings)
    lda_projection = lda_2d(subset_embeddings, subset_labels)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_projection(axes[0], pca_projection, subset_labels, target_labels, "PCA - Nesupervizat (5 persoane)")
    plot_projection(axes[1], lda_projection, subset_labels, target_labels, "LDA - Supervizat (5 persoane)")
    fig.suptitle("Vizualizarea embedding-urilor faciale pentru subsetul selectat", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    counts = {label: int(np.sum(subset_labels == label)) for label in target_labels}
    print(f"Detected embedding variable: {embeddings_key}")
    print(f"Detected label variable: {labels_key}")
    print(f"Selected labels: {target_labels}")
    print(f"Subset counts: {counts}")
    print(f"PDF saved to: {out_path}")


if __name__ == "__main__":
    main()
