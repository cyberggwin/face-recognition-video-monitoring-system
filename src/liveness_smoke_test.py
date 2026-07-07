"""Smoke test for EAR extraction from a single image.

Usage:
    .\venv311\Scripts\python.exe .\src\liveness_smoke_test.py --image .\dataset\some_person\img.jpg
"""

import argparse
from pathlib import Path

import cv2
import face_recognition

from liveness import mean_ear


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test: compute EAR from an image")
    p.add_argument("--image", required=True, help="Path to an image with a visible face")
    p.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Not found: {img_path}")
        return 2

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Failed to read image: {img_path}")
        return 2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model=args.detector)
    if not locs:
        print("No face detected")
        return 1

    lms = face_recognition.face_landmarks(rgb, [locs[0]])
    if not lms:
        print("No landmarks produced")
        return 1

    ear = mean_ear(lms[0])
    if ear is None:
        print("EAR could not be computed (eyes not found)")
        return 1

    print(f"EAR: {ear:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
