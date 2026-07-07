"""Compare liveness v1 vs v2 on a video/webcam source and export CSV summary.

Usage examples:
  .\\venv311\\Scripts\\python.exe .\\src\\compare_liveness_algorithms.py --source 0
  .\\venv311\\Scripts\\python.exe .\\src\\compare_liveness_algorithms.py --source .\\sample.mp4 --gt live --out reports\\liveness_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import face_recognition

from liveness import AdaptiveLiveness, BlinkLiveness, SimpleFaceTracker, mean_ear


def largest_face(locs):
    if not locs:
        return None

    def area(b):
        t, r, btm, l = b
        return max(0, btm - t) * max(0, r - l)

    return max(locs, key=area)


def main():
    p = argparse.ArgumentParser(description="Compare liveness v1 vs v2")
    p.add_argument("--source", default="0", help="0 webcam or video file")
    p.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    p.add_argument("--gt", choices=["live", "spoof"], default=None, help="Optional ground-truth for simple accuracy")
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    p.add_argument("--out", default=str(Path("reports") / "liveness_compare.csv"))
    args = p.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open source")
        return 2

    tracker_v1 = SimpleFaceTracker(max_center_dist_px=90.0, max_missing_s=1.5)
    tracker_v2 = SimpleFaceTracker(max_center_dist_px=90.0, max_missing_s=1.5)

    l1 = BlinkLiveness(
        ear_threshold=0.21,
        min_closed_seconds=0.06,
        grace_seconds=8.0,
        live_ttl_seconds=20.0,
        min_blinks=2,
        min_actions=2,
        head_turn_range=0.18,
        min_ear_samples_before_spoof=12,
    )

    l2 = AdaptiveLiveness(
        grace_seconds=8.0,
        live_ttl_seconds=25.0,
        min_blinks=1,
        min_actions=2,
        head_turn_range=0.16,
        min_ear_samples_before_spoof=8,
    )

    counts = {
        "v1": {"CHECKING": 0, "LIVE": 0, "SPOOF": 0},
        "v2": {"CHECKING": 0, "LIVE": 0, "SPOOF": 0},
    }

    frames = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        if args.max_frames > 0 and frames > args.max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model=args.detector)
        loc = largest_face(locs)
        if loc is None:
            continue

        t, r, btm, l = loc
        lm_list = face_recognition.face_landmarks(rgb, [loc])
        ear = mean_ear(lm_list[0]) if lm_list else None

        now = time.time()
        box = (l, t, r, btm)

        tid1 = tracker_v1.assign(box, now)
        st1 = tracker_v1.get_state(tid1)
        if st1 is not None:
            lbl1 = l1.update(st1, ear, lm_list[0] if lm_list else None, now, box=box)
            counts["v1"][lbl1] += 1

        tid2 = tracker_v2.assign(box, now)
        st2 = tracker_v2.get_state(tid2)
        if st2 is not None:
            lbl2 = l2.update(st2, ear, lm_list[0] if lm_list else None, now, box=box)
            counts["v2"][lbl2] += 1

    cap.release()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in ("v1", "v2"):
        c = counts[algo]
        total = c["CHECKING"] + c["LIVE"] + c["SPOOF"]
        live_ratio = (c["LIVE"] / total) if total > 0 else 0.0
        spoof_ratio = (c["SPOOF"] / total) if total > 0 else 0.0

        acc = ""
        if args.gt == "live":
            acc = f"{live_ratio:.4f}"
        elif args.gt == "spoof":
            acc = f"{spoof_ratio:.4f}"

        rows.append(
            {
                "algo": algo,
                "frames_scored": total,
                "checking_frames": c["CHECKING"],
                "live_frames": c["LIVE"],
                "spoof_frames": c["SPOOF"],
                "live_ratio": f"{live_ratio:.4f}",
                "spoof_ratio": f"{spoof_ratio:.4f}",
                "accuracy_if_gt": acc,
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo",
                "frames_scored",
                "checking_frames",
                "live_frames",
                "spoof_frames",
                "live_ratio",
                "spoof_ratio",
                "accuracy_if_gt",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    dur = time.time() - t0
    print(f"Frames read: {frames}")
    print(f"Duration: {dur:.1f}s")
    print(f"Saved: {out_path}")
    for r in rows:
        print(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
