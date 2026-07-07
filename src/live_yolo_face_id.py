"""Live person filtering (YOLOv8) + face identification (face_recognition)."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import face_recognition

from liveness import AdaptiveLiveness, BlinkLiveness, SimpleFaceTracker, mean_ear


def load_known(path: Path):
    """Load encodings and labels from a .pkl file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('encodings', []), data.get('names', [])


def identify(encodings, names, face_enc, thr=0.58, margin=0.0):
    """Return (name, distance). If distance >= thr, return 'necunoscut'."""
    if len(encodings) == 0:
        return "necunoscut", 1.0
    d = face_recognition.face_distance(encodings, face_enc)
    i = int(np.argmin(d))
    best_name = names[i]
    best_dist = float(d[i])
    if best_dist >= thr:
        return "necunoscut", best_dist

    if margin > 0:
        other = [float(di) for di, n in zip(d, names) if n != best_name]
        if other:
            second_best_other = min(other)
            if (second_best_other - best_dist) < margin:
                return "necunoscut", best_dist

    return best_name, best_dist


def main():
    """Rulează bucla live: citește cadre, aplică YOLO + identificare și afișează/salvează rezultatul."""
    p = argparse.ArgumentParser(description="YOLO person filter + face_recognition identification")
    p.add_argument('--encodings', default=str(Path('models') / 'known_faces.pkl'))
    p.add_argument('--source', default='0', help='0 for webcam, path to video file, or RTSP URL')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--thr', type=float, default=0.58)
    p.add_argument('--margin', type=float, default=0.0, help='Min distance margin vs next different identity (0 disables margin check)')
    p.add_argument('--liveness', action=argparse.BooleanOptionalAction, default=True,
                   help='Blink-based liveness (anti-photo). Use --no-liveness to disable.')
    p.add_argument('--liveness-algo', choices=['v1', 'v2'], default='v2',
                   help='v2 is adaptive and usually better for low FPS / glasses.')
    p.add_argument('--ear-thr', type=float, default=0.21, help='EAR threshold for blink detection (lower => stricter)')
    p.add_argument('--blink-min-close', type=float, default=0.06, help='Minimum eye-closed time in seconds to count as a blink')
    p.add_argument('--liveness-min-blinks', type=int, default=1, help='Minimum number of blink events for liveness')
    p.add_argument('--head-turn-range', type=float, default=0.20, help='Required yaw variation for head-turn liveness cue')
    p.add_argument('--liveness-min-actions', type=int, default=1, help='Minimum liveness actions required (blink and/or head turn)')
    p.add_argument('--liveness-min-samples', type=int, default=6, help='Minimum valid EAR samples before SPOOF is allowed')
    p.add_argument('--liveness-grace', type=float, default=10.0, help='Seconds allowed to provide liveness evidence before labeling SPOOF')
    p.add_argument('--liveness-ttl', type=float, default=20.0, help='Seconds after last blink to keep LIVE status')
    p.add_argument('--save', action='store_true', help='Save annotated output to file')
    p.add_argument('--out', default='output_identified.mp4')
    args = p.parse_args()

    # Interpretează sursa: "0" => webcam default
    source = 0 if args.source == '0' else args.source

    encs, names = load_known(Path(args.encodings))
    print(f"Loaded {len(encs)} encodings")

    # YOLOv8 nano (lightweight, fast on CPU)
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('Cannot open source'); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
        print(f"Saving to {args.out}")

    last_alert = 0.0  # debounce pentru alerte

    tracker = SimpleFaceTracker(max_center_dist_px=90.0, max_missing_s=1.5)
    if args.liveness_algo == 'v2':
        liveness = AdaptiveLiveness(
            grace_seconds=args.liveness_grace,
            live_ttl_seconds=args.liveness_ttl,
            min_blinks=args.liveness_min_blinks,
            min_actions=args.liveness_min_actions,
            head_turn_range=args.head_turn_range,
            min_ear_samples_before_spoof=args.liveness_min_samples,
        )
    else:
        liveness = BlinkLiveness(
            ear_threshold=args.ear_thr,
            min_closed_seconds=args.blink_min_close,
            grace_seconds=args.liveness_grace,
            live_ttl_seconds=args.liveness_ttl,
            min_blinks=args.liveness_min_blinks,
            min_actions=args.liveness_min_actions,
            head_turn_range=args.head_turn_range,
            min_ear_samples_before_spoof=args.liveness_min_samples,
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Rulează YOLO și extrage bbox pentru clasa "person" (id 0)
            yolo_res = model(frame, conf=args.conf, verbose=False)[0]
            boxes = []
            if yolo_res.boxes is not None:
                for b in yolo_res.boxes:
                    # Uneori tensori PyTorch, alteori valori scalare – acoperim ambele
                    cls_id = int(b.cls[0]) if hasattr(b.cls[0], 'item') is False else int(b.cls[0].item())
                    if cls_id != 0:  # 0 = person
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

            # Pentru fiecare persoană, căutăm o față în crop
            for (x1, y1, x2, y2) in boxes:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # model='hog' funcționează pe CPU; 'cnn' este mai precis dar necesită GPU/CUDA
                locs = face_recognition.face_locations(rgb, model='hog')
                face_encs = face_recognition.face_encodings(rgb, locs)
                for (top, right, bottom, left), fe in zip(locs, face_encs):
                    name, dist = identify(encs, names, fe, thr=args.thr, margin=args.margin)
                    now = time.time()
                    # Convert to global face box for tracking/liveness
                    fx1, fy1, fx2, fy2 = (x1 + left, y1 + top, x1 + right, y1 + bottom)

                    live_label = None
                    if args.liveness:
                        track_id = tracker.assign((fx1, fy1, fx2, fy2), now)
                        st = tracker.get_state(track_id)
                        if st is not None:
                            # Landmarks for current face (compute in crop coords)
                            lm_list = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
                            ear = mean_ear(lm_list[0]) if lm_list else None
                            live_label = liveness.update(
                                st,
                                ear,
                                lm_list[0] if lm_list else None,
                                now,
                                box=(fx1, fy1, fx2, fy2),
                            )

                    # If liveness is enabled and we are sure it's spoof, do not trust recognition.
                    display_name = name
                    if live_label == 'SPOOF':
                        display_name = 'SPOOF'

                    # Desenează în coordonate globale (nu relative la crop)
                    color = (0, 255, 0)
                    if live_label == 'SPOOF':
                        color = (0, 0, 255)
                    elif live_label == 'CHECKING':
                        color = (0, 200, 255)

                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)

                    label_parts = [f"{display_name}"]
                    if live_label is not None:
                        label_parts.append(live_label)
                    if live_label != 'SPOOF':
                        label_parts.append(f"{dist:.2f}")
                    text = " | ".join(label_parts)
                    cv2.putText(frame, text, (fx1, max(12, fy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if name == 'necunoscut' and time.time() - last_alert > 10 and live_label != 'SPOOF':
                        # hook: trimite alertă (ex: Telegram), salvează snapshot, etc.
                        last_alert = time.time()

            if writer is not None:
                writer.write(frame)
            else:
                cv2.imshow('YOLO + Face ID', frame)
                if cv2.waitKey(1) == 27:  # ESC
                    break
    except KeyboardInterrupt:
        print("\nStopping on Ctrl+C...")

    cap.release()
    if writer is not None:
        writer.release()
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
