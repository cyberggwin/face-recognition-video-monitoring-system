"""
Benchmark: evaluează sistemul pe un set de imagini/clipuri externe, presupunând că persoanele NU sunt în baza de encodări (ar trebui etichetate "necunoscut").
Produce statistici și CSV cu rezultate per fișier/cadru.

Comandă exemplu:
  python src/evaluate_benchmark.py --input C:\path\to\external_dataset --encodings models/known_faces.pkl --out results_benchmark.csv

Note:
- Procesează imagini (jpg/jpeg/png/bmp/webp). Pentru clipuri video poți indica un fișier cu --video, dar metricile sunt pe cadre.
- Metrici:
  - persons_detected: persoane detectate de YOLO
  - faces_found: fețe detectate în persoanele găsite
  - unknown_count: fețe etichetate "necunoscut"
  - false_known: fețe etichetate ca un nume cunoscut (ar trebui 0)
  - min_dist/avg_dist/max_dist: statistici distanțe față de encodările cunoscute
"""
import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import face_recognition
from tqdm import tqdm  # progress bar

IMG_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}


def load_known(path: Path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data.get('encodings', []), data.get('names', [])


def is_image(p: Path) -> bool:
    if not p.is_file():
        return False
    ext = p.suffix.lower().strip('.')
    return ext in IMG_EXTS


def identify(encodings, names, face_enc, thr=0.58):
    if len(encodings) == 0:
        return "necunoscut", 1.0
    d = face_recognition.face_distance(encodings, face_enc)
    i = int(np.argmin(d))
    return (names[i], float(d[i])) if float(d[i]) < thr else ("necunoscut", float(d[i]))


def process_image(model, encs, names, img_path: Path, conf: float, thr: float):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    yolo_res = model(img, conf=conf, verbose=False)[0]
    boxes = []
    if yolo_res.boxes is not None:
        for b in yolo_res.boxes:
            cls_id = int(b.cls[0]) if hasattr(b.cls[0], 'item') is False else int(b.cls[0].item())
            if cls_id != 0:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))

    persons = len(boxes)
    faces = 0
    unknown = 0
    false_known = 0
    dists = []

    for (x1, y1, x2, y2) in boxes:
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model='hog')
        face_encs = face_recognition.face_encodings(rgb, locs)
        for fe in face_encs:
            faces += 1
            name, dist = identify(encs, names, fe, thr=thr)
            dists.append(dist)
            if name == 'necunoscut':
                unknown += 1
            else:
                false_known += 1

    stats = {
        'file': str(img_path),
        'persons_detected': persons,
        'faces_found': faces,
        'unknown_count': unknown,
        'false_known': false_known,
        'min_dist': min(dists) if dists else '',
        'avg_dist': float(np.mean(dists)) if dists else '',
        'max_dist': max(dists) if dists else '',
    }
    return stats


def main():
    p = argparse.ArgumentParser(description='Benchmark unknown faces recognition behavior on external dataset')
    p.add_argument('--input', required=True, help='Folder cu imagini de test (persoane NECUNOSCUTE)')
    p.add_argument('--encodings', default=str(Path('models') / 'known_faces.pkl'))
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--thr', type=float, default=0.58)
    p.add_argument('--out', default='results_benchmark.csv')
    p.add_argument('--limit', type=int, default=None, help='Procesează doar primele N imagini (pentru test rapid)')
    args = p.parse_args()

    encs, names = load_known(Path(args.encodings))
    print(f"Loaded {len(encs)} known encodings")

    model = YOLO('yolov8n.pt')

    input_dir = Path(args.input)
    files = [p for p in input_dir.rglob('*') if is_image(p)]
    if args.limit is not None:
        files = files[:args.limit]
        print(f"Limiting to first {len(files)} images")
    else:
        print(f"Found {len(files)} images in {input_dir}")

    rows = []
    totals = {
        'persons_detected': 0,
        'faces_found': 0,
        'unknown_count': 0,
        'false_known': 0,
        'imgs': 0,
    }

    start = time.time()
    # Use tqdm progress bar for better feedback
    for f in tqdm(files, desc="Benchmarking images", unit="img"):
        r = process_image(model, encs, names, f, args.conf, args.thr)
        if r is None:
            continue
        rows.append(r)
        totals['imgs'] += 1
        totals['persons_detected'] += r['persons_detected']
        totals['faces_found'] += r['faces_found']
        totals['unknown_count'] += r['unknown_count']
        totals['false_known'] += r['false_known']
        # Periodic brief status in tqdm bar postfix
        if totals['faces_found'] > 0:
            unk_rate = totals['unknown_count'] / totals['faces_found']
            tqdm.write(f"Progress: imgs={totals['imgs']} faces={totals['faces_found']} unknown={totals['unknown_count']} false_known={totals['false_known']} unk_rate={unk_rate:.3f}")

    dur = time.time() - start

    # Scrie CSV detaliat
    out_path = Path(args.out)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'persons_detected', 'faces_found', 'unknown_count', 'false_known', 'min_dist', 'avg_dist', 'max_dist'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Rezumat
    print('--- Benchmark summary ---')
    print(f"Images processed: {totals['imgs']}")
    print(f"Persons detected: {totals['persons_detected']}")
    print(f"Faces found: {totals['faces_found']}")
    print(f"Unknown (expected high): {totals['unknown_count']}")
    print(f"False-known (should be 0): {totals['false_known']}")
    if totals['faces_found'] > 0:
        unk_rate = totals['unknown_count'] / totals['faces_found']
        print(f"Unknown rate: {unk_rate:.3f}")
    print(f"Duration: {dur:.1f}s")
    print(f"CSV saved to: {out_path}")


if __name__ == '__main__':
    main()
