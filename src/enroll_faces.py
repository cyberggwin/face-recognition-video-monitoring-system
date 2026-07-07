"""
Enroll face encodings from a local dataset structured into folders (one folder per person).
The result is saved in models/known_faces.pkl and is used later by live_yolo_face_id.py.
"""
import argparse
import os
import pickle
from pathlib import Path

import face_recognition


def is_image(name: str) -> bool:
    """Returns True if the name has an accepted image extension."""
    ext = name.lower().rsplit('.', 1)[-1] if '.' in name else ''
    return ext in {"jpg", "jpeg", "png", "bmp", "webp"}


def collect_images(dataset_dir: Path):
    """Generates (label, img_path) tuples by recursively traversing dataset/<person>/*.
    'label' is the name of the person's folder.
    """
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = person_dir.name
        for img_path in sorted(person_dir.rglob('*')):
            if img_path.is_file() and not img_path.name.startswith('.') and is_image(img_path.name):
                yield label, img_path


def main():
    """Loads images from dataset/, detects faces, and saves encodings + labels in a .pkl file."""
    parser = argparse.ArgumentParser(description="Enroll face encodings from a dataset of folders.")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder (subfolders per person)")
    parser.add_argument("--output", default=str(Path("models") / "known_faces.pkl"), help="Output .pkl path")
    parser.add_argument("--detector", choices=["hog", "cnn"], default="hog", help="face_recognition detector model")
    parser.add_argument("--jitters", type=int, default=2, help="face_recognition num_jitters for more robust encodings")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        print(f"Dataset folder not found: {dataset_dir}")
        return

    known_encodings = []
    known_names = []

    processed = 0
    detected = 0

    print(f"Scanning dataset: {dataset_dir}")
    for label, img_path in collect_images(dataset_dir):
        processed += 1
        try:
            # Load the image and detect face locations, then compute embeddings
            image = face_recognition.load_image_file(str(img_path))
            boxes = face_recognition.face_locations(image, model=args.detector)
            if not boxes:
                print(f"   No face in: {img_path}")
                continue

            # If multiple faces exist, keep the largest one.
            def area(b):
                top, right, bottom, left = b
                return max(0, bottom - top) * max(0, right - left)

            best_box = max(boxes, key=area)
            encs = face_recognition.face_encodings(image, [best_box], num_jitters=max(1, args.jitters))
            if encs:
                known_encodings.append(encs[0])
                known_names.append(label)
                detected += 1
            else:
                print(f"   No face in: {img_path}")
        except Exception as e:
            print(f"   Error {img_path}: {e}")

    data = {"encodings": known_encodings, "names": known_names}
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print("---------------------------------")
    print(f"Done. Processed images: {processed}, faces enrolled: {detected}")
    print(f"Saved encodings: {output_path}")


if __name__ == "__main__":
    main()
