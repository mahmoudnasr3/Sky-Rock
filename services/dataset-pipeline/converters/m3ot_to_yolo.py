import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/m3ot/M3OT/M3OT")
OUTPUT_ROOT = Path("datasets/processed/converted/m3ot")

SPLITS = ["train", "val", "test"]
MODALITIES = ["rgb", "ir"]
SCENES = ["1", "2"]

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


LABEL_TO_SKYROCK = {
    "uav": 0,
    "drone": 0,
    "person": 1,
    "pedestrian": 1,
    "people": 1,
    "car": 2,
    "vehicle": 2,
    "truck": 2,
    "bus": 2,
    "van": 2,
    "bicycle": 3,
    "bike": 3,
    "motorcycle": 3,
    "motorbike": 3,
    "tricycle": 3,
    "bird": 4,
    "airplane": 4,
    "aircraft": 4,
    "kite": 4,
}


def load_labels(labels_path):
    label_map = {}

    if not labels_path.exists():
        return label_map

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()

            if len(parts) < 2:
                continue

            try:
                label_id = int(parts[0])
            except ValueError:
                continue

            label_map[label_id] = parts[1].lower()

    return label_map


def parse_gt(gt_path, label_map):
    anns_by_frame = defaultdict(list)

    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 6:
                parts = line.strip().split()

            if len(parts) < 6:
                continue

            try:
                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue

            if w <= 1 or h <= 1:
                continue

            class_raw = None

            if len(parts) >= 8:
                try:
                    class_raw = int(float(parts[7]))
                except ValueError:
                    class_raw = None

            if class_raw is None:
                cls = 0
            else:
                name = label_map.get(class_raw)
                cls = LABEL_TO_SKYROCK.get(name, 0)

            anns_by_frame[frame_id].append((cls, x, y, w, h))

    return anns_by_frame


def find_image(img_dir, frame_id):
    stems = [
        f"{frame_id:06d}",
        f"{frame_id:05d}",
        f"{frame_id:04d}",
        str(frame_id),
    ]

    for stem in stems:
        for ext in IMAGE_EXTENSIONS:
            path = img_dir / f"{stem}{ext}"
            if path.exists():
                return path

    return None


def image_size(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None

    h, w = img.shape[:2]
    return w, h


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def convert_sequence(sequence_dir, split, scene, modality):
    gt_path = sequence_dir / "gt" / "gt.txt"
    labels_path = sequence_dir / "gt" / "labels.txt"
    img_dir = sequence_dir / "img1"

    if not gt_path.exists() or not img_dir.exists():
        return

    label_map = load_labels(labels_path)
    anns_by_frame = parse_gt(gt_path, label_map)

    out_img_dir = OUTPUT_ROOT / "images" / split
    out_lbl_dir = OUTPUT_ROOT / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for frame_id, anns in tqdm(anns_by_frame.items(), desc=f"M3OT {scene} {modality} {split}"):
        image_path = find_image(img_dir, frame_id)

        if image_path is None:
            continue

        img_w, img_h = image_size(image_path)

        if img_w is None:
            continue

        yolo_lines = []

        for cls, x, y, w, h in anns:
            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)

            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                continue

            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            continue

        output_stem = f"m3ot_{scene}_{modality}_{split}_{sequence_dir.name}_{frame_id:06d}"

        shutil.copy(image_path, out_img_dir / f"{output_stem}{image_path.suffix.lower()}")

        with open(out_lbl_dir / f"{output_stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")


def main():
    for scene in SCENES:
        for modality in MODALITIES:
            for split in SPLITS:
                split_root = RAW_ROOT / scene / modality / split

                if not split_root.exists():
                    continue

                for sequence_dir in split_root.iterdir():
                    if sequence_dir.is_dir():
                        convert_sequence(sequence_dir, split, scene, modality)


if __name__ == "__main__":
    main()