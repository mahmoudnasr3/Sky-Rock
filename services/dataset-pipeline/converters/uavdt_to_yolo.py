import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/uavdt")
OUTPUT_ROOT = Path("datasets/processed/converted/uavdt")

UAVDT_CLASS_ID = 2  # vehicle

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_images_root():
    candidates = [
        RAW_ROOT / "UAV-benchmark-M" / "UAV-benchmark-M",
        RAW_ROOT / "UAV-benchmark-M",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find UAVDT image root.")


def find_gt_root():
    candidates = [
        RAW_ROOT / "UAV-benchmark-MOTD_v1.0" / "UAV-benchmark-MOTD_v1.0" / "GT",
        RAW_ROOT / "UAV-benchmark-MOTD_v1.0" / "GT",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find UAVDT GT root.")


def clean_sequence_name(gt_path):
    name = gt_path.stem
    name = name.replace("_gt_whole", "")
    name = name.replace("_gt_ignore", "")
    name = name.replace("_gt", "")
    return name


def parse_gt_file(gt_path):
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

            anns_by_frame[frame_id].append((x, y, w, h))

    return anns_by_frame


def find_sequence_image_dir(images_root, sequence_name):
    path = images_root / sequence_name
    if path.exists():
        return path

    matches = list(images_root.rglob(sequence_name))
    for match in matches:
        if match.is_dir():
            return match

    return None


def find_image_by_frame(img_dir, frame_id):
    stems = [
        f"img{frame_id:06d}",
        f"img{frame_id:05d}",
        f"{frame_id:06d}",
        f"{frame_id:05d}",
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


def choose_split(index, total):
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    if index < train_end:
        return "train"

    if index < val_end:
        return "val"

    return "test"


def convert_uavdt():
    images_root = find_images_root()
    gt_root = find_gt_root()

    gt_files = sorted([
        p for p in gt_root.glob("*.txt")
        if p.stem.endswith("_gt")
    ])

    total_sequences = len(gt_files)

    for index, gt_path in enumerate(gt_files):
        split = choose_split(index, total_sequences)
        sequence_name = clean_sequence_name(gt_path)

        img_dir = find_sequence_image_dir(images_root, sequence_name)
        if img_dir is None:
            print(f"Skipping {sequence_name}: image folder not found")
            continue

        out_img_dir = OUTPUT_ROOT / "images" / split
        out_lbl_dir = OUTPUT_ROOT / "labels" / split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        anns_by_frame = parse_gt_file(gt_path)

        for frame_id, anns in tqdm(anns_by_frame.items(), desc=f"UAVDT {sequence_name}"):
            img_path = find_image_by_frame(img_dir, frame_id)

            if img_path is None:
                continue

            img_w, img_h = image_size(img_path)

            if img_w is None:
                continue

            yolo_lines = []

            for x, y, w, h in anns:
                xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)

                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                    continue

                yolo_lines.append(f"{UAVDT_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if not yolo_lines:
                continue

            output_stem = f"uavdt_{sequence_name}_{frame_id:06d}"

            shutil.copy(img_path, out_img_dir / f"{output_stem}{img_path.suffix.lower()}")

            with open(out_lbl_dir / f"{output_stem}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")


def main():
    convert_uavdt()


if __name__ == "__main__":
    main()