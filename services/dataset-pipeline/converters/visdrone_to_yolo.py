import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/visdrone")
OUTPUT_ROOT = Path("datasets/processed/converted/visdrone")

VISDRONE_TO_SKYROCK = {
    1: 1,   # pedestrian -> person
    2: 1,   # people -> person
    3: 3,   # bicycle -> two_wheeler
    4: 2,   # car -> vehicle
    5: 2,   # van -> vehicle
    6: 2,   # truck -> vehicle
    7: 3,   # tricycle -> two_wheeler
    8: 3,   # awning-tricycle -> two_wheeler
    9: 2,   # bus -> vehicle
    10: 3,  # motor -> two_wheeler
}

SPLIT_MAP = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def find_split_root(folder_name):
    """
    Supports both structures:

    datasets/raw/visdrone/VisDrone2019-DET-train/images

    and:

    datasets/raw/visdrone/VisDrone2019-DET-train/VisDrone2019-DET-train/images
    """

    normal_root = RAW_ROOT / folder_name
    nested_root = RAW_ROOT / folder_name / folder_name

    candidates = [
        nested_root,
        normal_root,
    ]

    for path in candidates:
        img_dir = path / "images"
        ann_dir = path / "annotations"

        if img_dir.exists() and ann_dir.exists():
            return path

    raise FileNotFoundError(
        "Missing VisDrone split folders. Tried:\n"
        + "\n".join(str(path) for path in candidates)
    )


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def convert_split(split):
    folder_name = SPLIT_MAP[split]
    split_root = find_split_root(folder_name)

    img_dir = split_root / "images"
    ann_dir = split_root / "annotations"

    out_img_dir = OUTPUT_ROOT / "images" / split
    out_lbl_dir = OUTPUT_ROOT / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting VisDrone {split}")
    print(f"Using image folder:      {img_dir}")
    print(f"Using annotation folder: {ann_dir}")

    ann_files = sorted(ann_dir.glob("*.txt"))

    copied_count = 0
    skipped_missing_images = 0
    skipped_empty_labels = 0

    for ann_path in tqdm(ann_files, desc=f"VisDrone {split}"):
        image_path = img_dir / f"{ann_path.stem}.jpg"

        if not image_path.exists():
            image_path = img_dir / f"{ann_path.stem}.png"

        if not image_path.exists():
            skipped_missing_images += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            skipped_missing_images += 1
            continue

        img_h, img_w = image.shape[:2]
        yolo_lines = []

        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                class_id = int(parts[5])

                if class_id not in VISDRONE_TO_SKYROCK:
                    continue

                if w <= 1 or h <= 1:
                    continue

                cls = VISDRONE_TO_SKYROCK[class_id]
                xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)

                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                    continue

                yolo_lines.append(
                    f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                )

        if not yolo_lines:
            skipped_empty_labels += 1
            continue

        output_stem = f"visdrone_{split}_{ann_path.stem}"

        dst_img = out_img_dir / f"{output_stem}{image_path.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{output_stem}.txt"

        shutil.copy(image_path, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        copied_count += 1

    print(f"Finished VisDrone {split}")
    print(f"Copied images: {copied_count}")
    print(f"Skipped missing images: {skipped_missing_images}")
    print(f"Skipped empty labels: {skipped_empty_labels}")


def main():
    for split in ["train", "val", "test"]:
        convert_split(split)


if __name__ == "__main__":
    main()