import json
import shutil
from pathlib import Path
from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/coco")
OUTPUT_ROOT = Path("datasets/processed/converted/coco")

COCO_TO_SKYROCK = {
    1: 1,    # person -> person
    2: 3,    # bicycle -> two_wheeler
    3: 2,    # car -> vehicle
    4: 3,    # motorcycle -> two_wheeler
    5: 4,    # airplane -> airborne_distractor
    6: 2,    # bus -> vehicle
    8: 2,    # truck -> vehicle
    16: 4,   # bird -> airborne_distractor
    33: 4,   # kite -> airborne_distractor
}


def get_annotation_file(split):
    """
    Supports both:
    datasets/raw/coco/annotations_trainval2017/annotations/instances_train2017.json
    datasets/raw/coco/annotations/instances_train2017.json
    """

    candidates = [
        RAW_ROOT / "annotations_trainval2017" / "annotations" / f"instances_{split}2017.json",
        RAW_ROOT / "annotations" / f"instances_{split}2017.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Missing annotation file. Tried:\n" +
        "\n".join(str(path) for path in candidates)
    )


def get_image_dir(split):
    """
    Supports both:
    datasets/raw/coco/train2017
    datasets/raw/coco/train2017/train2017

    and:
    datasets/raw/coco/val2017
    datasets/raw/coco/val2017/val2017
    """

    split_folder = f"{split}2017"

    candidates = [
        RAW_ROOT / split_folder / split_folder,  # nested folder: train2017/train2017
        RAW_ROOT / split_folder,                # normal folder: train2017
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Missing image folder. Tried:\n" +
        "\n".join(str(path) for path in candidates)
    )


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def convert_split(split):
    ann_file = get_annotation_file(split)
    img_dir = get_image_dir(split)

    out_img_dir = OUTPUT_ROOT / "images" / split
    out_lbl_dir = OUTPUT_ROOT / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting COCO {split}")
    print(f"Using annotation file: {ann_file}")
    print(f"Using image folder:     {img_dir}")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image = {}

    for ann in data["annotations"]:
        category_id = ann["category_id"]

        if category_id not in COCO_TO_SKYROCK:
            continue

        image_id = ann["image_id"]
        anns_by_image.setdefault(image_id, []).append(ann)

    copied_count = 0
    skipped_missing_images = 0

    for image_id, anns in tqdm(anns_by_image.items(), desc=f"COCO {split}"):
        img = images.get(image_id)

        if img is None:
            continue

        filename = img["file_name"]
        width = img["width"]
        height = img["height"]

        src_img = img_dir / filename

        if not src_img.exists():
            skipped_missing_images += 1
            continue

        output_stem = f"coco_{Path(filename).stem}"
        dst_img = out_img_dir / f"{output_stem}{src_img.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{output_stem}.txt"

        yolo_lines = []

        for ann in anns:
            cls = COCO_TO_SKYROCK[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            if w <= 1 or h <= 1:
                continue

            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, width, height)

            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= wn <= 1 and 0 <= hn <= 1):
                continue

            yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            continue

        shutil.copy(src_img, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        copied_count += 1

    print(f"Finished COCO {split}")
    print(f"Copied images: {copied_count}")
    print(f"Skipped missing images: {skipped_missing_images}")


def main():
    convert_split("train")
    convert_split("val")


if __name__ == "__main__":
    main()