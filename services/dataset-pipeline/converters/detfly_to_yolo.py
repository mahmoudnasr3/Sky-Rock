import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/detfly")
OUTPUT_ROOT = Path("datasets/processed/converted/detfly")

CLASS_ID = 0  # uav

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_annotations_root():
    candidates = [
        RAW_ROOT / "Annotations" / "Annotations",
        RAW_ROOT / "Annotations",
    ]

    for path in candidates:
        if path.exists() and list(path.rglob("*.xml")):
            return path

    raise FileNotFoundError("Could not find Det-Fly annotations.")


def find_images_root():
    candidates = [
        RAW_ROOT / "JPEGImages" / "JPEGImages",
        RAW_ROOT / "JPEGImages",
        RAW_ROOT / "Annotations" / "JPEGImages" / "JPEGImages",
        RAW_ROOT / "Annotations" / "JPEGImages",
    ]

    for path in candidates:
        if path.exists():
            for ext in IMAGE_EXTENSIONS:
                if list(path.rglob(f"*{ext}")):
                    return path

    raise FileNotFoundError("Could not find Det-Fly images.")


def find_image(images_root, xml_path):
    for ext in IMAGE_EXTENSIONS:
        matches = list(images_root.rglob(f"{xml_path.stem}{ext}"))
        if matches:
            return matches[0]
    return None


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return None, None, []

    img_w = int(float(size.find("width").text))
    img_h = int(float(size.find("height").text))

    boxes = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        boxes.append((xmin, ymin, xmax, ymax))

    return img_w, img_h, boxes


def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + w / 2
    y = ymin + h / 2

    return x / img_w, y / img_h, w / img_w, h / img_h


def choose_split(index, total):
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    if index < train_end:
        return "train"

    if index < val_end:
        return "val"

    return "test"


def main():
    ann_root = find_annotations_root()
    img_root = find_images_root()

    xml_files = sorted(list(ann_root.rglob("*.xml")))

    random.seed(42)
    random.shuffle(xml_files)

    total = len(xml_files)

    for index, xml_path in enumerate(tqdm(xml_files, desc="Det-Fly")):
        split = choose_split(index, total)

        out_img_dir = OUTPUT_ROOT / "images" / split
        out_lbl_dir = OUTPUT_ROOT / "labels" / split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        image_path = find_image(img_root, xml_path)

        if image_path is None:
            continue

        img_w, img_h, boxes = parse_xml(xml_path)

        if img_w is None or not boxes:
            continue

        yolo_lines = []

        for xmin, ymin, xmax, ymax in boxes:
            xc, yc, wn, hn = xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                continue

            yolo_lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            continue

        output_stem = f"detfly_{xml_path.parent.name}_{xml_path.stem}"

        shutil.copy(image_path, out_img_dir / f"{output_stem}{image_path.suffix.lower()}")

        with open(out_lbl_dir / f"{output_stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")


if __name__ == "__main__":
    main()