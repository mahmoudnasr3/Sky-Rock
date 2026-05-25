import json
from pathlib import Path

import cv2
from tqdm import tqdm


RAW_ROOT = Path("datasets/raw/anti_uav/Anti-UAV-RGBT")
OUTPUT_ROOT = Path("datasets/processed/converted/anti_uav")

CLASS_ID = 0  # uav

SPLITS = ["train", "val", "test"]
MODALITIES = ["visible", "infrared"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_video(sequence_dir, modality):
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        path = sequence_dir / f"{modality}{ext}"
        if path.exists():
            return path
    return None


def find_json(sequence_dir, modality):
    path = sequence_dir / f"{modality}.json"
    return path if path.exists() else None


def get_bboxes(data):
    if isinstance(data, dict):
        bboxes = (
            data.get("gt_rect")
            or data.get("bbox")
            or data.get("bboxes")
            or data.get("res")
        )
        exists = (
            data.get("exist")
            or data.get("exists")
            or data.get("visible")
            or data.get("visibility")
        )
        return bboxes, exists

    if isinstance(data, list):
        return data, None

    return None, None


def normalize_bbox(bbox):
    if bbox is None:
        return None

    if isinstance(bbox, dict):
        if "bbox" in bbox:
            bbox = bbox["bbox"]
        else:
            x = bbox.get("x", bbox.get("left"))
            y = bbox.get("y", bbox.get("top"))
            w = bbox.get("w", bbox.get("width"))
            h = bbox.get("h", bbox.get("height"))

            if None in [x, y, w, h]:
                return None

            return float(x), float(y), float(w), float(h)

    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    return None


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h,
    )


def convert_video(video_path, json_path, out_img_dir, out_lbl_dir, prefix):
    data = load_json(json_path)
    bboxes, exists = get_bboxes(data)

    if bboxes is None:
        print(f"No boxes found in {json_path}")
        return

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, len(bboxes))
    saved = 0

    for idx in tqdm(range(max_frames), desc=prefix):
        ok, frame = cap.read()

        if not ok:
            break

        bbox = normalize_bbox(bboxes[idx])

        object_exists = True
        if exists is not None and idx < len(exists):
            object_exists = bool(exists[idx])

        if bbox is None or not object_exists:
            continue

        img_h, img_w = frame.shape[:2]
        x, y, w, h = bbox

        if w <= 1 or h <= 1:
            continue

        xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w, img_h)

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
            continue

        image_name = f"{prefix}_{idx:06d}.jpg"
        label_name = f"{prefix}_{idx:06d}.txt"

        cv2.imwrite(str(out_img_dir / image_name), frame)

        with open(out_lbl_dir / label_name, "w", encoding="utf-8") as f:
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        saved += 1

    cap.release()
    print(f"Saved {saved} frames from {video_path}")


def convert_split(split):
    split_dir = RAW_ROOT / split

    out_img_dir = OUTPUT_ROOT / "images" / split
    out_lbl_dir = OUTPUT_ROOT / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        print(f"Missing split: {split_dir}")
        return

    for sequence_dir in split_dir.iterdir():
        if not sequence_dir.is_dir():
            continue

        for modality in MODALITIES:
            video_path = find_video(sequence_dir, modality)
            json_path = find_json(sequence_dir, modality)

            if video_path is None or json_path is None:
                continue

            prefix = f"anti_uav_{split}_{sequence_dir.name}_{modality}"

            convert_video(video_path, json_path, out_img_dir, out_lbl_dir, prefix)


def main():
    for split in SPLITS:
        convert_split(split)


if __name__ == "__main__":
    main()