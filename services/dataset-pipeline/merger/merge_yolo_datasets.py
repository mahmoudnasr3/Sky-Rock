import shutil
from pathlib import Path

from tqdm import tqdm


SOURCES = [
    "coco",
    "visdrone",
    "uavdt",
    "anti_uav",
    "detfly",
    "m3ot",
]

SPLITS = ["train", "val", "test"]

OUTPUT = Path("datasets/processed/skyrock_yolo")


def copy_split(dataset_name, split):
    src_img_dir = Path(f"datasets/processed/converted/{dataset_name}/images/{split}")
    src_lbl_dir = Path(f"datasets/processed/converted/{dataset_name}/labels/{split}")

    dst_img_dir = OUTPUT / "images" / split
    dst_lbl_dir = OUTPUT / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        print(f"Skipping missing image folder: {src_img_dir}")
        return

    if not src_lbl_dir.exists():
        print(f"Skipping missing label folder: {src_lbl_dir}")
        return

    copied = 0
    skipped = 0

    for img_path in tqdm(list(src_img_dir.glob("*.*")), desc=f"Merging {dataset_name} {split}"):
        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            skipped += 1
            continue

        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(label_path, dst_lbl_dir / label_path.name)

        copied += 1

    print(f"{dataset_name} {split}: copied={copied}, skipped={skipped}")


def main():
    for dataset in SOURCES:
        for split in SPLITS:
            copy_split(dataset, split)

    print("Merge completed.")


if __name__ == "__main__":
    main()