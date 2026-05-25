from pathlib import Path
import shutil


DATASET_ROOT = Path("datasets/processed/skyrock_yolo")
BACKUP_ROOT = Path("datasets/processed/skyrock_yolo_label_backup")

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def backup_labels():
    if BACKUP_ROOT.exists():
        print(f"Backup already exists: {BACKUP_ROOT}")
        return

    for split in SPLITS:
        src = DATASET_ROOT / "labels" / split
        dst = BACKUP_ROOT / "labels" / split

        if src.exists():
            shutil.copytree(src, dst)

    print(f"Label backup created at: {BACKUP_ROOT}")


def find_image_for_label(split: str, label_path: Path):
    image_dir = DATASET_ROOT / "images" / split

    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{label_path.stem}{ext}"
        if image_path.exists():
            return image_path

    return None


def clean_label_file(label_path: Path):
    cleaned_lines = []
    removed = 0

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                removed += 1
                continue

            try:
                cls = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                removed += 1
                continue

            if cls not in [0, 1, 2, 3, 4]:
                removed += 1
                continue

            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                removed += 1
                continue

            cleaned_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        if cleaned_lines:
            f.write("\n".join(cleaned_lines) + "\n")

    return removed


def remove_orphan_labels_and_images():
    total_removed_labels = 0
    total_removed_images = 0

    for split in SPLITS:
        label_dir = DATASET_ROOT / "labels" / split
        image_dir = DATASET_ROOT / "images" / split

        if not label_dir.exists() or not image_dir.exists():
            continue

        for label_path in label_dir.glob("*.txt"):
            image_path = find_image_for_label(split, label_path)

            if image_path is None:
                label_path.unlink()
                total_removed_labels += 1

        label_stems = {p.stem for p in label_dir.glob("*.txt")}

        for image_path in image_dir.glob("*.*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if image_path.stem not in label_stems:
                image_path.unlink()
                total_removed_images += 1

    print(f"Removed orphan labels: {total_removed_labels}")
    print(f"Removed orphan images: {total_removed_images}")


def clean_all_labels():
    total_removed_rows = 0

    for split in SPLITS:
        label_dir = DATASET_ROOT / "labels" / split

        if not label_dir.exists():
            continue

        for label_path in label_dir.glob("*.txt"):
            total_removed_rows += clean_label_file(label_path)

    print(f"Removed invalid label rows: {total_removed_rows}")


def main():
    backup_labels()
    clean_all_labels()
    remove_orphan_labels_and_images()
    print("Dataset preprocessing completed.")


if __name__ == "__main__":
    main()