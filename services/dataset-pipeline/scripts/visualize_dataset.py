from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


DATASET_ROOT = Path("datasets/processed/skyrock_yolo")
REPORTS_DIR = Path("reports/dataset")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

CLASS_NAMES = {
    0: "uav",
    1: "person",
    2: "vehicle",
    3: "two_wheeler",
    4: "airborne_distractor",
}


def count_images_and_labels():
    rows = []

    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        images = list(image_dir.glob("*.*")) if image_dir.exists() else []
        labels = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        rows.append(
            {
                "split": split,
                "images": len(images),
                "labels": len(labels),
            }
        )

    return pd.DataFrame(rows)


def count_classes():
    counter = Counter()
    rows = []

    for split in SPLITS:
        label_dir = DATASET_ROOT / "labels" / split

        if not label_dir.exists():
            continue

        split_counter = Counter()

        for label_path in label_dir.glob("*.txt"):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    cls = int(float(parts[0]))
                    counter[cls] += 1
                    split_counter[cls] += 1

        for cls, count in split_counter.items():
            rows.append(
                {
                    "split": split,
                    "class_id": cls,
                    "class_name": CLASS_NAMES.get(cls, "unknown"),
                    "count": count,
                }
            )

    total_rows = [
        {
            "split": "all",
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, "unknown"),
            "count": count,
        }
        for cls, count in counter.items()
    ]

    return pd.DataFrame(rows + total_rows)


def plot_split_counts(df):
    plt.figure(figsize=(8, 5))
    plt.bar(df["split"], df["images"])
    plt.title("Images per Split")
    plt.xlabel("Split")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "images_per_split.png", dpi=200)
    plt.close()


def plot_class_distribution(df):
    all_df = df[df["split"] == "all"].copy()
    all_df = all_df.sort_values("class_id")

    plt.figure(figsize=(10, 5))
    plt.bar(all_df["class_name"], all_df["count"])
    plt.title("Object Count per Class")
    plt.xlabel("Class")
    plt.ylabel("Objects")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=200)
    plt.close()


def main():
    split_df = count_images_and_labels()
    class_df = count_classes()

    split_df.to_csv(REPORTS_DIR / "split_counts.csv", index=False)
    class_df.to_csv(REPORTS_DIR / "class_counts.csv", index=False)

    plot_split_counts(split_df)
    plot_class_distribution(class_df)

    print("Dataset visualization completed.")
    print(f"Reports saved in: {REPORTS_DIR}")


if __name__ == "__main__":
    main()