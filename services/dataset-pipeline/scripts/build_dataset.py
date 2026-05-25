import subprocess
import sys
from pathlib import Path


COMMANDS = [
    "services/dataset-pipeline/converters/coco_to_yolo.py",
    "services/dataset-pipeline/converters/visdrone_to_yolo.py",
    "services/dataset-pipeline/converters/uavdt_to_yolo.py",
    "services/dataset-pipeline/converters/anti_uav_to_yolo.py",
    "services/dataset-pipeline/converters/detfly_to_yolo.py",
    "services/dataset-pipeline/converters/m3ot_to_yolo.py",
    "services/dataset-pipeline/merger/merge_yolo_datasets.py",
]


def run_script(script):
    path = Path(script)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\nRunning: {script}")
    subprocess.run([sys.executable, str(path)], check=True)


def main():
    for script in COMMANDS:
        run_script(script)

    print("\nDataset build completed.")


if __name__ == "__main__":
    main()