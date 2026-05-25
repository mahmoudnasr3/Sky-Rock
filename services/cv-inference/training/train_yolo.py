import gc
import subprocess

import torch
from ultralytics import YOLO


DATA_YAML = r"datasets\processed\skyrock_yolo\data.yaml"

# Change this to your downloaded YOLO26 weight file.
MODEL = r"yolo26s.pt"

RUN_NAME = "skyrock_yolo26s_optimized"

EPOCHS = 250
IMG_SIZE = 768
BATCH = 4
WORKERS = 8


def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def show_gpu():
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found")


def main():
    free_memory()
    show_gpu()

    model = YOLO(MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        cache=False,
        amp=True,
        cos_lr=True,
        patience=50,
        close_mosaic=15,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        project="runs/detect",
        name=RUN_NAME,
    )


if __name__ == "__main__":
    main()