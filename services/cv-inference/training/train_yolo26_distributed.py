import argparse
import gc
import os
import subprocess
from pathlib import Path

import torch
from ultralytics import YOLO


MODELS = {
    "small": {
        "weights": "yolo26s.pt",
        "run_name": "skyrock_yolo26s_ddp_300e",
        "batch": 4,
        "imgsz": 768,
    },
    "medium": {
        "weights": "yolo26m.pt",
        "run_name": "skyrock_yolo26m_ddp_300e",
        "batch": 4,
        "imgsz": 768,
    },
    "large": {
        "weights": "yolo26l.pt",
        "run_name": "skyrock_yolo26l_ddp_300e",
        "batch": 4,
        "imgsz": 768,
    },
}

PROJECT = "runs/detect"
EPOCHS = 300


def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def show_rank_info():
    print("RANK:", os.environ.get("RANK"))
    print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))
    print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))


def show_gpu():
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found")


def get_checkpoint(run_name: str):
    last_path = Path(PROJECT) / run_name / "weights" / "last.pt"
    return last_path if last_path.exists() else None


def train(args):
    cfg = MODELS[args.model]

    free_memory()
    show_rank_info()
    show_gpu()

    checkpoint = get_checkpoint(cfg["run_name"])

    if args.resume and checkpoint is not None:
        print(f"Resuming from: {checkpoint}")
        model = YOLO(str(checkpoint))
        resume_value = True
    else:
        print(f"Starting from weights: {cfg['weights']}")
        model = YOLO(cfg["weights"])
        resume_value = False

    model.train(
        data=args.data,
        epochs=EPOCHS,
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        workers=args.workers,
        cache=False,
        amp=True,

        optimizer="auto",
        cos_lr=True,
        patience=80,
        close_mosaic=20,

        save=True,
        save_period=10,
        plots=True,
        val=True,

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

        project=PROJECT,
        name=cfg["run_name"],
        exist_ok=True,
        resume=resume_value,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["small", "medium", "large"],
        required=True,
    )

    parser.add_argument(
        "--data",
        default="datasets/distributed_skyrock_yolo.yaml",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()