from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("models/detection/yolo26s/best.pt")
SOURCE = "datasets/processed/skyrock_yolo/images/test"

IMG_SIZE = 768
CONFIDENCE = 0.25


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Copy best.pt into models/detection/yolo26s/best.pt"
        )

    model = YOLO(str(MODEL_PATH))

    model.predict(
        source=SOURCE,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/inference",
        name="skyrock_predictions",
    )


if __name__ == "__main__":
    main()