from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("models/detection/yolo26s/best.pt")


class YOLOService:
    def __init__(self):
        self.model = None

    def load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Train the model first or copy best.pt."
            )

        if self.model is None:
            self.model = YOLO(str(MODEL_PATH))

        return self.model

    def predict(self, image_path: str):
        model = self.load_model()

        results = model.predict(
            source=image_path,
            imgsz=768,
            conf=0.25,
            save=False,
        )

        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                detections.append(
                    {
                        "class_id": cls_id,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

        return detections