import hashlib
import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from services.api.services.blockchain_service import BlockchainService
from services.api.services.yolo_service import YOLOService


router = APIRouter(prefix="/detect", tags=["Detection"])

UPLOAD_DIR = Path("services/api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

yolo_service = YOLOService()
blockchain_service = BlockchainService()

CLASS_NAMES = {
    0: "uav",
    1: "person",
    2: "vehicle",
    3: "two_wheeler",
    4: "airborne_distractor",
}


@router.post("/")
async def detect_image(file: UploadFile = File(...)):
    image_path = UPLOAD_DIR / file.filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections = yolo_service.predict(str(image_path))

    blockchain_logs = []

    for detection in detections:
        class_name = CLASS_NAMES.get(detection["class_id"], "unknown")

        metadata_hash = hashlib.sha256(
            str(detection).encode("utf-8")
        ).hexdigest()

        try:
            log_result = blockchain_service.log_detection(
                source_id=file.filename,
                class_name=class_name,
                confidence=detection["confidence"],
                metadata_hash=metadata_hash,
            )

            blockchain_logs.append(log_result)

        except Exception as exc:
            blockchain_logs.append(
                {
                    "error": str(exc),
                }
            )

    return {
        "filename": file.filename,
        "detections": detections,
        "blockchain_logs": blockchain_logs,
    }