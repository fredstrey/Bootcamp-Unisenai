from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import base64
import threading

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class DetectionItem:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass(slots=True)
class InferenceResult:
    filename: str
    count: int
    detections: list[DetectionItem]
    image_bytes: bytes
    image_mime_type: str = "image/png"

    def to_data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.image_mime_type};base64,{encoded}"


class CrackDetector:
    def __init__(self, model_path: Path, confidence: float = 0.25) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Weight file not found: {model_path}")

        self.model_path = model_path
        self.confidence = confidence
        self._model = YOLO(str(model_path))
        self._lock = threading.Lock()

    def predict(self, image_bytes: bytes, filename: str = "upload.png") -> InferenceResult:
        image = self._decode_image(image_bytes)

        with self._lock:
            prediction = self._model.predict(image, conf=self.confidence, verbose=False)[0]

        detections = self._extract_detections(prediction)
        annotated = self._annotate_count(prediction.plot(), len(detections))

        success, buffer = cv2.imencode(".png", annotated)
        if not success:
            raise RuntimeError("Failed to encode annotated image")

        return InferenceResult(
            filename=filename,
            count=len(detections),
            detections=detections,
            image_bytes=buffer.tobytes(),
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unsupported or corrupted image file")
        return image

    @staticmethod
    def _extract_detections(prediction) -> list[DetectionItem]:
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            return []

        class_names = prediction.names or {}
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        detections: list[DetectionItem] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True):
            idx = int(class_id)
            if isinstance(class_names, dict):
                class_name = str(class_names.get(idx, idx))
            else:
                class_name = str(class_names[idx]) if idx < len(class_names) else str(idx)

            detections.append(
                DetectionItem(
                    class_id=idx,
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=[float(coord) for coord in bbox.tolist()],
                )
            )

        return detections

    @staticmethod
    def _annotate_count(image: np.ndarray, count: int) -> np.ndarray:
        annotated = image.copy()
        label = f"Rachaduras detectadas: {count}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.85
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)

        padding = 14
        top_left = (12, 12)
        bottom_right = (
            top_left[0] + text_width + padding * 2,
            top_left[1] + text_height + baseline + padding * 2,
        )

        overlay = annotated.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)
        cv2.putText(
            annotated,
            label,
            (top_left[0] + padding, top_left[1] + text_height + padding),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return annotated
