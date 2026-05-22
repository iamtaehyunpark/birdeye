"""
detector.py — YOLO object detection wrapper.

Loaded ONCE at startup and kept in memory for the full session.
Ground-contact pixel = bottom-center of the bounding box (where the object
meets the ground plane), fed into the H-matrix projector.
"""
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around ultralytics YOLO for single-frame inference."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        from ultralytics import YOLO  # deferred import — avoids slow load at module level

        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        logger.info("YOLO model ready")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single BGR frame.

        Returns:
            list of detections, each a dict:
                class_name  (str)
                confidence  (float)
                bbox        [x1, y1, x2, y2]  — absolute pixel coords
                ground_px   [u, v]             — bottom-center ground contact
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections: List[Dict] = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf    = float(box.conf[0])
            cls_id  = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            # Ground-contact: bottom-center of the bounding box.
            # For vehicles / people this is where tires/feet touch the floor.
            u = (x1 + x2) / 2.0
            v = y2

            detections.append({
                "class_name": cls_name,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "ground_px":  [u, v],
            })

        return detections

    def unload(self) -> None:
        """Explicitly free model from memory (useful for testing)."""
        del self.model
        self.model = None  # type: ignore
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("YOLO model unloaded and GPU cache cleared")
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────
_detector: Optional[YOLODetector] = None


def get_detector(model_path: str = "yolov8n.pt", conf_threshold: float = 0.35) -> YOLODetector:
    """Return the shared detector, initialising it on first call."""
    global _detector
    if _detector is None:
        _detector = YOLODetector(model_path, conf_threshold)
    return _detector
