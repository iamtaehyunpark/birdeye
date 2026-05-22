"""
projector.py — H-matrix pixel → reference-image coordinate projection.

Pure OpenCV linear algebra — zero deep-learning overhead.
Typical latency: < 0.1 ms per point on CPU.
"""
import numpy as np
import cv2
from typing import Tuple


def project_point(u: float, v: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Project a single camera pixel (u, v) to reference image coordinates (rx, ry).

    Args:
        u: x-pixel in camera frame
        v: y-pixel in camera frame (bottom-center of detection bbox = ground contact)
        H: 3×3 homography matrix (camera → reference image)

    Returns:
        (rx, ry): float coordinates in reference image pixel space
    """
    pt = np.array([[[u, v]]], dtype=np.float64)
    dst = cv2.perspectiveTransform(pt, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Project N points at once.

    Args:
        points: float array of shape (N, 2)
        H:      3×3 homography matrix

    Returns:
        projected: float array of shape (N, 2)
    """
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    dst = cv2.perspectiveTransform(pts, H)
    return dst.reshape(-1, 2)


def reprojection_error(
    src_pts: np.ndarray, dst_pts: np.ndarray, H: np.ndarray
) -> float:
    """
    Compute mean reprojection error for a set of correspondences.
    Used for H validation.
    """
    projected = project_points(src_pts, H)
    errors = np.linalg.norm(projected - dst_pts, axis=1)
    return float(np.mean(errors))
