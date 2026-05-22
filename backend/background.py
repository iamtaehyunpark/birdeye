"""
background.py — Reference-image background differencing.

Core logic:
  After calibration, we know H (camera → reference).
  Invert H to warp the reference image into camera perspective → "what the
  camera *should* see if nothing moved".

  Diff = |camera_frame - warped_reference|

  High diff pixels = something changed since the reference was captured.
    • In camera but NOT in reference → new moving object → TRACK IT
    • In reference but NOT in camera → object has since left → YOLO won't
      detect anything there anyway, so it's naturally ignored.

  YOLO detections whose ground-contact point (or bottom bbox region) falls
  inside a high-diff area pass through. All others are classified as
  "background static" and suppressed from the tracker.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundDifferencer:
    """
    Compares the live camera frame against the homography-warped reference image
    to extract a foreground mask, then filters YOLO detections accordingly.
    """

    def __init__(
        self,
        diff_threshold: int = 40,
        blur_ksize: int = 5,
        morph_open_ksize: int = 5,
        morph_dilate_ksize: int = 9,
        dilate_iter: int = 2,
        region_foreground_ratio: float = 0.15,
    ):
        """
        Args:
            diff_threshold:          Pixel difference 0–255 above which a pixel
                                     is considered foreground. Lower = more
                                     sensitive, higher = ignores minor lighting/
                                     compression differences.
            blur_ksize:              Gaussian blur kernel size applied before
                                     differencing (reduces noise).
            morph_open_ksize:        Erosion then dilation kernel — removes tiny
                                     isolated foreground specks.
            morph_dilate_ksize:      Final dilation kernel — expands foreground
                                     blobs to fully enclose detected objects.
            dilate_iter:             Dilation iterations.
            region_foreground_ratio: Fraction of the bottom-quarter of a bbox
                                     that must be foreground to pass the object.
        """
        self.diff_threshold          = diff_threshold
        self.blur_ksize              = blur_ksize
        self.morph_open_ksize        = morph_open_ksize
        self.morph_dilate_ksize      = morph_dilate_ksize
        self.dilate_iter             = dilate_iter
        self.region_foreground_ratio = region_foreground_ratio

        self._ref_gray: Optional[np.ndarray] = None   # blurred gray, camera space
        self._ref_bgr:  Optional[np.ndarray] = None   # for visualization
        self._open_kernel:   Optional[np.ndarray] = None
        self._dilate_kernel: Optional[np.ndarray] = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ref_gray is not None

    def setup(
        self,
        ref_bgr: np.ndarray,
        H: np.ndarray,
        cam_w: int,
        cam_h: int,
    ) -> None:
        """
        Warp the reference image into camera perspective and pre-compute the
        blurred grayscale background template.

        Args:
            ref_bgr: Reference image (BGR, any resolution)
            H:       3×3 homography — maps camera pixels → reference pixels
            cam_w:   Camera frame width
            cam_h:   Camera frame height
        """
        # H maps cam → ref, so H_inv maps ref → cam
        H_inv = np.linalg.inv(H)

        warped = cv2.warpPerspective(ref_bgr, H_inv, (cam_w, cam_h))
        self._ref_bgr = warped

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 1:
            gray = cv2.GaussianBlur(
                gray, (self.blur_ksize, self.blur_ksize), 0
            )
        self._ref_gray = gray

        # Pre-build morphological kernels
        ks_open   = self.morph_open_ksize
        ks_dilate = self.morph_dilate_ksize
        self._open_kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ks_open, ks_open)
        )
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ks_dilate, ks_dilate)
        )

        logger.info(
            f"BackgroundDifferencer ready — warped ref {cam_w}×{cam_h}, "
            f"threshold={self.diff_threshold}"
        )

    def invalidate(self) -> None:
        """Call when H or reference image changes so setup() re-runs."""
        self._ref_gray = None
        self._ref_bgr  = None

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def compute_mask(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Compute a binary foreground mask.

        Returns:
            mask: uint8 (0 = background/static, 255 = foreground/moving)
        """
        if self._ref_gray is None:
            # Not set up — treat whole frame as foreground (don't suppress anything)
            return np.full(camera_frame.shape[:2], 255, dtype=np.uint8)

        cam_gray = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize > 1:
            cam_gray = cv2.GaussianBlur(
                cam_gray, (self.blur_ksize, self.blur_ksize), 0
            )

        # |camera - warped_ref| → high where scene differs from reference
        diff = cv2.absdiff(cam_gray, self._ref_gray)

        _, mask = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # Remove tiny noise blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        # Expand blobs to enclose the full object body
        mask = cv2.dilate(mask, self._dilate_kernel, iterations=self.dilate_iter)

        return mask

    def filter_detections(
        self,
        detections: List[Dict],
        mask: np.ndarray,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Classify detections as moving (foreground) or static (background).

        Strategy (two-pass):
          1. Check ground-contact pixel (u, v) — most reliable single point.
          2. If that pixel is background, check the bottom ¼ of the bounding
             box as a region. If >15% of that region is foreground, the object
             is still considered moving (handles partial overlap with static).

        Args:
            detections: list of detection dicts (with 'ground_px' and 'bbox')
            mask:       foreground mask from compute_mask()

        Returns:
            (moving, static) — two lists of detection dicts
        """
        h, w = mask.shape[:2]
        moving: List[Dict] = []
        static: List[Dict] = []

        for det in detections:
            u, v = det["ground_px"]
            ui, vi = int(round(u)), int(round(v))

            # ── Primary: ground-contact pixel ─────────────────────────────
            if 0 <= vi < h and 0 <= ui < w and mask[vi, ui] > 0:
                moving.append(det)
                continue

            # ── Fallback: bottom-quarter region check ──────────────────────
            x1, y1, x2, y2 = [int(round(c)) for c in det["bbox"]]
            ry1 = max(0,     int(y1 + (y2 - y1) * 0.65))  # bottom 35%
            ry2 = min(h - 1, y2)
            rx1 = max(0,     x1)
            rx2 = min(w - 1, x2)

            if ry2 > ry1 and rx2 > rx1:
                region = mask[ry1:ry2, rx1:rx2]
                fg_ratio = np.count_nonzero(region) / max(region.size, 1)
                if fg_ratio >= self.region_foreground_ratio:
                    moving.append(det)
                    continue

            static.append(det)

        return moving, static

    # ── Visualization helpers ─────────────────────────────────────────────────

    @property
    def warped_reference_bgr(self) -> Optional[np.ndarray]:
        """Reference image warped into camera space — for debug overlay."""
        return self._ref_bgr
