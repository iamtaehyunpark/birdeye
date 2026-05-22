"""
initializer.py — Automated Homography Initialization Layer.

Execution order (one-time, triggered on demand):
  1. SuperPoint + LightGlue  (GPU — best accuracy, cross-view capable)
  2. ORB + BFMatcher         (CPU fallback — no torch required)
  3. Manual 4-point pairs    (always available as guaranteed fallback)

The heavy models are deleted and GPU cache cleared immediately after H is solved.
"""
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Method 1: SuperPoint + LightGlue ────────────────────────────────────────

def compute_homography_superpoint_lightglue(
    cam_frame: np.ndarray,
    ref_image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Compute H using SuperPoint keypoints + LightGlue matching + RANSAC.

    Both images can be at wildly different perspectives — SuperPoint is
    view-invariant within reason; LightGlue handles the descriptor matching.

    Args:
        cam_frame:  BGR camera snapshot
        ref_image:  BGR reference image (satellite / floorplan / any flat view)

    Returns:
        (H, info_dict) — H is None on failure
    """
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch, rbd

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"SuperPoint+LightGlue using device: {device}")

        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        matcher   = LightGlue(features="superpoint").eval().to(device)

        def to_tensor(bgr: np.ndarray):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return numpy_image_to_torch(rgb).to(device)

        img0 = to_tensor(cam_frame)
        img1 = to_tensor(ref_image)

        with torch.no_grad():
            feats0   = extractor.extract(img0)
            feats1   = extractor.extract(img1)
            matches01 = matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        kpts0   = feats0["keypoints"].cpu().numpy()   # (N, 2)
        kpts1   = feats1["keypoints"].cpu().numpy()   # (M, 2)
        matches = matches01["matches"].cpu().numpy()  # (K, 2)

        # ── Explicitly free GPU resources ─────────────────────────────────────
        del extractor, matcher, feats0, feats1, matches01, img0, img1
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        if len(matches) < 8:
            return None, {
                "error": f"Too few matches: {len(matches)}",
                "method": "superpoint_lightglue",
            }

        matched0 = kpts0[matches[:, 0]]
        matched1 = kpts1[matches[:, 1]]

        H, mask = cv2.findHomography(matched0, matched1, cv2.RANSAC, ransacReprojThreshold=4.0)
        if H is None:
            return None, {"error": "RANSAC failed", "method": "superpoint_lightglue"}

        n_inliers = int(mask.sum())
        if n_inliers < 8:
            return None, {
                "error": f"Too few RANSAC inliers: {n_inliers}",
                "method": "superpoint_lightglue",
            }

        logger.info(
            f"[SuperPoint+LightGlue] H solved — matches={len(matches)}, inliers={n_inliers}"
        )
        return H, {
            "method":    "superpoint_lightglue",
            "n_matches": len(matches),
            "n_inliers": n_inliers,
            "device":    str(device),
        }

    except ImportError:
        logger.warning("LightGlue not installed — falling back to ORB")
        return compute_homography_orb(cam_frame, ref_image)

    except Exception as exc:
        logger.error(f"SuperPoint+LightGlue error: {exc}")
        return None, {"error": str(exc), "method": "superpoint_lightglue"}


# ─── Method 2: ORB + BFMatcher (CPU fallback) ────────────────────────────────

def compute_homography_orb(
    cam_frame: np.ndarray,
    ref_image: np.ndarray,
    n_features: int = 4000,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Fallback: classical ORB feature detection + brute-force matching + RANSAC.
    No GPU or torch dependency. Works well for scenes with rich texture.
    May struggle with extreme perspective changes (aerial vs. ground).
    """
    orb = cv2.ORB_create(nfeatures=n_features)

    gray0 = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

    kp0, des0 = orb.detectAndCompute(gray0, None)
    kp1, des1 = orb.detectAndCompute(gray1, None)

    if des0 is None or des1 is None:
        return None, {"error": "ORB found no descriptors", "method": "orb"}
    if len(kp0) < 8 or len(kp1) < 8:
        return None, {
            "error": f"Too few keypoints: cam={len(kp0)}, ref={len(kp1)}",
            "method": "orb",
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des0, des1), key=lambda m: m.distance)
    good = matches[: min(200, len(matches))]

    if len(good) < 8:
        return None, {"error": f"Too few ORB matches: {len(good)}", "method": "orb"}

    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
    if H is None:
        return None, {"error": "RANSAC failed (ORB)", "method": "orb"}

    n_inliers = int(mask.sum()) if mask is not None else 0
    if n_inliers < 8:
        return None, {
            "error": f"Too few RANSAC inliers: {n_inliers}",
            "method": "orb",
        }

    logger.info(f"[ORB] H solved — matches={len(good)}, inliers={n_inliers}")
    return H, {"method": "orb", "n_matches": len(good), "n_inliers": n_inliers}


# ─── Method 3: Manual point pairs ────────────────────────────────────────────

def compute_homography_manual(
    cam_pts: List[List[float]],
    ref_pts: List[List[float]],
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Compute H from ≥ 4 user-provided point correspondences.

    Args:
        cam_pts: [[u, v], ...] pixel coords clicked on the camera frame
        ref_pts: [[x, y], ...] pixel coords clicked on the reference image

    Returns:
        (H, info_dict)
    """
    n = len(cam_pts)
    if n < 4 or len(ref_pts) != n:
        return None, {"error": "Need at least 4 matching pairs (cam_pts == ref_pts length)"}

    pts0 = np.float32(cam_pts)
    pts1 = np.float32(ref_pts)

    if n == 4:
        # Exact solution (no RANSAC needed with exactly 4 points)
        H = cv2.getPerspectiveTransform(pts0, pts1)
        logger.info(f"[Manual] H from exactly 4 points (exact)")
        return H, {"method": "manual", "n_pairs": n, "n_inliers": 4}

    # With > 4 points, use RANSAC for robustness
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
    if H is None:
        return None, {"error": "RANSAC failed on manual points", "method": "manual"}

    n_inliers = int(mask.sum()) if mask is not None else 0
    logger.info(f"[Manual] H solved — pairs={n}, inliers={n_inliers}")
    return H, {"method": "manual", "n_pairs": n, "n_inliers": n_inliers}
