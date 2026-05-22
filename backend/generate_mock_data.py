#!/usr/bin/env python3
"""
generate_mock_data.py — Self-contained generator for BirdEye mock assets.

Produces:
  uploads/mock_reference.jpg   — synthetic top-down airfield / floor map
  uploads/mock_camera.mp4      — synthetic camera feed with moving vehicles
  uploads/mock_homography.npy  — ground-truth 3×3 H matrix (camera → reference)

Vehicles are drawn with OpenCV primitives in a perspective-view billboard
style, sized and positioned to be detectable by YOLOv8n as car / truck.
No external image files required.
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR  = BACKEND_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ── Synthetic vehicle sprite generator ───────────────────────────────────────

def _make_vehicle_sprite(
    body_bgr: tuple,
    width: int = 180,
    height: int = 90,
    vehicle_type: str = "car",   # "car" | "suv" | "truck"
) -> np.ndarray:
    """
    Draw a perspective-view vehicle sprite (BGRA).

    The sprite represents a vehicle seen from a slightly elevated, oblique
    angle — the way a roadside camera would see it passing by.
    Bottom-centre of the sprite is the ground-contact point.
    """
    w, h = width, height
    img = np.zeros((h, w, 4), dtype=np.uint8)

    def poly(pts, color_bgr, alpha=255):
        pts_np = np.array(pts, dtype=np.int32)
        bgr = np.ascontiguousarray(img[:, :, :3])
        cv2.fillPoly(bgr, [pts_np], color_bgr)
        img[:, :, :3] = bgr
        a = np.ascontiguousarray(img[:, :, 3])
        cv2.fillPoly(a, [pts_np], alpha)
        img[:, :, 3] = a

    def darker(c, amount=50):
        return tuple(max(0, v - amount) for v in c)

    def lighter(c, amount=60):
        return tuple(min(255, v + amount) for v in c)

    # ── Body silhouette ───────────────────────────────────────────────────────
    # Trapezoidal body — wider at bottom (viewer perspective)
    cabin_top = 0.30 if vehicle_type == "truck" else 0.22
    body = [
        (0,        h),
        (w,        h),
        (int(w * 0.92), int(h * 0.55)),
        (int(w * 0.08), int(h * 0.55)),
    ]
    poly(body, body_bgr)

    # Hood / bonnet (front, right side in sprite)
    hood = [
        (int(w * 0.62), int(h * 0.55)),
        (int(w * 0.92), int(h * 0.55)),
        (int(w * 0.95), int(h * 0.72)),
        (int(w * 0.62), int(h * 0.72)),
    ]
    poly(hood, darker(body_bgr, 25))

    # ── Cabin / roof ──────────────────────────────────────────────────────────
    cabin_bottom_y = int(h * 0.55)
    cabin_top_y    = int(h * cabin_top)
    cabin_l_bot = int(w * 0.18)
    cabin_r_bot = int(w * 0.65)
    cabin_l_top = int(w * 0.22) if vehicle_type == "truck" else int(w * 0.24)
    cabin_r_top = int(w * 0.60)

    cabin = [
        (cabin_l_bot, cabin_bottom_y),
        (cabin_r_bot, cabin_bottom_y),
        (cabin_r_top, cabin_top_y),
        (cabin_l_top, cabin_top_y),
    ]
    poly(cabin, darker(body_bgr, 40))

    # ── Windows ───────────────────────────────────────────────────────────────
    win_color = (200, 215, 235)  # pale blue-grey (BGR)

    # Windshield (front)
    wshield = [
        (int(w * 0.54), cabin_bottom_y - 2),
        (int(w * 0.63), cabin_bottom_y - 2),
        (int(w * 0.58), cabin_top_y    + 4),
        (int(w * 0.50), cabin_top_y    + 4),
    ]
    poly(wshield, win_color, 230)

    # Side windows
    side_win = [
        (cabin_l_bot + 4, cabin_bottom_y - 2),
        (int(w * 0.52),   cabin_bottom_y - 2),
        (int(w * 0.48),   cabin_top_y    + 4),
        (cabin_l_top + 4, cabin_top_y    + 4),
    ]
    poly(side_win, win_color, 210)

    # ── Wheels ────────────────────────────────────────────────────────────────
    wheel_bgr   = (25, 25, 25)
    rim_bgr     = (120, 120, 130)
    wheel_ry    = int(h * 0.10)
    wheel_rx    = int(w * 0.085)
    for cx in (int(w * 0.22), int(w * 0.78)):
        cy = h - 1
        bgr = np.ascontiguousarray(img[:, :, :3])
        cv2.ellipse(bgr, (cx, cy), (wheel_rx, wheel_ry), 0, 0, 360, wheel_bgr, -1)
        cv2.ellipse(bgr, (cx, cy), (wheel_rx // 2, wheel_ry // 2), 0, 0, 360, rim_bgr, -1)
        img[:, :, :3] = bgr
        a = np.ascontiguousarray(img[:, :, 3])
        cv2.ellipse(a, (cx, cy), (wheel_rx, wheel_ry), 0, 0, 360, 255, -1)
        img[:, :, 3] = a

    # ── Headlight (front) ─────────────────────────────────────────────────────
    hl_cx = int(w * 0.88)
    hl_cy = int(h * 0.68)
    bgr = np.ascontiguousarray(img[:, :, :3])
    cv2.circle(bgr, (hl_cx, hl_cy), int(w * 0.025), (220, 235, 255), -1)
    img[:, :, :3] = bgr

    # For trucks: add a cargo bed
    if vehicle_type == "truck":
        bed = [
            (int(w * 0.05), int(h * 0.55)),
            (int(w * 0.18), int(h * 0.55)),
            (int(w * 0.18), int(h * 0.25)),
            (int(w * 0.05), int(h * 0.25)),
        ]
        poly(bed, darker(body_bgr, 20))
        bed_inner = [
            (int(w * 0.07), int(h * 0.53)),
            (int(w * 0.16), int(h * 0.53)),
            (int(w * 0.16), int(h * 0.28)),
            (int(w * 0.07), int(h * 0.28)),
        ]
        poly(bed_inner, darker(body_bgr, 60))

    # For SUVs: taller cabin
    if vehicle_type == "suv":
        roof_rack = [
            (cabin_l_top + 8, cabin_top_y),
            (cabin_r_top - 8, cabin_top_y),
            (cabin_r_top - 8, cabin_top_y - int(h * 0.04)),
            (cabin_l_top + 8, cabin_top_y - int(h * 0.04)),
        ]
        poly(roof_rack, lighter(body_bgr, 30))

    return img


def _overlay_sprite(
    frame: np.ndarray,
    sprite: np.ndarray,
    u: float,
    v: float,
    scale: float = 1.0,
    flip_h: bool = False,
) -> None:
    """
    Alpha-composite a sprite onto frame so that its bottom-centre
    lands at (u, v) in camera pixel coordinates.
    """
    s = sprite
    if flip_h:
        s = cv2.flip(s, 1)

    sh, sw = s.shape[:2]
    tw = max(10, int(sw * scale))
    th = max(10, int(sh * scale))
    s = cv2.resize(s, (tw, th), interpolation=cv2.INTER_AREA)

    u1 = int(u - tw / 2)
    v1 = int(v - th)
    u2 = u1 + tw
    v2 = v1 + th

    fh, fw = frame.shape[:2]
    if u1 >= fw or v1 >= fh or u2 <= 0 or v2 <= 0:
        return

    su1, sv1 = max(0, -u1), max(0, -v1)
    su2, sv2 = tw - max(0, u2 - fw), th - max(0, v2 - fh)
    u1, v1 = max(0, u1), max(0, v1)
    u2, v2 = min(fw, u2), min(fh, v2)

    crop   = s[sv1:sv2, su1:su2]
    bg     = frame[v1:v2, u1:u2].astype(np.float32)
    alpha  = (crop[:, :, 3] / 255.0)[:, :, np.newaxis]
    blended = crop[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    frame[v1:v2, u1:u2] = np.clip(blended, 0, 255).astype(np.uint8)


# ── Reference map ─────────────────────────────────────────────────────────────

def generate_reference_map(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Synthetic bird's-eye airport/facility map with calibration targets."""
    img = np.full((height, width, 3), (28, 30, 35), dtype=np.uint8)

    # Grid
    for x in range(0, width, 100):
        cv2.line(img, (x, 0), (x, height), (38, 40, 46), 1)
    for y in range(0, height, 100):
        cv2.line(img, (0, y), (width, y), (38, 40, 46), 1)

    # Tarmac runway band
    cv2.rectangle(img, (100, 440), (1820, 640), (22, 22, 22), -1)
    cv2.line(img, (100, 440), (1820, 440), (0, 200, 220), 2)
    cv2.line(img, (100, 640), (1820, 640), (0, 200, 220), 2)
    for x in range(120, 1800, 60):
        cv2.line(img, (x, 540), (x + 30, 540), (210, 210, 210), 2)
    cv2.putText(img, "RUNWAY 09-27", (220, 495),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 220), 2, cv2.LINE_AA)
    cv2.putText(img, "09L", (140, 555),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 3, cv2.LINE_AA)

    # Taxiway
    cv2.rectangle(img, (900, 640), (1100, 900), (30, 30, 30), -1)
    cv2.line(img, (900, 640), (900, 900), (0, 160, 180), 1)
    cv2.line(img, (1100, 640), (1100, 900), (0, 160, 180), 1)

    # Apron area
    cv2.rectangle(img, (100, 100), (700, 380), (35, 35, 42), -1)
    cv2.rectangle(img, (100, 100), (700, 380), (50, 55, 70), 1)
    cv2.putText(img, "APRON", (350, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 80, 100), 1, cv2.LINE_AA)

    # Calibration targets  A B C D
    targets = {
        "A": ((400,  300), (50, 50, 220)),   # red-ish
        "B": ((1400, 250), (50, 200, 50)),   # green
        "C": ((1500, 850), (220, 50, 50)),   # blue
        "D": ((350,  800), (50, 150, 240)),  # orange
    }
    for label, (centre, color) in targets.items():
        cv2.circle(img, centre, 32, color, -1)
        cv2.circle(img, centre, 32, (255, 255, 255), 2)
        cv2.putText(img, label,
                    (centre[0] - 10, centre[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    return img, targets


# ── Ground-truth homography ───────────────────────────────────────────────────

def compute_ground_truth_H(targets: dict, cam_w: int, cam_h: int):
    """
    Define 4 camera-space positions that correspond to the 4 coloured circles
    on the reference map, then compute the exact 4-point homography.
    """
    cam_pts = np.float32([
        [250.0, 180.0],   # A
        [1030.0, 150.0],  # B
        [1120.0, 600.0],  # C
        [180.0,  580.0],  # D
    ])
    ref_pts = np.float32([targets[k][0] for k in ("A", "B", "C", "D")])
    H = cv2.getPerspectiveTransform(cam_pts, ref_pts)
    return H, cam_pts, ref_pts


# ── Motion paths (in reference-image coordinates) ─────────────────────────────

def _red_sedan_path(t: float):
    """Left-to-right on runway, t ∈ [0, 1]."""
    x = 150 + (1750 - 150) * t
    y = 520 + 8 * math.sin(t * 4 * math.pi)
    return x, y, 0.68, True    # (ref_x, ref_y, base_scale, flip_h)


def _blue_suv_path(t: float):
    """Top-right → bottom-left diagonal, t ∈ [0, 1]."""
    x = 1600 + (300  - 1600) * t
    y = 350  + (850  - 350)  * t
    return x, y, 0.72, False


def _white_pickup_path(t: float, total_frames: int, frame_idx: int):
    """Right-to-left on runway, starts after 2 s."""
    delay_frames = 30
    if frame_idx < delay_frames:
        return -999, -999, 0, False
    t2 = (frame_idx - delay_frames) / max(1, total_frames - delay_frames)
    x = 1700 + (300 - 1700) * t2
    y = 560.0
    return x, y, 0.78, False


# ── Main generator ────────────────────────────────────────────────────────────

def main():
    # ── Sprites ───────────────────────────────────────────────────────────────
    print("Generating vehicle sprites…")
    red_sedan   = _make_vehicle_sprite((30, 40, 200), width=190, height=95, vehicle_type="car")
    blue_suv    = _make_vehicle_sprite((180, 80, 30), width=200, height=105, vehicle_type="suv")
    white_pickup = _make_vehicle_sprite((230, 230, 225), width=220, height=95, vehicle_type="truck")

    # ── Reference map ─────────────────────────────────────────────────────────
    print("Generating reference map…")
    ref_img, targets = generate_reference_map(1920, 1080)
    mock_ref_path = UPLOAD_DIR / "mock_reference.jpg"
    cv2.imwrite(str(mock_ref_path), ref_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  → {mock_ref_path}")

    # ── Homography ────────────────────────────────────────────────────────────
    cam_w, cam_h = 1280, 720
    H_gt, cam_pts, ref_pts = compute_ground_truth_H(targets, cam_w, cam_h)
    H_inv = np.linalg.inv(H_gt)
    np.save(str(UPLOAD_DIR / "mock_homography.npy"), H_gt)
    print(f"  → {UPLOAD_DIR / 'mock_homography.npy'}")

    # ── Video ─────────────────────────────────────────────────────────────────
    fps           = 15
    duration_sec  = 20
    total_frames  = fps * duration_sec

    mock_video_path = UPLOAD_DIR / "mock_camera.mp4"
    writer = None
    for codec_str, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")]:
        try:
            fourcc     = cv2.VideoWriter_fourcc(*codec_str)
            test_path  = str(UPLOAD_DIR / f"_test{ext}")
            tw = cv2.VideoWriter(test_path, fourcc, fps, (cam_w, cam_h))
            if tw.isOpened():
                tw.release()
                os.remove(test_path)
                mock_video_path = UPLOAD_DIR / f"mock_camera{ext}"
                writer = cv2.VideoWriter(str(mock_video_path), fourcc, fps, (cam_w, cam_h))
                print(f"  Codec: {codec_str}{ext}")
                break
        except Exception as exc:
            print(f"  Codec {codec_str} failed: {exc}")

    if writer is None or not writer.isOpened():
        raise RuntimeError("No usable video codec found.")

    print(f"Rendering {total_frames} frames…")
    for f_idx in range(total_frames):
        t = f_idx / total_frames

        # Background: warp reference to camera view
        frame = cv2.warpPerspective(ref_img, H_inv, (cam_w, cam_h))

        def _place(sprite, rx, ry, base_scale, flip_h):
            if rx < -500:
                return
            pt  = np.array([[[rx, ry]]], dtype=np.float32)
            cam = cv2.perspectiveTransform(pt, H_inv)[0, 0]
            cu, cv_ = float(cam[0]), float(cam[1])
            # Perspective scale: objects lower in frame are larger
            scale = float(np.clip(base_scale * (cv_ / 520.0), 0.28, 0.95))
            if cv_ > 0:
                _overlay_sprite(frame, sprite, cu, cv_, scale, flip_h)

        rx, ry, bs, fh = _red_sedan_path(t)
        _place(red_sedan, rx, ry, bs, fh)

        rx, ry, bs, fh = _blue_suv_path(t)
        _place(blue_suv, rx, ry, bs, fh)

        rx, ry, bs, fh = _white_pickup_path(t, total_frames, f_idx)
        _place(white_pickup, rx, ry, bs, fh)

        # Subtle sensor noise
        noise = np.random.normal(0, 1.5, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

        if (f_idx + 1) % 50 == 0 or f_idx == total_frames - 1:
            print(f"  Frame {f_idx + 1}/{total_frames}")

    writer.release()
    print(f"\nDone! Generated:")
    print(f"  Reference image: {mock_ref_path}")
    print(f"  Camera video:    {mock_video_path}")
    print(f"  Homography:      {UPLOAD_DIR / 'mock_homography.npy'}")


if __name__ == "__main__":
    main()
