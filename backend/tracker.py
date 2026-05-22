"""
tracker.py — Geospatial Kalman Filter multi-object tracker.

State vector: [x, y, vx, vy] in reference-image pixel coordinates.
Association: Hungarian algorithm (scipy.optimize.linear_sum_assignment) on
             Euclidean distance between predicted track positions and new detections.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track:
    """Single object track with Kalman filter."""

    _id_counter: int = 0

    def __init__(self, ref_x: float, ref_y: float, class_name: str, confidence: float):
        Track._id_counter += 1
        self.id: int = Track._id_counter
        self.class_name: str = class_name
        self.confidence: float = confidence
        self.last_seen: float = time.monotonic()
        self.age: int = 0       # frames since creation (increments on predict)
        self.hits: int = 1      # matched detections count
        self.misses: int = 0    # consecutive unmatched frames

        # Trail of smoothed positions — kept for visualisation
        self.trail: List[Tuple[float, float]] = [(ref_x, ref_y)]
        self.max_trail_len: int = 40

        # ── Kalman filter (constant-velocity model) ──────────────────────────
        # State: [x, y, vx, vy], Measurement: [x, y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # Transition: x_{k} = F * x_{k-1}
        self.kf.F = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float64
        )
        # Observation: z = H * x
        self.kf.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float64
        )
        self.kf.R *= 20.0          # Measurement noise (higher → trust prediction more)
        self.kf.Q[2:, 2:] *= 0.5  # Process noise on velocity
        self.kf.P[2:, 2:] *= 200  # Initial uncertainty on velocity
        self.kf.x[:2] = [[ref_x], [ref_y]]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self.kf.x[2, 0]), float(self.kf.x[3, 0])

    @property
    def covariance_radius(self) -> float:
        """Scalar uncertainty estimate from position covariance diagonal."""
        px = float(self.kf.P[0, 0])
        py = float(self.kf.P[1, 1])
        return float(np.sqrt((px + py) / 2))

    # ── Kalman steps ──────────────────────────────────────────────────────────

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.misses += 1

    def update(self, ref_x: float, ref_y: float, class_name: str, confidence: float) -> None:
        self.kf.update(np.array([[ref_x], [ref_y]]))
        self.class_name = class_name
        self.confidence = confidence
        self.hits += 1
        self.misses = 0
        self.last_seen = time.monotonic()

        x, y = self.position
        self.trail.append((x, y))
        if len(self.trail) > self.max_trail_len:
            self.trail.pop(0)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        x, y = self.position
        vx, vy = self.velocity
        return {
            "id":         self.id,
            "class":      self.class_name,
            "confidence": round(self.confidence, 3),
            "ref_x":      round(x, 2),
            "ref_y":      round(y, 2),
            "vx":         round(vx, 3),
            "vy":         round(vy, 3),
            "cov_radius": round(self.covariance_radius, 2),
            "age":        self.age,
            "hits":       self.hits,
            "trail":      [(round(p[0], 1), round(p[1], 1)) for p in self.trail],
        }


class MultiTracker:
    """
    Manages a pool of Track instances.

    On each call to `update(detections)`:
    1. All tracks are predicted forward one step.
    2. Stale tracks (no match for > max_age_sec seconds) are removed.
    3. Hungarian matching assigns new detections to existing tracks.
    4. Unmatched detections spawn new tracks.
    5. Returns serialized list of confirmed tracks (hits >= min_hits).
    """

    def __init__(
        self,
        max_age_sec: float = 2.0,
        max_dist: float = 150.0,
        min_hits: int = 2,
    ):
        self.tracks: List[Track] = []
        self.max_age_sec = max_age_sec
        self.max_dist = max_dist
        self.min_hits = min_hits

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Args:
            detections: list of dicts with keys:
                        ref_x, ref_y, class_name, confidence

        Returns:
            list of confirmed track dicts (serialized)
        """
        # 1. Predict all tracks
        for t in self.tracks:
            t.predict()

        # 2. Remove stale tracks
        now = time.monotonic()
        self.tracks = [t for t in self.tracks if (now - t.last_seen) < self.max_age_sec]

        if not detections:
            return [t.to_dict() for t in self.tracks if t.hits >= self.min_hits]

        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(d["ref_x"], d["ref_y"], d["class_name"], d["confidence"]))
            return [t.to_dict() for t in self.tracks if t.hits >= self.min_hits]

        # 3. Cost matrix: Euclidean distance in reference-image space
        track_pos = np.array([t.position for t in self.tracks])   # (T, 2)
        det_pos   = np.array([(d["ref_x"], d["ref_y"]) for d in detections])  # (D, 2)
        cost = np.linalg.norm(track_pos[:, None] - det_pos[None, :], axis=2)  # (T, D)

        # 4. Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost)

        matched_tracks: set = set()
        matched_dets:   set = set()

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < self.max_dist:
                self.tracks[r].update(
                    detections[c]["ref_x"], detections[c]["ref_y"],
                    detections[c]["class_name"], detections[c]["confidence"],
                )
                matched_tracks.add(r)
                matched_dets.add(c)

        # 5. Spawn new tracks for unmatched detections
        for i, d in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(Track(d["ref_x"], d["ref_y"], d["class_name"], d["confidence"]))

        return [t.to_dict() for t in self.tracks if t.hits >= self.min_hits]

    def reset(self) -> None:
        self.tracks.clear()
        Track._id_counter = 0
