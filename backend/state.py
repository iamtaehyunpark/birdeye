"""
state.py — Shared in-memory system state.

All modules import the singleton `state` object.
Access protected by a threading.Lock for thread safety.
"""
import threading
from typing import Optional

import numpy as np


class SystemState:
    """Singleton holding the live system state."""

    def __init__(self):
        self.lock = threading.Lock()

        # Homography: 3×3 float64 numpy array, or None if not yet computed
        self.H: Optional[np.ndarray] = None

        # Init pipeline status
        self.init_status: str = "idle"  # idle | running | done | error
        self.init_error: Optional[str] = None
        self.init_info: Optional[dict] = None  # method, n_matches, n_inliers, etc.

        # Reference image
        self.ref_image_path: Optional[str] = None
        self.ref_image_size: tuple = (0, 0)  # (width, height)


# Module-level singleton — import this everywhere
state = SystemState()
