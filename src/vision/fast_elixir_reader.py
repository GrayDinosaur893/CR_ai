"""
FAST LOCAL ELIXIR READER (< 3ms latency)
Extracts elixir value directly from calibrated pixel ROI without using any VLM.

Method:
1. Crop physical bar ROI (180, 1060, 636, 1090) -> 456px total width = 45.6px / elixir.
2. Filter for purple/magenta elixir color in HSV space (Hue: 125-170, Sat: >80, Val: >70).
3. Measure horizontal column fill percentage.
4. Returns elixir float (0.0 to 10.0) with execution time in milliseconds.
"""

import time
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

class FastElixirReader:
    def __init__(
        self,
        bar_roi: Tuple[int, int, int, int] = (180, 1060, 636, 1090),
        droplet_roi: Tuple[int, int, int, int] = (130, 1055, 175, 1095)
    ):
        self.bar_roi = bar_roi # [x1, y1, x2, y2]
        self.droplet_roi = droplet_roi
        self.bar_width = bar_roi[2] - bar_roi[0] # 456 px
        self.px_per_elixir = self.bar_width / 10.0 # 45.6 px/elixir

        # Magenta / Purple elixir color bounds in OpenCV HSV
        self.lower_purple = np.array([125, 75, 70], dtype=np.uint8)
        self.upper_purple = np.array([170, 255, 255], dtype=np.uint8)

    def read_elixir(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Reads elixir level from full frame or pre-cropped bar.
        Returns {'elixir': float, 'latency_ms': float, 'fill_ratio': float}.
        """
        t0 = time.perf_counter()
        x1, y1, x2, y2 = self.bar_roi
        h, w = frame.shape[:2]

        # Bounds check
        if y2 > h or x2 > w:
            return {"elixir": 0.0, "latency_ms": 0.0, "error": "Frame smaller than ROI"}

        bar_crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_purple, self.upper_purple)

        # Compute column-wise purple presence (if >30% of vertical column is purple -> filled)
        col_purple_ratio = np.mean(mask > 0, axis=0) # shape: (width,)
        filled_cols = np.sum(col_purple_ratio > 0.3)

        elixir_float = round(float(filled_cols) / self.px_per_elixir, 2)
        elixir_float = max(0.0, min(10.0, elixir_float))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "elixir": elixir_float,
            "filled_pixels": int(filled_cols),
            "total_bar_pixels": self.bar_width,
            "latency_ms": round(latency_ms, 3)
        }
