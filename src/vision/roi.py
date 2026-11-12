"""
ROI (Region of Interest) Slicer for Clash Royale screens.
Supports common resolutions (1080x1920, 720x1280, etc.) and normalizes
bounding boxes across arena, player hand, elixir bar, and match timer.
"""

from typing import Dict, Any, Tuple

class ClashRoyaleROI:
    """
    Standard normalized bounding boxes [ymin, xmin, ymax, xmax] (0.0 to 1.0)
    relative to standard vertical mobile game screen.
    """
    NORMALIZED_REGIONS = {
        # Arena playing field where troops and buildings appear
        "arena": {
            "ymin": 0.08,
            "xmin": 0.02,
            "ymax": 0.82,
            "xmax": 0.98
        },
        # Match timer in the top area
        "timer": {
            "ymin": 0.02,
            "xmin": 0.65,
            "ymax": 0.07,
            "xmax": 0.95
        },
        # Left and Right Opponent Princess Towers
        "opponent_princess_left": {
            "ymin": 0.15,
            "xmin": 0.15,
            "ymax": 0.28,
            "xmax": 0.35
        },
        "opponent_princess_right": {
            "ymin": 0.15,
            "xmin": 0.65,
            "ymax": 0.28,
            "xmax": 0.85
        },
        # Player bottom hand cards (4 cards + next slot)
        "player_hand": {
            "ymin": 0.84,
            "xmin": 0.20,
            "ymax": 0.98,
            "xmax": 0.96
        },
        # Next card in queue (left of hand)
        "next_card_slot": {
            "ymin": 0.87,
            "xmin": 0.04,
            "ymax": 0.98,
            "xmax": 0.18
        },
        # Bottom elixir bar fill and counter
        "elixir_bar": {
            "ymin": 0.96,
            "xmin": 0.20,
            "ymax": 0.995,
            "xmax": 0.96
        }
    }

    def __init__(self, frame_width: int = 1080, frame_height: int = 1920):
        self.width = frame_width
        self.height = frame_height

    def get_pixel_crop(self, region_name: str) -> Tuple[int, int, int, int]:
        """
        Returns (ymin_px, xmin_px, ymax_px, xmax_px) for slicing numpy frame:
        cropped = frame[ymin:ymax, xmin:xmax]
        """
        if region_name not in self.NORMALIZED_REGIONS:
            raise ValueError(f"Unknown region '{region_name}'. Available: {list(self.NORMALIZED_REGIONS.keys())}")
        
        box = self.NORMALIZED_REGIONS[region_name]
        ymin = int(box["ymin"] * self.height)
        xmin = int(box["xmin"] * self.width)
        ymax = int(box["ymax"] * self.height)
        xmax = int(box["xmax"] * self.width)
        return (ymin, xmin, ymax, xmax)

    def get_all_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        return {name: self.get_pixel_crop(name) for name in self.NORMALIZED_REGIONS}
