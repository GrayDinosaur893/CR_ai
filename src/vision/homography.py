"""
Homography & Perspective Transformation for Clash Royale Arena.
Maps raw screen (x, y) coordinates to the discrete 18x32 tile grid.
"""

from typing import Tuple, Dict, Any

class ArenaTileProjector:
    """
    Clash Royale Arena Grid:
    - Width: 18 tiles (columns 0 to 17, from left bridge to right bridge)
    - Height: 32 tiles (rows 0 to 31, 0 = Opponent back, 31 = Player back)
    - River spans rows 15-16 (player bridge edge row = 16).
    - Player Princess Towers sit at col=3 (left) and col=14 (right), row=24.
    """
    GRID_COLS = 18
    GRID_ROWS = 32
    RIVER_CROSS_ROW = 16
    LEFT_TOWER_COL = 3
    RIGHT_TOWER_COL = 14
    TOWER_ROW = 24

    def __init__(self, arena_bounds_normalized: Dict[str, float] = None):
        bounds = arena_bounds_normalized or {
            "ymin": 0.10,
            "xmin": 0.05,
            "ymax": 0.80,
            "xmax": 0.95
        }
        self.ymin = bounds["ymin"]
        self.xmin = bounds["xmin"]
        self.ymax = bounds["ymax"]
        self.xmax = bounds["xmax"]

    def pixel_norm_to_tile(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """
        Convert normalized screen coordinates (0.0 to 1.0) to arena tile (col, row).
        Clamps to valid arena bounds [0, 17] and [0, 31].
        """
        # Linear approximation (perspective warp can be refined with 4-point homography matrix)
        clamped_x = max(self.xmin, min(self.xmax, norm_x))
        clamped_y = max(self.ymin, min(self.ymax, norm_y))

        rel_x = (clamped_x - self.xmin) / (self.xmax - self.xmin)
        rel_y = (clamped_y - self.ymin) / (self.ymax - self.ymin)

        col = int(rel_x * self.GRID_COLS)
        row = int(rel_y * self.GRID_ROWS)

        col = min(self.GRID_COLS - 1, max(0, col))
        row = min(self.GRID_ROWS - 1, max(0, row))
        return (col, row)

    def tile_to_norm(self, tile_col: int, tile_row: int) -> Tuple[float, float]:
        """Convert arena tile (col, row) back to center viewport-normalized (x, y)."""
        col = min(self.GRID_COLS - 1, max(0, int(tile_col)))
        row = min(self.GRID_ROWS - 1, max(0, int(tile_row)))
        norm_x = self.xmin + ((col + 0.5) / self.GRID_COLS) * (self.xmax - self.xmin)
        norm_y = self.ymin + ((row + 0.5) / self.GRID_ROWS) * (self.ymax - self.ymin)
        return (round(norm_x, 4), round(norm_y, 4))

    def get_lane(self, tile_col: int) -> str:
        """
        Determines if a tile falls in left or right lane.
        """
        return "left" if tile_col < (self.GRID_COLS // 2) else "right"

    def tile_to_relative(self, tile_col: int, tile_row: int, lane: str = "") -> Tuple[int, int]:
        """
        Convert absolute (col, row) to river/tower-relative (dy_river, dx_tower):
        - dy_river: tiles below river bridge edge (row - 16); positive = deeper in our half.
        - dx_tower: inward horizontal offset from that lane's Princess Tower; positive = toward center.
        """
        use_lane = lane if lane in ("left", "right") else self.get_lane(tile_col)
        dy_river = int(tile_row) - self.RIVER_CROSS_ROW
        if use_lane == "left":
            dx_tower = int(tile_col) - self.LEFT_TOWER_COL
        else:
            dx_tower = self.RIGHT_TOWER_COL - int(tile_col)
        return (dy_river, dx_tower)

    def relative_to_tile(self, dy_river: int, dx_tower: int, lane: str = "left") -> Tuple[int, int]:
        """Convert river/tower-relative (dy_river, dx_tower) back to clamped arena (col, row)."""
        row = self.RIVER_CROSS_ROW + int(dy_river)
        if lane == "left":
            col = self.LEFT_TOWER_COL + int(dx_tower)
        else:
            col = self.RIGHT_TOWER_COL - int(dx_tower)
        col = min(self.GRID_COLS - 1, max(0, col))
        row = min(self.GRID_ROWS - 1, max(0, row))
        return (col, row)
