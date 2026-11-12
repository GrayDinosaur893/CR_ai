"""
CLASH IQ — Viewport-Aware Clash Royale Component & Arena Perception Engine.
Extracts, verifies, and visually annotates all core Clash Royale components:
1. Active Game Viewport (strips GPG left sidebar & titlebar automatically)
2. Screen State Classification (battle_arena, lobby, gpg_idle, non_game)
3. Arena Grid (18x32 tiles via ArenaTileProjector, River, Left/Right Bridges)
4. King & Princess Towers (Alive vs Destroyed rubble detection)
5. Match Status & Timer
6. Elixir Bar & Droplet Region (dynamically calibrated to viewport width)
7. Player Hand Slots (Slots 1-4) + Next Card Queue + Playable/Cooldown State
8. Fast OpenCV Enemy & Ally Arena Entity / Threat Detection
"""

import os
import sys
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

# Prevent Windows OpenCV thread-pool wakeup jitter across 15 Hz sleep intervals
try:
    cv2.setNumThreads(2)
except Exception:
    pass

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.vision.roi import ClashRoyaleROI
from src.vision.homography import ArenaTileProjector
from src.vision.card_matcher import CardTemplateMatcher


class ComponentDetector:
    # Viewport-relative definitions [ymin, xmin, ymax, xmax] inside the active game viewport.
    # For bottom HUD items, vertical coordinates refine against the detected blue HUD strip.
    COMPONENTS = {
        "match_timer": {
            "name": "Match Timer & Score",
            "box": (0.012, 0.78, 0.115, 0.99),
            "color": (255, 200, 0),
            "category": "HUD",
        },
        "opponent_king_tower": {
            "name": "Opponent King Tower",
            "box": (0.06, 0.39, 0.20, 0.61),
            "color": (0, 0, 255),
            "category": "Tower",
        },
        "opponent_princess_left": {
            "name": "Opponent Princess (Left)",
            "box": (0.13, 0.14, 0.28, 0.36),
            "color": (50, 50, 255),
            "category": "Tower",
        },
        "opponent_princess_right": {
            "name": "Opponent Princess (Right)",
            "box": (0.13, 0.64, 0.28, 0.86),
            "color": (50, 50, 255),
            "category": "Tower",
        },
        "river_and_bridges": {
            "name": "River & Bridges (Row 15-16)",
            "box": (0.41, 0.06, 0.51, 0.94),
            "color": (255, 255, 0),
            "category": "Arena",
        },
        "player_princess_left": {
            "name": "Player Princess (Left)",
            "box": (0.59, 0.14, 0.73, 0.36),
            "color": (255, 120, 0),
            "category": "Tower",
        },
        "player_princess_right": {
            "name": "Player Princess (Right)",
            "box": (0.59, 0.64, 0.73, 0.86),
            "color": (255, 120, 0),
            "category": "Tower",
        },
        "player_king_tower": {
            "name": "Player King Tower",
            "box": (0.63, 0.39, 0.80, 0.61),
            "color": (255, 60, 0),
            "category": "Tower",
        },
        "next_card_slot": {
            "name": "Next Card Queue",
            "box": (0.865, 0.04, 0.975, 0.18),
            "color": (200, 200, 200),
            "category": "Hand",
        },
        "hand_slot_1": {
            "name": "Hand Slot 1",
            "box": (0.835, 0.22, 0.960, 0.40),
            "color": (0, 255, 120),
            "category": "Hand",
        },
        "hand_slot_2": {
            "name": "Hand Slot 2",
            "box": (0.835, 0.41, 0.960, 0.59),
            "color": (0, 255, 120),
            "category": "Hand",
        },
        "hand_slot_3": {
            "name": "Hand Slot 3",
            "box": (0.835, 0.60, 0.960, 0.78),
            "color": (0, 255, 120),
            "category": "Hand",
        },
        "hand_slot_4": {
            "name": "Hand Slot 4",
            "box": (0.835, 0.79, 0.960, 0.97),
            "color": (0, 255, 120),
            "category": "Hand",
        },
        "elixir_bar": {
            "name": "Elixir Bar (0-10)",
            "box": (0.955, 0.22, 0.996, 0.97),
            "color": (255, 0, 255),
            "category": "Elixir",
        },
    }

    def __init__(self, frame_width: int = 1080, frame_height: int = 1920):
        self.width = frame_width
        self.height = frame_height
        self.roi = ClashRoyaleROI(frame_width, frame_height)
        self.projector = ArenaTileProjector()
        self.card_matcher = CardTemplateMatcher()
        self._vp_cache_key: Optional[Tuple[int, int]] = None
        self._vp_cache_val: Optional[Dict[str, Any]] = None
        self._vp_cache_age: int = 0
        self._vp_cache_ttl: int = 120
        self._prev_arena_small: Optional[np.ndarray] = None
        self._prev_arena_entities: Dict[str, List[Dict[str, Any]]] = {"enemies": [], "allies": []}
        self._arena_static_skips: int = 0
        self._last_arena_changed: bool = True

    @staticmethod
    def get_source_fingerprint() -> str:
        """SHA-256 fingerprint of this detector file to tie benchmarks to code."""
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    def detect_game_viewport(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects active Clash Royale viewport inside the captured frame.
        Automatically strips Google Play Games PC left sidebar and top titlebar.
        Returns pixel box [vx1, vy1, vx2, vy2], normalized pct, and layout type.
        """
        h, w = frame.shape[:2]
        vx1, vy1, vx2, vy2 = 0, 0, w, h
        pillarboxed = False

        # 0. Detect fullscreen widescreen centered pillarbox (e.g. GPG fullscreen 1920x1200 with black left/right bars)
        if w > h * 0.90:
            left_edge = frame[:, :max(8, int(0.08 * w))]
            right_edge = frame[:, min(w - 8, int(0.92 * w)):]
            if left_edge.mean() < 20 and right_edge.mean() < 20:
                col_mid = frame[int(0.08 * h):int(0.92 * h), :, :].mean(axis=(0, 2))
                nb = np.where(col_mid > 20)[0]
                if len(nb) > 0:
                    cx1, cx2 = int(nb[0]), int(nb[-1]) + 1
                    if 0.25 * w <= (cx2 - cx1) <= 0.78 * w:
                        vx1, vx2 = cx1, cx2
                        pillarboxed = True

        # 1. Detect GPG left sidebar: uniform dark vertical rail followed by active game canvas
        if not pillarboxed and w >= 400:
            left_rail = frame[:, :max(8, int(0.08 * w))]
            if left_rail.mean() < 48 and left_rail.std() < 34:
                v_band = cv2.cvtColor(frame[int(0.15 * h):int(0.85 * h), :], cv2.COLOR_BGR2HSV)[:, :, 2]
                colm = v_band.mean(axis=0)
                bright_cols = np.where(colm > 65.0)[0]
                if len(bright_cols) > 0:
                    cand_x = int(bright_cols[0])
                    if int(0.05 * w) < cand_x < int(0.28 * w):
                        vx1 = cand_x
                if vx1 == 0:
                    col_mid = frame[int(0.08 * h):int(0.92 * h), :, :].mean(axis=(0, 2))
                    bright_cols2 = np.where(col_mid > 45)[0]
                    if len(bright_cols2) > 0:
                        cand_x = int(bright_cols2[0])
                        if int(0.05 * w) < cand_x < int(0.28 * w):
                            vx1 = cand_x

        if not pillarboxed and vx1 == 0:
            hud_band = frame[int(0.83 * h):int(0.94 * h), :, :].astype(np.int16)
            if hud_band.size > 0:
                b, g, r = hud_band[:, :, 0], hud_band[:, :, 1], hud_band[:, :, 2]
                blue_col = ((b - r > 45) & (b - g > 15) & (b > 95)).mean(axis=0)
                blue_hits = np.where(blue_col > 0.35)[0]
                if len(blue_hits) > 0 and int(0.05 * w) < int(blue_hits[0]) < int(0.28 * w):
                    vx1 = int(blue_hits[0])

        # 2. Detect GPG top window titlebar (dark uniform strip across active width)
        if vx1 > 0 and not pillarboxed:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for y in range(int(0.06 * h)):
                sl = gray[y, vx1:max(vx1 + 20, vx2 - 20)]
                if sl.size > 0 and sl.mean() < 48 and sl.std() < 35:
                    vy1 = y + 1
                else:
                    break

        # 3. Trim bottom black bar on 19.5:9 mobile frames if present
        vp_slice = frame[vy1:vy2, vx1:vx2]
        if vp_slice.size > 0:
            vh = vp_slice.shape[0]
            row_nonblack = vp_slice.mean(axis=(1, 2)) > 20
            nb = np.where(row_nonblack[int(0.92 * vh):])[0]
            if len(nb) > 0:
                last_nb = int(0.92 * vh) + int(nb[-1]) + 1
                if last_nb < vh - 4:
                    vy2 = vy1 + last_nb

        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        layout = (
            "gpg_fullscreen_pillarbox"
            if pillarboxed
            else ("gpg_windowed_sidebar" if vx1 > 0 else "full_viewport")
        )

        return {
            "box_px": [vx1, vy1, vx2, vy2],
            "box_pct": [round(vx1 / w, 4), round(vy1 / h, 4), round(vx2 / w, 4), round(vy2 / h, 4)],
            "width": vw,
            "height": vh,
            "layout": layout,
        }

    def _detect_hud_and_elixir(self, vp: np.ndarray) -> Dict[str, Any]:
        """Measures bottom royal-blue HUD bar top edge, magenta elixir fill, and battle HUD cues."""
        vh, vw = vp.shape[:2]
        search_start = int(0.76 * vh)
        search_end = int(0.92 * vh)
        band = vp[search_start:search_end, :]
        hud_detected = False
        y_hud_px = int(0.835 * vh)
        blue_hud_strength = 0.0
        blue_hud_peak = 0.0
        if band.size > 0:
            sub_band = band[::2, ::4] if (band.shape[0] >= 8 and band.shape[1] >= 16) else band
            b, g, r = sub_band[:, :, 0], sub_band[:, :, 1], sub_band[:, :, 2]
            # Royal blue card tray (B markedly above BOTH R and G so teal/cyan splash art never matches)
            blue_mask = (cv2.subtract(b, r) > 45) & (cv2.subtract(b, g) > 15) & (b > 95)
            row_blue = blue_mask.mean(axis=1)
            hud_rows = np.where(row_blue > 0.38)[0]
            hud_detected = len(hud_rows) > 0
            step_y = 2 if (band.shape[0] >= 8 and band.shape[1] >= 16) else 1
            rel_idx = int(hud_rows[0]) * step_y if hud_detected else max(0, y_hud_px - search_start)
            if hud_detected and step_y == 2 and rel_idx > 0:
                prev_row = band[rel_idx - 1, ::4] if band.shape[1] >= 16 else band[rel_idx - 1]
                pb, pg, pr = prev_row[:, 0], prev_row[:, 1], prev_row[:, 2]
                if ((cv2.subtract(pb, pr) > 45) & (cv2.subtract(pb, pg) > 15) & (pb > 95)).mean() > 0.38:
                    rel_idx -= 1
            y_hud_px = search_start + rel_idx
            band_span = max(4, int(0.03 * vh) // step_y)
            rel_sub = int(hud_rows[0]) if hud_detected else (rel_idx // step_y)
            blue_hud_strength = float(row_blue[rel_sub:min(len(row_blue), rel_sub + band_span)].mean())
            blue_hud_peak = float(row_blue.max())
        y_hud_rel = y_hud_px / float(vh)

        # Reported elixir bar ROI hugs lower HUD strip; pixel sampler uses calibrated bar edges past droplet glow
        bar_y1_rel = min(0.985, y_hud_rel + (1.0 - y_hud_rel) * 0.73)
        bar_y2_rel = min(0.996, y_hud_rel + (1.0 - y_hud_rel) * 0.96)
        sample_y1_rel = min(0.988, y_hud_rel + (1.0 - y_hud_rel) * 0.81)
        ey1, ey2 = int(sample_y1_rel * vh), int(bar_y2_rel * vh)
        ex1, ex2 = int(0.258 * vw), int(0.975 * vw)

        elixir_crop = vp[ey1:ey2, ex1:ex2]
        elixir_level = 0.0
        purple_cols = 0
        bar_width = max(1, ex2 - ex1)
        droplet_present = False

        if elixir_crop.size > 0:
            hsv = cv2.cvtColor(elixir_crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (130, 60, 60), (175, 255, 255))
            col_filled = (mask > 0).sum(axis=0) >= max(2, int(elixir_crop.shape[0] * 0.25))
            purple_indices = np.where(col_filled)[0]
            purple_cols = int(len(purple_indices))
            if purple_cols >= 3 and int(purple_indices[0]) < int(0.25 * bar_width):
                # Measure contiguous left-to-right fill so stray card-cost purple pixels never spike fill
                max_gap = max(6, int(0.035 * bar_width))
                fill_col = int(purple_indices[0])
                for col_idx in purple_indices[1:]:
                    if (int(col_idx) - fill_col) <= max_gap:
                        fill_col = int(col_idx)
                    else:
                        break
                elixir_level = round(min(10.0, max(0.0, (fill_col / float(bar_width)) * 10.0)), 1)

        # Check elixir droplet icon just left of the bar
        drop_crop = vp[ey1:ey2, int(0.14 * vw):int(0.24 * vw)]
        if drop_crop.size > 0:
            dhsv = cv2.cvtColor(drop_crop, cv2.COLOR_BGR2HSV)
            dmask = cv2.inRange(dhsv, (130, 50, 60), (175, 255, 255))
            droplet_present = (dmask > 0).mean() > 0.05

        return {
            "hud_detected": hud_detected,
            "y_hud_rel": y_hud_rel,
            "blue_hud_strength": round(blue_hud_strength, 3),
            "blue_hud_peak": round(blue_hud_peak, 3),
            "elixir_box_rel": [round(bar_y1_rel, 4), 0.22, round(bar_y2_rel, 4), 0.97],
            "elixir_estimate": elixir_level,
            "purple_cols": purple_cols,
            "bar_width_px": bar_width,
            "droplet_present": droplet_present,
        }

    def classify_screen(self, frame: np.ndarray, viewport: Dict[str, Any], hud_info: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classifies screen into 'battle_arena', 'lobby', 'gpg_idle', or 'non_game'
        with an explicit numerical confidence score based on measured visual anchors.
        """
        vx1, vy1, vx2, vy2 = viewport["box_px"]
        vp = frame[vy1:vy2, vx1:vx2]
        vh, vw = vp.shape[:2]
        aspect_ok = 0.38 <= (vw / float(max(1, vh))) <= 0.72

        # Battle HUD evidence score (require elixir droplet or substantial bar fill so Clan/Social tabs never match)
        elixir_ui_ok = bool(hud_info["droplet_present"] or hud_info["purple_cols"] >= 22)
        hud_score = 0.0
        if hud_info["hud_detected"]:
            hud_score += 0.45
        if hud_info.get("blue_hud_peak", 0.0) > 0.45 or hud_info["blue_hud_strength"] > 0.28:
            hud_score += 0.25
        if elixir_ui_ok:
            hud_score += 0.25
        if aspect_ok:
            hud_score += 0.05

        if (
            hud_info["hud_detected"]
            and (hud_info.get("blue_hud_peak", 0.0) > 0.45 or hud_info["blue_hud_strength"] > 0.28)
            and elixir_ui_ok
            and hud_score >= 0.70
            and aspect_ok
        ):
            return "battle_arena", round(min(0.99, hud_score), 3), {"hud_score": hud_score, "aspect_ok": aspect_ok}

        # Check dark/empty idle container
        if vp.mean() < 40 and vp.std() < 15:
            return "gpg_idle", 0.95, {"mean": float(vp.mean()), "std": float(vp.std())}

        # Check Clash Royale lobby (bright gold/yellow Battle button in lower-middle + 5 dark/blue bottom tabs)
        btn_crop = vp[int(0.58 * vh):int(0.76 * vh), int(0.25 * vw):int(0.75 * vw)]
        tabs_crop = vp[int(0.88 * vh):, :]
        lobby_score = 0.0
        if btn_crop.size > 0 and tabs_crop.size > 0 and aspect_ok:
            hsv_btn = cv2.cvtColor(btn_crop, cv2.COLOR_BGR2HSV)
            gold_mask = cv2.inRange(hsv_btn, np.array([15, 120, 140]), np.array([35, 255, 255]))
            gold_ratio = float((gold_mask > 0).mean())
            b = tabs_crop[:, :, 0].astype(np.int16)
            r = tabs_crop[:, :, 2].astype(np.int16)
            tab_blue = float(((b - r > 15) & (tabs_crop[:, :, 0] > 70)).mean())
            if gold_ratio > 0.12 and tab_blue > 0.18:
                lobby_score = min(0.97, 0.55 + gold_ratio + tab_blue)
                return "lobby", round(lobby_score, 3), {"gold_ratio": gold_ratio, "tab_blue": tab_blue}

        return "non_game", round(max(0.50, 1.0 - hud_score), 3), {"hud_score": hud_score, "aspect_ok": aspect_ok}

    def _vp_box_to_full_frame(
        self,
        vp_box: Tuple[float, float, float, float],
        viewport: Dict[str, Any],
        full_w: int,
        full_h: int,
    ) -> Tuple[List[float], List[int], List[float]]:
        """
        Converts viewport-relative [ymin, xmin, ymax, xmax] to:
          - box_pct: [x1, y1, x2, y2] in full frame 0..1 coords (Evaluator/VLM schema)
          - box_px:  [y1, x1, y2, x2] in full frame pixels (Phase 1 compatibility)
          - box_rel: [ymin, xmin, ymax, xmax] in full frame 0..1 coords
        """
        vx1, vy1, vx2, vy2 = viewport["box_px"]
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)

        ymin_vp, xmin_vp, ymax_vp, xmax_vp = vp_box
        x1_px = int(round(vx1 + xmin_vp * vw))
        x2_px = int(round(vx1 + xmax_vp * vw))
        y1_px = int(round(vy1 + ymin_vp * vh))
        y2_px = int(round(vy1 + ymax_vp * vh))

        x1_px = max(0, min(full_w - 1, x1_px))
        x2_px = max(x1_px + 1, min(full_w, x2_px))
        y1_px = max(0, min(full_h - 1, y1_px))
        y2_px = max(y1_px + 1, min(full_h, y2_px))

        box_pct = [
            round(x1_px / float(full_w), 4),
            round(y1_px / float(full_h), 4),
            round(x2_px / float(full_w), 4),
            round(y2_px / float(full_h), 4),
        ]
        box_rel = [box_pct[1], box_pct[0], box_pct[3], box_pct[2]]
        box_px = [y1_px, x1_px, y2_px, x2_px]
        return box_pct, box_px, box_rel

    def _inspect_component(self, key: str, comp_def: Dict[str, Any], crop: np.ndarray, hud_info: Dict[str, Any]) -> Dict[str, Any]:
        """Computes visual presence, confidence, and component-specific state from crop pixels."""
        if crop.size == 0:
            return {"seen": False, "confidence": 0.0, "status": "EMPTY"}

        extra: Dict[str, Any] = {}
        seen = True
        conf = 0.0

        if key == "elixir_bar":
            cols = hud_info["purple_cols"]
            droplet = hud_info["droplet_present"]
            std_val = float(crop[::2, ::2].std()) if (cols < 2 and not droplet) else 20.0
            seen = hud_info["hud_detected"] and (cols >= 2 or droplet or std_val > 15.0)
            conf = min(0.99, 0.55 + (0.25 if hud_info["hud_detected"] else 0.0) + (0.15 if cols >= 5 else 0.08 if droplet else 0.0))

        elif key.startswith("hand_slot_"):
            sub = crop if (crop.shape[0] < 40 or crop.shape[1] < 40) else crop[::4, ::4]
            hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            g_mean, g_std = cv2.meanStdDev(gray)
            s_mean, s_std = cv2.meanStdDev(hsv[:, :, 1])
            std_val = float(g_std[0, 0])
            mean_sat = float(s_mean[0, 0])
            sat_std = float(s_std[0, 0])
            # Empty cooldown slot is dominated by the royal-blue crown tray (>65% blue) or flat low std
            b_ch = sub[:, :, 0]
            r_ch = sub[:, :, 2]
            blue_tray_frac = float(((cv2.subtract(b_ch, r_ch) > 35) & (b_ch > 95)).mean())
            is_empty_cooldown = (blue_tray_frac >= 0.72 and std_val < 46.0) or (std_val < 28.0) or (sat_std < 22.0 and float(g_mean[0, 0]) < 105.0)
            # Playable card is vibrant/colorful; unplayable card is grayscale/whitewashed
            playable = (not is_empty_cooldown) and (mean_sat >= 62.0)
            seen = not is_empty_cooldown
            conf = min(0.98, 0.58 + min(0.36, std_val / 135.0))
            slot_num = int(key.rsplit("_", 1)[1])
            if is_empty_cooldown:
                # Mandatory cache invalidation on cooldown/empty slot transition
                self.card_matcher.invalidate_slot(slot_num)
                card_info = {
                    "card_name": f"slot_{slot_num}",
                    "card_cost": None,
                    "card_role": "unknown",
                    "spell_radius": 0.0,
                    "card_confidence": 0.0,
                }
            else:
                card_info = self.card_matcher.identify(crop, slot_num)
            card_cost = card_info.get("card_cost")
            est_elx = float(hud_info.get("elixir_estimate", 0.0))
            if est_elx < 0.85:
                # No card in Clash Royale costs 0 elixir
                playable = False
            elif card_cost is None and est_elx < 2.8:
                # Avoid clicking unidentified animated/highlighted slots at low elixir
                playable = False
            elif (
                playable
                and card_cost is not None
                and card_info.get("card_confidence", 0.0) >= 0.65
                and est_elx + 0.15 < float(card_cost)
            ):
                playable = False

            extra["slot"] = slot_num
            extra["playable"] = bool(playable)
            extra["slot_state"] = (
                "cooldown_empty" if is_empty_cooldown else ("playable" if playable else "unplayable_low_elixir")
            )
            extra["mean_saturation"] = round(mean_sat, 1)
            extra["card_name"] = card_info["card_name"]
            extra["card_cost"] = card_cost
            extra["card_role"] = card_info["card_role"]
            extra["spell_radius"] = card_info["spell_radius"]
            extra["card_confidence"] = card_info["card_confidence"]

        elif key == "next_card_slot":
            sub = crop if (crop.shape[0] < 32 or crop.shape[1] < 32) else crop[::2, ::2]
            std_val = float(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).std())
            seen = std_val > 18.0
            conf = min(0.96, 0.50 + min(0.42, std_val / 130.0))
            if seen:
                nc_info = self.card_matcher.identify(crop, 0)
                extra["card_name"] = nc_info["card_name"] if not nc_info["card_name"].startswith("slot_") else "next_unknown"
                extra["card_role"] = nc_info["card_role"]
                extra["card_cost"] = nc_info["card_cost"]

        elif key == "match_timer":
            sub = crop if (crop.shape[0] < 32 or crop.shape[1] < 32) else crop[::2, ::2]
            white_ratio = float((sub.min(axis=2) > 205).mean())
            dark_ratio = float((sub.max(axis=2) < 60).mean())
            seen = white_ratio > 0.03 and dark_ratio > 0.15
            conf = min(0.97, 0.60 + min(0.25, white_ratio * 2.0) + (0.10 if dark_ratio > 0.25 else 0.0))

        elif comp_def["category"] == "Tower":
            sub = crop[::6, ::6] if (crop.shape[0] >= 36 and crop.shape[1] >= 36) else crop
            b = sub[:, :, 0]
            g = sub[:, :, 1]
            r = sub[:, :, 2]
            std_val = float(g.std())
            is_opp = key.startswith("opponent_")
            if is_opp:
                hp_mask = (cv2.subtract(r, g) > 38) & (cv2.subtract(r, b) > 38) & (r > 120)
            else:
                hp_mask = (cv2.subtract(b, r) > 35) & (b > 120)
            team_ratio = float(hp_mask.mean())
            alive = team_ratio > 0.012
            seen = std_val > 18.0
            conf = min(0.97, 0.65 + min(0.28, team_ratio * 4.0) + (0.04 if seen else 0.0))
            extra["alive"] = bool(alive)
            extra["team"] = "opponent" if is_opp else "player"

        else:  # Arena / River (cheap decimated luminance std only, zero HSV/cvtColor allocation)
            sub = crop[::6, ::6, 1] if (crop.shape[0] >= 32 and crop.shape[1] >= 32) else crop[:, :, 1]
            std_val = float(sub.std())
            seen = std_val > 12.0
            conf = min(0.96, 0.70 + min(0.25, std_val / 120.0))

        return {
            "seen": bool(seen),
            "confidence": round(conf, 3),
            "status": "DETECTED" if seen else "LOW_SIGNAL",
            **extra,
        }

    def detect_arena_entities(self, frame: np.ndarray, viewport: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Level-badge + HP-bar OpenCV detection of active Red (enemy) and Blue (ally)
        deployed units inside the playable arena.
        Uses MORPH_CLOSE to bridge white digit strokes inside level badges, requires
        white-digit interior fraction on badges (suppressing 'Not enough Elixir!' red text),
        and filters static Princess/King tower badges & banners.
        """
        vx1, vy1, vx2, vy2 = viewport["box_px"]
        vp = frame[vy1:vy2, vx1:vx2]
        vh, vw = vp.shape[:2]

        ax1, ay1 = int(0.08 * vw), int(0.18 * vh)
        ax2, ay2 = int(0.92 * vw), int(0.76 * vh)
        arena_crop = vp[ay1:ay2, ax1:ax2]
        if arena_crop.size == 0:
            return {"enemies": [], "allies": []}

        ah, aw = arena_crop.shape[:2]
        arena_small = cv2.resize(arena_crop, (max(2, aw // 3), max(2, ah // 3)), interpolation=cv2.INTER_LINEAR)
        sh, sw = arena_small.shape[:2]
        active_mask: Optional[np.ndarray] = None
        if self._prev_arena_small is not None and self._prev_arena_small.shape == arena_small.shape:
            diff = cv2.absdiff(arena_small, self._prev_arena_small)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, changed_bin = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
            changed_frac = cv2.countNonZero(changed_bin) / float(max(1, sh * sw))
            if changed_frac < 0.004 and self._arena_static_skips < 3:
                self._arena_static_skips += 1
                self._last_arena_changed = False
                return self._prev_arena_entities
            self._last_arena_changed = changed_frac >= 0.004
            if 0.0 < changed_frac < 0.35:
                # Only scan changed arena regions + bounding boxes around previously tracked units
                active_mask = cv2.dilate(changed_bin, np.ones((15, 15), np.uint8))
                for group in (self._prev_arena_entities.get("enemies", []), self._prev_arena_entities.get("allies", [])):
                    for ent in group:
                        nx, ny = ent.get("norm_vp", [0.0, 0.0])
                        sx = int((nx * vw - ax1) / 3.0)
                        sy = int((ny * vh - ay1) / 3.0)
                        cv2.rectangle(
                            active_mask,
                            (max(0, sx - 16), max(0, sy - 16)),
                            (min(sw - 1, sx + 16), min(sh - 1, sy + 16)),
                            255,
                            -1,
                        )
        else:
            self._last_arena_changed = True

        hsv = cv2.cvtColor(arena_small, cv2.COLOR_BGR2HSV)
        # C++ SIMD cv2.inRange with tuple bounds avoids per-frame array allocations
        red1 = cv2.inRange(hsv, (0, 130, 102), (7, 255, 255))
        red_warm = cv2.inRange(hsv, (8, 170, 140), (12, 255, 255))
        red2 = cv2.inRange(hsv, (161, 130, 102), (179, 255, 255))
        cv2.bitwise_or(red1, red2, dst=red1)
        cv2.bitwise_or(red1, red_warm, dst=red1)
        red_raw = red1

        blue_raw = cv2.inRange(hsv, (92, 125, 98), (125, 255, 255))
        if active_mask is not None:
            cv2.bitwise_and(red_raw, active_mask, dst=red_raw)
            cv2.bitwise_and(blue_raw, active_mask, dst=blue_raw)
        close_kernel = np.ones((2, 2), np.uint8)

        def extract_entities(raw_mask: np.ndarray, team: str) -> List[Dict[str, Any]]:
            if cv2.countNonZero(raw_mask) < 6:
                return []
            closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            entities: List[Dict[str, Any]] = []
            max_side = max(9, int(0.024 * vw))
            min_side = max(3, int(0.0045 * vw))

            for c in cnts:
                x, y, cw, ch = cv2.boundingRect(c)
                area = cw * ch * 9
                if not (min_side <= cw <= max_side and min_side <= ch <= max_side and 144 <= area <= 1350):
                    continue
                aspect = cw / float(max(ch, 1))
                if not (0.45 <= aspect <= 2.4):
                    continue

                inv_px = 1.0 / float(cw * ch)
                color_frac = cv2.countNonZero(raw_mask[y:y + ch, x:x + cw]) * inv_px
                if color_frac < 0.25:
                    continue
                white_frac = cv2.countNonZero(cv2.inRange(hsv[y:y + ch, x:x + cw], (0, 0, 160), (179, 90, 255))) * inv_px
                # Badges always contain a white level numeral; 'Not enough Elixir!' glyphs have wf ~ 0.0
                if not (0.062 <= white_frac <= 0.65):
                    continue

                cx_vp = (ax1 + (x + cw * 0.5) * 3.0) / float(vw)
                cy_vp = (ay1 + (y + ch * 0.5) * 3.0) / float(vh)

                # Filter static tower badges / trim (enemy king/princess at top, our towers below y=0.60)
                if team == "enemy" and cy_vp < 0.242 and (
                    (0.18 <= cx_vp <= 0.30) or (0.44 <= cx_vp <= 0.56) or (0.70 <= cx_vp <= 0.82)
                ):
                    continue
                if team == "enemy" and cy_vp < 0.215:
                    continue
                if team == "enemy" and (0.452 <= cy_vp <= 0.474) and (ch * 3) <= 9 and (0.28 <= cx_vp <= 0.72):
                    continue
                if team == "enemy" and (0.58 <= cy_vp <= 0.68) and (
                    (0.18 <= cx_vp <= 0.31) or (0.69 <= cx_vp <= 0.82)
                ) and area < 550:
                    continue
                if team == "ally" and (
                    (cy_vp >= 0.605 and (cx_vp <= 0.32 or cx_vp >= 0.68))
                    or (cy_vp >= 0.665 and (0.40 <= cx_vp <= 0.60))
                ):
                    continue

                hp_frac = 1.0
                hp_visible = False
                if team == "enemy":
                    ah_s, aw_s = raw_mask.shape[:2]
                    bx1 = min(aw_s, x + cw + 1)
                    bx2 = min(aw_s, x + cw + max(12, int(0.025 * vw)))
                    by1 = max(0, y)
                    by2 = min(ah_s, y + ch)
                    if bx2 > bx1 + 2 and by2 > by1:
                        r_cols = int((raw_mask[by1:by2, bx1:bx2] > 0).any(axis=0).sum())
                        roi_bar = hsv[by1:by2, bx1:bx2]
                        d_cols = int(((roi_bar[:, :, 2] <= 105) & (roi_bar[:, :, 1] <= 95)).any(axis=0).sum())
                        if d_cols >= 2:
                            hp_visible = True
                            hp_frac = round(max(0.05, min(1.0, r_cols / float(max(1, r_cols + d_cols)))), 2)

                col, row = self.projector.pixel_norm_to_tile(cx_vp, cy_vp)
                lane = self.projector.get_lane(col)
                entities.append({
                    "team": team,
                    "norm_vp": [round(cx_vp, 4), round(cy_vp, 4)],
                    "tile": [col, row],
                    "col": col,
                    "row": row,
                    "lane": lane,
                    "in_our_half": bool(row >= 16),
                    "hp_frac": hp_frac,
                    "hp_visible": hp_visible,
                    "card": f"{team}_unit_{lane}",
                    "confidence": round(min(0.96, 0.55 + color_frac * 0.4 + white_frac * 0.6), 3),
                    "_area": area,
                })

            # Additional guard: suppress horizontal red warning banner text across y ~ 0.575..0.625
            if team == "enemy":
                banner_hits = [
                    e for e in entities
                    if 0.575 <= e["norm_vp"][1] <= 0.625 and 0.26 <= e["norm_vp"][0] <= 0.74 and e["_area"] < 500
                ]
                if len(banner_hits) >= 3:
                    span_x = max(e["norm_vp"][0] for e in banner_hits) - min(e["norm_vp"][0] for e in banner_hits)
                    if span_x >= 0.14:
                        entities = [
                            e for e in entities
                            if not (0.575 <= e["norm_vp"][1] <= 0.625 and 0.26 <= e["norm_vp"][0] <= 0.74 and e["_area"] < 500)
                        ]

            for e in entities:
                e.pop("_area", None)

            deduped: List[Dict[str, Any]] = []
            for ent in sorted(entities, key=lambda e: (-e["confidence"], -e["row"])):
                ex, ey = ent["norm_vp"]
                if not any(abs(ex - d["norm_vp"][0]) < 0.029 and abs(ey - d["norm_vp"][1]) < 0.029 for d in deduped):
                    deduped.append(ent)
            return deduped[:12]

        result = {
            "enemies": extract_entities(red_raw, "enemy"),
            "allies": extract_entities(blue_raw, "ally"),
        }
        self._prev_arena_small = arena_small
        self._prev_arena_entities = result
        self._arena_static_skips = 0
        return result

    def detect_components(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the given game frame:
        - Auto-detects game viewport (handles GPG sidebar & mobile/fullscreen)
        - Classifies screen state with measured visual confidence
        - Extracts all 14 components + playability + tower states + elixir + arena entities
        """
        h, w = frame.shape[:2]
        self.height = h
        self.width = w

        if (
            self._vp_cache_val is not None
            and self._vp_cache_key == (w, h)
            and self._vp_cache_age < self._vp_cache_ttl
        ):
            viewport = self._vp_cache_val
            self._vp_cache_age += 1
        else:
            viewport = self.detect_game_viewport(frame)
            self._vp_cache_key = (w, h)
            self._vp_cache_val = viewport
            self._vp_cache_age = 0
        vx1, vy1, vx2, vy2 = viewport["box_px"]
        vp = frame[vy1:vy2, vx1:vx2]
        vh, vw = vp.shape[:2]

        hud_info = self._detect_hud_and_elixir(vp)
        screen_name, screen_conf, screen_debug = self.classify_screen(frame, viewport, hud_info)
        if screen_name != "battle_arena":
            self._vp_cache_age = self._vp_cache_ttl  # re-detect fresh on non-battle transitions

        results: Dict[str, Any] = {
            "frame_size": {"width": w, "height": h},
            "viewport_px": viewport["box_px"],
            "viewport_pct": viewport["box_pct"],
            "layout": viewport["layout"],
            "screen": screen_name,
            "screen_confidence": screen_conf,
            "screen_debug": screen_debug,
            "match_detected": screen_name == "battle_arena",
            "elixir_estimate": hud_info["elixir_estimate"] if screen_name == "battle_arena" else 0.0,
            "droplet_present": hud_info["droplet_present"],
            "components": [],
            "hand_slots": [],
            "next_card": None,
            "tower_states": {},
            "arena_entities": {"enemies": [], "allies": []},
        }

        if screen_name != "battle_arena":
            return results

        # Dynamically refine hand slot vertical bounds to hug detected blue HUD deck strip
        y_hud = hud_info["y_hud_rel"]
        hud_h = max(0.08, 1.0 - y_hud)
        slot_y1 = min(0.95, y_hud + hud_h * 0.14)
        slot_y2 = min(0.985, y_hud + hud_h * 0.74)
        next_y1 = min(0.96, y_hud + hud_h * 0.42)
        next_y2 = min(0.99, y_hud + hud_h * 0.90)

        for key, comp in self.COMPONENTS.items():
            vp_box = comp["box"]
            if key == "elixir_bar":
                vp_box = tuple(hud_info["elixir_box_rel"])  # type: ignore[assignment]
            elif key.startswith("hand_slot_"):
                vp_box = (round(slot_y1, 4), comp["box"][1], round(slot_y2, 4), comp["box"][3])
            elif key == "next_card_slot":
                vp_box = (round(next_y1, 4), comp["box"][1], round(next_y2, 4), comp["box"][3])

            box_pct, box_px, box_rel = self._vp_box_to_full_frame(vp_box, viewport, w, h)
            y1_px, x1_px, y2_px, x2_px = box_px
            crop = frame[y1_px:y2_px, x1_px:x2_px]

            metrics = self._inspect_component(key, comp, crop, hud_info)
            item = {
                "id": key,
                "name": comp["name"],
                "category": comp["category"],
                "box_rel": box_rel,
                "box_pct": box_pct,
                "box_px": box_px,
                "width_px": x2_px - x1_px,
                "height_px": y2_px - y1_px,
                **metrics,
            }
            results["components"].append(item)

            if key.startswith("hand_slot_"):
                results["hand_slots"].append({
                    "slot": item["slot"],
                    "id": key,
                    "playable": item.get("playable", False),
                    "slot_state": item.get("slot_state", "unknown"),
                    "confidence": item["confidence"],
                    "card_name": item.get("card_name", f"slot_{item['slot']}"),
                    "card_cost": item.get("card_cost"),
                    "card_role": item.get("card_role", "unknown"),
                    "spell_radius": item.get("spell_radius", 0.0),
                    "card_confidence": item.get("card_confidence", 0.0),
                    "box_pct": box_pct,
                })
            elif key == "next_card_slot" and item.get("card_name"):
                results["next_card"] = {
                    "card_name": item["card_name"],
                    "card_role": item.get("card_role", "unknown"),
                    "card_cost": item.get("card_cost"),
                }
            elif comp["category"] == "Tower":
                results["tower_states"][key] = {
                    "alive": item.get("alive", True),
                    "team": item.get("team", "unknown"),
                    "confidence": item["confidence"],
                }

        results["arena_entities"] = self.detect_arena_entities(frame, viewport)
        return results

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: Dict[str, Any],
        planned_action: Optional[Dict[str, Any]] = None,
        gate_status: Optional[str] = None,
    ) -> np.ndarray:
        """Draws viewport, component boxes, card playability, arena threats, and planned drops."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        vx1, vy1, vx2, vy2 = detections.get("viewport_px", [0, 0, w, h])
        vw, vh = max(1, vx2 - vx1), max(1, vy2 - vy1)
        if vx1 > 0 or vy1 > 0:
            cv2.rectangle(annotated, (vx1, vy1), (vx2 - 1, vy2 - 1), (100, 200, 255), 1)

        for item in detections.get("components", []):
            key = item["id"]
            comp_info = self.COMPONENTS.get(key)
            if not comp_info:
                continue

            color = comp_info["color"]
            if key.startswith("hand_slot_"):
                color = (0, 255, 100) if item.get("playable") else (140, 140, 140)
            elif comp_info["category"] == "Tower" and not item.get("alive", True):
                color = (90, 90, 90)

            y1, x1, y2, x2 = item["box_px"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = key if key.startswith("hand_slot_") else item["name"]
            if key.startswith("hand_slot_"):
                tag = "READY" if item.get("playable") else "WAIT"
                cname = item.get("card_name", f"slot_{item['slot']}")
                ccost = item.get("card_cost")
                if cname.startswith("slot_"):
                    label = f"S{item['slot']}:{tag}"
                else:
                    cost_str = f"({ccost})" if ccost is not None else ""
                    short_name = cname[:8]
                    label = f"S{item['slot']}:{short_name}{cost_str}{'*' if item.get('playable') else ''}"
            elif comp_info["category"] == "Tower" and not item.get("alive", True):
                label = f"{key}(DOWN)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.32, min(0.48, w / 1600.0))
            (lw, lh), base = cv2.getTextSize(label, font, font_scale, 1)
            ly = y1 - 4 if y1 - lh - 4 > 0 else y1 + lh + 6
            cv2.rectangle(annotated, (x1, ly - lh - 2), (x1 + lw + 4, ly + base), (20, 20, 20), -1)
            cv2.putText(annotated, label, (x1 + 2, ly - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw detected enemy & ally arena entities
        entities = detections.get("arena_entities", {})
        for ent in entities.get("enemies", []):
            nx, ny = ent["norm_vp"]
            px, py = int(vx1 + nx * vw), int(vy1 + ny * vh)
            cv2.circle(annotated, (px, py), 7, (0, 0, 255), 2)
            cv2.putText(annotated, f"E{ent['tile']}", (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 80, 255), 1)
        for ent in entities.get("allies", []):
            nx, ny = ent["norm_vp"]
            px, py = int(vx1 + nx * vw), int(vy1 + ny * vh)
            cv2.circle(annotated, (px, py), 7, (255, 160, 0), 2)

        # Draw planned card drop arrow if provided
        if planned_action and planned_action.get("target_tile") and planned_action.get("slot"):
            slot_num = int(planned_action["slot"])
            col, row = planned_action["target_tile"]
            tx_vp, ty_vp = self.projector.tile_to_norm(col, row)
            tx_px, ty_px = int(vx1 + tx_vp * vw), int(vy1 + ty_vp * vh)
            slot_comp = next((c for c in detections.get("components", []) if c["id"] == f"hand_slot_{slot_num}"), None)
            if slot_comp:
                sy1, sx1, sy2, sx2 = slot_comp["box_px"]
                sx_px, sy_px = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                cv2.arrowedLine(annotated, (sx_px, sy_px), (tx_px, ty_px), (0, 255, 255), 2, tipLength=0.04)
            cv2.circle(annotated, (tx_px, ty_px), 10, (0, 255, 255), 2)

        gate_txt = f" | {gate_status}" if gate_status else ""
        header = (
            f"CLASH IQ | Screen:{detections.get('screen')} ({detections.get('screen_confidence', 0)*100:.0f}%) "
            f"| Elixir:{detections.get('elixir_estimate', 0):.1f}{gate_txt}"
        )
        cv2.rectangle(annotated, (0, 0), (w, 24), (15, 23, 42), -1)
        cv2.putText(annotated, header, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        return annotated


if __name__ == "__main__":
    detector = ComponentDetector()
    sample_path = "E:/cr_ai/frames/match_yt_01/00000354.png"
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
        data = detector.detect_components(img)
        print("Screen:", data["screen"], "Conf:", data["screen_confidence"], "Components:", len(data["components"]), "Elixir:", data["elixir_estimate"])
        out_img = detector.annotate_frame(img, data)
        out_path = "E:/cr_ai/frames/detected_components_test.png"
        cv2.imwrite(out_path, out_img)
        print(f"Saved annotated frame to: {out_path}")
    else:
        print("Sample frame not found.")
