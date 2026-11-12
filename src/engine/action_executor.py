"""
CLASH IQ — Safe Viewport-Aware Action Executor.
Maps planned card placements (slot + 18x32 tile) onto detected viewport pixels
and strictly blocks live Win32 mouse injection unless AccuracyGate (>= 90%) is UNLOCKED
and live_armed=True is explicitly enabled.
"""

import time
import ctypes
from typing import Dict, Any, Optional, Tuple

from src.vision.homography import ArenaTileProjector
from src.engine.accuracy_gate import AccuracyGate


user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
SW_RESTORE = 9
VK_ESCAPE = 0x1B


class ActionExecutor:
    def __init__(
        self,
        accuracy_gate: Optional[AccuracyGate] = None,
        live_armed: bool = False,
        min_play_interval_sec: float = 0.65,
    ):
        self.gate = accuracy_gate or AccuracyGate()
        self.live_armed = bool(live_armed)
        self.min_play_interval_sec = float(min_play_interval_sec)
        self.projector = ArenaTileProjector()
        self.last_action_ts = 0.0
        self.last_menu_ts = 0.0
        self.stats = {
            "planned_actions": 0,
            "shadow_logged": 0,
            "live_executed": 0,
            "menu_clicks": 0,
            "blocked_by_gate": 0,
        }

    def is_desktop_locked(self) -> bool:
        """Returns True if Windows Default Lock Screen is currently active."""
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
            fg = user32.GetForegroundWindow()
            if fg:
                length = user32.GetWindowTextLengthW(fg)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(fg, buf, length + 1)
                    if "lock screen" in buf.value.lower():
                        return True
        except Exception:
            pass
        return False

    def focus_window(self, hwnd: Optional[int]) -> None:
        """Brings the target game window to the foreground before mouse injection."""
        if not hwnd or not user32.IsWindow(hwnd):
            return
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
            if user32.GetForegroundWindow() != hwnd:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.05)
        except Exception:
            pass

    def _resolve_scale(
        self,
        detection: Dict[str, Any],
        slot_comp: Optional[Dict[str, Any]],
        client_size: Optional[Tuple[int, int]],
    ) -> Tuple[float, float]:
        if not client_size or client_size[0] <= 0 or client_size[1] <= 0:
            return 1.0, 1.0
        fsize = detection.get("frame_size") or {}
        fw = int(fsize.get("width") or 0)
        fh = int(fsize.get("height") or 0)
        if (fw <= 0 or fh <= 0) and slot_comp is not None:
            sy1, sx1, sy2, sx2 = slot_comp["box_px"]
            spct = slot_comp.get("box_pct")
            if spct and (spct[2] - spct[0]) > 0.01 and (spct[3] - spct[1]) > 0.01:
                fw = int(round((sx2 - sx1) / (spct[2] - spct[0])))
                fh = int(round((sy2 - sy1) / (spct[3] - spct[1])))
        if fw > 0 and fh > 0:
            return client_size[0] / float(fw), client_size[1] / float(fh)
        return 1.0, 1.0

    def compute_screen_targets(
        self,
        detection: Dict[str, Any],
        slot: int,
        target_tile: list,
        client_origin: Tuple[int, int] = (0, 0),
        client_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        """Converts slot & tile [col, row] to absolute screen (x, y) coordinates."""
        comps = detection.get("components", [])
        slot_comp = next((c for c in comps if c.get("id") == f"hand_slot_{slot}"), None)
        if slot_comp is None:
            return None

        sy1, sx1, sy2, sx2 = slot_comp["box_px"]
        slot_cx = (sx1 + sx2) / 2.0
        slot_cy = (sy1 + sy2) / 2.0

        vx1, vy1, vx2, vy2 = detection.get("viewport_px", [0, 0, 1, 1])
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)

        norm_x, norm_y = self.projector.tile_to_norm(target_tile[0], target_tile[1])
        tile_cx = vx1 + norm_x * vw
        tile_cy = vy1 + norm_y * vh

        scale_x, scale_y = self._resolve_scale(detection, slot_comp, client_size)

        ox, oy = client_origin
        slot_sx = int(round(ox + slot_cx * scale_x))
        slot_sy = int(round(oy + slot_cy * scale_y))
        tile_sx = int(round(ox + tile_cx * scale_x))
        tile_sy = int(round(oy + tile_cy * scale_y))

        if client_size and client_size[0] > 0 and client_size[1] > 0:
            cw, ch = client_size
            slot_sx = max(ox + 2, min(ox + cw - 3, slot_sx))
            slot_sy = max(oy + 2, min(oy + ch - 3, slot_sy))
            tile_sx = max(ox + 2, min(ox + cw - 3, tile_sx))
            tile_sy = max(oy + 2, min(oy + ch - 3, tile_sy))

        return {
            "slot_screen_xy": (slot_sx, slot_sy),
            "tile_screen_xy": (tile_sx, tile_sy),
        }

    def _win32_click(self, x: int, y: int, hold_sec: float = 0.015):
        """Performs a fast, low-latency Win32 left click at screen coordinates (x, y)."""
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.008)
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(hold_sec)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def execute_menu_click(
        self,
        label: str,
        viewport_xy: Tuple[float, float],
        detection: Dict[str, Any],
        client_origin: Tuple[int, int] = (0, 0),
        client_size: Optional[Tuple[int, int]] = None,
        hwnd: Optional[int] = None,
        cooldown_sec: float = 4.0,
    ) -> Dict[str, Any]:
        """Clicks a non-battle UI button (e.g., Lobby Battle or Post-Match OK) safely."""
        now = time.time()
        if (now - self.last_menu_ts) < cooldown_sec:
            return {"status": "MENU_COOLDOWN", "executed": False, "reason": f"Menu cooldown ({cooldown_sec:.1f}s)"}

        gate_unlocked, gate_reason, _ = self.gate.check()
        if not gate_unlocked:
            self.stats["blocked_by_gate"] += 1
            return {"status": "BLOCKED_ACCURACY_GATE", "executed": False, "reason": gate_reason}

        vx1, vy1, vx2, vy2 = detection.get("viewport_px", [0, 0, 1, 1])
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        fx = vx1 + float(viewport_xy[0]) * vw
        fy = vy1 + float(viewport_xy[1]) * vh
        scale_x, scale_y = self._resolve_scale(detection, None, client_size)
        ox, oy = client_origin
        sx = int(round(ox + fx * scale_x))
        sy = int(round(oy + fy * scale_y))
        if client_size and client_size[0] > 0 and client_size[1] > 0:
            sx = max(ox + 2, min(ox + client_size[0] - 3, sx))
            sy = max(oy + 2, min(oy + client_size[1] - 3, sy))

        if not self.live_armed:
            self.last_menu_ts = now
            self.stats["shadow_logged"] += 1
            return {
                "status": "SHADOW_MENU_DRY_RUN",
                "executed": False,
                "reason": f"Would click {label} at ({sx}, {sy})",
                "menu_screen_xy": (sx, sy),
            }

        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            return {"status": "ESC_ABORTED", "executed": False, "reason": "ESC held"}
        if self.is_desktop_locked():
            return {"status": "SCREEN_LOCKED", "executed": False, "reason": "Windows Lock Screen active"}

        self.focus_window(hwnd)
        self._win32_click(sx, sy, hold_sec=0.08)
        self.last_menu_ts = now
        self.stats["menu_clicks"] += 1
        return {
            "status": "LIVE_MENU_CLICKED",
            "executed": True,
            "reason": f"Clicked {label} at ({sx}, {sy})",
            "menu_screen_xy": (sx, sy),
        }

    def execute(
        self,
        decision: Dict[str, Any],
        detection: Dict[str, Any],
        client_origin: Tuple[int, int] = (0, 0),
        client_size: Optional[Tuple[int, int]] = None,
        hwnd: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates gates and either simulates (shadow mode) or executes live card deployment.
        """
        action = decision.get("action", "wait")
        if action == "wait" or not decision.get("slot") or not decision.get("target_tile"):
            return {"status": "WAIT", "executed": False, "reason": decision.get("reasoning", "No play")}

        now = time.time()
        if (now - self.last_action_ts) < self.min_play_interval_sec:
            return {
                "status": "COOLDOWN",
                "executed": False,
                "reason": f"Waiting play cooldown ({self.min_play_interval_sec:.1f}s)",
            }

        self.stats["planned_actions"] += 1
        slot = int(decision["slot"])
        tile = decision["target_tile"]
        coords = self.compute_screen_targets(
            detection, slot, tile, client_origin=client_origin, client_size=client_size
        )
        if coords is None:
            return {"status": "NO_SLOT_COORDS", "executed": False, "reason": f"Missing box for hand_slot_{slot}"}

        gate_unlocked, gate_reason, _ = self.gate.check()
        frame_ready, frame_reason = self.gate.is_live_frame_ready(detection, min_conf=0.90)

        if not gate_unlocked:
            self.stats["blocked_by_gate"] += 1
            self.last_action_ts = now
            return {
                "status": "BLOCKED_ACCURACY_GATE",
                "executed": False,
                "reason": gate_reason,
                "action": action,
                "slot": slot,
                "target_tile": tile,
                **coords,
            }

        if not frame_ready:
            self.stats["blocked_by_gate"] += 1
            return {
                "status": "BLOCKED_FRAME_CONFIDENCE",
                "executed": False,
                "reason": frame_reason,
                "action": action,
                "slot": slot,
                "target_tile": tile,
                **coords,
            }

        if not self.live_armed:
            t_tap_sent = time.perf_counter()
            self.stats["shadow_logged"] += 1
            self.last_action_ts = now
            return {
                "status": "SHADOW_DRY_RUN",
                "executed": False,
                "reason": f"Gate UNLOCKED ({gate_reason}), shadow mode active (pass --live to click)",
                "action": action,
                "slot": slot,
                "target_tile": tile,
                "t_tap_sent": t_tap_sent,
                "t_tap_done": t_tap_sent,
                **coords,
            }

        # Emergency ESC / Lock Screen check right before injecting clicks
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            return {"status": "ESC_ABORTED", "executed": False, "reason": "ESC held"}
        if hwnd and user32.GetForegroundWindow() != hwnd:
            if self.is_desktop_locked():
                return {"status": "SCREEN_LOCKED", "executed": False, "reason": "Windows Lock Screen active"}
            self.focus_window(hwnd)

        sx, sy = coords["slot_screen_xy"]
        tx, ty = coords["tile_screen_xy"]
        t_tap_sent = time.perf_counter()
        self._win32_click(sx, sy)
        time.sleep(0.020)
        self._win32_click(tx, ty)
        t_tap_done = time.perf_counter()

        self.last_action_ts = now
        self.stats["live_executed"] += 1
        return {
            "status": "LIVE_EXECUTED",
            "executed": True,
            "reason": f"Deployed slot {slot} ({decision.get('recommended_card')}) -> tile {tile}",
            "action": action,
            "slot": slot,
            "target_tile": tile,
            "t_tap_sent": t_tap_sent,
            "t_tap_done": t_tap_done,
            **coords,
        }
