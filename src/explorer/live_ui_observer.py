"""
CLASH IQ — Live UI Explorer & Window Anchor (Mode 1: Pure Observation)
Finds Clash Royale on Google Play Games PC, validates state, ensures foreground,
verifies window geometry invariants, captures screenshots, and detects UI elements
with 100% relative percentage coordinates.

NO CLICKS. ZERO INPUT INJECTION.
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import mss

# Win32 API setup
user32 = ctypes.windll.user32
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

class LiveUIObserver:
    def __init__(self, target_title_substr: str = "Clash Royale"):
        self.target_title_substr = target_title_substr
        self.hwnd = None
        self.window_title = ""
        self.baseline_rect = None
        self.sct = mss.MSS()

    def attach_desktop(self):
        """Attaches to the active interactive input desktop session."""
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)

    def find_window(self) -> bool:
        """Finds the Clash Royale window by substring match."""
        self.attach_desktop()
        found_hwnds = []

        def enum_cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    if self.target_title_substr.lower() in title.lower():
                        found_hwnds.append((hwnd, title))
            return True

        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        if not found_hwnds:
            return False

        # Pick the main window
        self.hwnd, self.window_title = found_hwnds[0]
        return True

    def check_minimized(self) -> bool:
        """Checks if window is currently minimized."""
        if not self.hwnd:
            return True
        return bool(user32.IsIconic(self.hwnd))

    def bring_to_foreground(self) -> bool:
        """Restores and brings the target window to foreground."""
        if not self.hwnd:
            return False

        # 1. If minimized, restore it
        if self.check_minimized():
            user32.ShowWindow(self.hwnd, 9) # SW_RESTORE
            time.sleep(0.4)

        # 2. Bring window to foreground
        user32.SetForegroundWindow(self.hwnd)
        time.sleep(0.3)

        fg_hwnd = user32.GetForegroundWindow()
        return fg_hwnd == self.hwnd

    def get_current_geometry(self):
        """Returns (left, top, width, height) in screen coordinates."""
        if not self.hwnd:
            return None
        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        return (rect.left, rect.top, w, h)

    def check_invariants(self) -> Tuple[bool, str]:
        """
        Re-verifies:
        1. Window still exists and is visible.
        2. Window is NOT minimized.
        3. Geometry (size/pos) has NOT shifted.
        """
        if not user32.IsWindow(self.hwnd) or not user32.IsWindowVisible(self.hwnd):
            return False, "Window was closed or hidden!"

        if self.check_minimized():
            return False, "Window was minimized!"

        current_geo = self.get_current_geometry()
        if self.baseline_rect is not None and current_geo != self.baseline_rect:
            return False, f"Window moved or resized! Was {self.baseline_rect}, now {current_geo}"

        return True, "OK"

    def capture_frame(self) -> np.ndarray:
        """Captures the exact window bounds as a BGR numpy array."""
        geo = self.get_current_geometry()
        if not geo:
            raise RuntimeError("Cannot get window geometry")

        left, top, w, h = geo
        monitor = {"top": top, "left": left, "width": w, "height": h}
        sct_img = self.sct.grab(monitor)
        frame_bgra = np.array(sct_img)
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        return frame_bgr

    def analyze_screen(self, frame: np.ndarray) -> dict:
        """
        Analyzes the frame, detects screen type, fixed anchors, and buttons
        with RELATIVE PERCENTAGE coordinates only.
        """
        h, w = frame.shape[:2]
        analysis = {
            "window_size": {"width": w, "height": h},
            "screen_name": "UNKNOWN",
            "anchor_element": "None",
            "confidence": "LOW",
            "detected_buttons": []
        }

        # 1. Check if window is just a dark container (loading / not rendered yet)
        mean_bgr = frame.mean(axis=(0, 1))
        if mean_bgr[0] < 45 and mean_bgr[1] < 45 and mean_bgr[2] < 45:
            # Check if there is an active banner or notification
            std_val = frame.std()
            if std_val < 15:
                analysis["screen_name"] = "GPG_CONTAINER_IDLE"
                analysis["anchor_element"] = "Dark Background Container (Mean BGR < 40)"
                analysis["confidence"] = "HIGH"
                return analysis

        # 2. Check for In-Game Arena (River, Elixir bar, HUD)
        # Bottom elixir bar area (Y: 95-99%, X: 20-95%)
        bar_crop = frame[int(0.95*h):int(0.99*h), int(0.2*w):int(0.95*w)]
        hsv_bar = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV) if bar_crop.size > 0 else None
        purple_px = 0
        if hsv_bar is not None:
            mask_purple = cv2.inRange(hsv_bar, np.array([130, 60, 60]), np.array([175, 255, 255]))
            purple_px = np.count_nonzero(mask_purple)

        if purple_px > 30:
            analysis["screen_name"] = "IN_BATTLE_ARENA"
            analysis["anchor_element"] = f"Magenta Elixir Bar ({purple_px} purple px)"
            analysis["confidence"] = "HIGH"
            # Buttons / interactable zones in battle:
            analysis["detected_buttons"] = [
                {"name": "hand_slot_1", "box_rel": [0.85, 0.23, 0.97, 0.39], "center_rel": [0.31, 0.91]},
                {"name": "hand_slot_2", "box_rel": [0.85, 0.42, 0.97, 0.58], "center_rel": [0.50, 0.91]},
                {"name": "hand_slot_3", "box_rel": [0.85, 0.61, 0.97, 0.77], "center_rel": [0.69, 0.91]},
                {"name": "hand_slot_4", "box_rel": [0.85, 0.80, 0.97, 0.96], "center_rel": [0.88, 0.91]},
                {"name": "emote_button", "box_rel": [0.85, 0.03, 0.95, 0.16], "center_rel": [0.09, 0.90]}
            ]
            return analysis

        # 3. Check for Main Menu / Home Lobby
        # In Home Lobby, the big yellow "Battle" button sits around Y: 60-70%, X: 30-70%
        # Bottom tabs sit at Y: 92-100%
        # Hamburger menu button (3 horizontal lines) sits at Top Right Y: 2-8%, X: 85-98%
        battle_zone = frame[int(0.60*h):int(0.72*h), int(0.25*w):int(0.75*w)]
        if battle_zone.size > 0:
            hsv_bz = cv2.cvtColor(battle_zone, cv2.COLOR_BGR2HSV)
            # Yellow/Gold Battle button HSV: H: 18-35, S > 120, V > 120
            yellow_mask = cv2.inRange(hsv_bz, np.array([18, 120, 120]), np.array([35, 255, 255]))
            yellow_px = np.count_nonzero(yellow_mask)
            if yellow_px > 500:
                analysis["screen_name"] = "LOBBY_HOME"
                analysis["anchor_element"] = f"Yellow Battle Button ({yellow_px} yellow px in Y:60-72%)"
                analysis["confidence"] = "HIGH"
                analysis["detected_buttons"] = [
                    {"name": "ranked_battle_button", "box_rel": [0.60, 0.28, 0.72, 0.72], "center_rel": [0.50, 0.66]},
                    {"name": "top_hamburger_menu", "box_rel": [0.02, 0.86, 0.08, 0.98], "center_rel": [0.92, 0.05]},
                    {"name": "tab_battle", "box_rel": [0.92, 0.40, 1.00, 0.60], "center_rel": [0.50, 0.96]},
                    {"name": "tab_cards", "box_rel": [0.92, 0.20, 1.00, 0.40], "center_rel": [0.30, 0.96]},
                    {"name": "tab_events", "box_rel": [0.92, 0.60, 1.00, 0.80], "center_rel": [0.70, 0.96]}
                ]
                return analysis

        # Default fallback: Unknown screen
        analysis["screen_name"] = "UNKNOWN"
        analysis["anchor_element"] = "Unmatched visual signature"
        analysis["confidence"] = "LOW"
        return analysis


def run_observer_session(max_frames: int = 10, interval_sec: float = 1.0):
    """
    Runs Mode 1 Live UI Explorer:
    - Finds window
    - Ensures not minimized
    - Sets foreground
    - Validates geometry invariants on every frame
    - Captures screenshot
    - Evaluates screen type and relative percentage button bounds
    - Listens for F12 killswitch (os._exit(0))
    """
    observer = LiveUIObserver()
    print("=" * 60)
    print("CLASH IQ — LIVE UI OBSERVER (MODE 1: ZERO CLICKS)")
    print("Press F12 at any moment to trigger IMMEDIATE HARD KILLSWITCH.")
    print("=" * 60)

    # 1. Window Discovery
    if not observer.find_window():
        print("[ERROR] Clash Royale window not found! Please ensure Google Play Games PC is running.")
        return False

    print(f"[FOUND] Window: \"{observer.window_title}\" (HWND: {observer.hwnd})")

    # 2. Check minimized & bring to foreground
    if observer.check_minimized():
        print("[INFO] Window is minimized. Attempting to restore...")
        user32.ShowWindow(observer.hwnd, 9)
        time.sleep(0.5)

    is_fg = observer.bring_to_foreground()
    print(f"[STATUS] Foreground focused: {is_fg}")

    # Set baseline geometry
    observer.baseline_rect = observer.get_current_geometry()
    left, top, w, h = observer.baseline_rect
    print(f"[GEOMETRY] Position: ({left}, {top}) | Size: {w}x{h} px")

    out_dir = "E:/cr_ai/frames/observer_sessions"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStarting live observation loop (Total: {max_frames} frames, {interval_sec}s interval)...\n")

    for frame_idx in range(1, max_frames + 1):
        # Check F12 Emergency Killswitch
        if user32.GetAsyncKeyState(0x7B) & 0x8000: # VK_F12
            print("\n" + "!" * 50)
            print("[EMERGENCY KILLSWITCH] F12 pressed! Executing os._exit(0) immediately...")
            print("!" * 50)
            os._exit(0)

        # Invariant check: window geometry, visibility, minimization
        ok, reason = observer.check_invariants()
        if not ok:
            print(f"\n[HALT TRIGGERED] Invariant broken on frame {frame_idx}: {reason}")
            print("System halted for safety. No actions will be taken.")
            return False

        # Capture frame
        frame = observer.capture_frame()
        save_path = f"{out_dir}/frame_{frame_idx:03d}.png"
        cv2.imwrite(save_path, frame)
        cv2.imwrite("E:/cr_ai/frames/live_cr_observer_latest.png", frame)

        # Analyze screen
        analysis = observer.analyze_screen(frame)

        # Print structured report for frame
        print(f"--- FRAME {frame_idx}/{max_frames} [{(frame_idx*interval_sec):.1f}s] ---")
        print(f"  Window Size:       {analysis['window_size']['width']}x{analysis['window_size']['height']} px")
        print(f"  Detected Screen:   {analysis['screen_name']}")
        print(f"  Anchor Element:    {analysis['anchor_element']}")
        print(f"  Confidence:        {analysis['confidence']}")
        print(f"  Buttons Visible:   {len(analysis['detected_buttons'])}")
        for btn in analysis['detected_buttons']:
            b_box = [round(x, 3) for x in btn['box_rel']]
            c_pos = [round(x, 3) for x in btn['center_rel']]
            print(f"    * '{btn['name']}': Rel Box [ymin={b_box[0]}, xmin={b_box[1]}, ymax={b_box[2]}, xmax={b_box[3]}], Center (x={c_pos[0]}, y={c_pos[1]})")

        time.sleep(interval_sec)

    print("\n[SUCCESS] Observer session completed cleanly. Zero clicks injected.")
    return True

if __name__ == "__main__":
    run_observer_session(max_frames=3, interval_sec=1.0)
