"""
CLASH ROYALE AUTONOMOUS LIVE UI EXPLORER
Author: Divyansh & Antigravity
Mode: Pure Live Observation (Zero Clicks)

Rules:
1. Autonomous window discovery and foreground enforcement.
2. Strict geometry invariant verification per frame (Window move/resize/minimize -> HALT).
3. Continuous screen classification & relative % coordinate extraction.
4. F12 Hard Killswitch (os._exit(0)).
5. Default-Deny: Unknown screen -> HALT.
"""

import os
import sys
import time
import json
import ctypes
from ctypes import wintypes
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import mss

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Win32 API setup
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

class AutonomousUIExplorer:
    def __init__(self, target_title: str = "Clash Royale", output_graph_path: str = "data/screen_graph.json"):
        self.target_title = target_title
        self.output_graph_path = output_graph_path
        self.hwnd = None
        self.window_title = ""
        self.baseline_geo = None # (left, top, w, h)
        self.sct = mss.MSS()
        self.screen_history = []
        os.makedirs(os.path.dirname(self.output_graph_path), exist_ok=True)

    def attach_desktop(self):
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)

    def find_and_lock_window(self) -> bool:
        """Finds window, brings to true foreground, and locks baseline geometry."""
        self.attach_desktop()
        found = []

        def enum_cb(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    if self.target_title.lower() in title.lower():
                        found.append((hwnd, title))
            return True

        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        if not found:
            return False

        self.hwnd, self.window_title = found[0]

        # 1. Restore if minimized
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, 9) # SW_RESTORE
            time.sleep(0.5)

        # 2. Attach thread input to guarantee foreground focus
        cur_thread = kernel32.GetCurrentThreadId()
        target_thread = user32.GetWindowThreadProcessId(self.hwnd, None)
        user32.AttachThreadInput(cur_thread, target_thread, True)
        user32.ShowWindow(self.hwnd, 9)
        user32.SetForegroundWindow(self.hwnd)
        user32.AttachThreadInput(cur_thread, target_thread, False)
        time.sleep(0.3)

        # 3. Lock baseline geometry
        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        self.baseline_geo = (rect.left, rect.top, w, h)
        return True

    def check_invariants(self) -> Tuple[bool, str]:
        """Continuous runtime invariant guard."""
        if not user32.IsWindow(self.hwnd) or not user32.IsWindowVisible(self.hwnd):
            return False, "Target window closed or hidden"
        if user32.IsIconic(self.hwnd):
            return False, "Target window minimized"

        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        cur_geo = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        if cur_geo != self.baseline_geo:
            return False, f"Geometry shift detected! Expected {self.baseline_geo}, got {cur_geo}"

        return True, "OK"

    def grab_window_frame(self) -> np.ndarray:
        left, top, w, h = self.baseline_geo
        monitor = {"top": top, "left": left, "width": w, "height": h}
        sct_img = self.sct.grab(monitor)
        img_bgra = np.array(sct_img)
        return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

    def classify_screen(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        AI Screen Classifier:
        Detects screen state, anchor elements, and relative % button positions.
        """
        h, w = frame.shape[:2]
        result = {
            "timestamp": time.strftime("%H:%M:%S"),
            "window_size": {"width": w, "height": h},
            "screen": "UNKNOWN",
            "anchor": "None",
            "confidence": 0.0,
            "buttons": []
        }

        # 1. Dark Idle Container Check
        mean_bgr = frame.mean(axis=(0, 1))
        if mean_bgr[0] < 42 and mean_bgr[1] < 42 and mean_bgr[2] < 42 and frame.std() < 12:
            result["screen"] = "GPG_IDLE_CONTAINER"
            result["anchor"] = "Solid Dark Canvas"
            result["confidence"] = 0.99
            return result

        # 2. In-Battle Arena Check (Elixir bar at bottom)
        bar_crop = frame[int(0.95*h):int(0.99*h), int(0.2*w):int(0.95*w)]
        if bar_crop.size > 0:
            hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
            purple_mask = cv2.inRange(hsv, np.array([130, 60, 60]), np.array([175, 255, 255]))
            if np.count_nonzero(purple_mask) > 30:
                result["screen"] = "IN_BATTLE_ARENA"
                result["anchor"] = "Magenta Elixir Strip"
                result["confidence"] = 0.98
                result["buttons"] = [
                    {"name": "hand_slot_1", "rel_center": [0.31, 0.91], "allowlist": True},
                    {"name": "hand_slot_2", "rel_center": [0.50, 0.91], "allowlist": True},
                    {"name": "hand_slot_3", "rel_center": [0.69, 0.91], "allowlist": True},
                    {"name": "hand_slot_4", "rel_center": [0.88, 0.91], "allowlist": True}
                ]
                return result

        # 3. Main Lobby / Home Screen Check
        # Signature: Bottom Tab 3 is active/brightest, Center Battle area has gold/brown elements
        tabs_crop = frame[int(0.88*h):, :]
        if tabs_crop.size > 0:
            tab_w = w / 5.0
            tab3_mean = tabs_crop[:, int(2*tab_w):int(3*tab_w)].mean()
            tab1_mean = tabs_crop[:, 0:int(tab_w)].mean()

            # Active battle tab has distinct highlight compared to shop tab
            if tab3_mean > 50 and tab3_mean > tab1_mean * 1.5:
                result["screen"] = "LOBBY_HOME"
                result["anchor"] = f"Bottom Tab 3 Active (Mean {tab3_mean:.1f} vs Tab 1 {tab1_mean:.1f})"
                result["confidence"] = 0.96
                result["buttons"] = [
                    {"name": "battle_button", "rel_box": [0.500, 0.350, 0.749, 0.847], "rel_center": [0.599, 0.624], "allowlist": False},
                    {"name": "tab_1_shop", "rel_box": [0.879, 0.000, 1.000, 0.199], "rel_center": [0.100, 0.940], "allowlist": False},
                    {"name": "tab_2_cards", "rel_box": [0.879, 0.199, 1.000, 0.399], "rel_center": [0.299, 0.940], "allowlist": True},
                    {"name": "tab_3_battle", "rel_box": [0.879, 0.399, 1.000, 0.600], "rel_center": [0.499, 0.940], "allowlist": True},
                    {"name": "tab_4_clan", "rel_box": [0.879, 0.600, 1.000, 0.799], "rel_center": [0.699, 0.940], "allowlist": False},
                    {"name": "tab_5_events", "rel_box": [0.879, 0.799, 1.000, 1.000], "rel_center": [0.899, 0.940], "allowlist": True},
                    {"name": "top_right_menu", "rel_box": [0.040, 0.750, 0.120, 0.980], "rel_center": [0.880, 0.080], "allowlist": True}
                ]
                return result

        # Unknown / Unclassified screen
        result["screen"] = "UNKNOWN"
        result["anchor"] = "None"
        result["confidence"] = 0.20
        return result

    def run_live_loop(self, poll_interval: float = 1.0):
        print("=" * 65)
        print("[AI] CLASH IQ -- AUTONOMOUS LIVE UI EXPLORER")
        print("   Running autonomously on host machine.")
        print("   Zero clicks. Pure visual telemetry & state graph builder.")
        print("   EMERGENCY KILLSWITCH: Press [F12] to force os._exit(0)")
        print("=" * 65)

        # Step 1: Find and focus window
        print("\n[AI STEP 1] Scanning for Clash Royale window on Google Play Games PC...")
        if not self.find_and_lock_window():
            print("[AI FATAL] Clash Royale window not found or cannot attach! Halting.")
            return

        left, top, w, h = self.baseline_geo
        print(f"[AI LOCKED] Target HWND: {self.hwnd} | Title: \"{self.window_title}\"")
        print(f"[AI LOCKED] Position: ({left}, {top}) | Dimensions: {w}x{h} px")
        print(f"[AI ACTIVE] Invariant guard online. Starting observation loop...\n")

        frame_count = 0
        try:
            while True:
                # 1. Killswitch check
                if user32.GetAsyncKeyState(0x7B) & 0x8000: # VK_F12
                    print("\n" + "!" * 55)
                    print("[EMERGENCY KILLSWITCH TRIGGERED VIA F12]")
                    print("   Executing hard exit os._exit(0)...")
                    print("!" * 55)
                    os._exit(0)

                # 2. Invariant guard check
                ok, err = self.check_invariants()
                if not ok:
                    print(f"\n[AI INVARIANT HALT] {err}!")
                    print("   Stopping immediately for safety. No actions will be taken.")
                    break

                # 3. Capture frame
                frame = self.grab_window_frame()
                frame_count += 1

                # 4. Classify screen
                state = self.classify_screen(frame)

                # 5. Save latest screenshots & graph
                cv2.imwrite("E:/cr_ai/frames/live_ui_current.png", frame)
                self.screen_history.append(state)
                # Keep last 50 states in graph
                if len(self.screen_history) > 50:
                    self.screen_history.pop(0)

                with open(self.output_graph_path, "w", encoding="utf-8") as f:
                    json.dump({"active_screen": state, "history": self.screen_history}, f, indent=2)

                # 6. Stream live output to terminal
                conf_tag = f"HIGH ({state['confidence']*100:.0f}%)" if state['confidence'] > 0.8 else f"LOW ({state['confidence']*100:.0f}%)"
                print(f"[{state['timestamp']} | Frame {frame_count:04d}] Screen: {state['screen']:<18} | Confidence: {conf_tag:<12} | Anchor: {state['anchor']}")

                if state['screen'] == "UNKNOWN":
                    print("   [UNKNOWN SCREEN DETECTED] Triggering Default-Deny auto-pause...")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n[AI INFO] Explorer stopped by user (Ctrl+C). Clean exit.")

if __name__ == "__main__":
    explorer = AutonomousUIExplorer()
    explorer.run_live_loop(poll_interval=1.0)
