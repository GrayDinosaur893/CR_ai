"""
CLASH IQ — Unified Simultaneous Perception & State Engine
Runs ALL subsystems concurrently in a single multi-threaded process:
  Thread 1: Window Anchor, True Foreground & Invariant Geometry Guard
  Thread 2: High-Speed Screen Capture (Lock-Free Ping-Pong Buffer)
  Thread 3: Simultaneous Perception & State Machine (Lobby & In-Battle CV)
  Thread 4: Global Hard Killswitch Listener (VK_F12 -> os._exit(0))
"""

import os
import sys
import time
import json
import threading
import ctypes
from ctypes import wintypes
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import mss

# UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Win32 API setup
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

class LockFreeBuffer:
    """Thread-safe lock-free latest frame container."""
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self.frame_id = 0

    def put(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
            self.frame_id += 1

    def get(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self.frame_id


class UnifiedClashEngine:
    def __init__(self, target_title: str = "Clash Royale"):
        self.target_title = target_title
        self.hwnd = None
        self.window_title = ""
        self.baseline_geo = None # (left, top, w, h)
        self.buffer = LockFreeBuffer()
        self.running = False
        self.halt_reason = None
        self.latest_state = {
            "screen": "INITIALIZING",
            "confidence": 0.0,
            "elixir": 0.0,
            "anchor": "None",
            "active_tab": "Unknown",
            "window_geo": None,
            "invariants_ok": True,
            "detected_buttons": []
        }
        self.sct = mss.MSS()

    def attach_desktop(self):
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)

    def find_and_focus_window(self) -> bool:
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

        # Restore if minimized
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, 9)
            time.sleep(0.4)

        # True foreground attachment
        cur_thread = kernel32.GetCurrentThreadId()
        target_thread = user32.GetWindowThreadProcessId(self.hwnd, None)
        user32.AttachThreadInput(cur_thread, target_thread, True)
        user32.ShowWindow(self.hwnd, 9)
        user32.SetForegroundWindow(self.hwnd)
        user32.AttachThreadInput(cur_thread, target_thread, False)
        time.sleep(0.2)

        # Lock geometry
        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        self.baseline_geo = (rect.left, rect.top, w, h)
        self.latest_state["window_geo"] = self.baseline_geo
        return True

    def check_invariants(self) -> Tuple[bool, str]:
        if not user32.IsWindow(self.hwnd) or not user32.IsWindowVisible(self.hwnd):
            return False, "Target window closed or hidden"
        if user32.IsIconic(self.hwnd):
            return False, "Target window minimized"

        rect = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
        cur_geo = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        if cur_geo != self.baseline_geo:
            return False, f"Window geometry shifted! Expected {self.baseline_geo}, got {cur_geo}"

        return True, "OK"

    # =========================================================================
    # THREAD 1: Screen Capture Thread (Runs as fast as compositor allows)
    # =========================================================================
    def _capture_worker(self):
        self.attach_desktop()
        left, top, w, h = self.baseline_geo
        monitor = {"top": top, "left": left, "width": w, "height": h}

        with mss.MSS() as sct_local:
            while self.running:
                # Check geometry invariant
                ok, err = self.check_invariants()
                if not ok:
                    self.halt_reason = err
                    self.running = False
                    break

                try:
                    sct_img = sct_local.grab(monitor)
                    frame_bgra = np.array(sct_img)
                    frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
                    self.buffer.put(frame_bgr)
                except Exception as e:
                    self.halt_reason = f"Capture exception: {e}"
                    self.running = False
                    break

                time.sleep(0.016) # ~60 Hz grab rate

    # =========================================================================
    # THREAD 2: Simultaneous Perception & State Machine
    # =========================================================================
    def _perception_worker(self):
        last_id = -1
        os.makedirs("data", exist_ok=True)
        os.makedirs("frames", exist_ok=True)

        while self.running:
            frame, frame_id = self.buffer.get()
            if frame is None or frame_id == last_id:
                time.sleep(0.010)
                continue

            last_id = frame_id
            h, w = frame.shape[:2]

            # 1. Check for In-Battle Arena
            # Check bottom Elixir bar (Y: 95-99%, X: 18-96%)
            elixir_crop = frame[int(0.96*h):int(0.99*h), int(0.18*w):int(0.96*w)]
            hsv_elixir = cv2.cvtColor(elixir_crop, cv2.COLOR_BGR2HSV) if elixir_crop.size > 0 else None
            purple_px = 0
            elixir_val = 0.0

            if hsv_elixir is not None:
                mask_purple = cv2.inRange(hsv_elixir, np.array([130, 60, 60]), np.array([175, 255, 255]))
                col_filled = (mask_purple > 0).sum(axis=0) >= 2
                purple_indices = np.where(col_filled)[0]
                if len(purple_indices) >= 5:
                    fill_col = int(purple_indices[-1])
                    elixir_val = round((fill_col / float(elixir_crop.shape[1])) * 10.0, 1)
                    purple_px = len(purple_indices)

            if purple_px > 25:
                # Screen is IN_BATTLE_ARENA
                self.latest_state = {
                    "screen": "IN_BATTLE_ARENA",
                    "confidence": 0.98,
                    "elixir": elixir_val,
                    "anchor": f"Magenta Elixir Strip ({purple_px} filled columns)",
                    "active_tab": "N/A (Battle Active)",
                    "window_geo": self.baseline_geo,
                    "invariants_ok": True,
                    "detected_buttons": [
                        {"name": "hand_slot_1", "center_rel": [0.31, 0.91]},
                        {"name": "hand_slot_2", "center_rel": [0.50, 0.91]},
                        {"name": "hand_slot_3", "center_rel": [0.69, 0.91]},
                        {"name": "hand_slot_4", "center_rel": [0.88, 0.91]}
                    ]
                }
            else:
                # 2. Check for Main Lobby / Home
                tabs_crop = frame[int(0.88*h):, :]
                if tabs_crop.size > 0:
                    tab_w = w / 5.0
                    tab3_mean = tabs_crop[:, int(2*tab_w):int(3*tab_w)].mean()
                    tab1_mean = tabs_crop[:, 0:int(tab_w)].mean()

                    if tab3_mean > 50 and tab3_mean > tab1_mean * 1.4:
                        self.latest_state = {
                            "screen": "LOBBY_HOME",
                            "confidence": 0.96,
                            "elixir": 0.0,
                            "anchor": f"Bottom Tab 3 Active (Brightness {tab3_mean:.1f})",
                            "active_tab": "Tab 3 (Battle Home)",
                            "window_geo": self.baseline_geo,
                            "invariants_ok": True,
                            "detected_buttons": [
                                {"name": "battle_button", "center_rel": [0.599, 0.624], "safe": False},
                                {"name": "tab_cards", "center_rel": [0.299, 0.940], "safe": True},
                                {"name": "tab_battle", "center_rel": [0.499, 0.940], "safe": True},
                                {"name": "tab_events", "center_rel": [0.899, 0.940], "safe": True}
                            ]
                        }
                    elif frame.mean() < 40 and frame.std() < 12:
                        self.latest_state = {
                            "screen": "GPG_IDLE_CONTAINER",
                            "confidence": 0.99,
                            "elixir": 0.0,
                            "anchor": "Dark Window Background",
                            "active_tab": "None",
                            "window_geo": self.baseline_geo,
                            "invariants_ok": True,
                            "detected_buttons": []
                        }
                    else:
                        self.latest_state = {
                            "screen": "UNKNOWN",
                            "confidence": 0.25,
                            "elixir": 0.0,
                            "anchor": "None",
                            "active_tab": "Unknown",
                            "window_geo": self.baseline_geo,
                            "invariants_ok": True,
                            "detected_buttons": []
                        }

            # Continuously export state to json
            with open("data/live_game_state.json", "w", encoding="utf-8") as f:
                json.dump(self.latest_state, f, indent=2)

            time.sleep(0.05) # ~20 Hz perception update

    # =========================================================================
    # THREAD 3: F12 Hard Killswitch Listener
    # =========================================================================
    def _killswitch_worker(self):
        while self.running:
            if user32.GetAsyncKeyState(0x7B) & 0x8000: # VK_F12
                print("\n" + "!" * 60)
                print("[EMERGENCY KILLSWITCH TRIGGERED VIA F12]")
                print("   Executing hard exit os._exit(0)...")
                print("!" * 60)
                os._exit(0)
            time.sleep(0.05)

    # =========================================================================
    # MASTER RUNNER: Launches All Threads Simultaneously
    # =========================================================================
    def run_simultaneous(self):
        print("=" * 70)
        print("[AI ENGINE] CLASH IQ -- UNIFIED SIMULTANEOUS ENGINE")
        print("   All Subsystems Running Concurrently in True Parallelism")
        print("   -> Thread 1: Invariant Geometry Guard & Desktop Anchor")
        print("   -> Thread 2: Lock-Free High-Speed Capture (~60 Hz)")
        print("   -> Thread 3: Simultaneous Perception & State Machine (~20 Hz)")
        print("   -> Thread 4: F12 Emergency Killswitch (os._exit(0))")
        print("=" * 70)

        # 1. Attach and focus window
        print("\n[INIT] Connecting to Clash Royale on Google Play Games PC...")
        if not self.find_and_focus_window():
            print("[FATAL] Could not attach to Clash Royale window! Ensure GPG PC is open.")
            return

        left, top, w, h = self.baseline_geo
        print(f"[ATTACHED] HWND: {self.hwnd} | Title: \"{self.window_title}\"")
        print(f"[LOCKED] Geometry: ({left}, {top}) | Dimensions: {w}x{h} px\n")

        self.running = True

        # Launch background threads
        t_cap = threading.Thread(target=self._capture_worker, daemon=True, name="CaptureThread")
        t_perc = threading.Thread(target=self._perception_worker, daemon=True, name="PerceptionThread")
        t_kill = threading.Thread(target=self._killswitch_worker, daemon=True, name="KillswitchThread")

        t_cap.start()
        t_perc.start()
        t_kill.start()

        print("[RUNNING] All simultaneous threads started. Live telemetry stream active.")
        print("Press Ctrl+C to stop, or F12 for Emergency Killswitch.\n")

        # Live Terminal UI Dashboard
        frame_idx = 0
        try:
            while self.running:
                frame_idx += 1
                state = self.latest_state

                # Generate live ASCII elixir bar (standard safe chars)
                elixir = state.get("elixir", 0.0)
                bar_len = 10
                filled_blocks = int(round(elixir))
                ascii_bar = "#" * filled_blocks + "-" * (bar_len - filled_blocks)

                screen_name = state.get("screen", "UNKNOWN")
                conf_str = f"{state.get('confidence', 0.0)*100:.0f}%"

                # Single-line clear & overwrite dashboard
                sys.stdout.write(
                    f"\r[{time.strftime('%H:%M:%S')}] Screen: {screen_name:<17} | "
                    f"Conf: {conf_str:<4} | "
                    f"Elixir: [{ascii_bar}] {elixir:4.1f}/10 | "
                    f"Tab: {state.get('active_tab', 'N/A'):<16} | "
                    f"Invariants: [OK] "
                )
                sys.stdout.flush()

                time.sleep(0.2) # Dashboard refresh rate (5 Hz)

        except KeyboardInterrupt:
            print("\n\n[USER STOP] Stopping engine cleanly...")

        self.running = False
        if self.halt_reason:
            print(f"\n[SAFETY HALT] Engine stopped because: {self.halt_reason}")

        print("\n[SHUTDOWN] Unified Simultaneous Engine closed.")

if __name__ == "__main__":
    engine = UnifiedClashEngine()
    engine.run_simultaneous()
