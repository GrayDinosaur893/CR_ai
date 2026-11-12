"""
CLASH IQ — Clash Royale Game Launcher & Window Anchor.
Responsible for:
1. Detecting installed Clash Royale (Google Play Games PC, BlueStacks, etc.).
2. Launching or bringing Clash Royale to the foreground.
3. Anchoring to the active game window and reporting exact pixel boundaries.
"""

import os
import sys
import time
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, Dict, Any

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long)
    ]

# Common Shortcut and Deep Link paths for Google Play Games PC
CR_SHORTCUT = os.path.expanduser(
    r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Google Play Games\Clash Royale.lnk"
)
CR_URI = "googleplaygames://launch/?id=com.supercell.clashroyale&lid=1&pid=1"
BOOTSTRAPPER_EXE = r"C:\Program Files\Google\Play Games\Bootstrapper.exe"

class CRGameLauncher:
    def __init__(self, target_keywords=("clash royale", "google play games")):
        self.target_keywords = [k.lower() for k in target_keywords]
        self.hwnd: Optional[int] = None
        self.window_rect: Optional[Tuple[int, int, int, int]] = None
        self.window_title: str = ""

    def find_existing_window(self) -> Optional[Tuple[int, str, Tuple[int, int, int, int]]]:
        """Scans visible Windows to find an active Clash Royale or Google Play Games window."""
        candidates = []

        def enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    title_lower = title.lower()

                    for kw in self.target_keywords:
                        if kw in title_lower:
                            rect = RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            w = rect.right - rect.left
                            h = rect.bottom - rect.top
                            # Ignore zero-sized or background helper windows
                            if w > 200 and h > 200:
                                candidates.append((hwnd, title, (rect.left, rect.top, rect.right, rect.bottom)))
                            break
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        # Prioritize windows explicitly titled "Clash Royale" over generic launchers
        for hwnd, title, box in candidates:
            if "clash royale" in title.lower():
                return (hwnd, title, box)

        if candidates:
            return candidates[0]
        return None

    def focus_window(self, hwnd: int) -> bool:
        """Brings the window to foreground and restores if minimized."""
        try:
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.5)
            rect = RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            self.hwnd = hwnd
            self.window_rect = (rect.left, rect.top, rect.right, rect.bottom)
            return True
        except Exception as e:
            print(f"[!] Could not focus window: {e}")
            return False

    def launch(self, timeout_sec: float = 12.0) -> Dict[str, Any]:
        """
        Ensures Clash Royale is running and focused:
        1. Checks if already open -> brings to front.
        2. If not open -> launches via official Google Play Games shortcut/URI.
        3. Waits and locks onto the game window.
        """
        print("[LAUNCHER] Checking for running Clash Royale window...")
        existing = self.find_existing_window()

        if existing:
            hwnd, title, rect = existing
            self.focus_window(hwnd)
            self.window_title = title
            w = rect[2] - rect[0]
            h = rect[3] - rect[1]
            print(f"[LAUNCHER] [OK] Attached to existing window: '{title}' ({w}x{h})")
            return {
                "status": "attached_existing",
                "hwnd": hwnd,
                "title": title,
                "rect": self.window_rect,
                "width": w,
                "height": h
            }

        print("[LAUNCHER] Game window not detected. Triggering launch sequence...")

        # 1. Try launching shortcut
        launched = False
        if os.path.exists(CR_SHORTCUT):
            try:
                os.startfile(CR_SHORTCUT)
                print(f"[LAUNCHER] Started shortcut: {CR_SHORTCUT}")
                launched = True
            except Exception as e:
                print(f"[LAUNCHER] Shortcut execution warning: {e}")

        # 2. Try URI protocol via os.system / powershell
        if not launched:
            try:
                import subprocess
                subprocess.Popen(["cmd", "/c", "start", "", f"googleplaygames://launch/?id=com.supercell.clashroyale&lid=1&pid=1"], shell=True)
                print("[LAUNCHER] Started via googleplaygames deep-link protocol.")
                launched = True
            except Exception as e:
                print(f"[LAUNCHER] Deep-link warning: {e}")

        # Wait for window to appear
        print(f"[LAUNCHER] Waiting up to {timeout_sec}s for game window to initialize...")
        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            time.sleep(1.0)
            existing = self.find_existing_window()
            if existing:
                hwnd, title, rect = existing
                self.focus_window(hwnd)
                self.window_title = title
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                print(f"[LAUNCHER] [SUCCESS] Detected and focused: '{title}' ({w}x{h})")
                return {
                    "status": "launched_new",
                    "hwnd": hwnd,
                    "title": title,
                    "rect": self.window_rect,
                    "width": w,
                    "height": h
                }

        # Fallback to desktop bounds if window title is hidden by host sandbox
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        self.window_rect = (0, 0, w, h)
        self.window_title = "Desktop Primary Display"
        print(f"[LAUNCHER] [INFO] Defaulting to Primary Display anchor ({w}x{h}).")
        return {
            "status": "fallback_desktop",
            "hwnd": None,
            "title": self.window_title,
            "rect": self.window_rect,
            "width": w,
            "height": h
        }

if __name__ == "__main__":
    launcher = CRGameLauncher()
    res = launcher.launch(timeout_sec=5.0)
    print("Result:", res)
