"""
OBSERVER MODULE 1: Window Guard & Invariant Sentinel.
Responsibilities:
1. Find or launch the Clash Royale / Google Play Games PC window (with configurable timeout).
2. Monitor window geometry: if minimized, moved, or resized -> trigger immediate HALT.
3. System-wide F12 hard killswitch listener (active even when game window has OS focus).
4. ABSOLUTE INVARIANT: ZERO input injection functions exist in this codebase.
   No mouse clicks, no key presses, no WM_LBUTTONDOWN. Pure observation only.
"""

import os
import sys
import time
import subprocess
import threading
import ctypes
from ctypes import wintypes
from typing import Optional, Tuple, Dict, Any, Callable

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long)
    ]

class POINT(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_long),
        ('y', ctypes.c_long)
    ]

VK_F12 = 0x7B
MOD_NOREPEAT = 0x4000
HOTKEY_ID = 9012

CR_SHORTCUT = os.path.expanduser(
    r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Google Play Games\Clash Royale.lnk"
)
CR_URI = "googleplaygames://launch/?id=com.supercell.clashroyale&lid=1&pid=1"

def attach_thread_desktop():
    """Attaches calling thread to active input desktop session."""
    try:
        hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
        if hdesk:
            user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


class GlobalKillswitch:
    """
    Listens for F12 globally across Windows using RegisterHotKey + GetAsyncKeyState backup.
    Works even when the game window is actively focused and capturing inputs.
    """
    def __init__(self, halt_callback: Callable[[str], None]):
        self.halt_callback = halt_callback
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_hotkey_pump, daemon=True)
        self.thread.start()

    def _run_hotkey_pump(self):
        attach_thread_desktop()
        registered = user32.RegisterHotKey(None, HOTKEY_ID, MOD_NOREPEAT, VK_F12)
        try:
            msg = wintypes.MSG()
            while not self.stop_event.is_set():
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1): # PM_REMOVE = 1
                    if msg.message == 0x0312 and msg.wParam == HOTKEY_ID: # WM_HOTKEY
                        print("\n[KILLSWITCH] >>> F12 PRESSED (WM_HOTKEY)! HALTING IMMEDIATELY <<<")
                        self.halt_callback("F12_PRESSED")
                        break
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))

                # Backup poll in case hotkey was intercepted or already occupied
                if user32.GetAsyncKeyState(VK_F12) & 0x8000:
                    print("\n[KILLSWITCH] >>> F12 PRESSED (ASYNC_KEY)! HALTING IMMEDIATELY <<<")
                    self.halt_callback("F12_PRESSED")
                    break

                time.sleep(0.02)
        finally:
            if registered:
                user32.UnregisterHotKey(None, HOTKEY_ID)

    def stop(self):
        self.stop_event.set()


class WindowGuard:
    """
    Manages window attachment, geometry validation, and zero-input enforcement.
    """
    def __init__(self, target_keywords=("clash royale", "google play games"), on_halt: Optional[Callable[[str], None]] = None):
        attach_thread_desktop()
        self.target_keywords = [k.lower() for k in target_keywords]
        self.on_halt = on_halt or self._default_halt
        self.hwnd: Optional[int] = None
        self.window_title: str = ""
        self.baseline_rect: Optional[Tuple[int, int, int, int]] = None
        self.baseline_client_rect: Optional[Tuple[int, int, int, int]] = None
        self.is_halted = False
        self.halt_reason: str = ""
        self.killswitch = GlobalKillswitch(self.trigger_halt)

    def _default_halt(self, reason: str):
        print(f"[WINDOW_GUARD] HALT TRIGGERED: {reason}")

    def trigger_halt(self, reason: str):
        if not self.is_halted:
            self.is_halted = True
            self.halt_reason = reason
            self.on_halt(reason)

    def find_existing_window(self) -> Optional[Tuple[int, str, Tuple[int, int, int, int]]]:
        """Finds any visible window matching target keywords with valid dimensions."""
        attach_thread_desktop()
        candidates = []

        def enum_cb(hwnd, _):
            if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    title_lower = title.lower()
                    if any(bad in title_lower for bad in ("brave", "chrome", "edge", "firefox", "claude", "antigravity", "visual studio", "gameplay aud")):
                        return True

                    for kw in self.target_keywords:
                        if kw in title_lower:
                            rect = RECT()
                            user32.GetWindowRect(hwnd, ctypes.byref(rect))
                            w = rect.right - rect.left
                            h = rect.bottom - rect.top
                            # Exclude helper/tray zero or tiny windows
                            if w > 200 and h > 200:
                                candidates.append((hwnd, title, (rect.left, rect.top, rect.right, rect.bottom)))
                            break
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

        # Prioritize exact Clash Royale game window ('Clash Royale' or 'Clash Royale - <account>')
        for hwnd, title, box in candidates:
            tl = title.lower()
            if tl == "clash royale" or tl.startswith("clash royale - "):
                return (hwnd, title, box)
        for hwnd, title, box in candidates:
            if "clash royale" in title.lower():
                return (hwnd, title, box)

        if candidates:
            return candidates[0]
        return None

    def launch_or_find(self, timeout_sec: Optional[float] = None) -> bool:
        """
        Locates game window or attempts launch via shortcut/URI, then waits.
        If timeout_sec is None, waits continuously until window appears.
        Does NOT inject any input.
        """
        existing = self.find_existing_window()
        if existing:
            self.hwnd, self.window_title, self.baseline_rect = existing
            self.baseline_client_rect = self.get_client_screen_rect()
            print(f"[WINDOW_GUARD] Found existing window: '{self.window_title}' (HWND: {self.hwnd})")
            print(f"[WINDOW_GUARD] Baseline Window Rect: {self.baseline_rect}")
            print(f"[WINDOW_GUARD] Baseline Client Rect: {self.baseline_client_rect}")
            return True

        print("[WINDOW_GUARD] Window not found. Launching via official GPG shortcut...")
        launched = False
        if os.path.exists(CR_SHORTCUT):
            try:
                os.startfile(CR_SHORTCUT)
                launched = True
            except Exception as e:
                print(f"[WINDOW_GUARD] Shortcut failed: {e}")

        if not launched:
            try:
                subprocess.Popen(["cmd", "/c", "start", "", CR_URI], shell=True)
                launched = True
            except Exception as e:
                print(f"[WINDOW_GUARD] URI launch failed: {e}")

        # Poll until window appears or timeout
        start_time = time.time()
        print("[WINDOW_GUARD] Waiting for Clash Royale window to appear on screen...")
        while True:
            if self.is_halted:
                return False
            existing = self.find_existing_window()
            if existing:
                self.hwnd, self.window_title, self.baseline_rect = existing
                self.baseline_client_rect = self.get_client_screen_rect()
                print(f"[WINDOW_GUARD] Window appeared: '{self.window_title}' (HWND: {self.hwnd})")
                return True

            if timeout_sec is not None and (time.time() - start_time) > timeout_sec:
                break
            time.sleep(1.0)

        self.trigger_halt("WINDOW_LAUNCH_TIMEOUT")
        return False

    def get_client_screen_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns the client area bounding box (left, top, right, bottom) in screen coordinates."""
        if not self.hwnd or not user32.IsWindow(self.hwnd):
            return None

        rect = RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            return None

        pt = POINT(rect.left, rect.top)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        return (pt.x, pt.y, pt.x + w, pt.y + h)

    def check_invariants(self) -> bool:
        """
        Checks if window is still valid, visible, not minimized, not moved, and not resized.
        HALTs immediately if any invariant is violated.
        """
        if self.is_halted:
            return False

        if not self.hwnd or not user32.IsWindow(self.hwnd):
            self.trigger_halt("WINDOW_CLOSED")
            return False

        if user32.IsIconic(self.hwnd):
            self.trigger_halt("WINDOW_MINIMIZED")
            return False

        current_rect = RECT()
        if not user32.GetWindowRect(self.hwnd, ctypes.byref(current_rect)):
            self.trigger_halt("GET_WINDOW_RECT_FAILED")
            return False

        cur_box = (current_rect.left, current_rect.top, current_rect.right, current_rect.bottom)
        if self.baseline_rect is not None and cur_box != self.baseline_rect:
            diff_msg = f"WINDOW_MOVED_OR_RESIZED (Initial: {self.baseline_rect}, Current: {cur_box})"
            self.trigger_halt(diff_msg)
            return False

        return True

    def close(self):
        self.killswitch.stop()
