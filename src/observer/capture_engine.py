"""
OBSERVER MODULE 2: Capture Engine & Pre-Filter Sentinel.
Responsibilities:
1. High-fidelity direct window capture using PrintWindow (PW_RENDERFULLCONTENT=2),
   which captures the window even if other windows overlap on top.
2. Fallback to desktop mss capture with OpenInputDesktop support.
3. Blank / Black frame detection based on pixel variance and mean brightness.
4. Stale frame detection (identical consecutive frames) via frame-to-frame MSE.
5. Pre-VLM PII Redaction: Mask account name / player tag directly on the image BEFORE
   any network transmission or disk storage.
6. Invalidation: Drop invalid frames with status CAPTURE_INVALID; do NOT send to VLM.
"""

import os
import time
import ctypes
import threading
from ctypes import wintypes
from typing import Optional, Tuple, Dict, Any, Callable
import cv2
import numpy as np
import mss

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 2

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD),
        ('biWidth', ctypes.c_long),
        ('biHeight', ctypes.c_long),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', ctypes.c_long),
        ('biYPelsPerMeter', ctypes.c_long),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD)
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ('bmiHeader', BITMAPINFOHEADER),
        ('bmiColors', wintypes.DWORD * 3)
    ]


class CaptureEngine:
    """
    Captures game frames, verifies quality/freshness, and sanitizes PII.
    """
    def __init__(
        self,
        min_variance_threshold: float = 100.0,
        min_mse_threshold: float = 2.0,
        mask_pii: bool = True
    ):
        self.attach_desktop()
        self.sct = mss.MSS()
        self.prev_gray: Optional[np.ndarray] = None
        # Calibratable thresholds (starting baseline, calibrated with real frames)
        self.min_variance_threshold = min_variance_threshold
        self.min_mse_threshold = min_mse_threshold
        self.mask_pii = mask_pii

    def attach_desktop(self):
        """Attaches calling thread to active interactive input desktop session."""
        try:
            hdesk = user32.OpenInputDesktop(0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception as e:
            print(f"[CAPTURE] Warning attaching desktop: {e}")

    def capture_printwindow(self, hwnd: int) -> Optional[np.ndarray]:
        """
        Direct hardware/GDI capture of the window buffer via PrintWindow.
        Does not get obscured by overlapping windows.
        """
        if not hwnd or not user32.IsWindow(hwnd):
            return None

        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None

        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 10 or h <= 10:
            return None

        hdc_screen = user32.GetDC(hwnd)
        if not hdc_screen:
            return None

        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_bm = gdi32.SelectObject(hdc_mem, hbm)

        success = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

        frame = None
        if success:
            bmi = BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = w
            bmi.bmiHeader.biHeight = -h  # top-down DIB
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = 0

            buf = (ctypes.c_ubyte * (w * h * 4))()
            lines = gdi32.GetDIBits(hdc_mem, hbm, 0, h, ctypes.byref(buf), ctypes.byref(bmi), 0)
            if lines == h:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
                frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # Cleanup GDI handles
        gdi32.SelectObject(hdc_mem, old_bm)
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_screen)

        return frame

    def capture_mss(self, client_rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Fallback capture using mss on screen coordinates.
        """
        self.attach_desktop()
        left, top, right, bottom = client_rect
        w = right - left
        h = bottom - top
        if w <= 10 or h <= 10:
            return None

        monitor = {"left": left, "top": top, "width": w, "height": h}
        try:
            sct_img = self.sct.grab(monitor)
            frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
            return frame
        except Exception as e:
            return None

    def redact_pii(self, frame: np.ndarray) -> np.ndarray:
        """
        Redacts player name and account info from the frame BEFORE sending to VLM.
        In Clash Royale lobby: The top-left corner contains player name, level, and trophy banner.
        Redacts region: [0% to 35% Width, 0% to 8% Height].
        """
        if not self.mask_pii or frame is None:
            return frame

        redacted = frame.copy()
        h, w = frame.shape[:2]
        x1 = 0
        y1 = 0
        x2 = int(w * 0.35)
        y2 = int(h * 0.08)

        # Draw opaque neutral banner box
        cv2.rectangle(redacted, (x1, y1), (x2, y2), (25, 25, 25), -1)
        cv2.putText(redacted, "[REDACTED_PII]", (x1 + 10, y1 + int(y2 * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        return redacted

    def evaluate_frame(self, frame: np.ndarray, skip_mse: bool = True) -> Tuple[str, Dict[str, float]]:
        """
        Computes subsampled brightness variance (and optional MSE) in pure NumPy
        without OpenCV calls so background capture threads avoid OpenCV thread-pool contention.
        """
        gray = frame[::8, ::8, 1]
        mean_val = float(gray.mean())
        variance_val = float(gray.var())

        mse_val = 0.0
        is_stale = False
        if (not skip_mse or self.min_mse_threshold > 0.0) and self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            diff = gray.astype(np.float32) - self.prev_gray.astype(np.float32)
            mse_val = float((diff * diff).mean())
            if self.min_mse_threshold > 0.0 and mse_val < self.min_mse_threshold:
                is_stale = True

        metrics = {
            "variance": variance_val,
            "mean": mean_val,
            "mse": mse_val
        }

        # Check for blank / pitch black frame
        if variance_val < self.min_variance_threshold or mean_val < 3.0:
            return "CAPTURE_INVALID_BLANK_OR_BLACK", metrics

        # Check for frozen / identical frame
        if is_stale:
            return "CAPTURE_INVALID_STALE", metrics

        if not skip_mse or self.min_mse_threshold > 0.0:
            self.prev_gray = gray
        return "VALID", metrics

    def process(
        self,
        hwnd: int,
        client_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[str, Optional[np.ndarray], Dict[str, float]]:
        """
        Primary capture entrypoint:
        Tries PrintWindow first (immune to overlapping windows) unless locked out by repeated blanks,
        falling back cleanly to mss without paying double-capture penalty on every tick.
        """
        prefer_mss = getattr(self, "_pw_fail_streak", 0) >= 3 and client_rect is not None
        probe_tick = getattr(self, "_cap_iter", 0)
        self._cap_iter = probe_tick + 1

        raw_frame = None
        if not prefer_mss or (probe_tick % 120 == 0):
            raw_frame = self.capture_printwindow(hwnd)
            if raw_frame is not None and float(np.mean(raw_frame[::8, ::8])) >= 3.0:
                self._pw_fail_streak = 0
            else:
                self._pw_fail_streak = getattr(self, "_pw_fail_streak", 0) + 1
                raw_frame = None

        if raw_frame is None and client_rect is not None:
            mss_frame = self.capture_mss(client_rect)
            if mss_frame is not None and float(np.mean(mss_frame[::8, ::8])) >= 3.0:
                raw_frame = mss_frame

        if raw_frame is None:
            return "CAPTURE_INVALID_GRAB_FAILED", None, {}

        status, metrics = self.evaluate_frame(raw_frame, skip_mse=(self.min_mse_threshold <= 0.0))
        if status != "VALID":
            return status, None, metrics

        sanitized_frame = self.redact_pii(raw_frame)
        return "VALID", sanitized_frame, metrics


class LatestFrameCapture:
    """
    Task 1: Dedicated daemon capture thread maintaining ONLY the latest frame.
    Never queues stale frames: slower consumers automatically drop overwritten intermediate frames.
    Instantiates CaptureEngine inside the worker thread for Windows GDI/mss thread safety.
    """

    def __init__(
        self,
        hwnd: Optional[int] = None,
        rect_provider: Optional[Callable[[], Optional[Tuple[int, int, int, int]]]] = None,
        target_capture_hz: float = 30.0,
        min_variance_threshold: float = 15.0,
        mask_pii: bool = False,
        custom_grab_fn: Optional[Callable[[], Tuple[str, Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]]] = None,
    ):
        self.hwnd = hwnd
        self.rect_provider = rect_provider
        self.target_capture_hz = max(5.0, float(target_capture_hz))
        self.min_variance_threshold = min_variance_threshold
        self.mask_pii = mask_pii
        self.custom_grab_fn = custom_grab_fn

        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest: Optional[Dict[str, Any]] = None
        self.captured_frames = 0
        self.consumed_frames = 0
        self.invalid_frames = 0
        self.last_consumed_seq = 0
        self.capture_ms_history = []

    def start(self) -> "LatestFrameCapture":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="LatestFrameCaptureThread", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        self._new_frame_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        engine = None if self.custom_grab_fn is not None else CaptureEngine(
            min_variance_threshold=self.min_variance_threshold,
            min_mse_threshold=0.0,
            mask_pii=self.mask_pii,
        )
        period = 1.0 / self.target_capture_hz

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            status = "CAPTURE_FAILED"
            frame = None
            rect = None

            try:
                if self.custom_grab_fn is not None:
                    status, frame, rect = self.custom_grab_fn()
                elif engine is not None and self.hwnd:
                    rect = self.rect_provider() if self.rect_provider else None
                    if rect:
                        status, frame, _ = engine.process(self.hwnd, rect)
            except Exception:
                status = "CAPTURE_EXCEPTION"
                frame = None

            t1 = time.perf_counter()
            cap_ms = (t1 - t0) * 1000.0

            if status == "VALID" and frame is not None:
                with self._lock:
                    self.captured_frames += 1
                    seq = self.captured_frames
                    self._latest = {
                        "seq_id": seq,
                        "t_capture": t1,
                        "capture_ms": cap_ms,
                        "status": status,
                        "rect": rect,
                        "frame": frame,
                    }
                    self.capture_ms_history.append(cap_ms)
                    if len(self.capture_ms_history) > 300:
                        self.capture_ms_history.pop(0)
                self._new_frame_event.set()
            else:
                with self._lock:
                    self.invalid_frames += 1

            elapsed = time.perf_counter() - t0
            wait_s = period - elapsed
            if wait_s > 0.0005:
                time.sleep(wait_s)

    def get_latest(self, timeout_sec: float = 0.05) -> Optional[Dict[str, Any]]:
        """
        Non-blocking/bounded-wait retrieval of ONLY the newest unconsumed frame.
        Any intermediate frames produced while consumer was busy are dropped cleanly.
        """
        if not self._new_frame_event.wait(timeout=timeout_sec):
            return None
        with self._lock:
            pkt = self._latest
            if pkt is None or pkt["seq_id"] <= self.last_consumed_seq:
                self._new_frame_event.clear()
                return None
            self.last_consumed_seq = pkt["seq_id"]
            self.consumed_frames += 1
            self._new_frame_event.clear()
            return pkt

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cap = self.captured_frames
            cons = self.consumed_frames
            dropped = max(0, cap - cons)
            drop_pct = round(100.0 * dropped / max(1, cap), 2)
            p95_cap = round(float(np.percentile(self.capture_ms_history, 95)), 2) if self.capture_ms_history else 0.0
            mean_cap = round(float(np.mean(self.capture_ms_history)), 2) if self.capture_ms_history else 0.0
            return {
                "captured_frames": cap,
                "consumed_frames": cons,
                "dropped_frames": dropped,
                "dropped_frames_pct": drop_pct,
                "invalid_frames": self.invalid_frames,
                "capture_ms_mean": mean_cap,
                "capture_ms_p95": p95_cap,
            }
