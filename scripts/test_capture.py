import ctypes
from ctypes import wintypes
import numpy as np
import cv2

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long)
    ]

def capture_window_hwnd(hwnd):
    rect = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        return None

    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mem_dc, bitmap)

    # Try PrintWindow PW_RENDERFULLCONTENT = 2
    PW_RENDERFULLCONTENT = 2
    res = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
    if not res:
        # Fallback PW_CLIENTONLY = 1 or standard 0
        user32.PrintWindow(hwnd, mem_dc, 0)

    bmp_info = ctypes.create_string_buffer(40) # BITMAPINFOHEADER size
    # Fill BITMAPINFOHEADER
    # biSize = 40, biWidth = w, biHeight = -h (top-down), biPlanes = 1, biBitCount = 32, biCompression = 0
    import struct
    struct.pack_into('<IiiHHIIiiII', bmp_info, 0, 40, w, -h, 1, 32, 0, w * h * 4, 0, 0, 0, 0)

    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, bmp_info, 0)

    # Cleanup
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)

    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return bgr

if __name__ == "__main__":
    # Test on desktop or any window
    hwnd = user32.GetDesktopWindow()
    img = capture_window_hwnd(hwnd)
    print("Desktop capture shape:", img.shape if img is not None else "Failed")
