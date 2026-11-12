"""
TEST RUNNER: Module 1 (Window Guard & F12 Killswitch) + Module 2 (Capture-only).
Runs without calling any VLM.
Verifies:
1. GPG / Clash Royale window discovery (HWND, client rect, title with account name masked).
2. Geometry invariant checking (HALT on move/resize/minimize).
3. Live client-area capture and quality metrics (real variance, mean, MSE).
4. Client-side PII redaction on saved frame.
5. F12 Hard Killswitch functionality.
"""

import os
import sys
import re
import time
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.observer.window_guard import WindowGuard
from src.observer.capture_engine import CaptureEngine

def mask_title(title: str) -> str:
    # Mask account name if present in title "Clash Royale - <account>"
    return re.sub(r"(Clash Royale\s*-\s*)(.*)", r"\1[MASKED_ACCOUNT]", title, flags=re.IGNORECASE)

def main():
    print("=" * 70)
    print("CR_BOT OBSERVER: MODULE 1 + CAPTURE-ONLY VERIFICATION TEST")
    print("=" * 70)
    print("[!] Instruction: Press F12 at any point (even with GPG focused) to test killswitch.")
    print("[!] Zero input-injection invariant: NO clicks or key injects exist.")

    output_dir = os.path.join("data", "observer_tests")
    os.makedirs(output_dir, exist_ok=True)

    halt_state = {"halted": False, "reason": ""}

    def on_halt(reason: str):
        halt_state["halted"] = True
        halt_state["reason"] = reason
        print(f"\n[EVENT] WindowGuard Halt: {reason}")

    guard = WindowGuard(target_keywords=("clash royale",), on_halt=on_halt)
    capture = CaptureEngine(min_variance_threshold=50.0, min_mse_threshold=0.5, mask_pii=True)

    try:
        print("\n[STEP 1] Finding active Clash Royale window...")
        attached = guard.launch_or_find(timeout_sec=5.0)

        if not attached or not guard.hwnd:
            print(f"[-] Could not lock window. Reason: {guard.halt_reason}")
            return

        client_rect = guard.get_client_screen_rect()
        safe_title = mask_title(guard.window_title)
        print(f"[+] Locked to window: '{safe_title}' (HWND: {guard.hwnd})")
        if client_rect:
            w = client_rect[2] - client_rect[0]
            h = client_rect[3] - client_rect[1]
            print(f"[+] Client Rect: {client_rect} (Width: {w}, Height: {h})")

        print("\n[STEP 2] Running live capture test (up to 5 frames or until F12)...")
        frame_idx = 0
        saved_paths = []

        while frame_idx < 5 and not guard.is_halted:
            loop_start = time.time()

            # 1. Invariant Check
            if not guard.check_invariants():
                print(f"[!] Invariant check failed: {guard.halt_reason}")
                break

            cur_rect = guard.get_client_screen_rect()
            # Pass hwnd and cur_rect to process
            status, frame, metrics = capture.process(guard.hwnd, cur_rect)

            print(f"[FRAME {frame_idx:02d}] Status: {status} | Variance: {metrics.get('variance', 0.0):.1f} | "
                  f"Mean: {metrics.get('mean', 0.0):.1f} | MSE: {metrics.get('mse', 0.0):.2f}")

            if status == "VALID" and frame is not None:
                save_path = os.path.join(output_dir, f"live_capture_{frame_idx:02d}.png")
                cv2.imwrite(save_path, frame)
                saved_paths.append(save_path)

            frame_idx += 1
            elapsed = time.time() - loop_start
            wait_time = max(0.0, 1.2 - elapsed)
            time.sleep(wait_time)

        print("\n[STEP 3] Capture Phase Complete.")
        print(f"Total Frames Processed: {frame_idx}")
        print(f"Saved Frames: {saved_paths}")
        print(f"Halt State: is_halted={guard.is_halted}, reason='{guard.halt_reason}'")

    finally:
        guard.close()
        print("[+] Test script finished cleanly.")

if __name__ == "__main__":
    main()
