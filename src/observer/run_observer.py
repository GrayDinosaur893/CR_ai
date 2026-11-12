"""
CR_BOT OBSERVER: Full Continuous Pipeline Runner.
Orchestrates:
1. WindowGuard: Window discovery/launch, geometry invariant checks, F12 global killswitch.
2. CaptureEngine: Frame grab, variance check, stale check, pre-VLM PII redaction.
3. FastElixirReader: Local sub-3ms pixel elixir telemetry.
4. QwenVLMClient: Vision reasoning (skip-on-busy, no queue).
5. SchemaVerifier: JSON validation, coordinate normalization, PII sanitization.
6. AuditLogger: Full screenshots, verbatim raw responses, component crops, JSONL audit trail.

ZERO input-injection code exists.
"""

import os
import sys
import time
import json
import threading

# Force unbuffered output so stdout flushes immediately to console/logs
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.observer.window_guard import WindowGuard
from src.observer.capture_engine import CaptureEngine
from src.observer.vlm_client import QwenVLMClient
from src.observer.verifier import SchemaVerifier
from src.observer.audit_logger import AuditLogger
from src.vision.fast_elixir_reader import FastElixirReader

def main():
    print("=" * 75, flush=True)
    print("CR_BOT OBSERVER PIPELINE (MODE: PURE OBSERVATION)", flush=True)
    print("=" * 75, flush=True)
    print("[INVARIANT] Zero input injection. No clicks or key inputs exist in code.", flush=True)
    print("[KILLSWITCH] Press F12 at any point (even when game window is active) to HALT.", flush=True)
    print("[WINDOW] Minimizing, resizing, or moving the window triggers instant HALT.", flush=True)

    guard = WindowGuard(target_keywords=("clash royale",))
    capture = CaptureEngine(min_variance_threshold=50.0, min_mse_threshold=0.5, mask_pii=True)
    vlm = QwenVLMClient()
    verifier = SchemaVerifier()
    logger = AuditLogger()
    elixir_reader = FastElixirReader()

    print(f"\n[CONFIG] VLM Model: {vlm.model}", flush=True)
    print(f"[CONFIG] Audit Log Session: {logger.session_dir}", flush=True)

    print("\n[INIT] Searching for Clash Royale window...", flush=True)
    print("[INIT] (Agar game band hai, toh screen par Clash Royale kholo - runner intezaar kar raha hai)", flush=True)

    attached = guard.launch_or_find(timeout_sec=None)

    if not attached or not guard.hwnd:
        print(f"\n[-] Game window nahi mila ya timeout ho gaya. Reason: {guard.halt_reason}", flush=True)
        guard.close()
        return

    client_rect = guard.get_client_screen_rect()
    w = client_rect[2] - client_rect[0] if client_rect else 0
    h = client_rect[3] - client_rect[1] if client_rect else 0
    print(f"[+] Attached to HWND {guard.hwnd} ({w}x{h})", flush=True)

    frame_idx = 0
    consecutive_invalids = 0
    loop_interval = 1.5 # ~1.5s tick

    print("\n[RUNNING] Observer loop active. Monitoring every ~1.5s...", flush=True)

    try:
        while not guard.is_halted:
            tick_start = time.time()

            # 1. Window Invariant Check
            if not guard.check_invariants():
                print(f"\n[HALT] Geometry invariant violated: {guard.halt_reason}", flush=True)
                break

            # 2. Capture Frame & Pre-Filter
            cur_rect = guard.get_client_screen_rect()
            status, frame, metrics = capture.process(guard.hwnd, cur_rect)

            if status != "VALID" or frame is None:
                consecutive_invalids += 1
                if consecutive_invalids % 4 == 1:
                    print(f"[{time.strftime('%H:%M:%S')}] Frame {frame_idx:04d}: {status} (var: {metrics.get('variance', 0):.1f}, mean: {metrics.get('mean', 0):.1f})", flush=True)
                time.sleep(1.0)
                continue

            consecutive_invalids = 0

            # 3. Fast Local Elixir Telemetry (< 3ms)
            elixir_data = elixir_reader.read_elixir(frame)

            # 4. Qwen VLM Inference (Skip-on-busy: drops frame if last request still pending)
            if vlm.is_busy():
                print(f"[{time.strftime('%H:%M:%S')}] Frame {frame_idx:04d}: VLM busy -> SKIPPED (no queue)", flush=True)
            else:
                def async_vlm_task(f_idx, f_img, e_data):
                    success, raw_resp, meta = vlm.infer(f_img)
                    if not success:
                        print(f"[{time.strftime('%H:%M:%S')}] Frame {f_idx:04d}: VLM Error -> {raw_resp}", flush=True)
                        return

                    # Verify Schema
                    is_valid, verify_status, verified_data = verifier.verify(raw_resp)
                    verified_data["status"] = verify_status
                    verified_data["fast_elixir"] = e_data.get("elixir", 0.0)

                    # Audit Log & Crop Extraction
                    record = logger.log_frame(
                        frame_idx=f_idx,
                        frame=f_img,
                        raw_vlm_response=raw_resp,
                        verified_data=verified_data,
                        metadata=meta
                    )

                    comp_names = [c["name"] for c in record.get("components", [])]
                    print(f"[{record['timestamp'][11:]}] Frame {f_idx:04d} | Screen: {record['screen']} "
                          f"| Elixir: {e_data.get('elixir', 0.0)} | VLM Latency: {record['latency_ms']}ms "
                          f"| Components ({record['components_count']}): {comp_names[:4]}...", flush=True)

                worker = threading.Thread(
                    target=async_vlm_task,
                    args=(frame_idx, frame.copy(), elixir_data),
                    daemon=True
                )
                worker.start()

            frame_idx += 1
            elapsed = time.time() - tick_start
            sleep_time = max(0.1, loop_interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt detected.", flush=True)
    finally:
        guard.close()
        print(f"\n[PIPELINE TERMINATED] Final Status: Halted={guard.is_halted}, Reason='{guard.halt_reason}'", flush=True)
        print(f"Session logs: {logger.log_file}", flush=True)

if __name__ == "__main__":
    main()
