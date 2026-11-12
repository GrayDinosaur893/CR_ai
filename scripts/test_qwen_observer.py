"""
TEST SCRIPT: Live Observer test with Qwen VLM.
Flow:
1. WindowGuard attaches to Clash Royale window.
2. CaptureEngine grabs client frame and applies PII redaction.
3. Frame is sent to Qwen VLM via OpenRouter API.
4. SchemaVerifier validates the response structure.
5. AuditLogger records full frame, verbatim raw response, and component crops.
6. Summary is printed strictly from the generated disk logs.
"""

import os
import sys
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.observer.window_guard import WindowGuard
from src.observer.capture_engine import CaptureEngine
from src.observer.vlm_client import QwenVLMClient
from src.observer.verifier import SchemaVerifier
from src.observer.audit_logger import AuditLogger

def main():
    print("=" * 70)
    print("CR_BOT OBSERVER: LIVE QWEN VLM INFERENCE & AUDIT LOG TEST")
    print("=" * 70)

    guard = WindowGuard(target_keywords=("clash royale",))
    capture = CaptureEngine(min_variance_threshold=50.0, min_mse_threshold=0.5, mask_pii=True)
    vlm = QwenVLMClient()
    verifier = SchemaVerifier()
    logger = AuditLogger()

    print(f"[CONFIG] Model: {vlm.model}")
    print(f"[CONFIG] Session Log Directory: {logger.session_dir}")

    try:
        # Step 1: Find game window
        print("\n[STEP 1] Locating Clash Royale window...")
        found = guard.launch_or_find(timeout_sec=5.0)
        if not found or not guard.hwnd:
            print(f"[-] Could not lock game window. Halt reason: {guard.halt_reason}")
            return

        client_rect = guard.get_client_screen_rect()
        print(f"[+] Attached to HWND {guard.hwnd} | Rect: {client_rect}")

        # Step 2: Capture live frame
        print("\n[STEP 2] Capturing client area frame...")
        status, frame, metrics = capture.process(guard.hwnd, client_rect)
        if status != "VALID" or frame is None:
            print(f"[-] Frame capture rejected by pre-filter: {status} (metrics: {metrics})")
            return

        print(f"[+] Frame captured successfully: {frame.shape} (Variance: {metrics['variance']:.1f}, Mean: {metrics['mean']:.1f})")
        print("[+] Pre-VLM PII Redaction applied.")

        # Step 3: Invoke Qwen VLM
        print("\n[STEP 3] Sending sanitized frame to Qwen VLM...")
        success, raw_resp, meta = vlm.infer(frame)
        if not success:
            print(f"[-] VLM call failed: {raw_resp}")
            print(f"[-] Metadata: {meta}")
            return

        print(f"[+] VLM returned in {meta.get('latency_ms', 0):.0f} ms.")

        # Step 4: Verify Schema
        print("\n[STEP 4] Verifying schema against contract...")
        is_valid, verify_status, verified_data = verifier.verify(raw_resp)
        verified_data["status"] = verify_status

        print(f"[+] Verification Result: {verify_status}")
        if not is_valid:
            print(f"[-] Verification Error: {verified_data.get('error')}")

        # Step 5: Audit Log & Crop Extraction
        print("\n[STEP 5] Logging audit record and extracting component crops...")
        record = logger.log_frame(
            frame_idx=0,
            frame=frame,
            raw_vlm_response=raw_resp,
            verified_data=verified_data,
            metadata=meta
        )

        # Step 6: Print Report STRICTLY from disk audit record
        print("\n" + "=" * 70)
        print("VERBATIM AUDIT RECORD (DIRECTLY FROM DISK LOG)")
        print("=" * 70)
        print(json.dumps(record, indent=2))
        print("=" * 70)
        print(f"[+] Audit log written to: {logger.log_file}")
        print(f"[+] Component crops count: {len(record.get('crops', []))}")

    finally:
        guard.close()

if __name__ == "__main__":
    main()
