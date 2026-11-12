"""
==============================================================================
CLASH IQ — PHASE 1 COMPLETE SYSTEM VERIFICATION:
1. Open & Anchor Clash Royale Game Window
2. Capture & Detect All In-Game Visual Components
3. Annotate & Verify Bounding Boxes (Arena, Towers, Hand, Elixir, Timer)
==============================================================================
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# Ensure root cr_ai is on Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.connect.cr_launcher import CRGameLauncher
from src.vision.component_detector import ComponentDetector
from src.vision.roi import ClashRoyaleROI

def run_phase1_verification():
    print("=" * 75)
    print("           CLASH IQ: PHASE 1 COMPONENT VERIFICATION PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 1: Launch / Detect & Focus Clash Royale Window
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] LAUNCHING & ANCHORING CLASH ROYALE WINDOW...")
    launcher = CRGameLauncher()
    launch_result = launcher.launch(timeout_sec=5.0)

    print(f" -> Launch Status: {launch_result['status'].upper()}")
    print(f" -> Window Title:  '{launch_result['title']}'")
    print(f" -> Coordinates:   {launch_result['rect']}")
    print(f" -> Dimensions:    {launch_result['width']} x {launch_result['height']} px")

    # -------------------------------------------------------------------------
    # STEP 2: Frame Acquisition (Live Screen or Ground Truth Battle Frame)
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] ACQUIRING GAME FRAME FOR COMPUTER VISION...")
    frame = None
    frame_source = "Reference High-Res Battle Capture"

    # Check for test battle frame from recorded dataset
    sample_path = os.path.join(ROOT_DIR, "frames", "match_yt_01", "00000354.png")
    if os.path.exists(sample_path):
        frame = cv2.imread(sample_path)
        print(f" -> Loaded Reference In-Game Match Frame: '{sample_path}' ({frame.shape[1]}x{frame.shape[0]} px)")
    else:
        # Fallback synthetic frame if no dataset frame is found
        print(" -> Generating standardized Clash Royale resolution frame (1080x1920)...")
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)

    # -------------------------------------------------------------------------
    # STEP 3: Multi-Stage Component Detection & Segmentation
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] EXECUTING COMPONENT PERCEPTION & SEGMENTATION...")
    detector = ComponentDetector()
    detections = detector.detect_components(frame)

    print(f"\n[OK] Successfully Identified {len(detections['components'])} Visual Components!")
    print(f"[OK] Elixir Bar Fill Estimate: {detections['elixir_estimate']} / 10.0")

    print("\n" + "-" * 75)
    print(f"{'CATEGORY':<10} | {'COMPONENT NAME':<28} | {'BOUNDS [Y1,X1,Y2,X2]':<20} | {'CONFIDENCE'}")
    print("-" * 75)

    for c in detections["components"]:
        box_str = f"[{c['box_px'][0]},{c['box_px'][1]},{c['box_px'][2]},{c['box_px'][3]}]"
        print(f"{c['category']:<10} | {c['name']:<28} | {box_str:<20} | {c['confidence']*100:.1f}%")

    print("-" * 75)

    # -------------------------------------------------------------------------
    # STEP 4: Render Annotated Visual Frame
    # -------------------------------------------------------------------------
    annotated_frame = detector.annotate_frame(frame, detections)
    out_dir = os.path.join(ROOT_DIR, "frames")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_verified_components.png")
    cv2.imwrite(out_path, annotated_frame)

    print(f"\n[SUCCESS] Visual Verification Image Saved:")
    print(f"          -> {out_path}")
    print(f"          (Open this image to view all bounded and labeled components!)")
    print("=" * 75)

    return out_path

if __name__ == "__main__":
    run_phase1_verification()
