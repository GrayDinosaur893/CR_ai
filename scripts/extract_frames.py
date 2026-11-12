"""
Extract timestamped frames and crop ROIs from recorded Clash Royale videos.
Compliant with 04_DATASET_SCHEMA.md and necessary.md.
"""

import cv2
import os
import sys
import json

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.roi import ClashRoyaleROI

def extract_match_frames(
    video_path: str,
    media_id: str = "match_yt_01",
    sample_interval_sec: float = 2.0,
    max_duration_sec: float = 60.0 # First 60 seconds for initial analysis
):
    output_frames_dir = os.path.join("frames", media_id)
    output_crops_dir = os.path.join("frames", media_id, "crops")
    os.makedirs(output_frames_dir, exist_ok=True)
    os.makedirs(output_crops_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"=== Video Info: {media_id} ===")
    print(f"Resolution: {width}x{height} | FPS: {fps:.2f} | Duration: {duration:.1f}s")

    roi = ClashRoyaleROI(frame_width=width, frame_height=height)
    regions = roi.get_all_regions()

    frame_interval = int(fps * sample_interval_sec)
    max_frames_to_process = int(fps * max_duration_sec)

    extracted_records = []
    frame_idx = 0
    saved_count = 0

    while cap.isOpened() and frame_idx < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = round(frame_idx / fps, 2)
            frame_filename = f"{frame_idx:08d}.png"
            full_frame_path = os.path.join(output_frames_dir, frame_filename)
            cv2.imwrite(full_frame_path, frame)

            # Crop and save hand ROI for preview
            ymin, xmin, ymax, xmax = regions["player_hand"]
            hand_crop = frame[ymin:ymax, xmin:xmax]
            hand_crop_path = os.path.join(output_crops_dir, f"hand_{frame_filename}")
            cv2.imwrite(hand_crop_path, hand_crop)

            extracted_records.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp_sec,
                "frame_path": full_frame_path,
                "hand_crop_path": hand_crop_path
            })
            saved_count += 1

        frame_idx += 1

    cap.release()

    # Save manifest
    manifest = {
        "media_id": media_id,
        "source_file": video_path,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "width": width,
        "height": height,
        "sampled_frames_count": saved_count,
        "sample_interval_sec": sample_interval_sec
    }
    manifest_path = os.path.join("data", "manifests", f"{media_id}_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Extracted {saved_count} frames to {output_frames_dir}")
    print(f"[OK] Saved manifest at {manifest_path}")
    return manifest, extracted_records

if __name__ == "__main__":
    extract_match_frames(
        video_path="media/source/match_yt_01.mp4",
        media_id="match_yt_01",
        sample_interval_sec=2.0,
        max_duration_sec=60.0
    )
