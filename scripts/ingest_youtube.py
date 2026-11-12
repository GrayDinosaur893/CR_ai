"""
YouTube Video Ingestion & Frame Sampler for Clash Royale Replays.
Allows passing a YouTube match URL to download and prepare frames for CV pipeline.
"""

import os
import sys
import subprocess
import json

def process_youtube_replay(youtube_url: str, output_id: str = "match_yt_01"):
    media_dir = os.path.join("data", "manifests")
    frames_dir = os.path.join("frames", output_id)
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    print(f"=== CLASH IQ: YouTube Ingest Pipeline ===")
    print(f"Target URL: {youtube_url}")
    print(f"Output Match ID: {output_id}")

    # Check for yt-dlp / ffmpeg
    try:
        import yt_dlp
    except ImportError:
        print("\n[!] 'yt-dlp' is required for direct YouTube downloading.")
        print("    Install it via: pip install yt-dlp")
        return

    video_path = os.path.join("media", f"{output_id}.mp4")
    os.makedirs("media", exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': video_path,
        'quiet': False
    }

    print("\n1. Downloading video via yt-dlp...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    manifest = {
        "media_id": output_id,
        "source_type": "youtube",
        "source_url": youtube_url,
        "local_file": video_path,
        "processing_status": "downloaded"
    }

    manifest_file = os.path.join(media_dir, f"{output_id}_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Downloaded to {video_path}")
    print(f"[OK] Created manifest at {manifest_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        process_youtube_replay(url)
    else:
        print("Usage: python scripts/ingest_youtube.py <YOUTUBE_URL>")
