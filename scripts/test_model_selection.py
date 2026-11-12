import os
import sys
sys.path.insert(0, ".")
from src.observer.vlm_client import QwenVLMClient
import cv2

frame = cv2.imread("data/observer_tests/live_capture_00.png")

models = [
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen/qwen-2.5-vl-72b-instruct",
    "qwen/qwen2.5-vl-72b-instruct"
]

for m in models:
    client = QwenVLMClient(model=m)
    print(f"Testing model: {m}...")
    success, resp, meta = client.infer(frame)
    print(f"Success: {success} | Latency: {meta.get('latency_ms', 0):.0f}ms")
    if success:
        print(f"Working model found: {m}")
        print("Raw response:\n", resp)
        break
    else:
        print("Error snippet:", resp[:250])
