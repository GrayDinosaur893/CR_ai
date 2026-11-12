"""
Vision frame analyzer using OpenRouter Multimodal Free Models.
Sends cropped frame directly to vision LLM to extract visible cards and game state.
"""

import base64
import os
import sys
import json

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client.openrouter import OpenRouterClient

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_clash_frame(frame_path: str):
    print(f"=== Analyzing Clash Frame: {frame_path} ===")
    if not os.path.exists(frame_path):
        print(f"[ERROR] Frame does not exist: {frame_path}")
        return

    base64_image = encode_image(frame_path)
    client = OpenRouterClient(model="nvidia/nemotron-3.5-super-vl:free")

    prompt = (
        "This is an in-game screenshot from Clash Royale. "
        "1. Identify the cards in hand (bottom area) or troops in the arena. "
        "2. Is the match in progress? "
        "3. Keep the output concise and structured as JSON."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    try:
        response = client.generate_chat_completion(messages=messages, max_tokens=1000)
        output = response["choices"][0]["message"].get("content") or response["choices"][0]["message"].get("reasoning")
        print("\n--- Vision Model Response ---")
        try:
            print(output)
        except UnicodeEncodeError:
            print(output.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
        print(f"\n[SUCCESS] Vision model '{response.get('model')}' perception completed!")
    except Exception as e:
        print(f"[ERROR] Vision analysis failed: {e}")

if __name__ == "__main__":
    import glob
    all_frames = sorted(glob.glob("frames/match_yt_01/*.png"))
    # Pick a frame when battle has started (around frame index 6 or 7)
    sample_frame = all_frames[min(6, len(all_frames)-1)] if all_frames else ""
    analyze_clash_frame(sample_frame)
