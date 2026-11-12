"""
OBSERVER MODULE 5: Audit Logger & Crop Extractor.
Responsibilities:
1. Save full captured screenshot for every frame.
2. Save RAW verbatim response from Qwen VLM.
3. Extract and save image crops for every detected component based on Qwen's box_pct.
   (Zero coordinate assumptions; crops are purely derived from model predictions).
4. Append structured audit metadata line to JSONL log.
5. All summaries and reports must be generated strictly from these disk logs.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

class AuditLogger:
    def __init__(self, base_dir: str = "data/observer_logs", session_id: Optional[str] = None):
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, self.session_id)
        self.frames_dir = os.path.join(self.session_dir, "frames")
        self.raw_dir = os.path.join(self.session_dir, "raw")
        self.crops_dir = os.path.join(self.session_dir, "crops")
        self.log_file = os.path.join(self.session_dir, "observer_audit.jsonl")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

    def sanitize_filename(self, name: str) -> str:
        """Sanitizes component name for safe filesystem path."""
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        return clean[:40]

    def log_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        raw_vlm_response: str,
        verified_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Logs a single frame's full visual and structured data.
        Returns the audit record.
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        # 1. Save Full Screenshot
        frame_filename = f"frame_{frame_idx:04d}.png"
        frame_path = os.path.join(self.frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # 2. Save RAW verbatim model response
        raw_filename = f"raw_response_{frame_idx:04d}.json"
        raw_path = os.path.join(self.raw_dir, raw_filename)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_vlm_response)

        # 3. Extract and save crops for each component
        h, w = frame.shape[:2]
        components = verified_data.get("components", [])
        saved_crops = []

        for c_idx, comp in enumerate(components):
            c_name = self.sanitize_filename(comp.get("name", f"comp_{c_idx}"))
            box = comp.get("box_pct", [0, 0, 0, 0])
            x1, y1, x2, y2 = box

            px1 = max(0, min(w - 1, int(x1 * w)))
            px2 = max(0, min(w, int(x2 * w)))
            py1 = max(0, min(h - 1, int(y1 * h)))
            py2 = max(0, min(h, int(y2 * h)))

            if px2 > px1 and py2 > py1:
                crop = frame[py1:py2, px1:px2]
                crop_filename = f"crop_{frame_idx:04d}_{c_idx:02d}_{c_name}.png"
                crop_path = os.path.join(self.crops_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                saved_crops.append({
                    "name": comp.get("name"),
                    "crop_path": crop_path,
                    "box_pct": box,
                    "pixel_box": [px1, py1, px2, py2]
                })

        # 4. Construct Audit Record
        record = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "status": verified_data.get("status", "VALID"),
            "screen": verified_data.get("screen", "UNKNOWN"),
            "confidence": verified_data.get("confidence", "unspecified"),
            "model": metadata.get("model", "unknown"),
            "latency_ms": round(metadata.get("latency_ms", 0.0), 2),
            "frame_path": frame_path,
            "raw_response_path": raw_path,
            "components_count": len(components),
            "components": verified_data.get("components", []),
            "crops": saved_crops,
            "unknown": verified_data.get("unknown", [])
        }

        # 5. Append to JSONL
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record
