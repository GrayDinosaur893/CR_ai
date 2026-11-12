"""
CLASH IQ — Live Gameplay & Decision Telemetry Recorder.
Records raw MP4 video, annotated AI-overlay MP4 video, and per-frame JSONL
decision/execution telemetry into recordings/session_<timestamp>/.
"""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np


class GameplayRecorder:
    def __init__(self, base_dir: str = "recordings", target_fps: float = 10.0, prefix: str = "session"):
        self.target_fps = max(1.0, float(target_fps))
        self.min_frame_interval = 1.0 / self.target_fps
        session_name = time.strftime(f"{prefix}_%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)

        self.raw_video_path = os.path.join(self.session_dir, "gameplay_raw.mp4")
        self.annotated_video_path = os.path.join(self.session_dir, "gameplay_annotated.mp4")
        self.events_path = os.path.join(self.session_dir, "events.jsonl")

        self._raw_writer: Optional[cv2.VideoWriter] = None
        self._ann_writer: Optional[cv2.VideoWriter] = None
        self._frame_size: Optional[Tuple[int, int]] = None
        self._events_file = open(self.events_path, "a", encoding="utf-8")
        self._last_video_ts = 0.0
        self.frames_written = 0
        self.events_logged = 0

    def _ensure_writers(self, frame_shape: Tuple[int, ...]) -> None:
        h, w = frame_shape[:2]
        if self._raw_writer is not None and self._frame_size == (w, h):
            return
        if self._raw_writer is not None:
            return  # Keep initial resolution consistent for MP4 stream
        self._frame_size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._raw_writer = cv2.VideoWriter(self.raw_video_path, fourcc, self.target_fps, (w, h))
        self._ann_writer = cv2.VideoWriter(self.annotated_video_path, fourcc, self.target_fps, (w, h))

    def record_step(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        payload: Dict[str, Any],
        elapsed_sec: float = 0.0,
    ) -> None:
        """Logs decision telemetry every step and writes video frames at target_fps."""
        exec_info = payload.get("execution") or {}
        decision = payload.get("decision") or {}
        record = {
            "t": round(elapsed_sec, 3),
            "timestamp": payload.get("timestamp"),
            "mode": payload.get("mode"),
            "screen": payload.get("screen"),
            "confidence": payload.get("confidence"),
            "elixir": payload.get("elixir"),
            "threats": payload.get("enemy_threats_count", 0),
            "action": decision.get("action"),
            "slot": decision.get("slot"),
            "recommended_card": decision.get("recommended_card"),
            "target_tile": decision.get("target_tile"),
            "reasoning": decision.get("reasoning"),
            "exec_status": exec_info.get("status"),
            "executed": bool(exec_info.get("executed")),
            "exec_reason": exec_info.get("reason"),
            "human_play": payload.get("human_play"),
            "qwen_brain": payload.get("qwen_brain"),
            "latency_ms": payload.get("pipeline_latency_ms"),
        }
        self._events_file.write(json.dumps(record) + "\n")
        self.events_logged += 1
        if self.events_logged % 20 == 0 or record["executed"]:
            self._events_file.flush()

        now = time.perf_counter()
        if not record["executed"] and (now - self._last_video_ts) < self.min_frame_interval:
            return

        self._ensure_writers(raw_frame.shape)
        assert self._frame_size is not None
        tw, th = self._frame_size
        raw_out = raw_frame if (raw_frame.shape[1], raw_frame.shape[0]) == (tw, th) else cv2.resize(raw_frame, (tw, th))
        ann_out = (
            annotated_frame
            if (annotated_frame.shape[1], annotated_frame.shape[0]) == (tw, th)
            else cv2.resize(annotated_frame, (tw, th))
        )
        if self._raw_writer is not None and self._raw_writer.isOpened():
            self._raw_writer.write(raw_out)
        if self._ann_writer is not None and self._ann_writer.isOpened():
            self._ann_writer.write(ann_out)
        self._last_video_ts = now
        self.frames_written += 1

    def close(self) -> Dict[str, Any]:
        """Flushes and finalizes MP4 video files and JSONL log."""
        if self._raw_writer is not None:
            self._raw_writer.release()
            self._raw_writer = None
        if self._ann_writer is not None:
            self._ann_writer.release()
            self._ann_writer = None
        if self._events_file is not None and not self._events_file.closed:
            self._events_file.flush()
            self._events_file.close()
        return {
            "session_dir": self.session_dir,
            "raw_video": self.raw_video_path,
            "annotated_video": self.annotated_video_path,
            "events_log": self.events_path,
            "frames_written": self.frames_written,
            "events_logged": self.events_logged,
        }
