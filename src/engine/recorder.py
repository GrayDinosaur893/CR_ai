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

    @staticmethod
    def _draw_elixir_hand_hud(frame: np.ndarray, payload: Dict[str, Any], record: Dict[str, Any]) -> None:
        """Overlay our + estimated enemy elixir & hand cards as readable text on annotated video."""
        h, w = frame.shape[:2]
        our_elx = float(record.get("elixir") or 0.0)
        elx_mult = float(record.get("elixir_multiplier") or 1.0)
        our_hand = record.get("our_hand") or []
        our_hand_txt = ", ".join(str(c)[:11] for c in our_hand[:4]) if our_hand else "unknown"

        opp_elx = record.get("opp_elixir")
        opp_range = record.get("opp_elixir_range")
        opp_hand = record.get("opp_hand") or []
        opp_plays = int(record.get("opp_plays_count") or 0)
        opp_elx_txt = f"{float(opp_elx):.1f}" if opp_elx is not None else "?"
        if isinstance(opp_range, (list, tuple)) and len(opp_range) == 2:
            opp_elx_txt += f" [{float(opp_range[0]):.1f}-{float(opp_range[1]):.1f}]"
        opp_hand_txt = ", ".join(str(c)[:11] for c in opp_hand[:4]) if opp_hand else "unknown"

        line1 = f"OUR ELX: {our_elx:.1f} ({int(elx_mult)}x) | HAND: [{our_hand_txt}]"
        line2 = f"OPP ELX: {opp_elx_txt} | OPP HAND: [{opp_hand_txt}] (plays:{opp_plays})"

        y_top, y_bot = 24, min(h, 62)
        if y_bot <= y_top + 10:
            return
        cv2.rectangle(frame, (0, y_top), (w, y_bot), (12, 18, 32), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(0.40, w / 1700.0))
        cv2.putText(frame, line1, (8, y_top + 14), font, scale, (100, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (8, y_top + 31), font, scale, (80, 190, 255), 1, cv2.LINE_AA)

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
        opp_intel = payload.get("opponent_intel") or {}
        our_hand = [
            f"{s.get('card_name') or ('slot_' + str(s.get('slot', '?')))}"
            for s in (payload.get("hand_slots") or [])
        ]
        opp_hand = list(opp_intel.get("opp_hand_archetypes") or [])
        opp_elx = opp_intel.get("opp_elixir", payload.get("opponent_elixir_est"))
        opp_range = opp_intel.get("opp_elixir_range")
        record = {
            "t": round(elapsed_sec, 3),
            "timestamp": payload.get("timestamp"),
            "mode": payload.get("mode"),
            "screen": payload.get("screen"),
            "confidence": payload.get("confidence"),
            "elixir": payload.get("elixir"),
            "elixir_multiplier": float(payload.get("elixir_multiplier") or opp_intel.get("elixir_multiplier") or 1.0),
            "our_hand": our_hand,
            "opp_elixir": opp_elx,
            "opp_elixir_range": opp_range,
            "opp_hand": opp_hand,
            "opp_plays_count": opp_intel.get("plays_count", 0),
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
            annotated_frame.copy()
            if (annotated_frame.shape[1], annotated_frame.shape[0]) == (tw, th)
            else cv2.resize(annotated_frame, (tw, th))
        )
        self._draw_elixir_hand_hud(ann_out, payload, record)
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
