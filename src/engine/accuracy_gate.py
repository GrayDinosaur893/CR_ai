"""
CLASH IQ — Strict >= 90% Accuracy Gate Controller.
Prevents autonomous card placement / live input injection until the perception
pipeline passes all ground-truth benchmark thresholds on real labeled frames.
"""

import os
import json
from typing import Dict, Any, Tuple

from src.vision.component_detector import ComponentDetector


DEFAULT_REPORT_PATH = os.path.join("data", "accuracy_report.json")


class AccuracyGate:
    """
    Enforces a strict accuracy lock:
      - screen_accuracy_pct >= min_accuracy_pct (90.0%)
      - component_precision_pct >= min_accuracy_pct (90.0%)
      - component_recall_pct >= min_accuracy_pct (90.0%)
      - overall_accuracy_pct >= min_accuracy_pct (90.0%)
      - mean_iou >= min_mean_iou (0.70)
      - evaluated_frames >= min_frames (8)
      - detector_fingerprint matches current ComponentDetector source SHA-256
    """

    def __init__(
        self,
        report_path: str = DEFAULT_REPORT_PATH,
        min_accuracy_pct: float = 90.0,
        min_mean_iou: float = 0.70,
        min_frames: int = 8,
    ):
        self.report_path = report_path
        self.min_accuracy_pct = float(min_accuracy_pct)
        self.min_mean_iou = float(min_mean_iou)
        self.min_frames = int(min_frames)

    def check(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluates whether autonomous execution is unlocked.
        Returns (unlocked: bool, reason: str, details: dict).
        """
        if not os.path.exists(self.report_path):
            return (
                False,
                f"ACCURACY_REPORT_MISSING ({self.report_path} not generated yet)",
                {"required_accuracy_pct": self.min_accuracy_pct},
            )

        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            return False, f"ACCURACY_REPORT_INVALID: {e}", {}

        current_fp = ComponentDetector.get_source_fingerprint()
        report_fp = report.get("detector_fingerprint")
        if report_fp != current_fp:
            return (
                False,
                f"DETECTOR_CHANGED_REBENCHMARK_REQUIRED (report={report_fp} != current={current_fp})",
                report,
            )

        eval_frames = int(report.get("evaluated_frames", 0))
        if eval_frames < self.min_frames:
            return (
                False,
                f"INSUFFICIENT_EVAL_FRAMES ({eval_frames} < {self.min_frames})",
                report,
            )

        checks = [
            ("screen_accuracy_pct", self.min_accuracy_pct),
            ("component_precision_pct", self.min_accuracy_pct),
            ("component_recall_pct", self.min_accuracy_pct),
            ("overall_accuracy_pct", self.min_accuracy_pct),
            ("mean_iou", self.min_mean_iou),
        ]

        failures = []
        for metric_key, threshold in checks:
            val = float(report.get(metric_key, 0.0))
            if val < threshold:
                failures.append(f"{metric_key}={val:.2f} < {threshold:.2f}")

        if failures:
            return False, "BELOW_90_THRESHOLD: " + "; ".join(failures), report

        return (
            True,
            f"UNLOCKED (overall={report['overall_accuracy_pct']:.1f}%, screen={report['screen_accuracy_pct']:.1f}%, mean_iou={report['mean_iou']:.3f})",
            report,
        )

    def is_live_frame_ready(self, detection: Dict[str, Any], min_conf: float = 0.90) -> Tuple[bool, str]:
        """Per-frame runtime confidence guard before any card drop executes."""
        screen = detection.get("screen")
        conf = float(detection.get("screen_confidence", 0.0))
        if screen != "battle_arena":
            return False, f"NOT_IN_BATTLE (screen={screen})"
        if conf < min_conf:
            return False, f"LOW_FRAME_CONFIDENCE ({conf:.2f} < {min_conf:.2f})"
        components = detection.get("components", [])
        seen_slots = sum(1 for c in components if c.get("id", "").startswith("hand_slot_") and c.get("seen"))
        if seen_slots < 3:
            return False, f"HAND_SLOTS_INCOMPLETE ({seen_slots}/4 seen)"
        return True, "FRAME_READY"
