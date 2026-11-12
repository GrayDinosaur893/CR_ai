"""
ACCURACY EVALUATOR: Benchmark Observer & ComponentDetector against Ground Truth.
Supports:
1. VLM Observer audit log benchmark against data/annotations/manual_labels.json
2. Full CV ComponentDetector benchmark (--benchmark) against data/annotations/benchmark_labels.json
   which writes data/accuracy_report.json for AccuracyGate (>= 90.0% lock).
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vision.component_detector import ComponentDetector
from src.engine.accuracy_gate import AccuracyGate, DEFAULT_REPORT_PATH

EVALUATED_COMPONENT_IDS = {
    "match_timer",
    "opponent_king_tower",
    "opponent_princess_left",
    "opponent_princess_right",
    "player_princess_left",
    "player_princess_right",
    "player_king_tower",
    "hand_slot_1",
    "hand_slot_2",
    "hand_slot_3",
    "hand_slot_4",
    "elixir_bar",
}


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def evaluate(ground_truth_path: str, audit_log_path: str, iou_thresh: float = 0.5):
    if not os.path.exists(ground_truth_path):
        print(f"[-] Ground truth file not found: {ground_truth_path}")
        return
    if not os.path.exists(audit_log_path):
        print(f"[-] Audit log file not found: {audit_log_path}")
        return

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    predictions = {}
    with open(audit_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                fname = os.path.basename(rec.get("frame_path", ""))
                predictions[fname] = rec

    total_gt_frames = len(gt_data.get("frames", []))
    correct_screen_count = 0
    total_tp = total_fp = total_fn = 0
    all_ious = []

    for gt_frame in gt_data.get("frames", []):
        f_name = gt_frame.get("frame_file")
        gt_screen = gt_frame.get("expected_screen")
        gt_components = gt_frame.get("components", [])

        pred = predictions.get(f_name)
        if not pred:
            total_fn += len(gt_components)
            continue

        if pred.get("screen") == gt_screen:
            correct_screen_count += 1

        pred_components = pred.get("components", [])
        matched_preds = set()

        for gt_c in gt_components:
            gt_name = gt_c["name"]
            gt_box = gt_c["box_pct"]
            best_iou = 0.0
            best_p_idx = -1
            for p_idx, p_c in enumerate(pred_components):
                if p_idx in matched_preds:
                    continue
                if p_c.get("name") == gt_name:
                    iou = compute_iou(gt_box, p_c.get("box_pct", [0, 0, 0, 0]))
                    if iou > best_iou:
                        best_iou = iou
                        best_p_idx = p_idx
            if best_iou >= iou_thresh and best_p_idx >= 0:
                total_tp += 1
                matched_preds.add(best_p_idx)
            else:
                total_fn += 1
            all_ious.append(best_iou)

        total_fp += len(pred_components) - len(matched_preds)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    print("=" * 70)
    print("VLM OBSERVER BENCHMARK SUMMARY")
    print(f"Screen State Accuracy:     {correct_screen_count}/{total_gt_frames} ({correct_screen_count/max(1, total_gt_frames)*100:.1f}%)")
    print(f"Component Precision:        {precision * 100:.1f}%")
    print(f"Component Recall:           {recall * 100:.1f}%")
    print(f"Mean IoU (Overall):         {mean_iou:.3f}")
    print("=" * 70)


def evaluate_detector_benchmark(
    benchmark_path: str = os.path.join("data", "annotations", "benchmark_labels.json"),
    output_report_path: str = DEFAULT_REPORT_PATH,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    detector = ComponentDetector()
    frames = gt_data.get("frames", [])

    evaluated = 0
    correct_screens = 0
    total_tp = total_fp = total_fn = 0
    all_ious: List[float] = []
    playability_correct = 0
    playability_total = 0
    elixir_errors: List[float] = []
    latencies_ms: List[float] = []

    print("=" * 75)
    print("CLASH IQ PERCEPTION ACCURACY BENCHMARK (GATE TARGET >= 90.0%)")
    print("=" * 75)

    for entry in frames:
        rel_path = entry["frame_path"]
        abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT_DIR, rel_path)
        img = cv2.imread(abs_path)
        if img is None:
            print(f"[!] Missing frame: {abs_path}")
            continue

        t0 = time.perf_counter()
        det = detector.detect_components(img)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)
        evaluated += 1

        gt_screen = entry["expected_screen"]
        pred_screen = det["screen"]
        screen_ok = pred_screen == gt_screen
        if screen_ok:
            correct_screens += 1

        # Filter evaluated components where detector reported seen=True
        pred_comps = [
            c for c in det.get("components", [])
            if c.get("id") in EVALUATED_COMPONENT_IDS and c.get("seen", False)
        ]
        gt_comps = entry.get("components", [])
        matched_preds = set()
        frame_ious = []

        for gt_c in gt_comps:
            gt_name = gt_c["name"]
            gt_box = gt_c["box_pct"]
            best_iou = 0.0
            best_idx = -1
            for idx, p_c in enumerate(pred_comps):
                if idx in matched_preds:
                    continue
                if p_c.get("id") == gt_name:
                    iou = compute_iou(gt_box, p_c.get("box_pct", [0, 0, 0, 0]))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            if best_iou >= iou_thresh and best_idx >= 0:
                total_tp += 1
                matched_preds.add(best_idx)
            else:
                total_fn += 1
                print(f"  [MISS] {os.path.basename(rel_path)} | {gt_name} IoU={best_iou:.3f}")
            all_ious.append(best_iou)
            frame_ious.append(best_iou)

        unmatched = [pred_comps[i]["id"] for i in range(len(pred_comps)) if i not in matched_preds]
        if unmatched:
            print(f"  [FP]   {os.path.basename(rel_path)} | Extra seen components: {unmatched}")
        total_fp += len(unmatched)

        # Card playability check
        if "expected_playable_slots" in entry:
            gt_play = entry["expected_playable_slots"]
            pred_play_map = {s["slot"]: bool(s["playable"]) for s in det.get("hand_slots", [])}
            for s_idx, expected_flag in enumerate(gt_play, start=1):
                playability_total += 1
                if pred_play_map.get(s_idx, False) == expected_flag:
                    playability_correct += 1
                else:
                    print(f"  [PLAY_ERR] {os.path.basename(rel_path)} slot {s_idx}: expected={expected_flag}, got={pred_play_map.get(s_idx)}")

        # Elixir check
        if "expected_elixir" in entry:
            err = abs(float(det.get("elixir_estimate", 0.0)) - float(entry["expected_elixir"]))
            elixir_errors.append(err)

        f_iou = sum(frame_ious) / len(frame_ious) if frame_ious else (1.0 if screen_ok else 0.0)
        print(
            f"[{evaluated:02d}] {os.path.basename(rel_path):18s} | "
            f"Screen: {pred_screen:<12s} ({'OK' if screen_ok else 'ERR'}) | "
            f"MeanIoU: {f_iou:.3f} | Elixir: {det.get('elixir_estimate', 0.0):4.1f} | {lat_ms:.1f}ms"
        )

    screen_acc = (correct_screens / max(1, evaluated)) * 100.0
    precision = (total_tp / max(1, total_tp + total_fp)) * 100.0
    recall = (total_tp / max(1, total_tp + total_fn)) * 100.0
    mean_iou = sum(all_ious) / max(1, len(all_ious))
    playability_acc = (playability_correct / max(1, playability_total)) * 100.0 if playability_total else 100.0
    elixir_mae = sum(elixir_errors) / max(1, len(elixir_errors)) if elixir_errors else 0.0
    elixir_within_1 = (
        sum(1 for e in elixir_errors if e <= 1.0) / max(1, len(elixir_errors)) * 100.0
        if elixir_errors else 100.0
    )

    overall_acc = (
        0.30 * screen_acc
        + 0.25 * precision
        + 0.25 * recall
        + 0.20 * playability_acc
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark_path": benchmark_path,
        "detector_fingerprint": ComponentDetector.get_source_fingerprint(),
        "evaluated_frames": evaluated,
        "iou_threshold": iou_thresh,
        "screen_accuracy_pct": round(screen_acc, 2),
        "component_precision_pct": round(precision, 2),
        "component_recall_pct": round(recall, 2),
        "playability_accuracy_pct": round(playability_acc, 2),
        "elixir_within_1_pct": round(elixir_within_1, 2),
        "elixir_mae": round(elixir_mae, 3),
        "mean_iou": round(mean_iou, 4),
        "overall_accuracy_pct": round(overall_acc, 2),
        "avg_latency_ms": round(sum(latencies_ms) / max(1, len(latencies_ms)), 2),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    abs_report = output_report_path if os.path.isabs(output_report_path) else os.path.join(ROOT_DIR, output_report_path)
    os.makedirs(os.path.dirname(abs_report), exist_ok=True)
    with open(abs_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    gate = AccuracyGate(report_path=abs_report)
    unlocked, gate_reason, _ = gate.check()

    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY & 90%+ GATE DECISION")
    print("=" * 75)
    print(f"Evaluated Frames:         {evaluated}")
    print(f"Screen State Accuracy:    {screen_acc:.2f}%")
    print(f"Component Precision:      {precision:.2f}% (TP={total_tp}, FP={total_fp})")
    print(f"Component Recall:         {recall:.2f}% (FN={total_fn})")
    print(f"Hand Playability Acc:     {playability_acc:.2f}% ({playability_correct}/{playability_total})")
    print(f"Elixir <=1.0 Agreement:   {elixir_within_1:.2f}% (MAE={elixir_mae:.3f})")
    print(f"Mean Component IoU:       {mean_iou:.3f}")
    print(f"Overall Composite Acc:    {overall_acc:.2f}%")
    print(f"Avg Frame Latency:        {report['avg_latency_ms']:.2f} ms")
    print(f"90%+ Accuracy Gate:       {'[UNLOCKED]' if unlocked else '[LOCKED]'} -> {gate_reason}")
    print(f"Saved Report:             {abs_report}")
    print("=" * 75)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash IQ Accuracy Evaluator")
    parser.add_argument("--benchmark", action="store_true", help="Run full CV ComponentDetector benchmark and update accuracy_report.json")
    parser.add_argument("--labels", default=os.path.join("data", "annotations", "benchmark_labels.json"))
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    if args.benchmark:
        evaluate_detector_benchmark(args.labels, args.report, args.iou)
    else:
        gt_file = os.path.join("data", "annotations", "manual_labels.json")
        audit_file = os.path.join("data", "observer_logs", "20261109_221244", "observer_audit.jsonl")
        evaluate(gt_file, audit_file)
