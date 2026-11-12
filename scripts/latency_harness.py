"""
Task 0: 3-Timestamp Latency Baseline Harness.

Measures the CURRENT unmodified system before real-time optimizations:
  1. t_capture  -> screen grab complete (live HWND PrintWindow/mss OR video decode)
  2. t_decision -> ComponentDetector + OpponentTracker + TacticalActionPlanner complete
  3. t_tap      -> ActionExecutor shadow/simulated tap & live Win32 tap path timing

Outputs exact breakdown (mean, p50, p95, max) for:
  - capture_ms
  - perception_ms
  - decision_ms
  - tap_dispatch_ms / tap_total_ms
  - capture_to_decision_ms
  - capture_to_tap_ms
  - dropped_frames_pct (when target interval budget is enforced)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.autonomous_player import AutonomousClashPlayer
from src.observer.capture_engine import CaptureEngine, LatestFrameCapture
from src.observer.window_guard import WindowGuard


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(arr.max()), 2),
    }


def benchmark_live_capture(frames: int = 80) -> Dict[str, Any]:
    """
    Benchmarks CaptureEngine.process() speed (PrintWindow + mss fallback).
    Uses Clash Royale HWND if present, else foreground desktop window so
    real Win32 GDI PrintWindow/mss latency is still measured accurately.
    """
    guard = WindowGuard(("clash royale", "google play games"))
    found = guard.find_existing_window()
    target_source = "clash_royale"
    if found:
        guard.hwnd, guard.window_title, _ = found
        rect = guard.get_client_screen_rect()
    else:
        target_source = "foreground_window_proxy"
        user32 = ctypes.windll.user32
        guard.hwnd = int(user32.GetForegroundWindow())
        rect = (0, 0, 720, 1200)
    guard.killswitch.stop()

    cap = CaptureEngine(min_variance_threshold=0.0, min_mse_threshold=0.0, mask_pii=False)
    pw_ms: List[float] = []
    mss_ms: List[float] = []
    proc_ms: List[float] = []

    for _ in range(frames):
        t0 = time.perf_counter()
        _ = cap.capture_printwindow(guard.hwnd)
        pw_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _ = cap.capture_mss(rect or (0, 0, 720, 1200))
        mss_ms.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        _, _, _ = cap.process(guard.hwnd, rect)
        proc_ms.append((time.perf_counter() - t2) * 1000.0)

    return {
        "source": target_source,
        "window_title": guard.window_title or "foreground_proxy",
        "hwnd": guard.hwnd,
        "frames": frames,
        "printwindow_ms": _stats(pw_ms),
        "mss_ms": _stats(mss_ms),
        "capture_process_ms": _stats(proc_ms),
    }


def run_baseline_video(
    video_path: str,
    max_frames: int = 250,
    stride: int = 2,
    simulate_win32_tap_sleep: bool = True,
    target_hz: float = 15.0,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    player = AutonomousClashPlayer(live_armed=False, auto_battle=False, use_vlm_card_id=False)

    capture_ms_list: List[float] = []
    perception_ms_list: List[float] = []
    decision_ms_list: List[float] = []
    tap_ms_list: List[float] = []
    capture_to_decision_ms_list: List[float] = []
    capture_to_tap_ms_list: List[float] = []

    # Simulate synchronous GDI PrintWindow cost per tick on top of frame decode?
    # Keep decode exact AND measure full perception+decision+tap directly, plus estimate synchronous loop drop rate.
    budget_ms = 1000.0 / target_hz
    over_budget = 0
    processed = 0
    action_taps = 0
    frame_idx = 0

    t_session = time.perf_counter()
    while processed < max_frames:
        t_cap_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % max(1, stride) != 0:
            continue
        t_capture = time.perf_counter()
        capture_ms_list.append((t_capture - t_cap_start) * 1000.0)

        # Perception (exact ComponentDetector w/ 8 templates)
        t_p0 = time.perf_counter()
        detection = player.detector.detect_components(frame)
        t_p1 = time.perf_counter()
        perception_ms_list.append((t_p1 - t_p0) * 1000.0)

        # Decision (exact OpponentTracker + TacticalActionPlanner)
        t_d0 = time.perf_counter()
        elapsed_sec = frame_idx / fps
        raw_battle = detection["screen"] == "battle_arena" and detection.get("screen_confidence", 0.0) >= 0.88
        player._battle_streak = player._battle_streak + 1 if raw_battle else 0
        in_battle = player._battle_streak >= 3
        if in_battle and not player._in_battle_prev:
            player.opp_tracker.reset()
            player.tracker.last_timestamp = elapsed_sec
        player._in_battle_prev = in_battle

        player.tracker.advance_time(elapsed_sec)
        enemies = detection.get("arena_entities", {}).get("enemies", [])
        opp_intel = (
            player.opp_tracker.update_from_frame(enemies, elapsed_sec, player.tracker)
            if in_battle
            else player.opp_tracker.get_intel(player.tracker, active_enemies=[])
        )
        hand_slots = detection.get("hand_slots", [])
        hand_names = [s.get("card_name", f"slot_{s['slot']}") for s in hand_slots]
        decision = {"action": "wait", "slot": None, "target_tile": None}
        if in_battle:
            decision = player.planner.decide_next_action(
                current_elixir=float(detection.get("elixir_estimate", 0.0)),
                opponent_estimated_elixir=float(opp_intel["opp_elixir"]),
                hand_cards=hand_names,
                arena_threats=enemies,
                hand_slots=hand_slots,
                tower_states=detection.get("tower_states"),
                opp_intel=opp_intel,
                next_card=detection.get("next_card"),
                qwen_advice=None,
            )
            player.scheduler.submit_decision(decision)
        t_decision = time.perf_counter()
        decision_ms_list.append((t_decision - t_d0) * 1000.0)
        capture_to_decision_ms_list.append((t_decision - t_cap_start) * 1000.0)

        # Tap stage (shadow gate compute + real Win32 2-click sleep cost if action triggered)
        t_tap0 = time.perf_counter()
        fired_action = None
        for _ in range(4 if decision.get("emergency") else 2):
            fired, act, _ = player.scheduler.tick()
            if fired and act is not None:
                fired_action = act
                break
        if in_battle and fired_action is not None:
            res = player.executor.execute(fired_action, detection, client_origin=(0, 0), client_size=(692, 1136))
            if res.get("status") == "SHADOW_DRY_RUN":
                action_taps += 1
                if simulate_win32_tap_sleep:
                    # Measure current _win32_click x2 wait duration in ActionExecutor (15+25 + 45 + 15+25 = 125ms)
                    time.sleep(0.125)
                player.planner.confirm_executed(fired_action)
        t_tap = time.perf_counter()

        tap_ms_list.append((t_tap - t_tap0) * 1000.0)
        c2t = (t_tap - t_cap_start) * 1000.0
        capture_to_tap_ms_list.append(c2t)
        if c2t > budget_ms:
            over_budget += 1
        processed += 1

    cap.release()
    total_sec = time.perf_counter() - t_session
    return {
        "video_path": video_path,
        "frames_processed": processed,
        "actions_triggered": action_taps,
        "target_hz": target_hz,
        "budget_ms": round(budget_ms, 2),
        "over_budget_pct": round(100.0 * over_budget / max(1, processed), 2),
        "throughput_hz": round(processed / max(1e-6, total_sec), 2),
        "templates_loaded": len(player.detector.card_matcher.templates),
        "capture_ms": _stats(capture_ms_list),
        "perception_ms": _stats(perception_ms_list),
        "decision_ms": _stats(decision_ms_list),
        "tap_ms": _stats(tap_ms_list),
        "capture_to_decision_ms": _stats(capture_to_decision_ms_list),
        "capture_to_tap_ms": _stats(capture_to_tap_ms_list),
    }


def run_threaded_latest_bench(
    video_path: str,
    target_hz: float = 15.0,
    duration_sec: float = 4.0,
) -> Dict[str, Any]:
    """
    Task 1 Gate Benchmark:
    Producer thread pushes frames at ~30 Hz (real decoded video buffer + lock overwrite);
    Consumer runs perception + planner + shadow tap at target_hz (15 Hz), dropping stale frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames_pool: List[np.ndarray] = []
    # Load active battle frames (skip first ~5s lobby)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    while len(frames_pool) < 120:
        ok, fr = cap.read()
        if not ok:
            break
        frames_pool.append(fr)
    cap.release()
    if not frames_pool:
        raise RuntimeError("No frames loaded for threaded benchmark")

    player = AutonomousClashPlayer(live_armed=False, auto_battle=False, use_vlm_card_id=False)
    # Warm up OpenCV/slot cache before measuring steady-state real-time loop
    _ = player.detector.detect_components(frames_pool[0])
    idx_box = {"i": 0}

    def _grab_from_pool():
        i = idx_box["i"]
        idx_box["i"] = (i + 1) % len(frames_pool)
        fr = frames_pool[i]
        return "VALID", fr, (0, 0, fr.shape[1], fr.shape[0])

    cap_thread = LatestFrameCapture(target_capture_hz=30.0, custom_grab_fn=_grab_from_pool).start()

    perc_ms: List[float] = []
    dec_ms: List[float] = []
    c2d_ms: List[float] = []
    c2t_ms: List[float] = []
    period = 1.0 / target_hz
    t_start = time.perf_counter()
    ticks = 0

    try:
        while (time.perf_counter() - t_start) < duration_sec:
            t_tick0 = time.perf_counter()
            pkt = cap_thread.get_latest(timeout_sec=0.1)
            if pkt is None:
                continue
            elapsed = t_tick0 - t_start
            t_p0 = time.perf_counter()
            res = player.process_frame(
                pkt["frame"],
                elapsed_sec=elapsed,
                client_origin=(0, 0),
                tick_steps=4,
                client_size=(pkt["frame"].shape[1], pkt["frame"].shape[0]),
                preview_every_n=999,
                t_capture=pkt["t_capture"],
            )
            lat = res.get("latency_breakdown", {})
            frame_age_ms = max(0.0, (t_tick0 - pkt["t_capture"]) * 1000.0)
            perc_v = float(lat.get("perception_ms", 0.0))
            c2d_v = float(lat.get("capture_to_decision_ms", 0.0))
            perc_ms.append(perc_v)
            c2d_ms.append(c2d_v)
            c2t_ms.append(float(lat.get("capture_to_tap_ms", 0.0)))
            dec_ms.append(max(0.0, c2d_v - perc_v - frame_age_ms))
            ticks += 1

            sleep_left = period - (time.perf_counter() - t_tick0)
            if sleep_left > 0.001:
                time.sleep(sleep_left)
    finally:
        cap_stats = cap_thread.stats()
        cap_thread.stop()

    return {
        "ticks": ticks,
        "target_hz": target_hz,
        "templates_loaded": len(player.detector.card_matcher.templates),
        "capture_stats": cap_stats,
        "perception_ms": _stats(perc_ms),
        "decision_ms": _stats(dec_ms),
        "capture_to_decision_ms": _stats(c2d_ms),
        "capture_to_tap_ms": _stats(c2t_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 0/1 Latency Baseline & Gate Harness")
    parser.add_argument("--video", default=os.path.join("data", "observer_logs", "Recording 2026-11-11 100928.mp4"))
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--live-bench-frames", type=int, default=60)
    parser.add_argument("--threaded", action="store_true", help="Run Task 1 threaded LatestFrameCapture benchmark.")
    parser.add_argument("--out", default=os.path.join("data", "benchmarks", "task0_baseline.json"))
    args = parser.parse_args()

    live_cap = benchmark_live_capture(frames=args.live_bench_frames)
    if args.threaded:
        threaded_bench = run_threaded_latest_bench(args.video, target_hz=15.0, duration_sec=4.0)
        p95_perc = threaded_bench["perception_ms"]["p95"]
        p95_c2t = threaded_bench["capture_to_tap_ms"]["p95"]
        report = {
            "task": "Task 1 Threaded LatestFrameCapture Gate",
            "live_capture_benchmark": live_cap,
            "threaded_15hz_benchmark": threaded_bench,
            "gates": {
                "perception_p95_le_33ms": {"measured": p95_perc, "target": 33.0, "passed": p95_perc <= 33.0},
                "capture_to_tap_p95_le_100ms": {"measured": p95_c2t, "target": 100.0, "passed": p95_c2t <= 100.0},
                "dropped_frames_pct_gt_0": {
                    "measured": threaded_bench["capture_stats"]["dropped_frames_pct"],
                    "passed": threaded_bench["capture_stats"]["dropped_frames_pct"] > 0.0,
                },
            },
        }
        out_path = os.path.join("data", "benchmarks", "task1_threaded_gate.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return 0 if (p95_perc <= 33.0 and p95_c2t <= 100.0) else 2

    video_bench = run_baseline_video(
        video_path=args.video,
        max_frames=args.max_frames,
        stride=args.stride,
        simulate_win32_tap_sleep=True,
        target_hz=15.0,
    )

    # Estimate combined synchronous live loop capture+perception+decision+tap p95
    est_live_sync_c2d_p95 = round(
        live_cap["capture_process_ms"]["p95"] + video_bench["perception_ms"]["p95"] + video_bench["decision_ms"]["p95"],
        2,
    )
    est_live_sync_c2t_p95 = round(
        live_cap["capture_process_ms"]["p95"] + video_bench["capture_to_tap_ms"]["p95"],
        2,
    )

    report = {
        "task": "Task 0 Baseline (pre-optimization)",
        "live_capture_benchmark": live_cap,
        "replay_baseline": video_bench,
        "combined_synchronous_estimate_ms": {
            "capture_to_decision_p95_ms": est_live_sync_c2d_p95,
            "capture_to_tap_p95_ms": est_live_sync_c2t_p95,
        },
        "gates": {
            "perception_p95_le_33ms": {
                "measured": video_bench["perception_ms"]["p95"],
                "target": 33.0,
                "passed": video_bench["perception_ms"]["p95"] <= 33.0,
            },
            "capture_to_tap_p95_le_100ms": {
                "measured": est_live_sync_c2t_p95,
                "target": 100.0,
                "passed": est_live_sync_c2t_p95 <= 100.0,
            },
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
