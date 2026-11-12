"""Unified CLI entry point for the accuracy-gated Clash Royale autonomous AI.

Safe by default: runs in SHADOW DRY-RUN unless `--live` is explicitly supplied
AND `AccuracyGate` confirms >= 90.0% component perception accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_accuracy import evaluate_detector_benchmark
from src.engine.accuracy_gate import AccuracyGate
from src.engine.autonomous_player import AutonomousClashPlayer


def _print_step_summary(result: Dict[str, Any], prefix: str = "") -> None:
    exec_info = result.get("execution", {}) or {}
    decision = result.get("decision") or {}
    tile = decision.get("target_tile")
    action = decision.get("action", "wait")
    plan_txt = (
        f"{action} slot={decision.get('slot')} -> tile={tuple(tile)}"
        if action != "wait" and tile
        else "wait"
    )
    print(
        f"{prefix}screen={result.get('screen')} "
        f"conf={result.get('confidence', 0.0):.2f} "
        f"elixir={result.get('elixir', 0.0):.1f} "
        f"threats={result.get('enemy_threats_count', 0)} "
        f"plan=[{plan_txt}] "
        f"exec={exec_info.get('status')} "
        f"gate={'OPEN' if result.get('gate_unlocked') else 'LOCKED'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 90%+ accuracy-gated autonomous Clash Royale player."
    )
    parser.add_argument(
        "--mode",
        choices=["live", "replay", "image", "benchmark", "human"],
        default="replay",
        help="Execution mode (default: replay; use 'human' to passively record your own match threshold).",
    )
    parser.add_argument(
        "--replay-dir",
        default=os.path.join("frames", "match_yt_01"),
        help="Directory of chronological frames for replay mode.",
    )
    parser.add_argument(
        "--image",
        default=os.path.join("frames", "match_yt_01", "00000354.png"),
        help="Single image path for --mode image.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Arm Win32 mouse clicks (only executes when AccuracyGate >= min-accuracy AND live frame is ready).",
    )
    parser.add_argument(
        "--auto-battle",
        action="store_true",
        help="Automatically click Lobby Battle and Post-Match OK buttons when not in battle_arena.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record raw + annotated gameplay MP4 videos and JSONL action logs to recordings/.",
    )
    parser.add_argument(
        "--record-dir",
        default="recordings",
        help="Output root directory for gameplay recordings (default: recordings).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional max duration in seconds for live mode (0 = run until stopped).",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Force-enable async Qwen VLM tactical brain (automatically ON in --mode live unless --no-vlm).",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable Qwen3-VL real-time brain (run CV+rules only).",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=90.0,
        help="Minimum composite/screen/precision/recall %% required to unlock live execution.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.066,
        help="Target loop period in seconds for live/replay loop (default: 0.066s ~= 15 Hz).",
    )
    parser.add_argument(
        "--benchmark-first",
        action="store_true",
        help="Re-run perception benchmark before launching play/replay.",
    )
    args = parser.parse_args()

    if args.mode == "benchmark" or args.benchmark_first:
        evaluate_detector_benchmark()
        if args.mode == "benchmark":
            gate = AccuracyGate(min_accuracy_pct=args.min_accuracy)
            unlocked, reason, _ = gate.check()
            print(f"[Gate] {reason}")
            return 0 if unlocked else 1

    is_human_mode = args.mode == "human"
    use_qwen_brain = False if (args.no_vlm or is_human_mode) else bool(args.vlm)

    player = AutonomousClashPlayer(
        live_armed=args.live,
        min_accuracy_pct=args.min_accuracy,
        use_vlm_card_id=use_qwen_brain,
        auto_battle=args.auto_battle,
        record=(True if is_human_mode else args.record),
        record_dir=args.record_dir,
        human_observe=is_human_mode,
    )

    if args.mode == "image":
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[Error] Could not read image: {args.image}")
            return 1
        result = player.process_frame(frame, elapsed_sec=1.0, tick_steps=8)
        _print_step_summary(result, prefix="[Image] ")
        print(json.dumps(result, indent=2))
        return 0

    if args.mode == "replay":
        results = player.run_replay(args.replay_dir, delay_sec=args.delay)
        if not results:
            print(f"[Error] No PNG frames found in: {args.replay_dir}")
            return 1
        statuses: Dict[str, int] = {}
        planned_count = 0
        for res in results:
            st = (res.get("execution") or {}).get("status", "UNKNOWN")
            statuses[st] = statuses.get(st, 0) + 1
            if (res.get("decision") or {}).get("action", "wait") != "wait":
                planned_count += 1
        print(
            f"[Replay Summary] frames={len(results)} non_wait_plans={planned_count} "
            f"statuses={statuses}"
        )
        return 0

    # args.mode in ("live", "human")
    player.run_live(tick_interval_sec=args.delay, max_duration_sec=args.duration)
    if player.recorder is not None:
        try:
            from scripts.human_vs_ai_benchmark import generate_all_benchmarks
            generate_all_benchmarks()
        except Exception as e:
            print(f"[!] Benchmark graph update warning: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
