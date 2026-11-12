"""
Extract gameplay telemetry, elixir-management patterns, play events, and
Learning Ledger / Policy updates from downloaded Ryley - Clash Royale videos.
"""

import glob
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import cv2

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.game_state.elixir_tracker import ElixirTracker
from src.game_state.opponent_tracker import OpponentTracker
from src.vision.component_detector import ComponentDetector


def process_video(
    video_path: str,
    detector: ComponentDetector,
    sample_fps: float = 2.0,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video": os.path.basename(video_path), "error": "failed_to_open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(0.5, sample_fps))))

    opp_tracker = OpponentTracker()
    elixir_tracker = ElixirTracker(initial_elixir=5.0)
    battle_frames = 0
    in_battle_streak = 0
    battle_start_t = None
    prev_elx = None
    last_play_t = -10.0

    events: List[Dict[str, Any]] = []
    ryley_plays: List[Dict[str, Any]] = []

    frame_idx = 0
    while True:
        if not cap.grab():
            break
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue
            t_sec = frame_idx / fps
            det = detector.detect_components(frame)
            raw_battle = det["screen"] == "battle_arena" and det.get("screen_confidence", 0.0) >= 0.88
            in_battle_streak = in_battle_streak + 1 if raw_battle else 0
            in_battle = in_battle_streak >= 2

            if in_battle:
                if battle_start_t is None:
                    battle_start_t = t_sec
                    elixir_tracker = ElixirTracker(initial_elixir=5.0)
                    opp_tracker.reset()
                match_sec = max(0.0, t_sec - battle_start_t)
                elixir_tracker.advance_time(match_sec)
                elx = float(det.get("elixir_estimate", 0.0))
                elixir_tracker.observe_player_elixir(elx, match_sec)

                enemies = det.get("arena_entities", {}).get("enemies", [])
                allies = det.get("arena_entities", {}).get("allies", [])
                opp_intel = opp_tracker.update_from_frame(enemies, match_sec, elixir_tracker)
                battle_frames += 1

                # Detect Ryley card deployment via sharp elixir drop (>= 0.9) with 0.8s debouncing
                play_evt = None
                if prev_elx is not None and (prev_elx - elx) >= 0.9 and (t_sec - last_play_t) >= 0.8:
                    spent = round(prev_elx - elx, 1)
                    in_our_half = any(e.get("row", 0) >= 16 for e in enemies)
                    play_type = "defend" if in_our_half else ("punish_or_push" if elx <= 5.5 else "slow_roll_or_cycle")
                    play_evt = {
                        "video": os.path.basename(video_path),
                        "t": round(t_sec, 2),
                        "match_sec": round(match_sec, 2),
                        "elixir_before": round(prev_elx, 1),
                        "elixir_after": round(elx, 1),
                        "spent": spent,
                        "opp_elixir_est": opp_intel["opp_elixir"],
                        "threats": len(enemies),
                        "allies": len(allies),
                        "multiplier": elixir_tracker.current_multiplier,
                        "play_type": play_type,
                    }
                    ryley_plays.append(play_evt)
                    last_play_t = t_sec

                prev_elx = elx
                events.append(
                    {
                        "video": os.path.basename(video_path),
                        "t": round(t_sec, 2),
                        "screen": "battle_arena",
                        "elixir": elx,
                        "elixir_multiplier": elixir_tracker.current_multiplier,
                        "opp_elixir": opp_intel["opp_elixir"],
                        "threats": len(enemies),
                        "allies": len(allies),
                        "action": play_evt["play_type"] if play_evt else "wait",
                        "executed": bool(play_evt),
                        "human_play": play_evt,
                    }
                )
            else:
                battle_start_t = None
                prev_elx = None

        frame_idx += 1

    cap.release()

    avg_elx_play = (
        round(sum(p["elixir_before"] for p in ryley_plays) / len(ryley_plays), 2)
        if ryley_plays
        else 0.0
    )
    defend_n = sum(1 for p in ryley_plays if p["play_type"] == "defend")
    off_n = len(ryley_plays) - defend_n

    return {
        "video": os.path.basename(video_path),
        "duration_sec": round(total_frames / fps, 1),
        "battle_frames_sampled": battle_frames,
        "ryley_plays_detected": len(ryley_plays),
        "defend_plays": defend_n,
        "offense_or_cycle_plays": off_n,
        "offense_pct": round((off_n / max(1, len(ryley_plays))) * 100.0, 1),
        "avg_elixir_at_play": avg_elx_play,
        "opp_plays_detected": opp_tracker.plays_count,
        "events": events,
        "plays": ryley_plays,
    }


def main() -> int:
    videos = sorted(glob.glob(str(ROOT_DIR / "media" / "ryley_videos" / "*.mp4")))
    if not videos:
        print("[-] No MP4 videos found in media/ryley_videos/")
        return 1

    detector = ComponentDetector()
    out_dir = ROOT_DIR / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "ryley_videos_events.jsonl"
    plays_path = out_dir / "ryley_videos_plays.jsonl"
    summary_path = out_dir / "ryley_videos_summary.json"

    summaries: List[Dict[str, Any]] = []
    all_plays: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    with open(events_path, "w", encoding="utf-8") as fe, open(plays_path, "w", encoding="utf-8") as fp:
        for idx, vp in enumerate(videos, start=1):
            vt0 = time.perf_counter()
            res = process_video(vp, detector, sample_fps=2.0)
            evts = res.pop("events", [])
            plays = res.pop("plays", [])
            for e in evts:
                fe.write(json.dumps(e) + "\n")
            for p in plays:
                fp.write(json.dumps(p) + "\n")
            all_plays.extend(plays)
            res["elapsed_proc_sec"] = round(time.perf_counter() - vt0, 1)
            summaries.append(res)
            print(
                f"[{idx}/{len(videos)}] {res['video']} | dur={res.get('duration_sec')}s "
                f"| battle_frames={res.get('battle_frames_sampled')} "
                f"| ryley_plays={res.get('ryley_plays_detected')} "
                f"| avg_elx_play={res.get('avg_elixir_at_play')} "
                f"| off={res.get('offense_pct')}% ({res['elapsed_proc_sec']}s)"
            )

    total_plays = len(all_plays)
    total_def = sum(s.get("defend_plays", 0) for s in summaries)
    total_off = sum(s.get("offense_or_cycle_plays", 0) for s in summaries)
    agg = {
        "videos_processed": len(summaries),
        "total_battle_frames_sampled": sum(s.get("battle_frames_sampled", 0) for s in summaries),
        "total_ryley_plays": total_plays,
        "total_defend_plays": total_def,
        "total_offense_or_cycle_plays": total_off,
        "overall_offense_pct": round((total_off / max(1, total_plays)) * 100.0, 1),
        "overall_avg_elixir_at_play": round(
            sum(p["elixir_before"] for p in all_plays) / max(1, total_plays), 2
        ),
        "total_proc_sec": round(time.perf_counter() - t0, 1),
        "per_video": summaries,
    }
    summary_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n[+] Saved {events_path}, {plays_path}, {summary_path}")
    print(json.dumps({k: v for k, v in agg.items() if k != "per_video"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
