"""
Analyzes an observer video recording (e.g., data/observer_logs/Recording 2026-11-11 100928.mp4)
and optional live daemon log, extracts frame-by-frame ComponentDetector + TacticalActionPlanner
metrics, exports keyframes + multi-panel matplotlib graphs + summary stats.
"""

import os
import sys
import re
import json
import time
import argparse
from collections import Counter
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vision.component_detector import ComponentDetector
from src.strategy.action_planner import TacticalActionPlanner


def parse_exec_log(log_path: Optional[str]) -> List[Dict[str, Any]]:
    """Parses [EXEC HH:MM:SS] lines from live engine log."""
    if not log_path or not os.path.exists(log_path):
        return []
    events = []
    pat = re.compile(
        r"\[EXEC\s+(\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+\|\s+(.*)"
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                events.append({"time": m.group(1), "status": m.group(2), "detail": m.group(3).strip()})
    return events


def analyze_video(
    video_path: str,
    out_dir: str,
    sample_fps: float = 5.0,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    det = ComponentDetector()
    planner = TacticalActionPlanner()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps
    step = max(1, int(round(fps / sample_fps)))

    key_targets_sec = [2, 15, 35, 60, 85, 110, 128, 136, 141]
    saved_keys = set()
    rows: List[Dict[str, Any]] = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t_sec = idx / fps
            t0 = time.perf_counter()
            d = det.detect_components(frame)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            enemies = d["arena_entities"]["enemies"]
            allies = d["arena_entities"]["allies"]
            playable_slots = [s["slot"] for s in d["hand_slots"] if s.get("playable")]
            towers = {k: v.get("alive", True) for k, v in d.get("tower_states", {}).items()}

            dec = {"action": "wait", "slot": None, "target_tile": None}
            if d["screen"] == "battle_arena":
                hand_names = [f"slot_{s['slot']}" for s in d["hand_slots"]]
                dec = planner.decide_next_action(
                    current_elixir=float(d["elixir_estimate"]),
                    opponent_estimated_elixir=10.0,
                    hand_cards=hand_names,
                    arena_threats=enemies,
                    hand_slots=d["hand_slots"],
                    tower_states=d.get("tower_states"),
                )

            rows.append({
                "frame": idx,
                "t": round(t_sec, 2),
                "screen": d["screen"],
                "conf": float(d["screen_confidence"]),
                "layout": d["layout"],
                "elixir": float(d["elixir_estimate"]),
                "droplet": bool(d["droplet_present"]),
                "playable_count": len(playable_slots),
                "enemies": len(enemies),
                "enemies_our_half": sum(1 for e in enemies if e.get("in_our_half")),
                "allies": len(allies),
                "action": dec["action"],
                "slot": dec.get("slot"),
                "target_tile": dec.get("target_tile"),
                "latency_ms": round(lat_ms, 2),
                "towers": towers,
            })

            for ks in key_targets_sec:
                if ks not in saved_keys and t_sec >= ks:
                    saved_keys.add(ks)
                    ann = det.annotate_frame(
                        frame,
                        d,
                        planned_action=dec if dec["action"] != "wait" else None,
                        gate_status="ANALYSIS",
                    )
                    cv2.imwrite(os.path.join(kf_dir, f"t_{ks:03d}s.png"), ann)
        idx += 1

    cap.release()

    screen_counts = Counter(r["screen"] for r in rows)
    action_counts = Counter(r["action"] for r in rows)
    layout_counts = Counter(r["layout"] for r in rows)
    battle_rows = [r for r in rows if r["screen"] == "battle_arena"]
    high_conf_battle = [r for r in battle_rows if r["conf"] >= 0.90]
    false_75_battle = [r for r in battle_rows if r["conf"] < 0.90]
    latencies = [r["latency_ms"] for r in rows]

    exec_events = parse_exec_log(log_path)

    summary = {
        "video_path": video_path,
        "resolution": [vid_w, vid_h],
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "sampled_frames": len(rows),
        "sample_fps": sample_fps,
        "screen_distribution": dict(screen_counts),
        "layout_distribution": dict(layout_counts),
        "action_distribution": dict(action_counts),
        "battle_frames_total": len(battle_rows),
        "battle_frames_high_conf_ge_090": len(high_conf_battle),
        "battle_frames_low_conf_lt_090": len(false_75_battle),
        "avg_elixir_high_conf_battle": round(float(np.mean([r["elixir"] for r in high_conf_battle])), 2) if high_conf_battle else 0.0,
        "avg_enemies_high_conf_battle": round(float(np.mean([r["enemies"] for r in high_conf_battle])), 2) if high_conf_battle else 0.0,
        "avg_enemies_our_half": round(float(np.mean([r["enemies_our_half"] for r in high_conf_battle])), 2) if high_conf_battle else 0.0,
        "latency_ms_mean": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "exec_events_count": len(exec_events),
        "exec_events": exec_events,
    }

    with open(os.path.join(out_dir, "video_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)

    plot_graphs(rows, summary, exec_events, out_dir)
    return summary


def plot_graphs(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    exec_events: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    ts = np.array([r["t"] for r in rows])
    elixir = np.array([r["elixir"] for r in rows])
    conf = np.array([r["conf"] * 100.0 for r in rows])
    enemies = np.array([r["enemies"] for r in rows])
    enemies_half = np.array([r["enemies_our_half"] for r in rows])
    allies = np.array([r["allies"] for r in rows])
    playable = np.array([r["playable_count"] for r in rows])
    latency = np.array([r["latency_ms"] for r in rows])

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.1, 1.1, 0.9]})
    fig.suptitle(
        f"CLASH IQ — Live Gameplay Recording Analysis ({summary['duration_sec']}s | {summary['sampled_frames']} sampled frames)",
        fontsize=14,
        fontweight="bold",
        color="#38BDF8",
        y=0.98,
    )

    # Panel 1: Elixir & Playable Hand Slots
    ax = axes[0]
    ax.plot(ts, elixir, color="#E879F9", linewidth=2.0, label="Measured Elixir (0-10)")
    ax.fill_between(ts, 0, elixir, color="#E879F9", alpha=0.18)
    ax.plot(ts, playable, color="#4ADE80", linewidth=1.3, linestyle="--", label="Playable Hand Slots (0-4)")
    # Shade non-battle or low-conf periods
    for r in rows:
        if r["screen"] != "battle_arena" or r["conf"] < 0.90:
            ax.axvspan(r["t"] - 0.1, r["t"] + 0.1, color="#EF4444", alpha=0.14)
    ax.set_ylabel("Elixir / Slots", fontsize=10)
    ax.set_ylim(-0.3, 10.8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("1. Elixir Curve, Playable Card Slots & Red Shaded Gate/Non-Battle Intervals", fontsize=10.5, loc="left", color="#E2E8F0")

    # Panel 2: Arena Threats vs Allies
    ax = axes[1]
    ax.plot(ts, enemies, color="#F87171", linewidth=1.7, label="Detected Enemy Units")
    ax.fill_between(ts, 0, enemies_half, color="#DC2626", alpha=0.35, label="Enemies in Our Half (row >= 16)")
    ax.plot(ts, allies, color="#60A5FA", linewidth=1.5, label="Detected Ally Units")
    ax.set_ylabel("Unit Count", fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("2. Arena Entity Detection (Enemy Pressure vs Ally Presence)", fontsize=10.5, loc="left", color="#E2E8F0")

    # Panel 3: Screen Classification Confidence & Planned Action Timeline
    ax = axes[2]
    ax.plot(ts, conf, color="#FBBF24", linewidth=1.8, label="Screen Confidence (%)")
    ax.axhline(90.0, color="#22C55E", linestyle=":", linewidth=1.4, label="AccuracyGate Live Threshold (90%)")
    defend_ts = [r["t"] for r in rows if r["action"] == "defend"]
    slow_ts = [r["t"] for r in rows if r["action"] == "slow_roll_back"]
    punish_ts = [r["t"] for r in rows if r["action"] == "opposite_lane_punish"]
    if defend_ts:
        ax.scatter(defend_ts, [58]*len(defend_ts), color="#F87171", s=16, label=f"Plan: defend ({len(defend_ts)})")
    if slow_ts:
        ax.scatter(slow_ts, [52]*len(slow_ts), color="#38BDF8", s=20, label=f"Plan: slow_roll_back ({len(slow_ts)})")
    if punish_ts:
        ax.scatter(punish_ts, [46]*len(punish_ts), color="#A78BFA", s=20, label=f"Plan: punish ({len(punish_ts)})")
    ax.set_ylabel("Confidence %", fontsize=10)
    ax.set_ylim(40, 105)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax.set_title("3. Frame Confidence vs 90% Live Gate & Tactical Action Triggers", fontsize=10.5, loc="left", color="#E2E8F0")

    # Panel 4: Pipeline Latency
    ax = axes[3]
    ax.plot(ts, latency, color="#2DD4BF", linewidth=1.3, label=f"Detector Latency (mean={summary['latency_ms_mean']}ms, p95={summary['latency_ms_p95']}ms)")
    ax.axhline(summary["latency_ms_p95"], color="#F59E0B", linestyle="--", linewidth=1.1, label=f"p95 = {summary['latency_ms_p95']}ms")
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_xlabel("Video Time (seconds)", fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("4. Perception Pipeline Latency Over Session", fontsize=10.5, loc="left", color="#E2E8F0")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "gameplay_timeline_analysis.png"), dpi=160)
    plt.close(fig)

    # Secondary summary breakdown chart (Screen states + Actions + Executions)
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig2.suptitle("CLASH IQ — Screen State & Tactical Action Distribution", fontsize=13, fontweight="bold", color="#38BDF8")

    s_labels = list(summary["screen_distribution"].keys())
    s_vals = list(summary["screen_distribution"].values())
    bx1.bar(s_labels, s_vals, color=["#22C55E" if k == "battle_arena" else "#64748B" for k in s_labels])
    bx1.set_title("Sampled Screen States")
    bx1.set_ylabel("Frame Count")
    for i, v in enumerate(s_vals):
        bx1.text(i, v + max(s_vals) * 0.02, str(v), ha="center", fontsize=9)

    a_labels = list(summary["action_distribution"].keys())
    a_vals = list(summary["action_distribution"].values())
    bx2.bar(a_labels, a_vals, color="#38BDF8")
    bx2.set_title("Planned Tactical Decisions")
    bx2.set_ylabel("Frame Count")
    for i, v in enumerate(a_vals):
        bx2.text(i, v + max(a_vals) * 0.02, str(v), ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(os.path.join(out_dir, "gameplay_distribution_summary.png"), dpi=160)
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"E:\cr_ai\data\observer_logs\Recording 2026-11-11 100928.mp4")
    parser.add_argument("--out-dir", default=r"E:\cr_ai\data\observer_logs\analysis_100928")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    res = analyze_video(args.video, args.out_dir, sample_fps=args.sample_fps, log_path=args.log)
    print(json.dumps({k: v for k, v in res.items() if k != "exec_events"}, indent=2))
    print(f"Total parsed [EXEC] events from log: {res['exec_events_count']}")
