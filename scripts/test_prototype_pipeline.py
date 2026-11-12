"""
Working Prototype Verification Runner (Zero Live Clicks / Shadow Dry-Run).
Runs the upgraded AutonomousClashPlayer pipeline (CardTemplateMatcher +
filtered ComponentDetector + spell/role-aware TacticalActionPlanner +
opponent elixir tracking) over Recording 2026-11-11 100928.mp4 and compares
against the baseline metrics in video_metrics.json.
"""

import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.engine.autonomous_player import AutonomousClashPlayer


def main() -> int:
    video_path = os.path.join(ROOT_DIR, "data", "observer_logs", "Recording 2026-11-11 100928.mp4")
    baseline_path = os.path.join(ROOT_DIR, "data", "observer_logs", "analysis_100928", "video_metrics.json")
    out_dir = os.path.join(ROOT_DIR, "frames", "prototype_test")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(video_path):
        print(f"[-] Recording not found: {video_path}")
        return 1

    baseline_summary = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_summary = json.load(f).get("summary", {})

    player = AutonomousClashPlayer(live_armed=False, use_vlm_card_id=False, auto_battle=False, record=False)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / 5.0)))  # 5 Hz sampling matching video_metrics.json

    key_secs = [2, 10, 35, 46, 110, 141]
    saved_keys = set()

    action_counts = Counter()
    slot_counts = Counter()
    card_counts = Counter()
    latencies = []
    enemies_total = []
    enemies_half = []
    non_cooldown_slots = 0
    identified_slots = 0
    unaffordable_plans = 0
    single_target_spell_plans = 0
    key_snapshots = []

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t_sec = idx / fps
            t0 = time.perf_counter()
            res = player.process_frame(frame, elapsed_sec=t_sec, tick_steps=2, preview_every_n=999999)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            dec = res["decision"]
            action_counts[dec.get("action", "wait")] += 1
            enemies_total.append(res["enemy_threats_count"])
            enemies_half.append(sum(1 for e in res["enemy_threats"] if e.get("in_our_half")))

            for hs in res["hand_slots"]:
                if hs.get("slot_state") != "cooldown_empty":
                    non_cooldown_slots += 1
                    if not str(hs.get("card_name", "")).startswith("slot_"):
                        identified_slots += 1

            if dec.get("action") != "wait" and dec.get("slot"):
                slot_counts[int(dec["slot"])] += 1
                cname = str(dec.get("recommended_card") or f"slot_{dec['slot']}")
                card_counts[cname] += 1
                chosen = next((s for s in res["hand_slots"] if s["slot"] == dec["slot"]), None)
                if chosen and chosen.get("card_cost") is not None:
                    if res["elixir"] + 0.15 < float(chosen["card_cost"]):
                        unaffordable_plans += 1
                if cname in ("fireball", "arrows") and "High-value" not in dec.get("reasoning", ""):
                    single_target_spell_plans += 1

            for ks in key_secs:
                if ks not in saved_keys and t_sec >= ks:
                    saved_keys.add(ks)
                    det = player.detector.detect_components(frame)
                    ann = player.detector.annotate_frame(
                        frame,
                        det,
                        planned_action=dec if dec.get("action") != "wait" else None,
                        gate_status="PROTO_SHADOW",
                    )
                    cv2.imwrite(os.path.join(out_dir, f"t_{ks:03d}s.png"), ann)
                    hand_str = [f"S{s['slot']}:{s['card_name']}{'*' if s['playable'] else ''}" for s in res["hand_slots"]]
                    key_snapshots.append({
                        "t": round(t_sec, 1),
                        "elixir": res["elixir"],
                        "opp_elixir": res["opponent_elixir_est"],
                        "enemies": res["enemy_threats_count"],
                        "hand": hand_str,
                        "action": dec.get("action"),
                        "card": dec.get("recommended_card"),
                        "slot": dec.get("slot"),
                        "tile": dec.get("target_tile"),
                        "reason": dec.get("reasoning"),
                    })
        idx += 1

    cap.release()

    id_rate = (identified_slots / max(1, non_cooldown_slots)) * 100.0
    print("=" * 78)
    print("CLASH IQ — WORKING PROTOTYPE EVALUATION (SHADOW DRY-RUN ON RECORDING)")
    print("=" * 78)
    print(f"Sampled Frames (5 Hz):       {len(latencies)}")
    print(f"Pipeline Latency Mean / p95: {np.mean(latencies):.2f} ms / {np.percentile(latencies, 95):.2f} ms")
    print(f"Card ID Rate (non-cooldown): {identified_slots}/{non_cooldown_slots} ({id_rate:.1f}%)")
    print(f"Avg Enemies (Before -> Now): {baseline_summary.get('avg_enemies_high_conf_battle', 'N/A')} -> {np.mean(enemies_total):.2f}")
    print(f"Avg Our-Half (Before->Now):  {baseline_summary.get('avg_enemies_our_half', 'N/A')} -> {np.mean(enemies_half):.2f}")
    print(f"Unaffordable Plans (>Elx):   {unaffordable_plans}")
    print(f"Low-Value 1-Unit Spells:     {single_target_spell_plans}")
    print(f"Actions (Before):            {baseline_summary.get('action_distribution', {})}")
    print(f"Actions (Prototype Now):     {dict(action_counts)}")
    print(f"Slot Usage (Prototype):      {dict(sorted(slot_counts.items()))}")
    print(f"Card Picks (Prototype):      {dict(card_counts)}")
    print("-" * 78)
    print("KEY MOMENT SNAPSHOTS:")
    for snap in key_snapshots:
        print(
            f"  t={snap['t']:5.1f}s | Elx:{snap['elixir']:4.1f} (Opp~{snap['opp_elixir']:4.1f}) "
            f"| En:{snap['enemies']} | {snap['hand']}"
        )
        print(
            f"           -> {snap['action']} | S{snap['slot']} ({snap['card']}) -> {snap['tile']} | {snap['reason']}"
        )
    print("=" * 78)
    print(f"Saved visual overlays to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
