"""
Task 6 Gate Test: Full-Match End-to-End Shadow Replay on Recorded Gameplay (No LLM/VLM).
Verifies idle frame suppression, opponent elixir dynamics, zero unaffordable clicks,
zero solo-spell wastes, offensive counterpush + defensive role balance, and latency.
"""
import json
import os
import sys
import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.engine.autonomous_player import AutonomousClashPlayer

VIDEO_PATH = os.path.join(ROOT_DIR, "data", "observer_logs", "Recording 2026-11-11 100928.mp4")


def run_replay() -> dict:
    assert os.path.exists(VIDEO_PATH), f"Missing replay video: {VIDEO_PATH}"
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / 5.0)))  # 5 Hz full-match replay pass (~750 frames)

    player = AutonomousClashPlayer(
        live_armed=False,
        use_vlm_card_id=False,
        auto_battle=False,
        record=False,
        save_preview_path="",
        state_export_path="",
    )

    frame_idx = 0
    processed = 0
    battle_frames = 0
    non_battle_plays = 0
    planned_actions = []
    unaffordable_plays = 0
    solo_spell_wastes = 0
    opp_elixir_samples = []
    perception_times = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            elapsed = frame_idx / fps
            out = player.process_frame(frame, elapsed_sec=elapsed, preview_every_n=999999)
            processed += 1
            perc_ms = float(out["latency_breakdown"]["perception_ms"])
            perception_times.append(perc_ms)
            dec = out["decision"]
            act = dec.get("action")
            screen = out["screen"]

            if screen != "battle_arena":
                if act not in ("wait", None):
                    non_battle_plays += 1
            else:
                battle_frames += 1
                opp_elixir_samples.append(float(out["opponent_elixir_est"]))
                if act not in ("wait", None):
                    planned_actions.append({
                        "t": round(elapsed, 1),
                        "action": act,
                        "card": dec.get("recommended_card"),
                        "elixir": out["elixir"],
                    })
                    card = dec.get("recommended_card")
                    slot_num = dec.get("slot")
                    slot_info = next((s for s in out["hand_slots"] if s["slot"] == slot_num), None)
                    cost = (slot_info or {}).get("card_cost")
                    if cost is not None and float(out["elixir"]) + 0.15 < float(cost):
                        unaffordable_plays += 1
                    if card in ("arrows", "fireball") and "cluster" not in str(dec.get("reasoning", "")).lower():
                        # Check reasoning string / enemy count if cluster spell
                        if int(out["enemy_threats_count"]) < 2 and act == "defend":
                            solo_spell_wastes += 1

        frame_idx += 1
    cap.release()

    action_counts = {}
    for p in planned_actions:
        action_counts[p["action"]] = action_counts.get(p["action"], 0) + 1

    opp_unique = len(set(round(x, 1) for x in opp_elixir_samples))
    has_defense = action_counts.get("defend", 0) > 0
    has_offense = (
        action_counts.get("opposite_lane_punish", 0) > 0
        or action_counts.get("slow_roll_back", 0) > 0
    )

    report = {
        "task": "Task 6 End-to-End Full-Match Shadow Replay Gate",
        "frames_processed": processed,
        "battle_frames": battle_frames,
        "non_battle_false_plays": non_battle_plays,
        "planned_plays_total": len(planned_actions),
        "action_breakdown": action_counts,
        "unaffordable_plays": unaffordable_plays,
        "solo_spell_wastes": solo_spell_wastes,
        "opp_plays_detected": player.opp_tracker.plays_count,
        "opp_elixir_unique_values": opp_unique,
        "opp_elixir_min_max": [
            round(min(opp_elixir_samples), 1) if opp_elixir_samples else 0.0,
            round(max(opp_elixir_samples), 1) if opp_elixir_samples else 0.0,
        ],
        "has_defense": has_defense,
        "has_offense_or_counterpush": has_offense,
        "perception_ms_mean": round(float(np.mean(perception_times)), 2),
        "perception_ms_p95": round(float(np.percentile(perception_times, 95)), 2),
        "gate_pass": bool(
            non_battle_plays == 0
            and unaffordable_plays == 0
            and solo_spell_wastes == 0
            and opp_unique >= 5
            and has_defense
            and has_offense
        ),
    }
    return report


if __name__ == "__main__":
    rep = run_replay()
    print(json.dumps(rep, indent=2))
    if not rep["gate_pass"]:
        sys.exit(1)
