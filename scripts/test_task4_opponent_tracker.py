"""
Task 4 Gate Test: OpponentTracker Spatial Persistence, Elixir Tracking & Threat Intel.
"""
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.game_state.elixir_tracker import ElixirTracker
from src.game_state.opponent_tracker import OpponentTracker


def run_tests() -> dict:
    tracker = ElixirTracker(initial_elixir=5.0)
    opp = OpponentTracker()

    # 1. Idle tick for 5s -> opp elixir regens from ~5.0 toward ~6.8, NOT stuck at 10.0 on spawn
    intel = opp.update_from_frame([], 5.0, tracker)
    idle_mid = intel["opp_elixir"]
    assert 6.5 <= idle_mid <= 7.0, f"Unexpected idle opp_elixir={idle_mid}"

    # 2. Enemy Hog/rusher/standard spawns at opponent bridge/side (t=6.0s)
    e1 = [{"col": 3, "row": 11, "lane": "left", "in_our_half": False, "norm_vp": [0.22, 0.36]}]
    intel = opp.update_from_frame(e1, 6.0, tracker)
    assert opp.plays_count == 1, f"Expected 1 play after first spawn, got {opp.plays_count}"
    after_spawn_mid = intel["opp_elixir"]
    assert after_spawn_mid < 6.0, f"Expected elixir drop after spawn, got {after_spawn_mid}"

    # 3. Smooth walk (t=6.5..8.5s), then unit blinks away for 0.6s (t=8.8..9.1s), reappears at t=9.2s
    for ts, r, y in [(6.8, 12, 0.38), (7.6, 13, 0.41), (8.5, 14, 0.44)]:
        opp.update_from_frame([{"col": 3, "row": r, "lane": "left", "in_our_half": False, "norm_vp": [0.22, y]}], ts, tracker)
    # Disappears for 0.6s (within 1.2s blink grace TTL)
    opp.update_from_frame([], 8.8, tracker)
    opp.update_from_frame([], 9.1, tracker)
    # Reappears 1.5 tiles lower at bridge (cooldown elapsed > 2.2s since t=6.0s -> WOULD double-count without track grace TTL!)
    e1_reappear = [{"col": 3, "row": 16, "lane": "left", "in_our_half": True, "norm_vp": [0.22, 0.49]}]
    intel = opp.update_from_frame(e1_reappear, 9.2, tracker)
    assert opp.plays_count == 1, f"Blink caused false duplicate play! plays_count={opp.plays_count}"

    # 4. Double-elixir regen rate verification (t=120s..122.8s -> +2.0 elixir)
    t2 = ElixirTracker(initial_elixir=5.0)
    t2.last_timestamp = 120.0
    t2.advance_time(122.8)
    double_gain = t2.get_state()["opponent_elixir_estimate"]["mid"] - 5.0
    assert abs(double_gain - 2.0) <= 0.15, f"Double elixir gain mismatch: {double_gain}"

    # 5. Simultaneous Skarmy/Goblin swarm burst (3 units on same frame t=15.0s) -> counts as 1 swarm play, push_detected=True
    swarm = [
        {"col": 14, "row": 12, "lane": "right", "in_our_half": False, "norm_vp": [0.76, 0.38]},
        {"col": 14, "row": 13, "lane": "right", "in_our_half": False, "norm_vp": [0.78, 0.40]},
        {"col": 15, "row": 13, "lane": "right", "in_our_half": False, "norm_vp": [0.81, 0.41]},
    ]
    intel = opp.update_from_frame(e1_reappear + swarm, 15.0, tracker)
    assert opp.plays_count == 2, f"Swarm should count as 1 burst play (total 2), got {opp.plays_count}"
    assert intel["right_lane_threats"] == 3
    assert intel["left_lane_threats"] == 1
    assert intel["our_half_threats"] == 1
    assert intel["push_detected"] is True
    assert intel["opp_elixir"] < 10.0

    # 6. Air unit crossing middle of river (outside bridge cols 2..4 & 13..15)
    air_u = [{"col": 9, "row": 15, "lane": "center", "in_our_half": False, "norm_vp": [0.50, 0.47]}]
    intel = opp.update_from_frame(air_u, 20.0, tracker)
    assert intel["active_air_threat"] is True
    assert opp.plays_count == 3

    return {
        "task": "Task 4 OpponentTracker & Elixir Counter Gate",
        "idle_opp_elixir": idle_mid,
        "after_spawn_opp_elixir": after_spawn_mid,
        "blink_dedup_ok": True,
        "double_elixir_2_8s_gain": round(double_gain, 2),
        "swarm_burst_plays_total": opp.plays_count,
        "lane_threat_summary": {
            "left": 1,
            "right": 3,
            "our_half": 1,
            "push_detected": True,
        },
        "gate_pass": True,
    }


if __name__ == "__main__":
    report = run_tests()
    print(json.dumps(report, indent=2))
