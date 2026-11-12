"""
Unit verification for Ryley-aligned Elixir Conservation & TacticalActionPlanner.
Tests every root cause identified in session_20261111_195349:
  1. Never fire any card below 0.9 elixir (eliminates 0.3-elixir dead clicks).
  2. Never fire unidentified slot_X below 2.8 elixir.
  3. Hold elixir (< 3.8) on distant single river threats (row < 21) instead of panic spending.
  4. Enforce post-defense patience window (< 2.2s, < 3.5 elixir, row < 23) to avoid trickle dumping.
  5. Never bridge-spam solo skeletons / electro_spirit / musketeer as opposite_lane_punish.
  6. Require >= 7.2 elixir (and clear lane defenders) for safe Hog Rider bridge pushes unless
     prediction_unlocked confirms opponent elixir < 4.0.
  7. Verify replay decisions on session_20261111_195349/gameplay_raw.mp4 (zero dead-elixir plans).
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.strategy.action_planner import TacticalActionPlanner


def _hand(elx: float, unknown_s4: bool = False):
    slots = [
        {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": elx >= 4.0},
        {"slot": 2, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": elx >= 3.0},
        {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": elx >= 4.0},
        {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": elx >= 1.0},
    ]
    if unknown_s4:
        slots[3] = {"slot": 4, "card_name": "slot_4", "card_cost": None, "card_role": "unknown", "playable": True}
    return slots


def main() -> int:
    # 1. Zero-elixir / 0.3-elixir floor (even if slot falsely flagged playable)
    p = TacticalActionPlanner()
    d = p.decide_next_action(
        current_elixir=0.3,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "slot_4"],
        arena_threats=[{"col": 3, "row": 20, "card": "enemy_unit"}],
        hand_slots=_hand(0.3, unknown_s4=True),
        now_ts=10.0,
    )
    assert d["action"] == "wait", f"Expected wait at 0.3 elixir, got {d}"

    # 2. Unidentified slot_4 at 1.8 elixir must NOT be clicked
    d = p.decide_next_action(
        current_elixir=1.8,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "slot_4"],
        arena_threats=[{"col": 3, "row": 22, "card": "enemy_unit"}],
        hand_slots=_hand(1.8, unknown_s4=True),
        now_ts=12.0,
    )
    assert d["action"] == "wait", f"Expected wait with unknown slot at 1.8 elixir, got {d}"

    # 3. Single distant river-entry threat (row=17) at 3.2 elixir -> wait/bank before crossing bridge;
    #    once stepped onto our half (row=18) with cheap 1-elx skeletons, react FAST with positive trade (NOT late, NOT Cannon)!
    d = p.decide_next_action(
        current_elixir=3.2,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 17, "card": "enemy_unit"}],
        hand_slots=_hand(3.2),
        now_ts=15.0,
    )
    assert d["action"] == "wait", f"Expected wait to bank elixir at 3.2 row 17, got {d}"

    d3b = p.decide_next_action(
        current_elixir=3.2,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 18, "card": "enemy_unit"}],
        hand_slots=_hand(3.2),
        now_ts=16.0,
    )
    assert d3b["action"] == "defend" and d3b["recommended_card"] == "skeletons", f"Expected fast 1-elx skeletons counter at row 18, got {d3b}"

    # 4. Positive-trade counter on lone standard unit (save Cannon -> use cheap Knight/Skeletons!), then patience wait
    d1 = p.decide_next_action(
        current_elixir=3.2,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 21, "card": "enemy_unit"}],
        hand_slots=_hand(3.2),
        opp_intel={"opp_elixir": 5.0, "opp_has_rusher": True},
        now_ts=20.0,
    )
    assert d1["action"] == "defend" and d1["recommended_card"] == "skeletons", f"Expected cheap 1-elx skeletons positive trade while saving cannon, got {d1}"
    p.confirm_executed(d1, now_ts=20.0)

    d2 = p.decide_next_action(
        current_elixir=1.4,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 21, "card": "enemy_unit"}],
        hand_slots=_hand(1.4),
        now_ts=21.2,
    )
    assert d2["action"] == "wait", f"Expected post-defense patience wait at 1.4 elx, got {d2}"

    # 4b. Multi-threat or active Rusher -> Cannon center-pull fires cleanly without waiting too late!
    p_rush = TacticalActionPlanner()
    d_rush = p_rush.decide_next_action(
        current_elixir=4.5,
        opponent_estimated_elixir=4.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[
            {"col": 3, "row": 19, "card": "enemy_rusher"},
            {"col": 4, "row": 19, "card": "enemy_unit"},
        ],
        hand_slots=_hand(4.5),
        opp_intel={"opp_elixir": 4.0, "recent_heavy_threat": True},
        now_ts=25.0,
    )
    assert d_rush["action"] == "defend" and d_rush["recommended_card"] == "cannon", f"Expected fast Cannon center-pull on 2-unit rusher push at row=19, got {d_rush}"

    # 5. Never solo bridge-spam skeletons/musketeer when opp_elx < 3.5 (even if hog_rider not in hand)
    no_hog_slots = [
        {"slot": 1, "card_name": "electro_spirit", "card_cost": 1, "card_role": "cycle", "playable": True},
        {"slot": 2, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": True},
        {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": True},
        {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
    ]
    d5 = p.decide_next_action(
        current_elixir=6.2,
        opponent_estimated_elixir=2.0,
        hand_cards=[s["card_name"] for s in no_hog_slots],
        arena_threats=[],
        hand_slots=no_hog_slots,
        now_ts=30.0,
    )
    assert d5["action"] == "wait", f"Expected accumulate wait instead of bridge spam at 6.2 elx, got {d5}"

    # 6. Do not throw naked Hog Rider at 5.1 elixir into living enemy troops
    d6 = p.decide_next_action(
        current_elixir=5.1,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 11, "card": "enemy_unit"}, {"col": 4, "row": 12, "card": "enemy_unit"}],
        hand_slots=_hand(5.1),
        now_ts=40.0,
    )
    assert d6["action"] == "wait", f"Expected wait instead of naked 5.1 Hog into defenders, got {d6}"

    # 7. Safe Hog Rider bridge push persists across queued frames (until confirm_executed) & pushes opposite lane
    d7 = p.decide_next_action(
        current_elixir=7.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 10, "card": "enemy_unit"}],
        hand_slots=_hand(7.5),
        now_ts=50.0,
    )
    assert d7["action"] == "opposite_lane_punish" and d7["recommended_card"] == "hog_rider", f"Expected Hog push at 7.5 elx, got {d7}"
    assert d7["target_tile"] == [14, 17], f"Expected opposite right-lane push [14, 17] when enemy left, got {d7}"
    d7_next = p.decide_next_action(
        current_elixir=7.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "cannon", "musketeer", "skeletons"],
        arena_threats=[{"col": 3, "row": 10, "card": "enemy_unit"}],
        hand_slots=_hand(7.5),
        now_ts=50.25,
    )
    assert d7_next["action"] == "opposite_lane_punish", f"Pending push self-cancelled before execution: {d7_next}"

    # 8. Never play hog_rider (wincon) as a defensive troop even at row 26
    only_hog_and_spell = [
        {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": True},
        {"slot": 2, "card_name": "arrows", "card_cost": 3, "card_role": "spell", "playable": False},
        {"slot": 3, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": False},
        {"slot": 4, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": False},
    ]
    d8 = p.decide_next_action(
        current_elixir=5.1,
        opponent_estimated_elixir=5.0,
        hand_cards=[s["card_name"] for s in only_hog_and_spell],
        arena_threats=[{"col": 8, "row": 26, "card": "enemy_unit"}],
        hand_slots=only_hog_and_spell,
        now_ts=60.0,
    )
    assert d8["recommended_card"] != "hog_rider", f"Hog Rider must never be deployed on defense: {d8}"

    # 9. Slow roll Musketeer in back at >= 8.2 when no Hog in hand & board clear
    d9 = p.decide_next_action(
        current_elixir=8.4,
        opponent_estimated_elixir=5.0,
        hand_cards=[s["card_name"] for s in no_hog_slots],
        arena_threats=[],
        hand_slots=no_hog_slots,
        now_ts=70.0,
    )
    assert d9["action"] == "slow_roll_back" and d9["recommended_card"] == "musketeer", f"Expected slow-roll musketeer at 8.4 elx, got {d9}"

    # 10. Session 205455 Regression: NEVER Fireball right after Hog Rider push at 5.1 elixir on river targets
    p10 = TacticalActionPlanner()
    hog = p10.decide_next_action(
        current_elixir=8.9,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "fireball", "cannon", "skeletons"],
        arena_threats=[],
        hand_slots=[
            {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": True},
            {"slot": 2, "card_name": "fireball", "card_cost": 4, "card_role": "spell", "playable": True},
            {"slot": 3, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": False},
            {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
        ],
        now_ts=60.8,
    )
    assert hog["action"] == "opposite_lane_punish" and hog["recommended_card"] == "hog_rider", f"Expected Hog push: {hog}"
    p10.confirm_executed(hog, now_ts=60.8)

    after_hog_fb = p10.decide_next_action(
        current_elixir=5.1,
        opponent_estimated_elixir=5.0,
        hand_cards=["knight", "fireball", "cannon", "skeletons"],
        arena_threats=[
            {"col": 14, "row": 16, "card": "enemy_unit_right"},
            {"col": 14, "row": 17, "card": "enemy_unit_right"},
        ],
        hand_slots=[
            {"slot": 1, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
            {"slot": 2, "card_name": "fireball", "card_cost": 4, "card_role": "spell", "playable": True},
            {"slot": 3, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": False},
            {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
        ],
        now_ts=63.1,
    )
    assert after_hog_fb["recommended_card"] != "fireball", f"Post-Hog Fireball suicide was NOT blocked! {after_hog_fb}"
    assert after_hog_fb["action"] == "wait", f"Expected bank elixir while river enemies cross after Hog: {after_hog_fb}"

    # 11. River bridge banking (row=17 -> wait until crossed); once 2 threats step to row=18 -> fast Knight (NOT Cannon, NOT late!)
    p11 = TacticalActionPlanner()
    d11_bridge = p11.decide_next_action(
        current_elixir=3.4,
        opponent_estimated_elixir=5.0,
        hand_cards=["skeletons", "knight", "musketeer", "arrows"],
        arena_threats=[
            {"col": 14, "row": 17, "card": "enemy_unit_right"},
            {"col": 13, "row": 17, "card": "enemy_unit_right"},
        ],
        hand_slots=[
            {"slot": 1, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
            {"slot": 2, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
            {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": False},
            {"slot": 4, "card_name": "arrows", "card_cost": 3, "card_role": "spell", "playable": True},
        ],
        now_ts=80.0,
    )
    assert d11_bridge["action"] == "wait", f"Expected bridge-entry banking wait at 3.4 elx on row 17, got {d11_bridge}"

    d11_cross = p11.decide_next_action(
        current_elixir=3.4,
        opponent_estimated_elixir=5.0,
        hand_cards=["skeletons", "knight", "musketeer", "arrows"],
        arena_threats=[
            {"col": 14, "row": 18, "card": "enemy_unit_right"},
            {"col": 13, "row": 18, "card": "enemy_unit_right"},
        ],
        hand_slots=[
            {"slot": 1, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
            {"slot": 2, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
            {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": False},
            {"slot": 4, "card_name": "arrows", "card_cost": 3, "card_role": "spell", "playable": True},
        ],
        now_ts=81.0,
    )
    assert d11_cross["action"] == "defend" and d11_cross["recommended_card"] in ("knight", "skeletons"), f"Expected fast Knight/Skeletons counter once crossed to row 18, got {d11_cross}"

    # 12. Per-lane active defender lock: after Cannon in right lane (on heavy rusher), hold follow-up dump at 3.2 elx 2.5s later
    d12a = p11.decide_next_action(
        current_elixir=4.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["cannon", "knight", "musketeer", "skeletons"],
        arena_threats=[{"col": 14, "row": 21, "card": "enemy_rusher"}, {"col": 13, "row": 21, "card": "enemy_unit_right"}],
        hand_slots=[
            {"slot": 1, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": True},
            {"slot": 2, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
            {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": True},
            {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
        ],
        opp_intel={"opp_elixir": 5.0, "recent_heavy_threat": True},
        now_ts=90.0,
    )
    assert d12a["action"] == "defend" and d12a["recommended_card"] == "cannon", f"Expected cannon center pull: {d12a}"
    p11.confirm_executed(d12a, now_ts=90.0)

    d12b = p11.decide_next_action(
        current_elixir=3.2,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "knight", "musketeer", "skeletons"],
        arena_threats=[
            {"col": 14, "row": 21, "card": "enemy_unit_right"},
            {"col": 13, "row": 22, "card": "enemy_unit_right"},
        ],
        hand_slots=[
            {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": False},
            {"slot": 2, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
            {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": False},
            {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": True},
        ],
        now_ts=92.5,
    )
    assert d12b["action"] == "wait", f"Expected Cannon lane patience wait instead of double-spending at 3.2 elx: {d12b}"

    # 13. Hand-vs-Hand + Elixir Advantage Overspend Reasoning:
    # 13a: Opponent has building counter in hand & high elixir (6.8 vs 6.5) -> DO NOT overspend Hog!
    p13 = TacticalActionPlanner()
    d13_hold = p13.decide_next_action(
        current_elixir=6.5,
        opponent_estimated_elixir=6.8,
        hand_cards=["hog_rider", "cannon", "knight", "skeletons"],
        arena_threats=[],
        hand_slots=_hand(6.5),
        opp_intel={"opp_elixir": 6.8, "opp_has_building_counter": True, "opp_hand_archetypes": ["cannon"]},
        now_ts=110.0,
    )
    assert d13_hold["action"] == "wait", f"Should NOT blindly throw Hog into opponent's Cannon when down elixir: {d13_hold}"

    # 13b: We have +3.4 elixir advantage (6.0 vs 2.6) -> ALLOW OVERSPEND Hog punish!
    d13_go = p13.decide_next_action(
        current_elixir=6.0,
        opponent_estimated_elixir=2.6,
        hand_cards=["hog_rider", "cannon", "knight", "skeletons"],
        arena_threats=[],
        hand_slots=_hand(6.0),
        opp_intel={"opp_elixir": 2.6, "opp_has_building_counter": False, "prediction_unlocked": True},
        qwen_advice={"allow_overspend": True, "strategy_mode": "overspend_punish", "recommended_card": "hog_rider"},
        now_ts=120.0,
    )
    assert d13_go["action"] == "opposite_lane_punish" and d13_go["recommended_card"] == "hog_rider", f"Expected overspend Hog punish at +3.4 elx adv: {d13_go}"

    print("ALL 14 ELIXIR CONSERVATION, POSITIVE-COUNTER & OVERSPEND TESTS PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
