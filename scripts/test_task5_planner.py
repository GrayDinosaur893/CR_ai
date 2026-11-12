"""
Task 5 Gate Test: TacticalActionPlanner Hard Guards, Spell Value, Role Priority & Anti-Leak.
"""
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.strategy.action_planner import TacticalActionPlanner


def _slots(names, costs=None, roles=None, playable=None):
    default_costs = {
        "skeletons": 1, "electro_spirit": 1, "arrows": 3, "cannon": 3,
        "knight": 3, "fireball": 4, "musketeer": 4, "hog_rider": 4,
        "goblin_barrel": 3, "minions": 3, "archers": 3,
    }
    default_roles = {
        "skeletons": "cycle", "electro_spirit": "cycle", "arrows": "spell", "cannon": "building",
        "knight": "minitank", "fireball": "spell", "musketeer": "ranged", "hog_rider": "wincon",
        "goblin_barrel": "wincon", "minions": "air", "archers": "ranged",
    }
    out = []
    for idx, n in enumerate(names, start=1):
        out.append({
            "slot": idx,
            "card_name": n,
            "card_cost": (costs or {}).get(n, default_costs.get(n)),
            "card_role": (roles or {}).get(n, default_roles.get(n, "unknown")),
            "playable": (playable[idx - 1] if playable else True),
        })
    return out


def run_tests() -> dict:
    planner = TacticalActionPlanner()

    # 1. Hard unaffordable card guard (even when hand_cards fallback is used)
    d1 = planner.decide_next_action(
        current_elixir=2.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "fireball", "cannon", "arrows"],
        arena_threats=[{"col": 3, "row": 21, "lane": "left", "in_our_half": True}],
        now_ts=10.0,
    )
    assert d1["action"] == "wait", f"Played unaffordable card at 2.0 elixir: {d1}"

    # 2. Spell guard against single unit even deep near tower (row=26) -> never solo-Fireball/Arrows
    d2 = planner.decide_next_action(
        current_elixir=4.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["fireball", "arrows", "hog_rider", "knight"],
        hand_slots=_slots(["fireball", "arrows", "hog_rider", "knight"]),
        arena_threats=[{"col": 3, "row": 26, "lane": "left", "in_our_half": True}],
        now_ts=20.0,
    )
    assert d2.get("recommended_card") not in ("fireball", "arrows"), f"Solo spell waste! {d2}"
    assert d2.get("recommended_card") == "knight", f"Expected knight positive trade, got {d2}"

    # 3. Spell guard with only spells + wincon on lone deep unit -> MUST NOT solo-spell
    d3 = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["fireball", "arrows", "hog_rider", "slot_4"],
        hand_slots=_slots(["fireball", "arrows", "hog_rider", "slot_4"], playable=[True, True, True, False]),
        arena_threats=[{"col": 3, "row": 26, "lane": "left", "in_our_half": True}],
        now_ts=30.0,
    )
    assert d3["action"] == "wait", f"Must wait instead of solo spell or defensive hog! {d3}"

    # 4. Clustered enemy swarm (>=3 enemies within spell radius) -> triggers spell on cluster
    swarm = [
        {"col": 3, "row": 20, "lane": "left", "in_our_half": True},
        {"col": 4, "row": 20, "lane": "left", "in_our_half": True},
        {"col": 4, "row": 21, "lane": "left", "in_our_half": True},
    ]
    d4 = planner.decide_next_action(
        current_elixir=6.0,
        opponent_estimated_elixir=4.0,
        hand_cards=["fireball", "arrows", "hog_rider", "knight"],
        hand_slots=_slots(["fireball", "arrows", "hog_rider", "knight"]),
        arena_threats=swarm,
        now_ts=40.0,
    )
    assert d4["action"] == "defend" and d4["recommended_card"] in ("arrows", "fireball"), f"Missed cluster spell! {d4}"

    # 5. No slot-1 bias: slot 1 is Hog Rider, slot 2 is Knight -> defends with Knight (never slot-1 Hog)
    d5 = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["hog_rider", "knight", "fireball", "arrows"],
        hand_slots=_slots(["hog_rider", "knight", "fireball", "arrows"]),
        arena_threats=[{"col": 14, "row": 19, "lane": "right", "in_our_half": True}],
        now_ts=50.0,
    )
    assert d5["slot"] == 2 and d5["recommended_card"] == "knight", f"Slot-1 bias regression: {d5}"

    # 6. Counterpush / offense guard: elixir >= 7.0 & no threats in our half -> punishes opposite lane
    d6 = planner.decide_next_action(
        current_elixir=7.4,
        opponent_estimated_elixir=5.0,
        hand_cards=["cannon", "hog_rider", "arrows", "fireball"],
        hand_slots=_slots(["cannon", "hog_rider", "arrows", "fireball"]),
        arena_threats=[],
        now_ts=65.0,
    )
    assert d6["action"] == "opposite_lane_punish" and d6["recommended_card"] == "hog_rider", f"Counterpush blocked! {d6}"

    # 7. Anti-leak at >= 9.2 elixir even when holding cannon/spells -> cycles instead of capping at 10.0
    d7 = planner.decide_next_action(
        current_elixir=9.5,
        opponent_estimated_elixir=8.0,
        hand_cards=["cannon", "arrows", "fireball", "slot_4"],
        hand_slots=_slots(["cannon", "arrows", "fireball", "slot_4"], playable=[True, True, True, False]),
        arena_threats=[],
        opp_intel={"elixir_leak_warning": True},
        now_ts=80.0,
    )
    assert d7["action"] != "wait", f"Leaked elixir at 9.5! {d7}"

    # 8. Air defense: never sends Knight (ground-only) or solo Fireball against flying threat when Musketeer available
    d8 = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["knight", "fireball", "musketeer", "hog_rider"],
        hand_slots=_slots(["knight", "fireball", "musketeer", "hog_rider"]),
        arena_threats=[{"col": 9, "row": 16, "lane": "center", "in_our_half": True}],
        opp_intel={"active_air_threat": True},
        now_ts=95.0,
    )
    assert d8["recommended_card"] == "musketeer", f"Expected musketeer vs air, got {d8}"

    # 9. Tower-will-kill skip: lone low-HP unit within tower defense -> wait (0 elixir spent)
    d9 = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["skeletons", "knight", "cannon", "musketeer"],
        hand_slots=_slots(["skeletons", "knight", "cannon", "musketeer"]),
        arena_threats=[{"col": 3, "row": 21, "lane": "left", "in_our_half": True, "hp_visible": True, "hp_frac": 0.35}],
        now_ts=110.0,
    )
    assert d9["action"] == "wait", f"Over-defended dying unit! {d9}"

    # 10. Minimum-elixir defense & cycle kite placement: prefers 1-elx skeletons over 3-elx knight/cannon
    # Threat at [3, 18] is 6.0 tiles from Princess Tower [3, 24] -> cycle places at center-kite [7, 20]
    d10 = planner.decide_next_action(
        current_elixir=6.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["musketeer", "knight", "skeletons", "cannon"],
        hand_slots=_slots(["musketeer", "knight", "skeletons", "cannon"]),
        arena_threats=[{"col": 3, "row": 18, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=125.0,
    )
    assert d10["recommended_card"] == "skeletons" and d10["slot"] == 3, f"Expected 1-elx skeletons, got {d10}"
    assert d10["target_tile"][0] in (7, 10) and d10["target_tile"][1] == 20, f"Expected center-kite [7/10, 20], got {d10}"

    # 11. Role-based safe placements: ranged -> tower pocket (row >= 25), building -> center pull ([8/9, 21-22])
    d11_ranged = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["musketeer", "hog_rider", "fireball", "arrows"],
        hand_slots=_slots(["musketeer", "hog_rider", "fireball", "arrows"]),
        arena_threats=[{"col": 14, "row": 20, "lane": "right", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=140.0,
    )
    assert d11_ranged["recommended_card"] == "musketeer" and d11_ranged["target_tile"][1] >= 25 and d11_ranged["target_tile"][0] in (4, 13), f"Unsafe ranged spot: {d11_ranged}"

    d11_building = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["cannon", "hog_rider", "fireball", "arrows"],
        hand_slots=_slots(["cannon", "hog_rider", "fireball", "arrows"]),
        arena_threats=[{"col": 14, "row": 19, "lane": "right", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=155.0,
    )
    assert d11_building["recommended_card"] == "cannon" and d11_building["target_tile"][0] in (8, 9) and 21 <= d11_building["target_tile"][1] <= 22, f"Bad cannon pull: {d11_building}"

    # 12. Unknown slot suppression: picks known knight instead of unidentified slot_1
    d12 = planner.decide_next_action(
        current_elixir=5.0,
        opponent_estimated_elixir=5.0,
        hand_cards=["slot_1", "knight", "fireball", "arrows"],
        hand_slots=_slots(["slot_1", "knight", "fireball", "arrows"]),
        arena_threats=[{"col": 4, "row": 20, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=170.0,
    )
    assert d12["recommended_card"] == "knight" and d12["slot"] == 2, f"Picked unknown slot_1! {d12}"

    # 13. Lure range checks: outer-flank kite stays <= 4.6 tiles; late deep threat shifts cannon inline
    d13_outer_kite = planner.decide_next_action(
        current_elixir=5.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["skeletons", "knight", "musketeer", "fireball"],
        hand_slots=_slots(["skeletons", "knight", "musketeer", "fireball"]),
        arena_threats=[{"col": 1, "row": 18, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=185.0,
    )
    assert d13_outer_kite["recommended_card"] == "skeletons" and d13_outer_kite["target_tile"][0] <= 5, f"Out-of-range outer kite: {d13_outer_kite}"

    d13_late_cannon = planner.decide_next_action(
        current_elixir=5.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["cannon", "hog_rider", "fireball", "arrows"],
        hand_slots=_slots(["cannon", "hog_rider", "fireball", "arrows"]),
        arena_threats=[{"col": 14, "row": 23, "lane": "right", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=200.0,
    )
    assert d13_late_cannon["recommended_card"] == "cannon" and d13_late_cannon["target_tile"] == [12, 22], f"Missed close-lane cannon block: {d13_late_cannon}"

    # 14. Anti-splash inner pocket (threat on tower) & King anchor (Princess dead)
    d14_splash = planner.decide_next_action(
        current_elixir=5.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["musketeer", "hog_rider", "fireball", "arrows"],
        hand_slots=_slots(["musketeer", "hog_rider", "fireball", "arrows"]),
        arena_threats=[{"col": 3, "row": 24, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        now_ts=215.0,
    )
    assert d14_splash["recommended_card"] == "musketeer" and d14_splash["target_tile"] == [6, 26], f"Splash vulnerable pocket: {d14_splash}"

    d14_king = planner.decide_next_action(
        current_elixir=5.5,
        opponent_estimated_elixir=5.0,
        hand_cards=["musketeer", "hog_rider", "fireball", "arrows"],
        hand_slots=_slots(["musketeer", "hog_rider", "fireball", "arrows"]),
        arena_threats=[{"col": 4, "row": 20, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0}],
        tower_states={"player_princess_left": {"alive": False}, "player_princess_right": {"alive": True}},
        now_ts=230.0,
    )
    assert d14_king["recommended_card"] == "musketeer" and d14_king["target_tile"] == [7, 25], f"Did not anchor King with dead tower: {d14_king}"

    # 15. Soak damage when behind on elixir vs lone moderate-HP chip
    d15_soak = planner.decide_next_action(
        current_elixir=3.2,
        opponent_estimated_elixir=6.0,
        hand_cards=["knight", "cannon", "musketeer", "hog_rider"],
        hand_slots=_slots(["knight", "cannon", "musketeer", "hog_rider"]),
        arena_threats=[{"col": 3, "row": 21, "lane": "left", "in_our_half": True, "hp_visible": True, "hp_frac": 0.60}],
        now_ts=245.0,
    )
    assert d15_soak["action"] == "wait" and "SOAK_DAMAGE" in d15_soak["reasoning"], f"Failed to soak moderate chip when behind: {d15_soak}"

    # 16. Full counter vs multi-unit/heavy push -> Cannon center lure
    d16_full = planner.decide_next_action(
        current_elixir=6.0,
        opponent_estimated_elixir=3.0,
        hand_cards=["skeletons", "knight", "cannon", "musketeer"],
        hand_slots=_slots(["skeletons", "knight", "cannon", "musketeer"]),
        arena_threats=[
            {"col": 3, "row": 18, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0},
            {"col": 4, "row": 18, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0},
        ],
        now_ts=260.0,
    )
    assert d16_full["recommended_card"] == "cannon" and d16_full.get("posture") == "FULL_COUNTER" and d16_full["target_tile"][0] in (7, 8) and d16_full["target_tile"][1] == 21, f"Expected FULL_COUNTER cannon lure, got {d16_full}"

    # 17. Dying hp=0.05 unit ignored even with Princess tower down (no wasted Cannon/Minions)
    planner_17 = TacticalActionPlanner()
    d17_dying = planner_17.decide_next_action(
        current_elixir=5.5,
        opponent_estimated_elixir=4.0,
        hand_cards=["minions", "cannon", "knight", "skeletons"],
        hand_slots=_slots(["minions", "cannon", "knight", "skeletons"]),
        arena_threats=[
            {"col": 3, "row": 24, "lane": "left", "in_our_half": True, "hp_visible": True, "hp_frac": 0.05},
        ],
        tower_states={"player_princess_left": {"alive": False}, "player_princess_right": {"alive": True}},
        now_ts=300.0,
    )
    assert d17_dying["action"] == "wait", f"Expected wait on hp=0.05 dying unit, got {d17_dying}"

    # 18. Lone river poke at high elixir (>=6.5) with clear opposite lane -> opposite_lane_punish instead of over-defending
    planner_18 = TacticalActionPlanner()
    d18_punish = planner_18.decide_next_action(
        current_elixir=7.2,
        opponent_estimated_elixir=4.5,
        hand_cards=["goblin_barrel", "cannon", "knight", "skeletons"],
        hand_slots=_slots(["goblin_barrel", "cannon", "knight", "skeletons"]),
        arena_threats=[
            {"col": 3, "row": 18, "lane": "left", "in_our_half": True, "hp_visible": False, "hp_frac": 1.0},
        ],
        now_ts=320.0,
    )
    assert d18_punish["action"] == "opposite_lane_punish" and d18_punish["recommended_card"] == "goblin_barrel" and d18_punish["target_tile"] == [14, 7], f"Expected opposite_lane_punish goblin_barrel [14, 7], got {d18_punish}"

    # 19. slow_roll_back cooldown prevents back-to-back double-fire at 8.9 elixir
    planner_19 = TacticalActionPlanner()
    d19_first = planner_19.decide_next_action(
        current_elixir=8.9,
        opponent_estimated_elixir=5.0,
        hand_cards=["knight", "archers", "arrows", "cannon"],
        hand_slots=_slots(["knight", "archers", "arrows", "cannon"]),
        arena_threats=[],
        now_ts=100.0,
    )
    assert d19_first["action"] == "slow_roll_back", f"Expected initial slow_roll_back, got {d19_first}"
    planner_19.confirm_executed(d19_first, now_ts=100.0)
    d19_second = planner_19.decide_next_action(
        current_elixir=8.9,
        opponent_estimated_elixir=5.0,
        hand_cards=["knight", "archers", "arrows", "cannon"],
        hand_slots=_slots(["knight", "archers", "arrows", "cannon"]),
        arena_threats=[],
        now_ts=101.0,
    )
    assert d19_second["action"] == "wait", f"Expected wait during slow_roll_back cooldown, got {d19_second}"

    return {
        "task": "Task 5 ActionPlanner Hard Guards & Tactics Gate",
        "unaffordable_blocked": True,
        "solo_spell_blocked": True,
        "cluster_spell_triggered": d4["recommended_card"],
        "role_priority_no_slot1_bias": d5["recommended_card"],
        "counterpush_active_ge_7": d6["action"],
        "anti_leak_action": d7["action"],
        "anti_air_selection": d8["recommended_card"],
        "tower_will_kill_skip": d9["action"] == "wait",
        "min_elixir_defense": d10["recommended_card"],
        "ranged_safe_pocket": d11_ranged["target_tile"],
        "building_center_pull": d11_building["target_tile"],
        "unknown_slot_suppressed": d12["recommended_card"],
        "outer_kite_lure_tile": d13_outer_kite["target_tile"],
        "late_cannon_intercept_tile": d13_late_cannon["target_tile"],
        "splash_safe_pocket": d14_splash["target_tile"],
        "dead_tower_king_anchor": d14_king["target_tile"],
        "soak_chip_when_behind": d15_soak["action"] == "wait",
        "full_counter_posture": d16_full.get("posture"),
        "ignore_dying_hp_005": d17_dying["action"] == "wait",
        "ignore_poke_to_punish_opp": d18_punish["action"] == "opposite_lane_punish",
        "slow_roll_cooldown_blocks_repeat": d19_second["action"] == "wait",
        "gate_pass": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_tests(), indent=2))
