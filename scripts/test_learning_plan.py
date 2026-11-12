"""
LEARNING_PLAN.md Phases P0-P4 Acceptance Tests & Offline Session Trainer.
"""
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.learning.exchange_ledger import ExchangeLedgerBuilder
from src.learning.tables import LearningTables
from src.strategy.action_planner import TacticalActionPlanner
from src.vision.homography import ArenaTileProjector


def run_acceptance() -> dict:
    # P0: Tile-grid river/tower relative calibration on 5/5 test pairs
    proj = ArenaTileProjector()
    p0_cases = [
        (5, 4, "left", (7, 21)),     # 4-2 left Cannon
        (5, 5, "left", (8, 21)),     # center left Cannon
        (5, 4, "right", (10, 21)),   # 4-2 right Cannon
        (4, 4, "left", (7, 20)),     # cycle center-kite left
        (10, 1, "right", (13, 26)),  # ranged safe pocket right
    ]
    for dy, dx, lane, expected in p0_cases:
        tile = proj.relative_to_tile(dy, dx, lane=lane)
        assert tile == expected, f"P0 calibration mismatch ({dy},{dx},{lane}): {tile} != {expected}"
        rel_back = proj.tile_to_relative(tile[0], tile[1], lane=lane)
        assert rel_back == (dy, dx), f"P0 inverse mismatch: {rel_back} != {(dy, dx)}"

    # P1: Ledger Builder on recorded Human + Bot sessions (creates exchanges.jsonl + process metrics)
    builder = ExchangeLedgerBuilder(projector=proj)
    human_res = builder.build_from_events(
        "recordings/human_20261111_203505/events.jsonl",
        output_jsonl_path="recordings/human_20261111_203505/exchanges.jsonl",
        is_human=True,
    )
    bot_res = builder.build_from_events(
        "recordings/session_20261112_113120/events.jsonl",
        output_jsonl_path="recordings/session_20261112_113120/exchanges.jsonl",
        is_human=False,
    )
    assert human_res["metrics"]["exchanges_total"] >= 5, f"Too few human exchanges: {human_res['metrics']}"
    assert bot_res["metrics"]["exchanges_total"] >= 5, f"Too few bot exchanges: {bot_res['metrics']}"
    assert bot_res["metrics"]["unaffordable_clicks"] == 0, f"Unaffordable click regression: {bot_res['metrics']}"

    # P2 + P4: Seed tables, update from ledger, verify constrained bandit safety gates
    tables = LearningTables(base_dir="data/learning", rng_seed=42)
    upd = tables.update_from_exchanges(
        bot_res["exchanges"], metrics=bot_res["metrics"], match_id=bot_res["match_id"]
    )
    assert os.path.exists("data/learning/rollback/placement_table.json"), "Rollback snapshot missing!"

    # Safety: NEVER explore on spells, low tower HP, >4 elx threat, or reserve violation
    assert tables.select_placement_arm("spell", "multi_push", 6.0, 4.0, allow_explore=True) is None
    safe_arm = tables.select_placement_arm(
        "building", "heavy_or_rusher", current_elixir=8.0, card_cost=3.0, enemy_cost_est=4.0, tower_hp_ok=True, allow_explore=True
    )
    assert safe_arm is not None and "dy_river" in safe_arm and "dx_tower" in safe_arm

    # Verify Reflex planner reads tables & management params cleanly
    planner = TacticalActionPlanner(learning_tables=tables, allow_explore=False)
    slots = [
        {"slot": 1, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": True},
        {"slot": 2, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": True},
        {"slot": 3, "card_name": "goblin_barrel", "card_cost": 3, "card_role": "wincon", "playable": True},
        {"slot": 4, "card_name": "minions", "card_cost": 3, "card_role": "air_dps", "playable": True},
    ]
    d_barrel = planner.decide_next_action(
        current_elixir=7.2,
        opponent_estimated_elixir=4.0,
        hand_cards=[s["card_name"] for s in slots],
        hand_slots=slots,
        arena_threats=[],
        now_ts=100.0,
    )
    assert d_barrel["recommended_card"] == "goblin_barrel" and d_barrel["target_tile"] == [3, 7], (
        f"Expected Goblin Barrel on tower [3, 7], got {d_barrel}"
    )

    return {
        "p0_relative_calibration_5_of_5": True,
        "p1_human_exchanges": human_res["metrics"],
        "p1_bot_exchanges": bot_res["metrics"],
        "p2_p4_table_updates": upd,
        "p3_leak_sec": bot_res["metrics"]["leak_sec"],
        "p3_starved_sec": bot_res["metrics"]["starved_sec"],
        "goblin_barrel_tower_tile": d_barrel["target_tile"],
        "gate_pass": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_acceptance(), indent=2))
