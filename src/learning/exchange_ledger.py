"""
Exchange Ledger Builder & Elixir Management Metrics (LEARNING_PLAN.md Sections 2 & 5).
Cuts every match (events.jsonl) into discrete exchanges (one push + our response)
and computes process metrics (leak_sec, starved_sec, reserve_violations, unaffordable_clicks).
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.vision.card_matcher import CARD_METADATA
from src.vision.homography import ArenaTileProjector

DEFAULT_CARD_COSTS: Dict[str, float] = {
    k: float(v["cost"]) for k, v in CARD_METADATA.items() if v.get("cost") is not None
}
DEFAULT_CARD_COSTS.update({
    "zap": 2.0, "the_log": 2.0, "log": 2.0, "snowball": 2.0, "miner": 3.0,
    "tombstone": 3.0, "poison": 4.0, "tesla": 4.0, "bomb_tower": 4.0,
    "freeze": 4.0, "ram_rider": 5.0, "giant": 5.0, "balloon": 5.0,
    "inferno_tower": 5.0, "royal_giant": 6.0, "rocket": 6.0,
})
DEFAULT_CARD_ROLES: Dict[str, str] = {
    k: str(v["role"]) for k, v in CARD_METADATA.items() if v.get("role")
}


class ExchangeLedgerBuilder:
    K_TOWER_PENALTY = 8.0
    EXCHANGE_TIMEOUT_SEC = 12.0
    CLEAR_CLOSE_GRACE_SEC = 1.5

    def __init__(self, projector: Optional[ArenaTileProjector] = None):
        self.projector = projector or ArenaTileProjector()

    @staticmethod
    def _infer_lane(row: Dict[str, Any]) -> str:
        reason = str(row.get("reasoning") or "").lower()
        if "right" in reason or "(right)" in reason or "enemy_unit_right" in reason:
            return "right"
        if "left" in reason or "(left)" in reason or "enemy_unit_left" in reason:
            return "left"
        tile = row.get("target_tile")
        if tile and len(tile) >= 1:
            return "left" if int(tile[0]) < 9 else "right"
        return "left"

    @staticmethod
    def _infer_threat_sig(rows_window: List[Dict[str, Any]], peak_threats: int) -> str:
        joined = " ".join(str(r.get("reasoning") or "").lower() for r in rows_window)
        if "air" in joined or "flying" in joined:
            return "air_threat"
        if "rusher" in joined or "heavy" in joined:
            return "heavy_or_rusher"
        if peak_threats >= 2 or "clustered" in joined or "multi" in joined:
            return "multi_push"
        return "single_ground"

    def build_from_events(
        self,
        events_path: str,
        output_jsonl_path: Optional[str] = None,
        reserve_floor: float = 2.0,
        is_human: bool = False,
    ) -> Dict[str, Any]:
        """Parse events.jsonl into closed exchanges + management metrics."""
        if not os.path.exists(events_path):
            raise FileNotFoundError(f"events file not found: {events_path}")

        with open(events_path, "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]

        match_id = os.path.basename(os.path.dirname(events_path)) or "unknown_match"
        exchanges: List[Dict[str, Any]] = []
        active: Dict[str, Dict[str, Any]] = {}  # lane -> open exchange

        # Process metrics (§5)
        leak_sec = 0.0
        starved_sec = 0.0
        reserve_violations = 0
        unaffordable_clicks = 0
        elx_at_play: List[float] = []
        last_battle_t: Optional[float] = None
        ex_counter = 0

        def _close_exchange(lane_key: str, close_t: float) -> None:
            ex = active.pop(lane_key)
            our_cost = round(sum(float(p["cost"]) for p in ex["our_plays"]), 2)
            # Without opponent card OCR, approximate enemy spending conservatively from threat waves
            est_enemy_cost = float(ex["peak_threats"] * 2.5) if ex["enemy_cost_known"] == 0 else ex["enemy_cost_known"]
            enemy_cost = round(max(2.0 if ex["peak_threats"] > 0 else 0.0, min(12.0, est_enemy_cost)), 2)
            enemy_known_all = bool(ex["enemy_cost_known_all"])
            tower_dmg_frac = round(ex.get("tower_dmg_frac", 0.0), 3)
            score = round((enemy_cost - our_cost) - self.K_TOWER_PENALTY * tower_dmg_frac, 2)
            cls = "positive" if score > 0.5 else ("negative" if score < -0.5 else "even")
            exchanges.append({
                "match_id": match_id,
                "ex_id": ex["ex_id"],
                "t_open": round(ex["t_open"], 2),
                "t_close": round(close_t, 2),
                "lane": lane_key,
                "phase": "double" if close_t >= 120.0 else "single",
                "threat_sig": self._infer_threat_sig(ex["reason_rows"], ex["peak_threats"]),
                "peak_threats": ex["peak_threats"],
                "enemy_cost": enemy_cost,
                "enemy_cost_known_all": enemy_known_all,
                "our_plays": ex["our_plays"],
                "our_cost": our_cost,
                "tower_dmg_frac": tower_dmg_frac,
                "neutralized": True,
                "elixir_before": round(ex["elixir_before"], 2),
                "elixir_after": round(ex["elixir_after"], 2),
                "reserve_violation": ex["reserve_violation"],
                "score": score,
                "class": cls,
            })

        for row in events:
            if row.get("screen") != "battle_arena":
                last_battle_t = None
                continue

            t = float(row.get("t", 0.0))
            elx = float(row.get("elixir", 0.0))
            threats = int(row.get("threats") or 0)
            reason = str(row.get("reasoning") or "")

            if last_battle_t is not None:
                dt = max(0.0, min(1.0, t - last_battle_t))
                if elx >= 9.5:
                    leak_sec += dt
                elif elx < 2.0:
                    starved_sec += dt
            last_battle_t = t

            lane = self._infer_lane(row)
            is_play = bool(row.get("human_play")) if is_human else ("LIVE_EXECUTED" in str(row.get("exec_status", "")))

            # Open exchange on active threat or defensive play
            act = str(row.get("action") or "")
            if threats > 0 or (is_play and act in ("defend", "predict_building")):
                if lane not in active:
                    ex_counter += 1
                    active[lane] = {
                        "ex_id": ex_counter,
                        "t_open": t,
                        "last_threat_t": t,
                        "peak_threats": max(1, threats),
                        "enemy_cost_known": 0.0,
                        "enemy_cost_known_all": False,
                        "our_plays": [],
                        "reason_rows": [row],
                        "elixir_before": elx,
                        "elixir_after": elx,
                        "reserve_violation": False,
                        "tower_dmg_frac": 0.0,
                    }
                else:
                    ex = active[lane]
                    if threats > 0:
                        ex["last_threat_t"] = t
                    ex["peak_threats"] = max(ex["peak_threats"], threats)
                    if len(ex["reason_rows"]) < 25:
                        ex["reason_rows"].append(row)

            # Attach our play
            if is_play:
                if is_human:
                    hp = row["human_play"]
                    card = str(hp.get("card") or "unknown")
                    cost = float(hp.get("cost") or DEFAULT_CARD_COSTS.get(card, 3.0))
                    slot = int(hp.get("slot") or 0)
                    play_elx = float(hp.get("elixir_at_play", elx))
                    tile = row.get("target_tile") or ([4, 21] if lane == "left" else [13, 21])
                else:
                    card = str(row.get("recommended_card") or "unknown")
                    # fallback: extract card from reasoning if needed
                    if card == "unknown":
                        m = re.search(r":\s*([a-z_]+)\s+vs\s+|slow-rolling\s+([a-z_]+)|push at .* with ([a-z_]+)", reason)
                        if m:
                            card = next(g for g in m.groups() if g)
                    cost = float(DEFAULT_CARD_COSTS.get(card, 3.0))
                    slot = int(row.get("slot") or 0)
                    play_elx = elx
                    tile = row.get("target_tile") or [8, 21]

                elx_at_play.append(play_elx)
                if cost > play_elx + 0.05:
                    unaffordable_clicks += 1

                after_elx = max(0.0, play_elx - cost)
                is_emergency = threats > 0
                is_reserve_violation = (not is_emergency) and (after_elx < reserve_floor) and (act not in ("slow_roll_back", "opposite_lane_punish"))
                if is_reserve_violation:
                    reserve_violations += 1

                if lane in active and act in ("defend", "predict_building", "wait", "attack"):
                    dy_r, dx_t = self.projector.tile_to_relative(int(tile[0]), int(tile[1]), lane=lane)
                    active[lane]["our_plays"].append({
                        "t": round(t, 2),
                        "card": card,
                        "cost": cost,
                        "role": DEFAULT_CARD_ROLES.get(card, "unknown"),
                        "dy_river": dy_r,
                        "dx_tower": dx_t,
                        "tile": [int(tile[0]), int(tile[1])],
                        "slot": slot,
                    })
                    active[lane]["elixir_after"] = after_elx
                    active[lane]["reserve_violation"] = active[lane]["reserve_violation"] or is_reserve_violation

            # Close timed-out or cleared exchanges
            for k in list(active.keys()):
                ex = active[k]
                idle_clear = (threats == 0) and ((t - ex["last_threat_t"]) >= self.CLEAR_CLOSE_GRACE_SEC)
                timed_out = (t - ex["t_open"]) >= self.EXCHANGE_TIMEOUT_SEC
                if idle_clear or timed_out:
                    _close_exchange(k, t)

        final_t = float(events[-1].get("t", 0.0)) if events else 0.0
        for k in list(active.keys()):
            _close_exchange(k, final_t)

        if output_jsonl_path:
            os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                for row in exchanges:
                    f.write(json.dumps(row) + "\n")

        pos = sum(1 for e in exchanges if e["class"] == "positive")
        neg = sum(1 for e in exchanges if e["class"] == "negative")
        even = sum(1 for e in exchanges if e["class"] == "even")
        return {
            "match_id": match_id,
            "exchanges": exchanges,
            "metrics": {
                "exchanges_total": len(exchanges),
                "positive": pos,
                "even": even,
                "negative": neg,
                "mean_exchange_score": round(sum(e["score"] for e in exchanges) / max(1, len(exchanges)), 2),
                "leak_sec": round(leak_sec, 2),
                "starved_sec": round(starved_sec, 2),
                "reserve_violations": reserve_violations,
                "unaffordable_clicks": unaffordable_clicks,
                "avg_elixir_at_play": round(sum(elx_at_play) / max(1, len(elx_at_play)), 2),
            },
        }
