"""
Strategic Action & Tactical Move Planner.
Given the current arena state, identified hand cards/slots, next card, tower states,
elixir estimate, and OpponentTracker intelligence (hand cycle + elixir confidence):
  1. Spell-value gating (require >= 2 targets in spell radius unless tower emergency).
  2. Elixir-starvation guard (hold < 3.0 elixir on distant single threats).
  3. Anti-air & all-flying deck adaptation (use musketeer/arrows/electro_spirit vs air,
     cycle or pressure opposite lane if hand lacks anti-air).
  4. Gated prediction plays ONLY when BOTH opponent elixir count AND opponent hand cycle are known.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from src.learning.tables import LearningTables
from src.vision.homography import ArenaTileProjector


SPELL_HINTS = {"the_log", "log", "zap", "arrows", "fireball", "poison", "rocket", "snowball", "freeze"}
BUILDING_HINTS = {"cannon", "tesla", "inferno_tower", "bomb_tower", "tombstone", "goblin_cage"}
WINCON_HINTS = {"hog_rider", "goblin_barrel", "miner", "giant", "royal_giant", "balloon", "graveyard", "ram_rider"}
ANTI_AIR_CARDS = {"musketeer", "archers", "minions", "arrows", "electro_spirit", "fireball", "zap", "tesla"}
CARD_COSTS = {
    "skeletons": 1,
    "electro_spirit": 1,
    "zap": 2,
    "the_log": 2,
    "log": 2,
    "snowball": 2,
    "arrows": 3,
    "cannon": 3,
    "knight": 3,
    "archers": 3,
    "minions": 3,
    "miner": 3,
    "goblin_barrel": 3,
    "tombstone": 3,
    "fireball": 4,
    "musketeer": 4,
    "hog_rider": 4,
    "poison": 4,
    "tesla": 4,
    "bomb_tower": 4,
    "freeze": 4,
    "ram_rider": 5,
    "giant": 5,
    "balloon": 5,
    "inferno_tower": 5,
    "royal_giant": 6,
    "rocket": 6,
}

# Default single standard ground threat: least-elixir positive trades first (1-elx cycle -> 3-elx knight -> cannon -> musketeer)
TROOP_DEFENSE_PRIORITY = {
    "cycle": 6,       # Skeletons / Electro Spirit (1 elx) minimum-elixir defense & kite
    "minitank": 5,    # Knight (3 elx) positive-trade tank & counter-push
    "air_dps": 4,     # Minions (3 elx) high-DPS vs ground melee & air
    "building": 3,    # Save Cannon for rushers/tanks or 2+ unit pushes unless no cheaper counter
    "ranged": 2,      # Archers (3 elx) / Musketeer (4 elx) behind tower
    "unknown": 0,     # Never blind-click un-ID'd cycling slots unless tower emergency
    "wincon": 0,      # Hog Rider / Goblin Barrel ignores troops and wastes elixir on defense
    "spell": 0,
}
# When enemy is a fast rusher / heavy tank / 2+ unit push: Cannon center-pull becomes top priority
HEAVY_PULL_PRIORITY = {
    "building": 5,    # Cannon center-pull drags tanks/Hog into both Princess Towers
    "minitank": 4,    # Knight tanks & protects backline
    "air_dps": 3,     # Minions melt ground tanks quickly
    "ranged": 3,      # Archers/Musketeer high DPS safely behind tower
    "cycle": 2,
    "unknown": 0,
    "wincon": 0,
    "spell": 0,
}
AIR_DEFENSE_PRIORITY = {
    "ranged": 5,      # Archers/Musketeer primary DPS vs flying
    "air_dps": 5,     # Minions directly shred air
    "cycle": 3,       # Electro Spirit stuns/resets air + cycles
    "building": 2,    # Cannon pulls Balloon/Lava Hound to center
    "spell": 0,       # Spells strictly gated through multi-target cluster check
    "minitank": 0,    # Knight cannot hit air
    "wincon": 0,
    "unknown": 0,
}
BACK_ROLL_PRIORITY = {
    "ranged": 4,
    "minitank": 3,
    "cycle": 2,
    "air_dps": 2,
    "wincon": 1,
    "unknown": 0,
    "building": 0,
    "spell": 0,
}
SPELL_RADII = {
    "arrows": 4.0,
    "fireball": 2.5,
    "zap": 2.5,
    "snowball": 2.5,
    "poison": 3.5,
    "freeze": 3.0,
    "rocket": 2.0,
}


class TacticalActionPlanner:
    def __init__(
        self,
        learning_tables: Optional[LearningTables] = None,
        allow_explore: bool = False,
    ):
        self.learning_tables = learning_tables or LearningTables()
        self.allow_explore = bool(allow_explore)
        self.projector = ArenaTileProjector()
        self._last_slot_used: Optional[int] = None
        self._last_push_ts: float = -100.0
        self._last_push_col: int = 3
        self._last_defend_ts: float = -100.0
        self._last_defend_by_lane: Dict[str, Tuple[float, str]] = {}
        self._last_slow_roll_ts: float = -100.0
        self._last_building_ts: float = -100.0
        self._leak_start_ts: Optional[float] = None
        self._bank_start_ts: Optional[float] = None

    def confirm_executed(self, decision: Dict[str, Any], now_ts: Optional[float] = None) -> None:
        """
        Updates push/defend cooldown timestamps ONLY after ActionExecutor actually dispatches
        the action, so scheduler queue frames do not cancel pending decisions on the next frame.
        """
        if not decision:
            return
        ts = float(time.time() if now_ts is None else now_ts)
        act = decision.get("action")
        if decision.get("slot") is not None:
            self._last_slot_used = decision.get("slot")
        if act == "opposite_lane_punish":
            self._last_push_ts = ts
            tile = decision.get("target_tile")
            if tile and len(tile) >= 1:
                self._last_push_col = int(tile[0])
        elif act in ("slow_roll_back", "cycle_for_anti_air"):
            self._last_slow_roll_ts = ts
        elif act == "defend":
            self._last_defend_ts = ts
            lane = str(decision.get("target_lane") or "")
            if lane not in ("left", "right"):
                tile = decision.get("target_tile")
                lane = "left" if (tile and int(tile[0]) < 9) else "right"
            cname = str(decision.get("recommended_card") or "").lower().replace(" ", "_")
            role = str(decision.get("card_role") or self._infer_role(cname, None))
            self._last_defend_by_lane[lane] = (ts, role)
            if role == "building":
                self._last_building_ts = ts
        elif act in ("predict_building", "anti_leak_building"):
            self._last_building_ts = ts

    @staticmethod
    def _norm_name(slot_info: Dict[str, Any]) -> str:
        return str(slot_info.get("card_name") or f"slot_{slot_info.get('slot', 1)}").lower().replace(" ", "_")

    @staticmethod
    def _infer_role(name: str, explicit_role: Optional[str]) -> str:
        if explicit_role and explicit_role != "unknown":
            return explicit_role
        if name in SPELL_HINTS:
            return "spell"
        if name in BUILDING_HINTS:
            return "building"
        if name in WINCON_HINTS:
            return "wincon"
        if name in {"knight"}:
            return "minitank"
        if name in {"minions"}:
            return "air_dps"
        if name in {"musketeer", "archers"}:
            return "ranged"
        if name in {"skeletons", "electro_spirit"}:
            return "cycle"
        return "unknown"

    def _playable_candidates(
        self,
        hand_cards: List[str],
        hand_slots: Optional[List[Dict[str, Any]]],
        current_elixir: float,
    ) -> List[Dict[str, Any]]:
        if current_elixir < 0.9:
            return []

        if hand_slots:
            out = []
            for s in hand_slots:
                if not s.get("playable", False):
                    continue
                cost = s.get("card_cost")
                if cost is None:
                    # Unknown slot cost: require >= 2.8 elixir so we never click dead slots at 0.3-2.0 elixir
                    if current_elixir < 2.8:
                        continue
                elif current_elixir + 0.15 < float(cost):
                    continue
                out.append(s)
            return out

        if hand_cards:
            out = []
            for i, c in enumerate(hand_cards):
                cname = str(c).lower().replace(" ", "_")
                cost = CARD_COSTS.get(cname)
                if cost is not None and current_elixir + 0.15 < float(cost):
                    continue
                out.append({"slot": i + 1, "card_name": c, "card_cost": cost, "playable": True})
            return out
        return []

    def _select_playable_slot(
        self,
        hand_cards: List[str],
        hand_slots: Optional[List[Dict[str, Any]]],
        preferred_set: Optional[set] = None,
        current_elixir: float = 10.0,
    ) -> Optional[Dict[str, Any]]:
        """Picks a playable slot matching preferred_set if provided."""
        candidates = self._playable_candidates(hand_cards, hand_slots, current_elixir)
        if not candidates:
            return None
        if preferred_set:
            for s in candidates:
                if self._norm_name(s) in preferred_set:
                    return s
            return None
        if len(candidates) > 1 and self._last_slot_used == candidates[0].get("slot"):
            return candidates[1]
        return candidates[0]

    @staticmethod
    def _best_spell_cluster(
        threats: List[Dict[str, Any]],
        radius: float,
    ) -> Tuple[int, int, int]:
        """Finds target [col, row] maximizing enemy count within `radius` tiles."""
        best_hits = 0
        best_col, best_row = 9, 20
        for anchor in threats:
            ac = int(anchor.get("col", 9))
            ar = int(anchor.get("row", 20))
            in_rad = [
                t for t in threats
                if math.hypot(int(t.get("col", 9)) - ac, int(t.get("row", 20)) - ar) <= radius
            ]
            if len(in_rad) > best_hits:
                best_hits = len(in_rad)
                best_col = round(sum(int(t.get("col", 9)) for t in in_rad) / len(in_rad))
                best_row = round(sum(int(t.get("row", 20)) for t in in_rad) / len(in_rad))
        return best_hits, max(1, min(16, best_col)), max(15, min(27, best_row))

    def _pick_by_priority(
        self,
        candidates: List[Dict[str, Any]],
        priority_map: Dict[str, int],
        min_priority: int = 1,
    ) -> Optional[Dict[str, Any]]:
        scored = []
        for s in candidates:
            name = self._norm_name(s)
            role = self._infer_role(name, s.get("card_role"))
            prio = priority_map.get(role, 0)
            if prio < min_priority:
                continue
            # Prefer minimum elixir cost within same priority tier
            cost_bonus = -float(s.get("card_cost") or 3) * 0.08
            repeat_penalty = -0.02 if s.get("slot") == self._last_slot_used else 0.0
            scored.append((prio + cost_bonus + repeat_penalty, s))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    @staticmethod
    def _is_air_unit(ent: Dict[str, Any]) -> bool:
        col = int(ent.get("col", 9))
        row = int(ent.get("row", 0))
        return 14 <= row <= 16 and not (2 <= col <= 4 or 13 <= col <= 15)

    def _classify_defense_posture(
        self,
        immediate_threats: List[Dict[str, Any]],
        t_hp: float,
        t_hp_vis: bool,
        t_dist: float,
        t_row: int,
        t_card: str,
        is_air_ctx: bool,
        recent_heavy_threat: bool,
        our_lane_alive: bool,
        current_elixir: float,
        elixir_adv: float,
        has_cheap_counter: bool,
    ) -> Tuple[str, str]:
        """Classify defensive response: SOAK_TOWER_KILLS, SOAK_DAMAGE, FULL_COUNTER, or CHEAP_TRADE."""
        max_hp_vis = all(bool(t.get("hp_visible", False)) for t in immediate_threats)
        max_hp = max(float(t.get("hp_frac", 1.0)) for t in immediate_threats)
        is_heavy = bool(recent_heavy_threat or "rusher" in t_card.lower() or "heavy" in t_card.lower())

        # 1) Tower finishes alone -> 0 elixir (Princess Tower or King Tower)
        if (
            len(immediate_threats) == 1
            and t_hp_vis
            and (
                (our_lane_alive and (t_hp <= 0.45 or (t_hp <= 0.62 and t_dist >= 4.2 and not is_heavy)))
                or (not our_lane_alive and (t_hp <= 0.38 or (t_hp <= 0.45 and t_dist <= 6.5 and not is_heavy)))
            )
        ):
            return "SOAK_TOWER_KILLS", f"lone enemy hp={t_hp:.2f} at dist={t_dist:.1f} dies to tower."
        if (
            len(immediate_threats) <= 2
            and max_hp_vis
            and max_hp <= 0.32
            and (our_lane_alive and t_dist >= 2.8 or (not our_lane_alive and max_hp <= 0.30 and t_dist <= 6.5))
        ):
            return "SOAK_TOWER_KILLS", f"{len(immediate_threats)} low-HP enemies (max_hp={max_hp:.2f}) die to tower."

        # 2) Soak minor single-unit chip damage instead of over-defending every poke
        if len(immediate_threats) == 1 and not is_air_ctx and not is_heavy and our_lane_alive:
            if t_hp_vis and t_hp <= 0.68 and (elixir_adv <= -1.0 or current_elixir < 3.6) and t_dist >= 2.2:
                return "SOAK_DAMAGE", (
                    f"Soaking moderate chip (hp={t_hp:.2f}, elx_adv={elixir_adv:+.1f}) to save elixir."
                )
            if t_hp_vis and t_hp <= 0.72 and t_row <= 21:
                return "SOAK_DAMAGE", (
                    f"Soaking injured single unit (hp={t_hp:.2f}, row={t_row}); tower handles it."
                )
            if t_row <= 19 and (current_elixir <= 4.2 or current_elixir >= 6.5):
                return "SOAK_DAMAGE", (
                    f"Ignoring minor river poke (row={t_row}, elx={current_elixir:.1f}) to save/counterpush."
                )
            if not has_cheap_counter and current_elixir < 4.5 and (
                (t_hp_vis and t_hp <= 0.80) or elixir_adv <= -1.5
            ):
                return "SOAK_DAMAGE", (
                    f"Soaking minor poke (no <=3 elx counter, elx={current_elixir:.1f}) to avoid negative trade."
                )

        # 3) Full counter vs multi-unit, heavy/rusher, air, or exposed King lane
        if len(immediate_threats) >= 2 or is_heavy or is_air_ctx or not our_lane_alive or (t_row >= 22 and t_hp >= 0.72):
            return "FULL_COUNTER", "Multi/heavy/deep push requires full shutdown."

        return "CHEAP_TRADE", "Single ground unit -> minimum-elixir positive trade."

    @staticmethod
    def _compute_defensive_tile(
        picked_role: str,
        t_col: int,
        t_row: int,
        t_lane: str,
        t_hp: float,
        t_dist: float,
        is_air_ctx: bool,
        threat_count: int,
        our_lane_alive: bool,
        bandit_tile: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int, str]:
        """Compute lure/range-validated placement tile and tactical note."""
        if picked_role == "building":
            # Cannon 3x3 center pull pulls up to ~5.5 tiles center-to-unit; deeper threats (row>=22) need same-lane block
            if bandit_tile is not None:
                base_col, base_row = int(bandit_tile[0]), int(bandit_tile[1])
            else:
                base_col = 8 if t_lane == "left" else 9
                base_row = 22 if t_row >= 20 else 21
            center_dist = math.hypot(t_col - base_col, t_row - base_row)
            if t_row >= 22:
                def_col = 5 if t_lane == "left" else 12
                return def_col, 22, f"close-lane intercept (center pull out of range at row={t_row})"
            if center_dist > 5.5:
                shift_col = max(5, t_col + 4) if t_lane == "left" else min(12, t_col - 4)
                eff_dist = math.hypot(t_col - shift_col, t_row - base_row)
                return shift_col, base_row, f"adjusted center lure (lure_dist={eff_dist:.1f}<=5.5)"
            return base_col, base_row, f"double-tower center lure (lure_dist={center_dist:.1f}<=5.5)"

        if picked_role == "spell":
            return max(1, min(16, t_col)), max(16, min(27, t_row + 1)), "spell impact"

        if picked_role == "ranged":
            if not our_lane_alive:
                # Princess tower dead -> anchor safely beside King tower
                return (7 if t_lane == "left" else 10), 25, "King-anchor safe pocket (Princess down)"
            if bandit_tile is not None:
                pocket_col, pocket_row = int(bandit_tile[0]), int(bandit_tile[1])
            else:
                pocket_col = 4 if t_lane == "left" else 13
                pocket_row = 26 if t_row >= 20 else 25
            if math.hypot(t_col - pocket_col, t_row - pocket_row) < 3.2:
                # Avoid splash/spell value on both tower and ranged defender
                inward_col = 6 if t_lane == "left" else 11
                return inward_col, 26, "anti-splash inner pocket (keeps >3 tiles from tower)"
            return pocket_col, pocket_row, "safe tower backline pocket"

        if picked_role == "air_dps":
            # Minions: drop right above/slightly behind threat for immediate DPS lock without splash
            def_col = max(2, min(15, t_col))
            def_row = max(18, min(25, t_row))
            return def_col, def_row, "aerial direct shred"

        if picked_role == "cycle":
            # Kite within ~4.5-tile sight/aggro range when far & high-HP; else front intercept
            if t_dist >= 3.5 and t_row <= 21 and t_hp >= 0.55 and not is_air_ctx:
                if bandit_tile is not None:
                    kite_col, kite_row = int(bandit_tile[0]), int(bandit_tile[1])
                else:
                    kite_col = 7 if t_lane == "left" else 10
                    kite_row = 20
                if math.hypot(t_col - kite_col, t_row - kite_row) > 4.6:
                    kite_col = min(7, t_col + 3) if t_lane == "left" else max(10, t_col - 3)
                kdist = math.hypot(t_col - kite_col, t_row - kite_row)
                return kite_col, kite_row, f"center kite (lure_dist={kdist:.1f}<=4.6)"
            center_bias = 1 if t_lane == "left" else -1
            def_col = max(2, min(15, t_col + center_bias))
            def_row = max(19, min(24, t_row + 1))
            return def_col, def_row, "front surround/intercept"

        # Minitank (Knight): intercept swarm/backline directly or pull melee slightly toward center for counterpush
        center_bias = (1 if t_lane == "left" else -1) if (threat_count >= 2 or t_row >= 22) else (2 if t_lane == "left" else -2)
        def_col = max(3, min(14, t_col + center_bias))
        def_row = max(19, min(23, t_row + 1))
        if def_row >= 22 and (2 <= def_col <= 4 or 13 <= def_col <= 15):
            def_col = 6 if def_col <= 4 else 11
        return def_col, def_row, "tank-absorb + counterpush pull"

    def decide_next_action(
        self,
        current_elixir: float,
        opponent_estimated_elixir: float,
        hand_cards: List[str],
        arena_threats: List[Dict[str, Any]],
        hand_slots: Optional[List[Dict[str, Any]]] = None,
        tower_states: Optional[Dict[str, Any]] = None,
        opp_intel: Optional[Dict[str, Any]] = None,
        next_card: Optional[Dict[str, Any]] = None,
        now_ts: Optional[float] = None,
        qwen_advice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Determines the optimal tactical action.
        Blends real-time Qwen3-VL Brain advice (macro posture, air unit visual alerts, lane focus)
        with fast deterministic CV execution invariants.
        """
        now_ts = float(time.time() if now_ts is None else now_ts)
        candidates = self._playable_candidates(hand_cards, hand_slots, current_elixir)
        if not candidates:
            return {
                "action": "wait",
                "slot": None,
                "recommended_card": None,
                "target_tile": None,
                "confidence": 0.99,
                "emergency": False,
                "reasoning": "No playable/affordable card slot detected yet (waiting on elixir/cooldown).",
            }

        opp_intel = opp_intel or {}
        qwen_advice = qwen_advice or {}
        qwen_mode = str(qwen_advice.get("strategy_mode") or "").lower()
        qwen_lane = str(qwen_advice.get("preferred_lane") or "auto").lower()
        qwen_rec_card = str(qwen_advice.get("recommended_card") or "").lower().replace(" ", "_")
        qwen_floor = float(qwen_advice.get("elixir_floor") or 0.0)
        qwen_air = bool(qwen_advice.get("anti_air_needed", False))
        qwen_overspend = bool(qwen_advice.get("allow_overspend", False)) or (qwen_mode == "overspend_punish")
        qwen_max_def_cost = float(qwen_advice.get("max_defense_cost") or 10.0)

        prediction_unlocked = bool(opp_intel.get("prediction_unlocked", False))
        opp_hand = opp_intel.get("opp_hand_archetypes", [])
        opp_elx = float(opp_intel.get("opp_elixir", opponent_estimated_elixir))
        elixir_adv = current_elixir - opp_elx
        opp_has_rusher = bool(opp_intel.get("opp_has_rusher", False))
        opp_has_building_counter = bool(opp_intel.get("opp_has_building_counter", False))
        recent_heavy_threat = bool(opp_intel.get("recent_heavy_threat", False))
        enemy_all_flying = bool(opp_intel.get("enemy_all_flying", False))
        active_air_threat = bool(opp_intel.get("active_air_threat", False)) or qwen_air
        can_overspend = bool(
            qwen_overspend
            or (elixir_adv >= 2.0 and current_elixir >= 5.5)
            or (prediction_unlocked and opp_elx < 3.5 and current_elixir >= 5.5)
        )

        # 1. Immediate defensive threats in our half (row >= 16), ignoring nearly-dead (<=0.30 HP) units
        immediate_threats = sorted(
            [
                t for t in arena_threats
                if t.get("row", 0) >= 16
                and not (bool(t.get("hp_visible", False)) and float(t.get("hp_frac", 1.0)) <= 0.30)
            ],
            key=lambda t: -t.get("row", 0),
        )

        if immediate_threats:
            our_left_alive = (tower_states or {}).get("player_princess_left", {}).get("alive", True)
            our_right_alive = (tower_states or {}).get("player_princess_right", {}).get("alive", True)
            tower_anchors = [(9, 28)]
            if our_left_alive:
                tower_anchors.append((3, 24))
            if our_right_alive:
                tower_anchors.append((14, 24))

            def _tower_dist(ent: Dict[str, Any]) -> float:
                ec, er = int(ent.get("col", 9)), int(ent.get("row", 18))
                return min(math.hypot(ec - tc, er - tr) for tc, tr in tower_anchors)

            # Prioritize threats closest to our towers (tie-break deeper row & higher HP)
            immediate_threats = sorted(
                immediate_threats,
                key=lambda t: (_tower_dist(t), -float(t.get("hp_frac", 1.0)), -int(t.get("row", 16))),
            )
            threat = immediate_threats[0]
            t_col = int(threat.get("col", 4))
            t_row = int(threat.get("row", 18))
            t_lane = threat.get("lane") or ("left" if t_col < 9 else "right")
            t_card = threat.get("card", "enemy_unit")
            t_hp = float(threat.get("hp_frac", 1.0))
            t_hp_vis = bool(threat.get("hp_visible", False))
            t_dist = _tower_dist(threat)

            is_air_ctx = active_air_threat or any(self._is_air_unit(t) for t in immediate_threats)
            our_lane_alive = our_left_alive if t_lane == "left" else our_right_alive
            has_cheap_counter = any(
                float(s.get("card_cost") or 3) <= 3.0
                and self._infer_role(self._norm_name(s), s.get("card_role")) in ("cycle", "minitank", "spell")
                for s in candidates
            )

            # 0) Defense posture classification: SOAK_TOWER_KILLS / SOAK_DAMAGE / FULL_COUNTER / CHEAP_TRADE
            posture, posture_reason = self._classify_defense_posture(
                immediate_threats=immediate_threats,
                t_hp=t_hp,
                t_hp_vis=t_hp_vis,
                t_dist=t_dist,
                t_row=t_row,
                t_card=str(t_card),
                is_air_ctx=is_air_ctx,
                recent_heavy_threat=recent_heavy_threat,
                our_lane_alive=our_lane_alive,
                current_elixir=current_elixir,
                elixir_adv=elixir_adv,
                has_cheap_counter=has_cheap_counter,
            )
            if posture in ("SOAK_TOWER_KILLS", "SOAK_DAMAGE"):
                # If we have >= 6.5 elixir and a wincon ready with clear opposite lane, punish instead of waiting
                opp_push_col = 14 if t_col < 9 else 3
                opp_lane_defenders = any(
                    ((int(t.get("col", 9)) < 9) == (opp_push_col < 9)) and int(t.get("row", 0)) >= 6
                    for t in arena_threats
                )
                wincon_pick = (
                    self._select_playable_slot(hand_cards, hand_slots, WINCON_HINTS, current_elixir)
                    if (current_elixir >= 6.5 and not opp_lane_defenders and (now_ts - self._last_push_ts) >= 2.2)
                    else None
                )
                if wincon_pick is not None:
                    wname = self._norm_name(wincon_pick)
                    target_tile = [14 if opp_push_col >= 9 else 3, 7] if wname == "goblin_barrel" else [opp_push_col, 15]
                    self._last_slot_used = wincon_pick.get("slot")
                    return {
                        "action": "opposite_lane_punish",
                        "slot": wincon_pick["slot"],
                        "recommended_card": wname,
                        "target_tile": target_tile,
                        "confidence": 0.93,
                        "emergency": False,
                        "reasoning": (
                            f"{posture}: ignoring minor poke (row={t_row}) to punish opposite lane "
                            f"with {wname} at {target_tile} ({current_elixir:.1f} elx)."
                        ),
                    }
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.96,
                    "emergency": False,
                    "reasoning": f"{posture}: {posture_reason}",
                }

            since_push = now_ts - self._last_push_ts
            since_defend = now_ts - self._last_defend_ts
            last_lane_ts, last_lane_role = self._last_defend_by_lane.get(t_lane, (-100.0, ""))
            since_lane_defend = now_ts - last_lane_ts

            # Positive-Trade & Faster-Response Defense Guards:
            # 1) River boundary (row < 18): don't fight on bridge unless >=4.4 elx (or post-Hog 6.2s lock)
            eff_river_floor = max(4.4, min(5.8, qwen_floor)) if qwen_mode == "conserve_elixir" else 4.4
            if since_push < 4.5:
                eff_river_floor = max(eff_river_floor, 6.2)
            if t_row < 18 and not (len(immediate_threats) >= 3 and current_elixir >= 3.8):
                if current_elixir < eff_river_floor:
                    return {
                        "action": "wait",
                        "slot": None,
                        "recommended_card": None,
                        "target_tile": None,
                        "confidence": 0.96,
                        "emergency": False,
                        "reasoning": f"Banking elixir ({current_elixir:.1f}/{eff_river_floor:.1f}) while threat steps onto bridge (row {t_row}).",
                    }

            # 2) In our half (row >= 18): react PROMPTLY (>= 3.2 elx) if we hold a 1-3 elixir positive counter,
            #    otherwise conserve until 3.8+ (avoids acting super late while enemies stack on tower!)
            eff_def_floor = 3.1 if (has_cheap_counter and current_elixir >= 3.1 and since_push >= 4.5) else (max(3.8, min(5.5, qwen_floor)) if qwen_mode == "conserve_elixir" else 3.8)
            if since_push < 4.5 and t_row < 20:
                eff_def_floor = max(eff_def_floor, 6.2)
            if current_elixir < eff_def_floor and t_row < 20 and len(immediate_threats) < 2:
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.96,
                    "emergency": False,
                    "reasoning": f"Banking elixir ({current_elixir:.1f}/{eff_def_floor:.1f}) for positive counter; lone threat at row {t_row}.",
                }

            # 3) Hard low-elixir floor (< 2.5): allow 1-elx positive Skeletons/Spirit kite at >= 1.5 elx ONLY if deep tower hit (row >= 22)
            if current_elixir < 2.5 and not (t_row >= 22 and since_lane_defend >= 2.8 and current_elixir >= 1.5 and any(float(s.get("card_cost") or 3) <= 1.0 for s in candidates)):
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.96,
                    "emergency": False,
                    "reasoning": f"Protecting elixir bank ({current_elixir:.1f}); refusing low-elixir panic dump at row {t_row}.",
                }

            # 4) Active per-lane defender patience: let deployed defender + Princess Tower finish cleaning up
            lane_lock_sec = 5.0 if last_lane_role == "building" else (3.6 if last_lane_role in ("minitank", "ranged") else 2.5)
            lane_floor = 5.5 if last_lane_role == "building" else 4.6
            can_ignore_ground_lock = is_air_ctx and last_lane_role in ("minitank", "cycle")
            if (
                since_lane_defend < lane_lock_sec
                and current_elixir < lane_floor
                and not can_ignore_ground_lock
                and len(immediate_threats) < 4
                and t_row < 24
            ):
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.95,
                    "emergency": False,
                    "reasoning": f"Letting active {last_lane_role} + tower defend {t_lane} lane while banking ({current_elixir:.1f}/{lane_floor:.1f}).",
                }

            # 5) Global & panic-dump cooldowns: prevent dumping 2+ squishies in <2.5s or double-Cannon within 12s
            if since_lane_defend < 2.4 and len(immediate_threats) < 3 and t_row < 22:
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.95,
                    "emergency": False,
                    "reasoning": f"Anti-panic pacing ({since_lane_defend:.1f}s since last {t_lane} defender); letting unit+tower work.",
                }
            if since_defend < 2.6 and current_elixir < 3.8 and t_row < 23:
                return {
                    "action": "wait",
                    "slot": None,
                    "recommended_card": None,
                    "target_tile": None,
                    "confidence": 0.95,
                    "emergency": False,
                    "reasoning": f"Conserving elixir ({current_elixir:.1f}) while previous defender engages (row {t_row}).",
                }

            # 6) Disciplined Spell Value Check (NO immediate Hog -> Fireball suicide, require real defensive cluster)
            spell_zone_threats = [t for t in immediate_threats if int(t.get("row", 16)) >= 18]
            spell_pick = None
            spell_tile = None
            spell_score = -1.0
            spell_hits = 0
            for s in candidates:
                sname = self._norm_name(s)
                if self._infer_role(sname, s.get("card_role")) != "spell":
                    continue
                scost = float(s.get("card_cost") or 3)
                pool = immediate_threats if (since_push >= 5.5 and scost <= 3) else spell_zone_threats
                if not pool:
                    continue
                rad = float(s.get("spell_radius") or SPELL_RADII.get(sname, 2.5))
                hits, sc, sr = self._best_spell_cluster(pool, rad)
                # Guard expensive spells (Fireball >= 4 elixir):
                # - Never throw within 5.5s of a Hog push unless >= 3 enemies deep (row >= 21) AND elixir >= 6.0
                # - Otherwise require (>=3 hits @ >=4.5 elx) OR (>=2 deep hits row >=20 @ >=6.0 elx)
                if scost >= 4.0:
                    if since_push < 5.5 and not (hits >= 3 and t_row >= 21 and current_elixir >= 6.0):
                        continue
                    if not ((hits >= 3 and current_elixir >= 4.5) or (hits >= 2 and sr >= 20 and current_elixir >= 6.0)):
                        continue
                else:
                    # Cheaper 2-3 elixir spells (Arrows / Log): require >=3 hits or (>=2 deep hits row >= 19 & elx >= 3.8)
                    if since_push < 5.0 and hits < 3:
                        continue
                    if not (hits >= 3 or (hits >= 2 and sr >= 19 and current_elixir >= 3.8)):
                        continue

                score = hits * 2.0 + (0.6 if scost <= 3.0 else 0.0)
                if score > spell_score:
                    spell_score = score
                    spell_hits = hits
                    spell_pick = s
                    spell_tile = [sc, sr]

            if spell_pick is not None and spell_tile is not None:
                picked = spell_pick
                picked_name = self._norm_name(picked)
                return {
                    "action": "defend",
                    "slot": picked["slot"],
                    "recommended_card": picked_name,
                    "target_tile": spell_tile,
                    "target_lane": t_lane,
                    "card_role": "spell",
                    "confidence": 0.94,
                    "emergency": bool(t_row >= 19),
                    "reasoning": f"High-value {picked_name} on {spell_hits} clustered enemies near {spell_tile}.",
                }

            # 7) Positive-Trade Counter Selection (Stop Always Dumping Cannon!)
            # Respect explicit LLM tactical recommendation when affordable & safe (non-wincon)
            picked = None
            if qwen_rec_card and qwen_rec_card not in ("none", "") and qwen_rec_card not in WINCON_HINTS:
                q_slot = self._select_playable_slot(hand_cards, hand_slots, {qwen_rec_card}, current_elixir)
                if q_slot is not None and not (is_air_ctx and self._infer_role(qwen_rec_card, q_slot.get("card_role")) == "minitank"):
                    picked = q_slot

            if picked is None:
                need_cannon_pull = bool(
                    len(immediate_threats) >= 2
                    or recent_heavy_threat
                    or "rusher" in str(t_card).lower()
                    or "heavy" in str(t_card).lower()
                )
                if is_air_ctx:
                    prio_table = AIR_DEFENSE_PRIORITY
                elif need_cannon_pull and not (opp_has_rusher and len(immediate_threats) < 2 and current_elixir < 5.0):
                    prio_table = HEAVY_PULL_PRIORITY
                else:
                    prio_table = TROOP_DEFENSE_PRIORITY

                # Avoid double-Cannon within 12s while previous building is still active (unless >=7 elx)
                since_building = now_ts - self._last_building_ts
                def_pool = [
                    s for s in candidates
                    if not (
                        self._infer_role(self._norm_name(s), s.get("card_role")) == "building"
                        and since_building < 12.0
                        and t_row < 24
                        and current_elixir < 7.0
                    )
                ] or candidates

                # When conserving (not overspending) against lone unit, enforce low-cost positive trade (<= max_defense_cost) if cheap counter exists
                budget_candidates = def_pool
                if not can_overspend and len(immediate_threats) == 1 and not is_air_ctx:
                    cheap_pool = [
                        s for s in def_pool
                        if float(s.get("card_cost") or 3) <= min(3.0, qwen_max_def_cost)
                        and self._infer_role(self._norm_name(s), s.get("card_role")) in ("cycle", "minitank", "air_dps")
                    ]
                    if cheap_pool:
                        budget_candidates = cheap_pool

                # Consult learned counter_table (only overrides priority once n>=5 & 1 SE beats prior)
                threat_sig = (
                    "air_threat" if is_air_ctx
                    else ("heavy_or_rusher" if (recent_heavy_threat or "rusher" in str(t_card).lower() or "heavy" in str(t_card).lower())
                          else ("multi_push" if len(immediate_threats) >= 2 else "single_ground"))
                )
                learned_card = self.learning_tables.get_best_counter(
                    threat_sig, [self._norm_name(s) for s in budget_candidates], require_learned=True
                )
                if learned_card is not None:
                    l_slot = next((s for s in budget_candidates if self._norm_name(s) == learned_card), None)
                    if l_slot is not None and prio_table.get(self._infer_role(learned_card, l_slot.get("card_role")), 0) >= 2:
                        picked = l_slot

                # Fallback / standard priority selection; never play wincon on defense
                if picked is None:
                    picked = self._pick_by_priority(budget_candidates, prio_table, min_priority=2)
                if picked is None:
                    picked = self._pick_by_priority(def_pool, prio_table, min_priority=2)
                if picked is None and (t_row >= 25 or current_elixir >= 4.5):
                    picked = self._pick_by_priority(def_pool, prio_table, min_priority=1)

            else:
                threat_sig = (
                    "air_threat" if is_air_ctx
                    else ("heavy_or_rusher" if need_cannon_pull
                          else ("multi_push" if len(immediate_threats) >= 2 else "single_ground"))
                ) if "need_cannon_pull" in locals() else "single_ground"

            non_wincon_troops = [
                s for s in candidates
                if self._infer_role(self._norm_name(s), s.get("card_role")) not in ("wincon", "spell")
            ]
            if picked is None:
                # If air threat & only ground cards playable, wait for anti-air unless tower emergency
                if is_air_ctx:
                    cycle_air = self._select_playable_slot(
                        hand_cards, hand_slots, {"skeletons", "electro_spirit"}, current_elixir
                    )
                    if cycle_air and current_elixir >= 6.5:
                        picked = cycle_air
                    elif t_row >= 25 and current_elixir >= 2.0 and non_wincon_troops:
                        picked = non_wincon_troops[0]
                    else:
                        return {
                            "action": "wait",
                            "slot": None,
                            "recommended_card": None,
                            "target_tile": None,
                            "confidence": 0.95,
                            "emergency": False,
                            "reasoning": "Air threat active; conserving elixir and holding ground cards for anti-air.",
                        }
                elif t_row >= 25 and current_elixir >= 2.5 and non_wincon_troops:
                    picked = non_wincon_troops[0]
                else:
                    return {
                        "action": "wait",
                        "slot": None,
                        "recommended_card": None,
                        "target_tile": None,
                        "confidence": 0.95,
                        "emergency": False,
                        "reasoning": f"Conserving elixir ({current_elixir:.1f}); holding wincon/spells against threat at [{t_col}, {t_row}].",
                    }

            picked_name = self._norm_name(picked)
            picked_role = self._infer_role(picked_name, picked.get("card_role"))
            picked_cost = float(picked.get("card_cost") or CARD_COSTS.get(picked_name, 3))
            bandit_arm = self.learning_tables.select_placement_arm(
                role=picked_role,
                threat_sig=threat_sig,
                current_elixir=current_elixir,
                card_cost=picked_cost,
                enemy_cost_est=float(len(immediate_threats) * 3.0),
                tower_hp_ok=our_lane_alive,
                allow_explore=self.allow_explore,
            )
            bandit_tile = (
                self.projector.relative_to_tile(bandit_arm["dy_river"], bandit_arm["dx_tower"], lane=t_lane)
                if bandit_arm is not None else None
            )
            defensive_col, defensive_row, place_note = self._compute_defensive_tile(
                picked_role=picked_role,
                t_col=t_col,
                t_row=t_row,
                t_lane=t_lane,
                t_hp=t_hp,
                t_dist=t_dist,
                is_air_ctx=is_air_ctx,
                threat_count=len(immediate_threats),
                our_lane_alive=our_lane_alive,
                bandit_tile=bandit_tile,
            )

            return {
                "action": "defend",
                "slot": picked["slot"],
                "recommended_card": picked_name,
                "target_tile": [defensive_col, defensive_row],
                "target_lane": t_lane,
                "card_role": picked_role,
                "posture": posture,
                "learning_arm": bandit_arm.get("arm_id") if bandit_arm else None,
                "explored": bool(bandit_arm.get("explored", False)) if bandit_arm else False,
                "confidence": 0.93,
                "emergency": bool(t_row >= 18),
                "reasoning": (
                    f"{posture} ({place_note}): {picked_name} vs {t_card} "
                    f"(hp={t_hp:.2f}, dist={t_dist:.1f}) at [{defensive_col}, {defensive_row}] ({t_lane})."
                ),
            }

        # Determine active opposite-lane push column dynamically based on alive towers, Qwen brain, & buildup
        opp_left_alive = (tower_states or {}).get("opponent_princess_left", {}).get("alive", True)
        opp_right_alive = (tower_states or {}).get("opponent_princess_right", {}).get("alive", True)
        left_count = sum(1 for t in arena_threats if int(t.get("col", 9)) < 9 and int(t.get("row", 0)) >= 6)
        right_count = sum(1 for t in arena_threats if int(t.get("col", 9)) >= 9 and int(t.get("row", 0)) >= 6)
        if qwen_lane == "right" and opp_right_alive:
            push_col = 14
        elif qwen_lane == "left" and opp_left_alive:
            push_col = 3
        elif opp_left_alive and opp_right_alive:
            push_col = 14 if left_count > right_count else 3
        else:
            push_col = 3 if opp_left_alive else (14 if opp_right_alive else 9)

        # 2. GATED PREDICTION PLAYS (Strictly ONLY when opponent elixir AND hand cycle are known)
        if prediction_unlocked:
            # 2a. Predictive Arrows on Princess defense spot right after our Hog push if opponent has swarm in hand
            if (
                (now_ts - self._last_push_ts) <= 3.5
                and "swarm" in opp_hand
                and opp_elx >= 3.0
                and current_elixir >= 3.0
            ):
                arrows_pick = self._select_playable_slot(hand_cards, hand_slots, {"arrows"}, current_elixir)
                if arrows_pick is not None:
                    self._last_slot_used = arrows_pick.get("slot")
                    self._last_push_ts = -100.0  # one prediction per push
                    return {
                        "action": "predict_spell",
                        "slot": arrows_pick["slot"],
                        "recommended_card": "arrows",
                        "target_tile": [self._last_push_col, 7],
                        "confidence": 0.91,
                        "emergency": False,
                        "reasoning": (
                            f"Prediction unlocked (opp_elx~{opp_elx:.1f}, opp_hand={opp_hand}): "
                            f"pre-aiming Arrows at [{self._last_push_col}, 7] for swarm counter."
                        ),
                    }

            # 2b. Predictive Cannon pull if opponent has rusher in hand + elixir >= 4 and we are near cap
            if (
                "rusher" in opp_hand
                and opp_elx >= 4.0
                and current_elixir >= 7.5
            ):
                cannon_pick = self._select_playable_slot(hand_cards, hand_slots, BUILDING_HINTS, current_elixir)
                if cannon_pick is not None:
                    self._last_slot_used = cannon_pick.get("slot")
                    return {
                        "action": "predict_building",
                        "slot": cannon_pick["slot"],
                        "recommended_card": self._norm_name(cannon_pick),
                        "target_tile": [8, 21],
                        "confidence": 0.91,
                        "emergency": False,
                        "reasoning": (
                            f"Prediction unlocked (opp_elx~{opp_elx:.1f}, rusher in opp_hand): "
                            "pre-placing Cannon at [8, 21]."
                        ),
                    }

        # 3. Supported / Elixir-Safe Opposite-Lane Punish (avoid naked low-elixir bridge spam)
        elixir_mult = max(1.0, float(opp_intel.get("elixir_multiplier", 1.0)))
        total_threats = len(arena_threats)
        lane_defenders = [
            t for t in arena_threats
            if ((int(t.get("col", 9)) < 9) == (push_col < 9)) and int(t.get("row", 0)) >= 6
        ]
        if (now_ts - self._last_push_ts) < (2.2 / elixir_mult) or (
            (now_ts - self._last_push_ts) < (4.5 / elixir_mult) and current_elixir < 8.5
        ):
            return {
                "action": "wait",
                "slot": None,
                "recommended_card": None,
                "target_tile": None,
                "confidence": 0.96,
                "emergency": False,
                "reasoning": f"Recovering elixir ({current_elixir:.1f}) after recent push before committing again.",
            }

        mgmt = self.learning_tables.management_params
        leak_thresh = float(mgmt.get("leak_threshold", 9.5))
        leak_max_sec = float(mgmt.get("leak_max_sec", 3.0)) / elixir_mult
        counterpush_elx = float(mgmt.get("counterpush_elixir", 7.0))
        slow_roll_cd = float(mgmt.get("slow_roll_cooldown_sec", 7.5)) / elixir_mult

        if current_elixir >= leak_thresh:
            if self._leak_start_ts is None:
                self._leak_start_ts = now_ts
        else:
            self._leak_start_ts = None
        forced_leak_spend = bool(self._leak_start_ts is not None and (now_ts - self._leak_start_ts) >= leak_max_sec)
        leak_warning = bool(opp_intel.get("elixir_leak_warning", False) or current_elixir >= 9.2 or forced_leak_spend)
        wincon_pick = self._select_playable_slot(hand_cards, hand_slots, WINCON_HINTS, current_elixir)
        qwen_greenlight = (qwen_mode in ("punish_opposite_lane", "overspend_punish") or qwen_overspend) and (qwen_rec_card in WINCON_HINTS or qwen_overspend) and current_elixir >= 5.8
        qwen_hold_off = qwen_mode == "conserve_elixir" and current_elixir < 8.5 and not can_overspend and not leak_warning
        # If enemy holds a Cannon/Tesla/PEKKA counter in hand and is NOT down elixir, don't donate Hog blindly (unless leaking near cap)
        counter_blocks_hog = opp_has_building_counter and not can_overspend and current_elixir < 8.4 and not leak_warning
        can_safe_push = (
            wincon_pick is not None
            and len(lane_defenders) == 0
            and not qwen_hold_off
            and not counter_blocks_hog
            and (
                current_elixir >= counterpush_elx
                or forced_leak_spend
                or qwen_greenlight
                or (can_overspend and current_elixir >= 5.8)
                or (prediction_unlocked and opp_elx < 4.0 and current_elixir >= 5.8)
            )
        )
        if can_safe_push:
            picked_name = self._norm_name(wincon_pick)
            if picked_name in ("goblin_barrel", "miner", "graveyard"):
                push_tile = [9, 3] if (not opp_left_alive and not opp_right_alive) else [push_col, 7]
            else:
                push_tile = [push_col, 17]
            return {
                "action": "opposite_lane_punish",
                "slot": wincon_pick["slot"],
                "recommended_card": picked_name,
                "target_tile": push_tile,
                "confidence": 0.93,
                "emergency": False,
                "reasoning": (
                    f"Elixir banked ({current_elixir:.1f}, lane clear); push at {push_tile} "
                    f"with {picked_name}, keeping >=3 elixir in reserve."
                ),
            }

        # 4. Air-heavy enemy adaptation: if opponent runs mostly flying units and our hand lacks anti-air,
        # proactively cycle a cheap card in the back at >= 7.0 elixir so we don't starve ourselves
        has_anti_air_in_hand = any(self._norm_name(s) in ANTI_AIR_CARDS for s in (hand_slots or []))
        if enemy_all_flying and not has_anti_air_in_hand and current_elixir >= 7.0:
            cycle_pick = self._select_playable_slot(
                hand_cards, hand_slots, {"skeletons", "electro_spirit", "knight"}, current_elixir
            )
            if cycle_pick is not None:
                cname = self._norm_name(cycle_pick)
                back_col = 4 if push_col < 9 else 13
                self._last_slot_used = cycle_pick.get("slot")
                return {
                    "action": "cycle_for_anti_air",
                    "slot": cycle_pick["slot"],
                    "recommended_card": cname,
                    "target_tile": [back_col, 28],
                    "confidence": 0.91,
                    "emergency": False,
                    "reasoning": f"Enemy air-heavy & hand lacks anti-air; cycling {cname} in back at {current_elixir:.1f} elixir.",
                }

        # Build ONE slow-roll counter-push in the back (>= 9.0 elixir or forced leak spend, slow_roll_cd cooldown)
        near_board_threats = sum(1 for t in arena_threats if int(t.get("row", 0)) >= 10)
        since_slow_roll = now_ts - self._last_slow_roll_ts
        if (since_slow_roll >= slow_roll_cd or forced_leak_spend) and near_board_threats == 0 and (
            current_elixir >= 9.0 or (current_elixir >= 8.4 and len(arena_threats) > 0) or forced_leak_spend
        ):
            back_pick = self._pick_by_priority(candidates, BACK_ROLL_PRIORITY, min_priority=2)
            if back_pick is not None:
                back_col = 4 if push_col < 9 else 13
                picked_name = self._norm_name(back_pick)
                self._last_slot_used = back_pick.get("slot")
                return {
                    "action": "slow_roll_back",
                    "slot": back_pick["slot"],
                    "recommended_card": picked_name,
                    "target_tile": [back_col, 28],
                    "confidence": 0.92,
                    "emergency": False,
                    "reasoning": f"High elixir ({current_elixir:.1f}/10.0) & no immediate push; slow-rolling {picked_name} in back [{back_col}, 28].",
                }

        # 5. Passive elixir accumulation below 8.5 (unless leak_warning >= 9.2)
        if current_elixir < 8.5 and not leak_warning:
            return {
                "action": "wait",
                "slot": None,
                "recommended_card": None,
                "target_tile": None,
                "confidence": 0.97,
                "emergency": False,
                "reasoning": f"Accumulating elixir ({current_elixir:.1f}/10.0) with no immediate threat in our half.",
            }

        # 6. Anti-leak cycle / slow roll in back at >= 8.5 (or >= 9.2 leak_warning)
        if not (since_slow_roll >= slow_roll_cd or forced_leak_spend):
            return {
                "action": "wait",
                "slot": None,
                "recommended_card": None,
                "target_tile": None,
                "confidence": 0.95,
                "emergency": False,
                "reasoning": (
                    f"Slow-roll cooldown ({since_slow_roll:.1f}s < {slow_roll_cd:.1f}s); "
                    f"holding elixir ({current_elixir:.1f}/10.0)."
                ),
            }
        back_pick = self._pick_by_priority(candidates, BACK_ROLL_PRIORITY, min_priority=1)
        if back_pick is None and leak_warning:
            # Last resort anti-leak: cycle non-spell building defensively center instead of wasting elixir at 10.0
            b_pick = self._select_playable_slot(hand_cards, hand_slots, BUILDING_HINTS, current_elixir)
            if b_pick is not None:
                self._last_slot_used = b_pick.get("slot")
                return {
                    "action": "anti_leak_building",
                    "slot": b_pick["slot"],
                    "recommended_card": self._norm_name(b_pick),
                    "target_tile": [8, 21],
                    "confidence": 0.90,
                    "emergency": False,
                    "reasoning": f"Elixir leak prevention ({current_elixir:.1f}/10.0): pre-placing {self._norm_name(b_pick)} center [8, 21].",
                }
        if back_pick is None:
            return {
                "action": "wait",
                "slot": None,
                "recommended_card": None,
                "target_tile": None,
                "confidence": 0.95,
                "emergency": False,
                "reasoning": f"At {current_elixir:.1f} elixir, holding spells/building instead of wasting in back.",
            }

        back_col = 4 if push_col < 9 else 13
        picked_name = self._norm_name(back_pick)
        self._last_slot_used = back_pick.get("slot")
        return {
            "action": "slow_roll_back",
            "slot": back_pick["slot"],
            "recommended_card": picked_name,
            "target_tile": [back_col, 28],
            "confidence": 0.91,
            "emergency": False,
            "reasoning": f"Near elixir cap ({current_elixir:.1f}/10.0); slow-rolling {picked_name} in back [{back_col}, 28].",
        }
