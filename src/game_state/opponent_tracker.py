"""
Opponent Deck, Hand Cycle, Archetype & Prediction Readiness Tracker.
Tracks opponent plays from observed arena spawns/crossings, estimates card roles
(air, swarm, rusher, heavy, spell, standard), maintains the 4-card hand / 8-card cycle,
and gates predictive plays strictly behind BOTH:
  1. Confident opponent elixir count (enough observed plays + tight uncertainty).
  2. Known opponent hand cycle (>= 4 plays observed so hand queue is populated).
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


ARCHETYPE_COSTS = {
    "swarm": 3,
    "air": 3,
    "rusher": 4,
    "heavy": 5,
    "spell": 3,
    "standard": 3,
}

# Real Clash Royale card costs & tactical archetypes so VLM-spotted enemy troops calibrate elixir + hand tracking
CARD_CATALOG: Dict[str, Tuple[int, str]] = {
    "hog_rider": (4, "rusher"),
    "ram_rider": (5, "rusher"),
    "battle_ram": (4, "rusher"),
    "royal_hogs": (5, "rusher"),
    "balloon": (5, "air"),
    "lava_hound": (7, "air"),
    "baby_dragon": (4, "air"),
    "mega_minion": (3, "air"),
    "minions": (3, "air"),
    "minion_horde": (5, "air"),
    "bats": (2, "air"),
    "skeleton_barrel": (3, "air"),
    "giant": (5, "heavy"),
    "golem": (8, "heavy"),
    "pekka": (7, "heavy"),
    "mega_knight": (7, "heavy"),
    "royal_giant": (6, "heavy"),
    "goblin_giant": (6, "heavy"),
    "electro_giant": (7, "heavy"),
    "giant_skeleton": (6, "heavy"),
    "skeleton_army": (3, "swarm"),
    "goblin_gang": (3, "swarm"),
    "guards": (3, "swarm"),
    "goblin_barrel": (3, "swarm"),
    "valkyrie": (4, "standard"),
    "knight": (3, "standard"),
    "mini_pekka": (4, "heavy"),
    "musketeer": (4, "standard"),
    "wizard": (5, "standard"),
    "witch": (5, "standard"),
    "prince": (5, "rusher"),
    "dark_prince": (4, "rusher"),
    "bandit": (3, "rusher"),
    "miner": (3, "rusher"),
    "cannon": (3, "building"),
    "tesla": (4, "building"),
    "inferno_tower": (5, "building"),
    "bomb_tower": (4, "building"),
    "tombstone": (3, "building"),
    "fireball": (4, "spell"),
    "arrows": (3, "spell"),
    "zap": (2, "spell"),
    "the_log": (2, "spell"),
    "poison": (4, "spell"),
    "tornado": (3, "spell"),
}


class OpponentTracker:
    """Tracks opponent deck composition, 4-card cycle, and prediction unlock status."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.plays_count: int = 0
        self.seen_archetypes: List[str] = []
        self.known_opp_cards: List[str] = []
        self.hand_slots: List[str] = []      # cards likely in opponent's current 4-card hand
        self.cycle_queue: Deque[str] = deque()  # recently played cards waiting to cycle back
        self.play_history: List[Dict[str, Any]] = []
        self._last_spawn_ts: float = -10.0
        self._last_vlm_card_ts: Dict[str, float] = {}
        self._last_enemy_tiles: List[Tuple[int, int]] = []
        self._last_opp_side_count: int = 0
        self._active_tracks: List[Dict[str, Any]] = []
        self._next_track_id: int = 1
        self._track_grace_sec: float = 1.2

    def ingest_vlm_cards(
        self,
        opp_cards_seen: List[str],
        elapsed_sec: float,
        elixir_tracker: Any = None,
    ) -> None:
        """
        Records named opponent cards spotted by the LLM tactical brain so hand cycle and
        elixir deduction use true card costs (e.g. Giant=5, Balloon=5, Mega Knight=7).
        """
        if not opp_cards_seen:
            return
        for raw in opp_cards_seen:
            cname = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
            if not cname or cname in ("none", "unknown"):
                continue
            last_seen = self._last_vlm_card_ts.get(cname, -100.0)
            if (elapsed_sec - last_seen) < 9.0:
                continue
            self._last_vlm_card_ts[cname] = elapsed_sec
            if cname not in self.known_opp_cards and len(self.known_opp_cards) < 8:
                self.known_opp_cards.append(cname)
            cost, role = CARD_CATALOG.get(cname, (3, "standard"))
            self.record_play(role, cost, elapsed_sec, lane="vlm", card_name=cname)
            # Only trigger extra elixir subtraction if CV spawn didn't just fire in last 2.2s
            if elixir_tracker is not None and (elapsed_sec - self._last_spawn_ts) >= 2.2:
                elixir_tracker.register_card_played("opponent", cost, elapsed_sec)
                self._last_spawn_ts = elapsed_sec

    @staticmethod
    def _is_air_crossing(ent: Dict[str, Any]) -> bool:
        """
        In Clash Royale, ground units must cross the river (row 14..16) via left bridge (col 2..4)
        or right bridge (col 13..15). Crossing outside bridges indicates a flying/air unit.
        """
        col = int(ent.get("col", 9))
        row = int(ent.get("row", 0))
        if 14 <= row <= 16:
            on_left_bridge = 2 <= col <= 4
            on_right_bridge = 13 <= col <= 15
            return not (on_left_bridge or on_right_bridge)
        return False

    def _classify_spawn_burst(
        self,
        new_enemies: List[Dict[str, Any]],
        all_enemies: List[Dict[str, Any]],
    ) -> Tuple[str, int]:
        if not new_enemies:
            return "standard", 3

        if any(self._is_air_crossing(e) for e in all_enemies):
            return "air", 3

        if len(new_enemies) >= 3:
            return "swarm", 3

        # Fast bridge/river appearance on our side without prior opp-half travel -> rusher/spell
        if all(int(e.get("row", 0)) >= 15 for e in new_enemies):
            return "rusher", 4

        return "standard", 3

    def record_play(
        self,
        archetype: str,
        est_cost: int,
        elapsed_sec: float,
        lane: str = "center",
        card_name: Optional[str] = None,
    ) -> None:
        self.plays_count += 1
        token = card_name or archetype
        tag = f"{token}_{len(self.seen_archetypes) + 1}" if token not in self.seen_archetypes else token
        if len(self.seen_archetypes) < 8 and token not in self.seen_archetypes:
            self.seen_archetypes.append(token)

        # Rotate 4-card hand / 4-card cycle queue (8-card Clash Royale deck invariant)
        if token in self.hand_slots:
            self.hand_slots.remove(token)
        self.cycle_queue.append(token)
        while len(self.cycle_queue) > 4:
            returned = self.cycle_queue.popleft()
            if len(self.hand_slots) < 4:
                self.hand_slots.append(returned)

        # Before all 4 cycle slots fill, keep observed cards/archetypes visible in known hand estimate
        if self.plays_count <= 4 and len(self.hand_slots) < 4 and token not in self.hand_slots:
            self.hand_slots.append(token)

        self.play_history.append({
            "t": round(elapsed_sec, 2),
            "archetype": archetype,
            "card": token,
            "tag": tag,
            "cost": est_cost,
            "lane": lane,
        })

    @staticmethod
    def _enemy_norm_pos(ent: Dict[str, Any]) -> Tuple[float, float]:
        vp = ent.get("norm_vp")
        if isinstance(vp, (list, tuple)) and len(vp) >= 2:
            return float(vp[0]), float(vp[1])
        col = float(ent.get("col", 9))
        row = float(ent.get("row", 0))
        return max(0.0, min(1.0, col / 18.0)), max(0.0, min(1.0, row / 32.0))

    def _update_spatial_tracks(
        self,
        enemies: List[Dict[str, Any]],
        elapsed_sec: float,
    ) -> List[Dict[str, Any]]:
        """
        Matches frame detections against active tracks (within 1.2s blink grace TTL)
        using predicted motion so flickering detections never re-trigger phantom spawns.
        Returns ONLY newly spawned enemy dicts that did not match any active track.
        """
        # Expire tracks past grace TTL
        self._active_tracks = [
            tr for tr in self._active_tracks
            if (elapsed_sec - tr["last_seen"]) <= self._track_grace_sec
        ]

        det_points = []
        for idx, e in enumerate(enemies):
            cx, cy = self._enemy_norm_pos(e)
            det_points.append((idx, cx, cy, int(e.get("col", 9)), int(e.get("row", 0))))

        # Score candidate (track, detection) pairs by spatial proximity + mild motion extrapolation
        pairs: List[Tuple[float, int, int]] = []
        for t_idx, tr in enumerate(self._active_tracks):
            dt = max(0.0, elapsed_sec - tr["last_seen"])
            pred_x = tr["cx"] + tr["vx"] * min(dt, 0.8)
            pred_y = tr["cy"] + tr["vy"] * min(dt, 0.8)
            max_dist = 0.11 + min(0.08, dt * 0.09)
            for d_idx, cx, cy, ecol, erow in det_points:
                dist = ((cx - pred_x) ** 2 + (cy - pred_y) ** 2) ** 0.5
                tile_dist = abs(ecol - tr["col"]) + abs(erow - tr["row"])
                if dist <= max_dist or tile_dist <= 3:
                    pairs.append((dist, t_idx, d_idx))

        pairs.sort(key=lambda p: p[0])
        matched_tracks = set()
        matched_dets = set()
        for _, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            tr = self._active_tracks[t_idx]
            _, cx, cy, ecol, erow = det_points[d_idx]
            dt = max(0.03, elapsed_sec - tr["last_seen"])
            if dt <= 1.0:
                inst_vx = max(-0.25, min(0.25, (cx - tr["cx"]) / dt))
                inst_vy = max(-0.25, min(0.25, (cy - tr["cy"]) / dt))
                tr["vx"] = 0.65 * tr["vx"] + 0.35 * inst_vx
                tr["vy"] = 0.65 * tr["vy"] + 0.35 * inst_vy
            tr["cx"] = cx
            tr["cy"] = cy
            tr["col"] = ecol
            tr["row"] = erow
            tr["lane"] = enemies[d_idx].get("lane", tr["lane"])
            tr["last_seen"] = elapsed_sec
            tr["misses"] = 0
            if self._is_air_crossing(enemies[d_idx]):
                tr["is_air"] = True

        for t_idx, tr in enumerate(self._active_tracks):
            if t_idx not in matched_tracks:
                tr["misses"] += 1

        new_spawns: List[Dict[str, Any]] = []
        for d_idx, cx, cy, ecol, erow in det_points:
            if d_idx in matched_dets:
                continue
            e = enemies[d_idx]
            is_air = self._is_air_crossing(e)
            self._active_tracks.append({
                "id": self._next_track_id,
                "cx": cx,
                "cy": cy,
                "vx": 0.0,
                "vy": 0.04,  # default gentle downward advance toward player towers
                "col": ecol,
                "row": erow,
                "lane": e.get("lane", "center"),
                "first_seen": elapsed_sec,
                "last_seen": elapsed_sec,
                "misses": 0,
                "is_air": is_air,
            })
            self._next_track_id += 1
            new_spawns.append(e)

        return new_spawns

    def update_from_frame(
        self,
        enemies: List[Dict[str, Any]],
        elapsed_sec: float,
        elixir_tracker: Any,
    ) -> Dict[str, Any]:
        """
        Inspects current enemy entities using spatial persistence (grace TTL against blinks),
        detects genuine new spawn bursts, updates ElixirTracker, and returns lane threat intel.
        """
        if elixir_tracker is not None and elapsed_sec > getattr(elixir_tracker, "last_timestamp", elapsed_sec):
            elixir_tracker.advance_time(elapsed_sec)
        unmatched_spawns = self._update_spatial_tracks(enemies, elapsed_sec)
        opp_side_spawns = [e for e in unmatched_spawns if int(e.get("row", 0)) < 15]
        air_bridge_spawns = [e for e in unmatched_spawns if self._is_air_crossing(e)]
        direct_drop_spawns = [
            e for e in unmatched_spawns
            if int(e.get("row", 0)) >= 15 and not self._is_air_crossing(e)
        ]

        # Only trigger a new opponent play if unmatched spawns occur outside spawn cooldown
        trigger_burst = opp_side_spawns or air_bridge_spawns or (
            direct_drop_spawns and len(self._active_tracks) == len(unmatched_spawns)
        )
        if trigger_burst and (elapsed_sec - self._last_spawn_ts) >= 2.0:
            burst = opp_side_spawns or air_bridge_spawns or direct_drop_spawns
            archetype, cost = self._classify_spawn_burst(burst, enemies)
            lane = burst[0].get("lane", "center") if burst else "center"
            self.record_play(archetype, cost, elapsed_sec, lane=lane)
            elixir_tracker.register_card_played("opponent", cost, elapsed_sec)
            self._last_spawn_ts = elapsed_sec

        self._last_opp_side_count = sum(1 for e in enemies if int(e.get("row", 0)) < 15)
        self._last_enemy_tiles = [(int(e.get("col", 9)), int(e.get("row", 0))) for e in enemies]

        return self.get_intel(elixir_tracker, active_enemies=enemies)

    def get_intel(
        self,
        elixir_tracker: Any,
        active_enemies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        est = elixir_tracker.get_state()["opponent_elixir_estimate"]
        spread = float(est["max"]) - float(est["min"])

        elixir_known = self.plays_count >= 2 and spread <= 2.5
        hand_known = self.plays_count >= 4 and len(self.hand_slots) >= 2
        prediction_unlocked = bool(elixir_known and hand_known)

        air_plays = sum(1 for h in self.play_history if h["archetype"] == "air")
        active_air = any(self._is_air_crossing(e) for e in (active_enemies or []))
        enemy_all_flying = bool(
            (self.plays_count >= 2 and air_plays / float(self.plays_count) >= 0.6)
            or (active_air and air_plays >= 1)
        )

        opp_hand_roles = {CARD_CATALOG.get(x, (3, x))[1] for x in self.hand_slots}
        opp_has_rusher = bool("rusher" in opp_hand_roles or "heavy" in opp_hand_roles)
        opp_has_building_counter = bool("building" in opp_hand_roles or "tornado" in self.hand_slots or "pekka" in self.hand_slots or "mini_pekka" in self.hand_slots)
        recent_heavy_seen = any(h.get("archetype") in ("rusher", "heavy") for h in self.play_history[-2:])

        enemies_now = active_enemies or []
        left_threats = sum(1 for e in enemies_now if e.get("lane") == "left")
        right_threats = sum(1 for e in enemies_now if e.get("lane") == "right")
        our_half_threats = sum(1 for e in enemies_now if e.get("in_our_half") or int(e.get("row", 0)) >= 16)
        push_detected = bool(
            left_threats >= 2 or right_threats >= 2 or (our_half_threats >= 1 and recent_heavy_seen)
        )

        return {
            "opp_elixir": float(est["mid"]),
            "opp_elixir_range": [float(est["min"]), float(est["max"])],
            "opp_elixir_known": elixir_known,
            "opp_hand_known": hand_known,
            "opp_plays_observed": self.plays_count,
            "opp_hand_archetypes": list(self.hand_slots),
            "opp_known_cards": list(self.known_opp_cards),
            "opp_cycle_queue": list(self.cycle_queue),
            "opp_seen_archetypes": list(self.seen_archetypes),
            "opp_has_rusher": opp_has_rusher,
            "opp_has_building_counter": opp_has_building_counter,
            "recent_heavy_threat": recent_heavy_seen,
            "enemy_all_flying": enemy_all_flying,
            "active_air_threat": active_air,
            "left_lane_threats": left_threats,
            "right_lane_threats": right_threats,
            "our_half_threats": our_half_threats,
            "push_detected": push_detected,
            "active_track_count": len(self._active_tracks),
            "prediction_unlocked": prediction_unlocked,
        }
