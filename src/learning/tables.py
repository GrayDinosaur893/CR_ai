"""
Counter Table, Placement Bandit Table, & Management Params (LEARNING_PLAN.md Sections 3, 4, 5, 7).
Nothing here trains neural weights; it stores prior-seeded tables with constrained
Thompson sampling for safe online adaptation and versioned rollback snapshots.
"""
import copy
import json
import math
import os
import random
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_COUNTER_TABLE: Dict[str, Dict[str, Any]] = {
    "single_ground": {
        "prior": "skeletons",
        "answers": {
            "skeletons": {"cost": 1.0, "n": 0, "w": 0.0, "mean": 1.5, "m2": 0.5, "prior_mean": 1.5},
            "electro_spirit": {"cost": 1.0, "n": 0, "w": 0.0, "mean": 1.3, "m2": 0.5, "prior_mean": 1.3},
            "knight": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.8, "m2": 0.5, "prior_mean": 0.8},
            "minions": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.7, "m2": 0.5, "prior_mean": 0.7},
            "cannon": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.3, "m2": 0.5, "prior_mean": 0.3},
        },
    },
    "heavy_or_rusher": {
        "prior": "cannon",
        "answers": {
            "cannon": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 1.5, "m2": 0.5, "prior_mean": 1.5},
            "knight": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.8, "m2": 0.5, "prior_mean": 0.8},
            "minions": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.7, "m2": 0.5, "prior_mean": 0.7},
            "musketeer": {"cost": 4.0, "n": 0, "w": 0.0, "mean": 0.6, "m2": 0.5, "prior_mean": 0.6},
            "skeletons": {"cost": 1.0, "n": 0, "w": 0.0, "mean": 0.4, "m2": 0.5, "prior_mean": 0.4},
        },
    },
    "multi_push": {
        "prior": "cannon",
        "answers": {
            "cannon": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 1.4, "m2": 0.5, "prior_mean": 1.4},
            "knight": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
            "archers": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 0.9, "m2": 0.5, "prior_mean": 0.9},
            "musketeer": {"cost": 4.0, "n": 0, "w": 0.0, "mean": 0.8, "m2": 0.5, "prior_mean": 0.8},
        },
    },
    "air_threat": {
        "prior": "musketeer",
        "answers": {
            "musketeer": {"cost": 4.0, "n": 0, "w": 0.0, "mean": 1.2, "m2": 0.5, "prior_mean": 1.2},
            "archers": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 1.4, "m2": 0.5, "prior_mean": 1.4},
            "minions": {"cost": 3.0, "n": 0, "w": 0.0, "mean": 1.3, "m2": 0.5, "prior_mean": 1.3},
            "electro_spirit": {"cost": 1.0, "n": 0, "w": 0.0, "mean": 0.6, "m2": 0.5, "prior_mean": 0.6},
        },
    },
}

DEFAULT_PLACEMENT_TABLE: Dict[str, Dict[str, Any]] = {
    "building:heavy_or_rusher": {
        "prior": "4-2_pull",
        "arms": {
            "4-2_pull": {"dy_river": 5, "dx_tower": 4, "n": 0, "w": 0.0, "mean": 1.2, "m2": 0.5, "prior_mean": 1.2},
            "center_pull": {"dy_river": 5, "dx_tower": 5, "n": 0, "w": 0.0, "mean": 1.1, "m2": 0.5, "prior_mean": 1.1},
            "4-3_pull": {"dy_river": 6, "dx_tower": 4, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
        },
    },
    "building:multi_push": {
        "prior": "center_pull",
        "arms": {
            "center_pull": {"dy_river": 5, "dx_tower": 5, "n": 0, "w": 0.0, "mean": 1.2, "m2": 0.5, "prior_mean": 1.2},
            "4-2_pull": {"dy_river": 5, "dx_tower": 4, "n": 0, "w": 0.0, "mean": 1.1, "m2": 0.5, "prior_mean": 1.1},
            "4-3_pull": {"dy_river": 6, "dx_tower": 4, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
        },
    },
    "cycle:single_ground": {
        "prior": "mid_kite",
        "arms": {
            "mid_kite": {"dy_river": 4, "dx_tower": 4, "n": 0, "w": 0.0, "mean": 1.2, "m2": 0.5, "prior_mean": 1.2},
            "tight_kite": {"dy_river": 4, "dx_tower": 2, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
            "front_surround": {"dy_river": 5, "dx_tower": 1, "n": 0, "w": 0.0, "mean": 0.8, "m2": 0.5, "prior_mean": 0.8},
        },
    },
    "ranged:air_threat": {
        "prior": "tower_pocket",
        "arms": {
            "tower_pocket": {"dy_river": 10, "dx_tower": 1, "n": 0, "w": 0.0, "mean": 1.2, "m2": 0.5, "prior_mean": 1.2},
            "inner_pocket": {"dy_river": 10, "dx_tower": 3, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
        },
    },
    "ranged:multi_push": {
        "prior": "tower_pocket",
        "arms": {
            "tower_pocket": {"dy_river": 10, "dx_tower": 1, "n": 0, "w": 0.0, "mean": 1.1, "m2": 0.5, "prior_mean": 1.1},
            "inner_pocket": {"dy_river": 10, "dx_tower": 3, "n": 0, "w": 0.0, "mean": 1.0, "m2": 0.5, "prior_mean": 1.0},
        },
    },
}

DEFAULT_MANAGEMENT_PARAMS: Dict[str, Any] = {
    "version": 1,
    "reserve_floor": 2.0,
    "bank_until": 8.5,
    "counterpush_elixir": 7.0,
    "leak_threshold": 9.5,
    "leak_max_sec": 3.0,
    "slow_roll_cooldown_sec": 7.5,
}


class LearningTables:
    """Persistent counter, placement bandit, and elixir-management tables."""

    MIN_REPLACE_SAMPLES = 5

    def __init__(self, base_dir: str = "data/learning", rng_seed: Optional[int] = None):
        self.base_dir = base_dir
        self.rollback_dir = os.path.join(base_dir, "rollback")
        self.counter_path = os.path.join(base_dir, "counter_table.json")
        self.placement_path = os.path.join(base_dir, "placement_table.json")
        self.params_path = os.path.join(base_dir, "management_params.json")
        self.history_path = os.path.join(base_dir, "learning_history.jsonl")
        self._rng = random.Random(rng_seed)

        self.counter_table: Dict[str, Dict[str, Any]] = {}
        self.placement_table: Dict[str, Dict[str, Any]] = {}
        self.management_params: Dict[str, Any] = {}
        self.load_or_seed()

    def load_or_seed(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        self.counter_table = self._load_json(self.counter_path, DEFAULT_COUNTER_TABLE)
        self.placement_table = self._load_json(self.placement_path, DEFAULT_PLACEMENT_TABLE)
        self.management_params = self._load_json(self.params_path, DEFAULT_MANAGEMENT_PARAMS)

    @staticmethod
    def _load_json(path: str, default_obj: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        obj = copy.deepcopy(default_obj)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        return obj

    def save(self, create_rollback: bool = True) -> None:
        if create_rollback:
            os.makedirs(self.rollback_dir, exist_ok=True)
            for p in (self.counter_path, self.placement_path, self.params_path):
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(self.rollback_dir, os.path.basename(p)))
        for p, obj in (
            (self.counter_path, self.counter_table),
            (self.placement_path, self.placement_table),
            (self.params_path, self.management_params),
        ):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)

    @staticmethod
    def _std_error(entry: Dict[str, Any]) -> float:
        n = int(entry.get("n", 0))
        if n <= 1:
            return 999.0
        var = float(entry.get("m2", 0.5)) / max(1, n - 1)
        return math.sqrt(max(1e-4, var) / n)

    def get_best_counter(
        self, threat_sig: str, available_cards: List[str], require_learned: bool = False
    ) -> Optional[str]:
        """Return lowest-cost learned winner (>=5 samples & 1 SE above prior), else prior (or None if require_learned)."""
        bucket = self.counter_table.get(threat_sig)
        if not bucket or not available_cards:
            return None
        answers = bucket.get("answers", {})
        avail_set = {c.lower().replace(" ", "_") for c in available_cards}
        candidates = [(c, answers[c]) for c in avail_set if c in answers]
        if not candidates:
            return None

        prior_name = bucket.get("prior")
        prior_entry = answers.get(prior_name) if prior_name in avail_set else None
        prior_ref = float(prior_entry["prior_mean"]) if prior_entry else 0.0

        eligible = [
            (c, st) for c, st in candidates
            if int(st.get("n", 0)) >= self.MIN_REPLACE_SAMPLES
            and (float(st["mean"]) - self._std_error(st)) >= prior_ref
        ]
        if eligible:
            # Prefer cheapest answer whose mean score stays within noise of the highest mean
            best_mean = max(float(st["mean"]) for _, st in eligible)
            within_noise = [
                (c, st) for c, st in eligible
                if float(st["mean"]) >= best_mean - max(0.25, self._std_error(st))
            ]
            within_noise.sort(key=lambda x: (float(x[1].get("cost", 3.0)), -float(x[1]["mean"])))
            return within_noise[0][0]

        if require_learned:
            return None
        if prior_name in avail_set:
            return prior_name
        candidates.sort(key=lambda x: (-float(x[1].get("prior_mean", 0.0)), float(x[1].get("cost", 3.0))))
        return candidates[0][0]

    def select_placement_arm(
        self,
        role: str,
        threat_sig: str,
        current_elixir: float,
        card_cost: float,
        enemy_cost_est: float = 3.0,
        tower_hp_ok: bool = True,
        time_remaining_sec: float = 120.0,
        allow_explore: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Select placement arm (dy_river, dx_tower) using constrained Thompson sampling (§4).
        Exploration is strictly gated off unless safety checks pass.
        """
        if role == "spell":
            return None
        ctx_key = f"{role}:{threat_sig}"
        bucket = self.placement_table.get(ctx_key) or self.placement_table.get(f"{role}:single_ground")
        if not bucket:
            return None
        arms: Dict[str, Dict[str, Any]] = bucket.get("arms", {})
        if not arms:
            return None

        reserve = float(self.management_params.get("reserve_floor", 2.0))
        safe_to_explore = bool(
            allow_explore
            and tower_hp_ok
            and enemy_cost_est <= 4.0
            and current_elixir >= (card_cost + reserve)
            and time_remaining_sec > 30.0
        )

        min_n = min(int(a.get("n", 0)) for a in arms.values())
        eps = 0.05 if min_n >= 8 else 0.20

        if safe_to_explore and self._rng.random() < eps:
            best_arm_id = None
            best_sample = -1e9
            for arm_id, st in arms.items():
                sigma = max(0.15, self._std_error(st) if st.get("n", 0) >= 2 else 0.6)
                draw = self._rng.gauss(float(st.get("mean", 1.0)), sigma)
                if draw > best_sample:
                    best_sample = draw
                    best_arm_id = arm_id
            chosen = dict(arms[best_arm_id])
            chosen["arm_id"] = best_arm_id
            chosen["explored"] = True
            return chosen

        # Exploit: promote non-prior arm only after >= 5 samples + 1 SE beat over prior
        prior_id = bucket.get("prior")
        prior_mean = float(arms.get(prior_id, {}).get("mean", 1.0)) if prior_id in arms else 0.0
        best_id = prior_id if prior_id in arms else next(iter(arms.keys()))
        best_score = float(arms[best_id].get("mean", 1.0))
        for arm_id, st in arms.items():
            if int(st.get("n", 0)) >= self.MIN_REPLACE_SAMPLES:
                if (float(st["mean"]) - self._std_error(st)) > prior_mean and float(st["mean"]) > best_score:
                    best_id = arm_id
                    best_score = float(st["mean"])
        chosen = dict(arms[best_id])
        chosen["arm_id"] = best_id
        chosen["explored"] = False
        return chosen

    @staticmethod
    def _welford_update(entry: Dict[str, Any], value: float, weight: float) -> None:
        n_old = int(entry.get("n", 0))
        w_old = float(entry.get("w", 0.0))
        w_new = w_old + weight
        mean_old = float(entry.get("mean", value)) if w_old > 0 else value
        delta = value - mean_old
        mean_new = mean_old + (weight / max(1e-6, w_new)) * delta
        m2_new = float(entry.get("m2", 0.0)) + weight * delta * (value - mean_new)
        entry["n"] = n_old + 1
        entry["w"] = round(w_new, 3)
        entry["mean"] = round(mean_new, 3)
        entry["m2"] = round(max(0.0, m2_new), 4)

    def update_from_exchanges(
        self,
        exchanges: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
        match_id: str = "",
    ) -> Dict[str, Any]:
        """Update counter_table & placement_table offline from a match's exchange rows."""
        counter_updates = 0
        placement_updates = 0

        for ex in exchanges:
            plays = ex.get("our_plays") or []
            if not plays:
                continue
            score = float(ex.get("score", 0.0))
            weight = 1.0 if ex.get("enemy_cost_known_all") else 0.3
            sig = str(ex.get("threat_sig") or "single_ground")

            # Primary answer = first non-unknown defensive card played in exchange
            main_play = plays[0]
            card = str(main_play.get("card") or "")
            role = str(main_play.get("role") or "unknown")

            if sig in self.counter_table and card in self.counter_table[sig].get("answers", {}):
                self._welford_update(self.counter_table[sig]["answers"][card], score, weight)
                counter_updates += 1

            ctx_key = f"{role}:{sig}"
            if ctx_key in self.placement_table:
                dy_r = int(main_play.get("dy_river", 5))
                dx_t = int(main_play.get("dx_tower", 4))
                # Match closest arm within Manhattan <= 1 tile
                arms = self.placement_table[ctx_key]["arms"]
                best_arm = None
                best_d = 99
                for arm_id, st in arms.items():
                    d = abs(int(st["dy_river"]) - dy_r) + abs(int(st["dx_tower"]) - dx_t)
                    if d < best_d:
                        best_d = d
                        best_arm = arm_id
                if best_arm is not None and best_d <= 2:
                    self._welford_update(arms[best_arm], score, weight)
                    placement_updates += 1

        self.save(create_rollback=True)
        summary = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "match_id": match_id,
            "counter_updates": counter_updates,
            "placement_updates": placement_updates,
            "metrics": metrics or {},
        }
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        return summary
