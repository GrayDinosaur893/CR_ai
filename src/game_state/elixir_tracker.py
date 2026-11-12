"""
Rules-based Elixir Tracking Engine according to 02_DOMAIN_CONTEXT.md.
"""

from typing import Dict, Any, Tuple

class ElixirTracker:
    # Match time benchmarks in seconds (Standard 3:00 min match)
    SINGLE_ELIXIR_RATE = 1.0 / 2.8  # 1 elixir every 2.8s
    DOUBLE_ELIXIR_RATE = 1.0 / 1.4  # 1 elixir every 1.4s
    TRIPLE_ELIXIR_RATE = 1.0 / 0.9  # 1 elixir every 0.9s

    def __init__(self, initial_elixir: float = 5.0):
        self.player_elixir = initial_elixir
        self.opponent_elixir_range = (4.0, 6.0) # (min, max) uncertainty range
        self.last_timestamp = 0.0
        self.visual_multiplier: float = 1.0
        self.current_multiplier: float = 1.0
        self._obs_window_start_ts: float = 0.0
        self._obs_window_start_elx: float = initial_elixir
        self._last_obs_elx: float = initial_elixir

    def get_elixir_multiplier(self, match_seconds: float) -> float:
        """Dynamic elixir phase multiplier: 1x (0-2m), 2x (2-4m / OT1), 3x (4m+ / OT2+)."""
        if match_seconds >= 240.0:
            timer_mult = 3.0
        elif match_seconds >= 120.0:
            timer_mult = 2.0
        else:
            timer_mult = 1.0
        self.current_multiplier = max(timer_mult, self.visual_multiplier)
        return self.current_multiplier

    def observe_player_elixir(self, measured_elixir: float, current_seconds: float) -> float:
        """Visual backup: infer 2x/3x from actual elixir bar fill slope if timer offset."""
        if measured_elixir < self._last_obs_elx - 0.5 or measured_elixir >= 9.7:
            self._obs_window_start_ts = current_seconds
            self._obs_window_start_elx = measured_elixir
            self._last_obs_elx = measured_elixir
            return self.get_elixir_multiplier(current_seconds)

        dt = current_seconds - self._obs_window_start_ts
        if dt >= 2.0:
            slope = (measured_elixir - self._obs_window_start_elx) / dt
            if slope >= 0.92:
                self.visual_multiplier = max(self.visual_multiplier, 3.0)
            elif slope >= 0.60:
                self.visual_multiplier = max(self.visual_multiplier, 2.0)
            self._obs_window_start_ts = current_seconds
            self._obs_window_start_elx = measured_elixir
        self._last_obs_elx = measured_elixir
        return self.get_elixir_multiplier(current_seconds)

    def get_generation_rate(self, match_seconds: float) -> float:
        mult = self.get_elixir_multiplier(match_seconds)
        if mult >= 3.0:
            return self.TRIPLE_ELIXIR_RATE
        if mult >= 2.0:
            return self.DOUBLE_ELIXIR_RATE
        return self.SINGLE_ELIXIR_RATE

    def advance_time(self, current_seconds: float):
        dt = max(0.0, current_seconds - self.last_timestamp)
        rate = self.get_generation_rate(current_seconds)
        gain = dt * rate

        self.player_elixir = min(10.0, self.player_elixir + gain)
        min_opp = min(10.0, self.opponent_elixir_range[0] + gain)
        max_opp = min(10.0, self.opponent_elixir_range[1] + gain)
        self.opponent_elixir_range = (min_opp, max_opp)
        self.last_timestamp = current_seconds

    def register_card_played(self, player: str, elixir_cost: int, current_seconds: float):
        self.advance_time(current_seconds)
        if player == "player":
            self.player_elixir = max(0.0, self.player_elixir - elixir_cost)
        else:
            min_opp = max(0.0, self.opponent_elixir_range[0] - elixir_cost)
            max_opp = max(0.0, self.opponent_elixir_range[1] - elixir_cost)
            self.opponent_elixir_range = (min_opp, max_opp)

    def get_state(self) -> Dict[str, Any]:
        return {
            "player_elixir": round(self.player_elixir, 1),
            "opponent_elixir_estimate": {
                "min": round(self.opponent_elixir_range[0], 1),
                "max": round(self.opponent_elixir_range[1], 1),
                "mid": round((self.opponent_elixir_range[0] + self.opponent_elixir_range[1]) / 2.0, 1)
            }
        }
