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

    def get_generation_rate(self, match_seconds: float) -> float:
        if match_seconds < 120.0:  # First 2 mins: single elixir
            return self.SINGLE_ELIXIR_RATE
        elif match_seconds < 240.0: # 2 to 4 mins: double elixir
            return self.DOUBLE_ELIXIR_RATE
        else: # Overtime triple elixir
            return self.TRIPLE_ELIXIR_RATE

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
