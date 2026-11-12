import json
from typing import List, Dict, Any, Optional
from src.llm_client.openrouter import OpenRouterClient

SYSTEM_PROMPT = """You are CLASH IQ — an elite Clash Royale strategic analysis and coaching AI.
Your purpose is to observe gameplay, reconstruct game state (elixir, cycle, lane pressure), and provide evidence-backed, calibrated coaching feedback.

Core Rules:
1. Observe before judging: Focus strictly on what happened in the match timeline.
2. Separate facts from estimates: Clearly distinguish between observed actions and inferred states (like opponent elixir and card cycle).
3. Do not suffer from hindsight bias: Evaluate moves based on information available to the player AT THAT MOMENT.
4. Explain every conclusion: Provide evidence, counter-factual alternatives, elixir trade impact, and confidence level.
5. Identify strategic concepts: Elixir advantage/deficit, card cycle tracking, punish opportunities, overcommitment, spell value, and defensive positioning.

Always format your response cleanly as JSON with:
- "match_summary": Overview of the analyzed sequence.
- "elixir_and_cycle_analysis": Key estimates for player and opponent.
- "critical_decisions": List of important plays, whether they were optimal or mistakes, and why.
- "alternative_lines": Better alternative plays with reasoning.
- "coaching_takeaways": Actionable rules of thumb for improvement.
"""

class ReplayCoach:
    def __init__(self, client: Optional[OpenRouterClient] = None):
        self.client = client or OpenRouterClient()

    def analyze_replay_events(
        self,
        events: List[Dict[str, Any]],
        commentary: Optional[List[Dict[str, Any]]] = None,
        player_name: str = "Player",
        opponent_name: str = "Opponent",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a list of match events (JSON schema compliant) and return coaching feedback.
        """
        events_str = json.dumps(events, indent=2)
        comm_str = json.dumps(commentary or [], indent=2)

        prompt = f"""Analyze the following Clash Royale match segment:
Player: {player_name}
Opponent: {opponent_name}

=== MATCH EVENTS (Timeline) ===
{events_str}

=== EXPERT COMMENTARY SEGMENTS ===
{comm_str}

Perform a rigorous strategic analysis:
1. Calculate elixir balance swings across the plays.
2. Track card cycle advantages or mistakes (e.g. playing win condition into counter, or playing when counter is out of cycle).
3. Evaluate key decisions made by {player_name}.
4. Provide structured JSON coaching output.
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = self.client.generate_chat_completion(
            messages=messages,
            model=model,
            temperature=0.2
        )

        msg = response["choices"][0]["message"]
        content = msg.get("content")
        if not content and "reasoning" in msg:
            content = msg.get("reasoning")

        return {
            "model_used": response.get("model", model or self.client.model),
            "raw_output": content,
            "reasoning": msg.get("reasoning"),
            "usage": response.get("usage", {})
        }
