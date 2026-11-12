"""
Script to prepare and format Clash Royale match annotations & commentary into
fine-tuning training datasets (JSONL format for Hugging Face SFT / LoRA / Unsloth).
"""

import json
import os
from typing import List, Dict, Any

SAMPLE_MATCH_EVENTS = [
    {
        "event_id": "evt_000001",
        "start_seconds": 12.0,
        "end_seconds": 12.5,
        "event_type": "card_deployed",
        "player": "opponent",
        "card_id": "golem",
        "lane": "left",
        "elixir_cost": 8,
        "confidence": 0.98
    },
    {
        "event_id": "evt_000002",
        "start_seconds": 13.0,
        "end_seconds": 13.5,
        "event_type": "card_deployed",
        "player": "player",
        "card_id": "hog_rider",
        "lane": "right",
        "elixir_cost": 4,
        "confidence": 0.99
    },
    {
        "event_id": "evt_000003",
        "start_seconds": 13.8,
        "end_seconds": 14.2,
        "event_type": "card_deployed",
        "player": "player",
        "card_id": "goblins",
        "lane": "right",
        "elixir_cost": 2,
        "confidence": 0.95
    }
]

SAMPLE_COACHING_GROUND_TRUTH = {
    "match_summary": "Opponent invested heavy 8 elixir Golem in the back left lane at normal elixir time.",
    "elixir_and_cycle_analysis": "Opponent spent 8 elixir and dropped to near 0 elixir. Player had 9 elixir and immediately punished opposite lane.",
    "critical_decisions": [
        {
            "timestamp": 13.0,
            "decision": "Opposite lane punish with Hog Rider + Goblins (6 elixir total)",
            "evaluation": "Excellent",
            "reasoning": "When an opponent drops an 8-elixir Golem at normal elixir, they cannot adequately defend a fast opposite lane push without taking massive tower damage."
        }
    ],
    "alternative_lines": [
        "Defending the Golem lane early would allow the opponent to build an unstoppable support push behind Golem."
    ],
    "coaching_takeaways": [
        "Always apply opposite lane pressure when opponent makes a heavy elixir investment in single elixir."
    ]
}

def create_training_pair(events: List[Dict[str, Any]], target_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates an instruction-tuning (chat format) sample for fine-tuning LLMs.
    """
    system_message = (
        "You are CLASH IQ, an expert Clash Royale replay analysis and coaching model. "
        "Analyze match events, compute elixir advantage, and evaluate strategic decisions."
    )
    user_prompt = (
        "Given the following timeline of match events, provide a strategic evaluation and coaching breakdown:\n"
        f"{json.dumps(events, indent=2)}"
    )
    assistant_response = json.dumps(target_analysis, indent=2)

    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def main():
    output_dir = os.path.join("data", "derived")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "finetune_train_sample.jsonl")

    sample_sample = create_training_pair(SAMPLE_MATCH_EVENTS, SAMPLE_COACHING_GROUND_TRUTH)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_sample) + "\n")

    print(f"[OK] Generated sample fine-tuning dataset at: {output_file}")
    print("[INFO] Add more annotated matches to this file to train/fine-tune a custom model using LoRA / QLoRA.")

if __name__ == "__main__":
    main()
