"""
Correlate Bot defense & pacing directly against Human and Ryley gameplay logs.
"""
import argparse
import collections
import glob
import json
import os
from typing import Any, Dict, List


def _extract_plays(events_path: str, is_bot: bool) -> Dict[str, Any]:
    with open(events_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if is_bot:
        plays = [r for r in rows if "LIVE_EXECUTED" in str(r.get("exec_status", ""))]
    else:
        plays = [r for r in rows if r.get("human_play")]

    cards = collections.Counter()
    actions = collections.Counter()
    elx_vals: List[float] = []
    rapid_bursts = 0
    zero_threat_dumps = 0
    repeat_building_lt_12s = 0
    prev_t = -999.0
    prev_building_t = -999.0

    for r in plays:
        t = float(r.get("t", 0.0))
        if is_bot:
            card = r.get("recommended_card") or "unknown"
            act = r.get("action") or "unknown"
            elx = r.get("elixir")
            thr = int(r.get("threats") or 0)
        else:
            hp = r["human_play"]
            card = hp.get("card") or hp.get("intent") or "unknown"
            act = r.get("action") or hp.get("intent") or "play"
            elx = hp.get("elixir_at_play", hp.get("elixir", r.get("elixir")))
            thr = hp.get("our_half_threats")
            if thr is None:
                thr = int(r.get("threats") or 0)

        cards[str(card)] += 1
        actions[str(act)] += 1
        if elx is not None:
            elx_vals.append(float(elx))
        if (t - prev_t) < 2.4:
            rapid_bursts += 1
        if thr == 0 and str(act) in ("slow_roll_back", "defend", "wait"):
            zero_threat_dumps += 1
        if str(card) in ("cannon", "tesla", "inferno_tower", "tombstone"):
            if (t - prev_building_t) < 12.0:
                repeat_building_lt_12s += 1
            prev_building_t = t
        prev_t = t

    return {
        "session": os.path.basename(os.path.dirname(events_path)),
        "plays_total": len(plays),
        "avg_elixir_at_play": round(sum(elx_vals) / max(1, len(elx_vals)), 2),
        "rapid_burst_lt_2_4s": rapid_bursts,
        "zero_threat_slow_roll_dumps": zero_threat_dumps,
        "repeat_building_lt_12s": repeat_building_lt_12s,
        "top_cards": dict(cards.most_common(8)),
        "actions": dict(actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Human vs Ryley vs Bot correlation audit.")
    parser.add_argument("--bot-session", type=str, default="", help="Path to bot session dir (defaults to latest session_*)")
    args = parser.parse_args()

    bot_dir = args.bot_session
    if not bot_dir:
        sessions = sorted(glob.glob("recordings/session_*"))
        if not sessions:
            raise FileNotFoundError("No recordings/session_* directory found.")
        bot_dir = sessions[-1]

    report = {
        "human": _extract_plays("recordings/human_20261111_203505/events.jsonl", is_bot=False),
        "ryley": _extract_plays("recordings/human_phone_PCxoUkeC6JA/events.jsonl", is_bot=False),
        "bot": _extract_plays(os.path.join(bot_dir, "events.jsonl"), is_bot=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
