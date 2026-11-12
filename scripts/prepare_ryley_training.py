"""
Build Ryley CR Subtitle SFT Dataset & State-Action Policy Corpus.

Parses downloaded Ryley - Clash Royale YouTube English subtitles (VTT -> clean transcripts),
pairs timestamped coaching segments with Clash Royale 2.6 Hog Cycle strategic annotations
(positive elixir trades, out-cycling counters, musketeer counter-push, spell discipline),
and extracts state->action supervision examples from recorded gameplay logs using
our elixir-conserving TacticalActionPlanner.
"""

import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.strategy.action_planner import TacticalActionPlanner

RYLEY_VIDEOS = [
    {
        "video_id": "h16sj_76LSg",
        "title": "How to Play 2.6 Hog Cycle Like a PRO!!",
        "url": "https://www.youtube.com/watch?v=h16sj_76LSg",
        "channel": "Ryley - Clash Royale",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "h16sj_76LSg_transcript.txt",
    },
    {
        "video_id": "uyXYTIJgM4w",
        "title": "The RETURN of 2.6 Hog Cycle on Top Ladder!!",
        "url": "https://www.youtube.com/watch?v=uyXYTIJgM4w",
        "channel": "Ryley - Clash Royale",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "uyXYTIJgM4w_transcript.txt",
    },
    {
        "video_id": "SEBfDB81vZg",
        "title": "Can 2.0 Hog Cycle WIN on Top Ladder?",
        "url": "https://www.youtube.com/watch?v=SEBfDB81vZg",
        "channel": "Ryley - Clash Royale",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "SEBfDB81vZg_transcript.txt",
    },
    {
        "video_id": "oIWHwBc80EI",
        "title": "How to COUNT Elixir in Clash Royale (2024)",
        "url": "https://www.youtube.com/watch?v=oIWHwBc80EI",
        "channel": "Clash Royale Pro Guide",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "oIWHwBc80EI_transcript.txt",
    },
    {
        "video_id": "ytWqIi08SZs",
        "title": "How to COUNT Elixir in Clash Royale (2022)",
        "url": "https://www.youtube.com/watch?v=ytWqIi08SZs",
        "channel": "Clash Royale Pro Guide",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "ytWqIi08SZs_transcript.txt",
    },
    {
        "video_id": "ZF2rjDrZV1A",
        "title": "5 Tips For Better Elixir Management",
        "url": "https://www.youtube.com/watch?v=ZF2rjDrZV1A",
        "channel": "Clash Royale Pro Guide",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "ZF2rjDrZV1A_transcript.txt",
    },
    {
        "video_id": "ISSqA8mex3s",
        "title": "How to get SO good at Clash Royale you can't stop winning",
        "url": "https://www.youtube.com/watch?v=ISSqA8mex3s",
        "channel": "Clash Royale Pro Guide",
        "transcript_path": ROOT_DIR / "data" / "ryley_subs" / "ISSqA8mex3s_transcript.txt",
    },
]

TOPIC_RULES: List[Tuple[str, List[str], str]] = [
    (
        "positive_elixir_trade",
        ["positive trade", "positive elixir", "elixir trade", "spend a little bit less", "up elixir", "down elixir", "overcommit", "overspend", "worth"],
        "Prioritize low-cost defensive trades (Skeletons/Spirit/Cannon pull) instead of dumping equal or more elixir; bank surplus for a protected counter-push.",
    ),
    (
        "patience_and_elixir_hold",
        ["wait", "hold up", "hold", "patient", "in the back", "early", "first phase", "single elixir"],
        "Do not panic-play at the river on 1-3 elixir. Wait for troops to cross into tower/cannon range or build Musketeer/Knight in the back at >=8.2 elixir.",
    ),
    (
        "outcycle_hog_counters",
        ["outcycle", "out-cycle", "counters to", "hog counter", "tornado", "building", "wasted", "in hand"],
        "Track the opponent's Hog Rider counter and elixir before committing Hog Rider. Never send Hog blindly into a waiting defender at <=6 elixir.",
    ),
    (
        "counterpush_synergy",
        ["counter push", "counterpush", "musketeer", "musky", "second one", "hog rider here with", "with this"],
        "Use surviving defensive Musketeer or Knight to tank/support a Hog Rider push rather than solo-spamming cycle troops at the bridge.",
    ),
    (
        "spell_value_discipline",
        ["fireball", "log", "arrows", "clip", "tower damage", "tornado"],
        "Hold spells until clipping multiple units or finishing critical value; avoid wasting single-target spells at low elixir unless emergency.",
    ),
]


def _ts_to_seconds(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) != 3:
        return 0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))


def classify_segment(text: str) -> Tuple[str, str]:
    low = text.lower()
    for topic, kws, rule in TOPIC_RULES:
        if any(k in low for k in kws):
            return topic, rule
    return (
        "general_2_6_macro",
        "Maintain fast 2.6 defensive cycle discipline while preserving at least 3-4 elixir after committing a win condition.",
    )


def load_transcript_chunks(path: pathlib.Path, chunk_size: int = 8, stride: int = 6) -> List[Dict[str, Any]]:
    line_re = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s*(.+)$")
    parsed: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = line_re.match(raw.strip())
        if m:
            parsed.append((m.group(1), m.group(2).strip()))

    chunks: List[Dict[str, Any]] = []
    for idx in range(0, max(1, len(parsed) - chunk_size + 1), stride):
        window = parsed[idx : idx + chunk_size]
        if not window:
            continue
        start_ts = window[0][0]
        end_ts = window[-1][0]
        text = " ".join(txt for _, txt in window)
        topic, principle = classify_segment(text)
        chunks.append(
            {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "start_sec": _ts_to_seconds(start_ts),
                "text": text,
                "topic": topic,
                "principle": principle,
            }
        )
    return chunks


def build_subtitle_sft_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sft_records: List[Dict[str, Any]] = []
    video_summaries: List[Dict[str, Any]] = []

    system_msg = (
        "You are Clash IQ, an AI Clash Royale 2.6 Hog Cycle tactician trained on Ryley CR "
        "pro gameplay and live arena telemetry. Output disciplined elixir conservation decisions."
    )

    for v in RYLEY_VIDEOS:
        tpath: pathlib.Path = v["transcript_path"]
        if not tpath.exists():
            continue
        chunks = load_transcript_chunks(tpath)
        topic_counts: Dict[str, int] = {}
        for ch in chunks:
            topic_counts[ch["topic"]] = topic_counts.get(ch["topic"], 0) + 1
            # Keep strategy-rich chunks to focus dataset quality
            if ch["topic"] == "general_2_6_macro" and len(sft_records) % 3 != 0:
                continue
            target = {
                "source": "ryley_cr_subtitles",
                "video_id": v["video_id"],
                "timestamp": f"{ch['start_ts']}-{ch['end_ts']}",
                "topic": ch["topic"],
                "tactical_rule": ch["principle"],
            }
            user_prompt = (
                f"Ryley CR commentary [{v['title']} @ {ch['start_ts']}]:\n"
                f"\"{ch['text']}\"\n\n"
                "Extract the structured elixir management & Hog Cycle decision rule."
            )
            sft_records.append(
                {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                    ],
                    "meta": {
                        "video_id": v["video_id"],
                        "url_at_ts": f"{v['url']}&t={ch['start_sec']}s",
                        "topic": ch["topic"],
                    },
                }
            )
        video_summaries.append(
            {
                "video_id": v["video_id"],
                "title": v["title"],
                "url": v["url"],
                "chunks_parsed": len(chunks),
                "topic_counts": topic_counts,
            }
        )
    return sft_records, video_summaries


def _map_decision_class(dec: Dict[str, Any]) -> str:
    act = dec.get("action")
    card = str(dec.get("recommended_card") or "").lower()
    if act == "defend":
        if card in ("skeletons", "electro_spirit"):
            return "cheap_cycle_defend"
        if card in ("cannon", "tesla"):
            return "cannon_pull_defend"
        return "knight_or_musk_counter"
    if act in ("wait", "opposite_lane_punish", "slow_roll_back"):
        return str(act)
    return "wait"


def build_overspend_reasoning_sft() -> List[Dict[str, Any]]:
    """
    Non-imitation Tactical Reasoning SFT corpus: teaches explicit hand-vs-hand + elixir-math
    decisions on WHEN to overspend vs conserve, and WHEN to pick 1-3 elixir counters instead of Cannon.
    """
    sys_msg = (
        "You are CR_AI Tactical Brain. Never imitate blindly: compare OUR hand + elixir vs OPPONENT hand + elixir "
        "to decide whether overspending is profitable or whether to force a positive 1-3 elixir counter trade."
    )
    examples: List[Dict[str, Any]] = []
    scenarios = [
        (5.5, 8.0, ["knight", "skeletons", "cannon", "hog_rider"], ["hog_rider", "fireball"], "Lone enemy knight crossing row=19", False, 3, "skeletons", "Defend +2 positive trade with Skeletons (1) or Knight (3); opponent has 8.0 elx and Hog in hand so SAVE Cannon and DO NOT overspend."),
        (7.8, 3.0, ["hog_rider", "knight", "skeletons", "cannon"], ["musketeer", "ice_spirit"], "Opponent just dropped Golem/PEKKA or wasted 6 elx; Hog counter out of hand", True, 4, "hog_rider", "Elixir advantage is +4.8 and building counter is out of opp hand -> ALLOW OVERSPEND with Hog Rider opposite lane."),
        (6.0, 2.5, ["hog_rider", "arrows", "knight", "cannon"], ["skeleton_army", "bats"], "Opponent at 2.5 elx with Swarm in hand, right lane clear", True, 4, "hog_rider", "Elixir adv +3.5; opponent too broke (2.5) to full-defend clean Hog punish."),
        (6.2, 6.5, ["hog_rider", "cannon", "knight", "skeletons"], ["cannon", "valkyrie", "log"], "Board quiet, opponent holds Cannon + Log at 6.5 elx", False, 3, "none", "Elixir adv -0.3 and opponent holds Cannon in hand -> DO NOT overspend Hog into waiting Cannon; wait/bank to 8.4+ or defend cheap."),
        (4.2, 5.5, ["cannon", "skeletons", "electro_spirit", "musketeer"], ["giant", "zap"], "Lone enemy musketeer at row=19, opp holds Giant at 5.5 elx", False, 1, "skeletons", "Save Cannon for Giant! Counter lone Musketeer with 1-elx Skeletons/E-Spirit for a +3 positive trade."),
        (4.5, 4.0, ["cannon", "knight", "arrows", "hog_rider"], ["hog_rider", "fireball"], "Fast Hog Rider + Minions push crossing row=18", False, 4, "cannon", "Building-targeter / multi-unit rusher crossing row=18 -> react immediately with 4-3 Cannon center-pull `[8/9, 21]` (no late delay)."),
        (4.0, 6.0, ["knight", "arrows", "musketeer", "skeletons"], ["balloon", "freeze"], "Enemy Balloon crossing river at row=16", False, 4, "musketeer", "Air wincon Balloon incoming while opp holds 6 elx -> anti_air_needed=true; deploy safe Musketeer at row=23+."),
        (5.1, 4.5, ["fireball", "knight", "cannon", "skeletons"], ["musketeer", "knight"], "Just played Hog Rider 2s ago; 2 enemies at river row=17", False, 3, "none", "Post-Hog spell lock: DO NOT burn Fireball at 5.1 elx on river targets; let them walk to tower and keep 3+ elx to defend."),
    ]
    for my_e, opp_e, my_h, opp_h, sit, allow_ov, max_c, rec_c, why in scenarios:
        adv = round(my_e - opp_e, 1)
        user_txt = (
            f"State: {sit}\n"
            f"My Elixir={my_e}, Hand={my_h}\n"
            f"Opp Elixir Est={opp_e} (adv={adv:+.1f}), Opp Estimated Hand={opp_h}\n"
            "Decide overspend budget, counter choice, and reasoning."
        )
        ans = {
            "allow_overspend": allow_ov,
            "max_defense_cost": max_c,
            "recommended_card": rec_c,
            "elixir_advantage": adv,
            "reasoning": why,
        }
        examples.append({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_txt},
                {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)},
            ],
            "meta": {"topic": "overspend_and_counter_reasoning"},
        })
    return examples


def build_replay_policy_samples() -> List[Dict[str, Any]]:
    """
    Re-evaluates recorded live events (195349, 194003, 205455, 234457) alongside an extensive
    Hand-vs-Hand + Elixir-Advantage + Counter-Type curriculum (12 features, 6 classes).
    """
    planner = TacticalActionPlanner()
    samples: List[Dict[str, Any]] = []

    for sess in ["session_20261111_195349", "session_20261111_194003", "session_20261111_205455", "session_20261111_234457"]:
        ep = ROOT_DIR / "recordings" / sess / "events.jsonl"
        if not ep.exists():
            continue
        lines = [json.loads(ln) for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, row in enumerate(lines):
            if row.get("screen") != "battle_arena":
                continue
            elx = float(row.get("elixir") or 0.0)
            opp_e = float(row.get("opponent_elixir_est") or 5.0)
            thr_cnt = int(row.get("threats") or 0)
            threats = []
            if thr_cnt > 0:
                tile = row.get("target_tile") or [4, 19]
                t_row = int(tile[1]) if row.get("action") == "defend" else 18
                t_col = int(tile[0])
                threats.append({"col": t_col, "row": max(4, min(27, t_row - 1)), "card": "enemy_unit"})
                for k in range(1, min(thr_cnt, 4)):
                    threats.append({"col": (t_col + k) % 16, "row": 12 if k % 2 else 19, "card": "enemy_unit"})

            hand_slots = [
                {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": elx >= 4.0},
                {"slot": 2, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": elx >= 3.0},
                {"slot": 3, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": elx >= 3.0},
                {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": elx >= 1.0},
            ]
            if i % 3 == 0:
                hand_slots[2] = {"slot": 3, "card_name": "musketeer", "card_cost": 4, "card_role": "ranged", "playable": elx >= 4.0}
            if i % 4 == 0:
                hand_slots[3] = {"slot": 4, "card_name": "arrows", "card_cost": 3, "card_role": "spell", "playable": elx >= 3.0}

            opp_has_rusher = (i % 5 == 0)
            opp_has_swarm = (i % 4 == 0)
            opp_intel = {
                "opp_elixir": opp_e,
                "opp_has_rusher": opp_has_rusher,
                "opp_has_building_counter": (i % 6 == 0),
                "opp_hand_archetypes": ["rusher"] if opp_has_rusher else [],
                "prediction_unlocked": True,
            }
            dec = TacticalActionPlanner().decide_next_action(
                current_elixir=elx,
                opponent_estimated_elixir=opp_e,
                hand_cards=[s["card_name"] for s in hand_slots],
                arena_threats=threats,
                hand_slots=hand_slots,
                opp_intel=opp_intel,
                now_ts=float(i) * 0.25,
            )
            our_half = sum(1 for t in threats if t["row"] >= 16)
            max_row = max((t["row"] for t in threats if t["row"] >= 16), default=0)
            adv_norm = max(0.0, min(1.0, (elx - opp_e + 5.0) / 10.0))
            samples.append(
                {
                    "features": [
                        round(elx / 10.0, 3),
                        round(opp_e / 10.0, 3),
                        round(adv_norm, 3),
                        1.0 if opp_has_rusher else 0.0,
                        1.0 if opp_has_swarm else 0.0,
                        min(1.0, thr_cnt / 5.0),
                        min(1.0, our_half / 4.0),
                        round(max_row / 31.0, 3),
                        1.0 if opp_has_rusher and our_half >= 1 else 0.0,
                        1.0 if any(s["playable"] and s["card_role"] == "wincon" for s in hand_slots) else 0.0,
                        1.0 if any(s["playable"] and s["card_role"] == "cycle" for s in hand_slots) else 0.0,
                        1.0 if any(s["playable"] and s["card_role"] == "building" for s in hand_slots) else 0.0,
                    ],
                    "target_action": _map_decision_class(dec),
                    "elixir": elx,
                }
            )

    # Rich Hand-vs-Hand + Elixir-Advantage + Counter-Type Curriculum (covers positive trades & overspend decisions)
    # (elx, opp_elx, our_half, max_row, opp_half, opp_rusher, opp_bldg_cnt, is_rusher_or_air, wc, knight_or_musk, cycle, cannon)
    curriculum_grid = [
        (0.3, 4.0, 1, 20, 0, False, False, False, False, False, False, False),
        (1.6, 5.0, 1, 23, 0, False, False, False, False, False, True, False),
        (2.4, 5.0, 1, 18, 0, False, False, False, False, False, True, False),
        (3.3, 5.0, 1, 19, 0, True, False, False, False, True, True, True),   # Save Cannon for rusher -> Knight/Skeletons!
        (3.5, 5.0, 1, 20, 0, False, False, False, False, True, True, True),  # Cheap Knight counter instead of Cannon!
        (4.2, 5.0, 2, 19, 0, True, False, True, False, True, True, True),    # 2 threats / rusher -> Cannon center-pull!
        (4.5, 6.0, 1, 21, 0, False, False, True, False, True, True, True),   # Heavy/rusher -> Cannon center-pull!
        (5.9, 2.5, 0, 0, 1, False, False, False, True, True, True, True),    # +3.4 elx adv -> OVERSPEND Hog punish!
        (6.4, 6.8, 0, 0, 0, False, True, False, True, True, True, True),     # Opp has building counter & high elx -> WAIT!
        (7.5, 4.5, 0, 0, 0, False, False, False, True, True, True, True),    # Safe Hog punish!
        (8.6, 5.0, 0, 0, 0, False, False, False, False, True, True, True),   # Slow roll Knight/Musk in back!
    ]
    for rep in range(35):
        for step_idx, (elx, opp_e, oh, mrow, opph, opp_rush, opp_bldg, is_rh, wc, km, cy, cn) in enumerate(curriculum_grid):
            cname = "enemy_rusher" if is_rh else "enemy_unit"
            threats = [{"col": 4, "row": mrow, "card": cname}] * oh + [{"col": 3, "row": 11, "card": "enemy_unit"}] * opph
            slots = [
                {"slot": 1, "card_name": "hog_rider", "card_cost": 4, "card_role": "wincon", "playable": wc and elx >= 4.0},
                {"slot": 2, "card_name": "cannon", "card_cost": 3, "card_role": "building", "playable": cn and elx >= 3.0},
                {"slot": 3, "card_name": "knight", "card_cost": 3, "card_role": "minitank", "playable": km and elx >= 3.0},
                {"slot": 4, "card_name": "skeletons", "card_cost": 1, "card_role": "cycle", "playable": cy and elx >= 1.0},
            ]
            opp_intel = {
                "opp_elixir": opp_e,
                "opp_has_rusher": opp_rush,
                "opp_has_building_counter": opp_bldg,
                "recent_heavy_threat": is_rh,
                "opp_hand_archetypes": (["rusher"] if opp_rush else []) + (["cannon"] if opp_bldg else []),
                "prediction_unlocked": True,
            }
            dec = TacticalActionPlanner().decide_next_action(
                current_elixir=elx,
                opponent_estimated_elixir=opp_e,
                hand_cards=[s["card_name"] for s in slots],
                arena_threats=threats,
                hand_slots=slots,
                opp_intel=opp_intel,
                now_ts=100.0 + rep * 10.0 + step_idx,
            )
            adv_norm = max(0.0, min(1.0, (elx - opp_e + 5.0) / 10.0))
            samples.append(
                {
                    "features": [
                        round(elx / 10.0, 3),
                        round(opp_e / 10.0, 3),
                        round(adv_norm, 3),
                        1.0 if opp_rush else 0.0,
                        1.0 if opp_bldg else 0.0,
                        min(1.0, (oh + opph) / 5.0),
                        min(1.0, oh / 4.0),
                        round(mrow / 31.0, 3),
                        1.0 if is_rh else 0.0,
                        1.0 if (wc and elx >= 4.0) else 0.0,
                        1.0 if (cy and elx >= 1.0) else 0.0,
                        1.0 if (cn and elx >= 3.0) else 0.0,
                    ],
                    "target_action": _map_decision_class(dec),
                    "elixir": elx,
                }
            )
    return samples


def main() -> int:
    out_dir = ROOT_DIR / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_records, video_summaries = build_subtitle_sft_records()
    reasoning_sft = build_overspend_reasoning_sft()
    sft_records.extend(reasoning_sft)

    # Merge user's mobile phone SFT if available
    phone_sft_path = out_dir / "phone_human_sft.jsonl"
    phone_count = 0
    if phone_sft_path.exists():
        for ln in phone_sft_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                sft_records.append(json.loads(ln))
                phone_count += 1

    sft_path = out_dir / "ryley_finetune_train.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for rec in sft_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    policy_samples = build_replay_policy_samples()
    policy_path = out_dir / "ryley_policy_samples.jsonl"
    with open(policy_path, "w", encoding="utf-8") as f:
        for s in policy_samples:
            f.write(json.dumps(s) + "\n")

    manifest_path = out_dir / "ryley_sources_summary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "videos": video_summaries,
                "reasoning_sft_pairs": len(reasoning_sft),
                "phone_human_sft_pairs": phone_count,
                "sft_pairs": len(sft_records),
                "policy_samples": len(policy_samples),
                "sft_path": str(sft_path),
                "policy_samples_path": str(policy_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Built Expanded SFT JSONL: {sft_path} ({len(sft_records)} examples across {len(video_summaries)} videos + {len(reasoning_sft)} reasoning + {phone_count} phone)")
    print(f"Built Counter+Overspend Policy Samples: {policy_path} ({len(policy_samples)} 12-dim examples)")
    for vs in video_summaries:
        print(f"  - {vs['video_id']}: {vs['title']} | chunks={vs['chunks_parsed']} | topics={vs['topic_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
