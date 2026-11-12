"""
Analyze a user's Clash Royale mobile/phone gameplay clip (including 9:20 Replay HUD
videos like YouTube Short PCxoUkeC6JA.mp4), extract verified CV telemetry +
player/opponent elixir curves + card deployment events, and query Qwen3-VL
for tactical coaching & SFT dataset generation.
"""

import json
import os
import sys
import time
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vision.component_detector import ComponentDetector
from src.observer.vlm_client import QwenVLMClient


# Calibrated Replay HUD pixel geometry for 1080x2400 mobile Clash Royale replays
REPLAY_ELIXIR_X1 = 105
REPLAY_ELIXIR_X2 = 1035
BOT_ELIXIR_Y1, BOT_ELIXIR_Y2 = 2335, 2362
TOP_ELIXIR_Y1, TOP_ELIXIR_Y2 = 156, 176

# Bottom 4 hand card slots in 1080x2400 Replay HUD: [DartGob, Log, Tesla, Princess]
REPLAY_SLOTS_1080x2400 = [
    (1, 138, 2145, 250, 2295),
    (2, 263, 2145, 375, 2295),
    (3, 388, 2145, 500, 2295),
    (4, 513, 2145, 625, 2295),
]


def _scale_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * w / 1080.0)),
        int(round(y1 * h / 2400.0)),
        int(round(x2 * w / 1080.0)),
        int(round(y2 * h / 2400.0)),
    )


def read_replay_elixir(frame_bgr: np.ndarray, top: bool = False) -> Tuple[float, bool]:
    """Reads magenta elixir fill from mobile Replay HUD (bottom=player, top=opponent)."""
    h, w = frame_bgr.shape[:2]
    x1 = int(round(REPLAY_ELIXIR_X1 * w / 1080.0))
    x2 = int(round(REPLAY_ELIXIR_X2 * w / 1080.0))
    y1, y2 = (TOP_ELIXIR_Y1, TOP_ELIXIR_Y2) if top else (BOT_ELIXIR_Y1, BOT_ELIXIR_Y2)
    y1 = int(round(y1 * h / 2400.0))
    y2 = int(round(y2 * h / 2400.0))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0, False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mag = (
        (hsv[:, :, 0] >= 132)
        & (hsv[:, :, 0] <= 172)
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 100)
    )
    col_fill = mag.mean(axis=0) > 0.25
    fill_ratio = float(col_fill.mean())
    cols = np.where(col_fill)[0]
    right_ratio = float((cols.max() + 1) / max(1, col_fill.shape[0])) if len(cols) > 0 else 0.0
    # When track is clean, right edge matches mean fill closely; guard stray sparkles
    est = fill_ratio * 10.0
    if right_ratio - fill_ratio <= 0.12 and right_ratio > fill_ratio:
        est = (0.5 * fill_ratio + 0.5 * right_ratio) * 10.0
    return round(min(10.0, max(0.0, est)), 2), bool(fill_ratio > 0.02)


def is_replay_battle_frame(frame_bgr: np.ndarray) -> bool:
    """Fast check whether frame shows live mobile Replay battle HUD (excludes VS intro & post-match)."""
    h, w = frame_bgr.shape[:2]
    # 1. Right-side bottom royal blue banner under player name
    by1, by2 = int(0.895 * h), int(0.955 * h)
    bx1, bx2 = int(0.65 * w), int(0.96 * w)
    patch = frame_bgr[by1:by2, bx1:bx2].astype(np.int16)
    if patch.size == 0:
        return False
    b, g, r = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    blue_ratio = float(((b - r > 35) & (b > 90)).mean())
    if blue_ratio <= 0.35:
        return False

    # 2. Elixir track pixels must be EITHER filled magenta OR empty dark slate (excludes sky-blue VS intro & gold/orange banners)
    ey1, ey2 = int(round(BOT_ELIXIR_Y1 * h / 2400.0)), int(round(BOT_ELIXIR_Y2 * h / 2400.0))
    ex1, ex2 = int(round(REPLAY_ELIXIR_X1 * w / 1080.0)), int(round(REPLAY_ELIXIR_X2 * w / 1080.0))
    track = frame_bgr[ey1:ey2, ex1:ex2]
    if track.size == 0:
        return False
    b_t = track[:, :, 0].astype(np.int16)
    g_t = track[:, :, 1].astype(np.int16)
    r_t = track[:, :, 2].astype(np.int16)
    mag_px = (r_t > 115) & (b_t > 115) & (r_t - g_t > 40)
    dark_px = np.maximum(np.maximum(b_t, g_t), r_t) < 115
    return float((mag_px | dark_px).mean()) > 0.60


def extract_arena_entities_clean(det: ComponentDetector, frame_bgr: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
    """Extracts enemy/ally presence strictly inside 9:20 mobile arena (half-res for 4x speed), separating our half below river."""
    small = cv2.resize(frame_bgr, (540, 1200), interpolation=cv2.INTER_AREA)
    res = det.detect_components(small)
    raw = res.get("arena_entities", {"enemies": [], "allies": []})
    enemies = [e for e in raw.get("enemies", []) if 0.22 <= e.get("norm_vp", [0.5, 0.5])[1] <= 0.87]
    allies = [a for a in raw.get("allies", []) if 0.22 <= a.get("norm_vp", [0.5, 0.5])[1] <= 0.87]
    for e in enemies:
        ny = float(e.get("norm_vp", [0.5, 0.5])[1])
        e["in_our_half"] = bool(ny >= 0.54)
    return {"enemies": enemies, "allies": allies}


def make_storyboard(frames_with_meta: List[Tuple[np.ndarray, str]], out_path: Path, cols: int = 3) -> None:
    """Builds a clean annotated horizontal/grid storyboard image for Qwen3-VL."""
    if not frames_with_meta:
        return
    tile_w, tile_h = 360, 800  # Preserves 9:20 mobile aspect ratio cleanly
    tiles: List[np.ndarray] = []
    for frame, label in frames_with_meta:
        resized = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (tile_w, 34), (15, 15, 25), -1)
        cv2.putText(resized, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 220), 2)
        tiles.append(resized)
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
    grid = np.vstack(grid_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def ask_qwen_analyze_phone_clip(
    storyboards: List[Path],
    summary_stats: Dict[str, Any],
    play_log: List[Dict[str, Any]],
) -> Tuple[bool, str, Dict[str, Any]]:
    """Sends annotated phone storyboards + telemetry to Qwen3-VL for deep tactical breakdown + SFT pairs."""
    client = QwenVLMClient()
    image_parts = []
    for sb in storyboards:
        b64 = base64.b64encode(sb.read_bytes()).decode("utf-8")
        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    plays_text = "\n".join(
        f"- t={p['t']:.1f}s: player_elx={p['elixir_before']:.1f}->{p['elixir_after']:.1f} "
        f"(spent~{p['est_cost']:.1f}), opp_elx={p['opp_elixir']:.1f}, adv={p['elixir_adv']:+.1f}, "
        f"enemies={p['enemies']} (our_half={p['enemies_our_half']}), mode={p['intent']}"
        for p in play_log[:25]
    )

    prompt = (
        "You are coaching CR_AI by analyzing the USER'S OWN mobile phone Clash Royale gameplay clip "
        "(YouTube Short PCxoUkeC6JA — 2.8 Underleveled Logbait Replay view showing Opponent top HUD and Player 'PapaNIK' bottom HUD).\n\n"
        f"MEASURED CV TELEMETRY:\n{json.dumps(summary_stats, indent=2)}\n\n"
        f"DETECTED CARD PLAYS:\n{plays_text}\n\n"
        "Look closely at BOTH chronological storyboards (first half + second half of the match).\n"
        "Return ONLY valid JSON (keep every list to 3-4 short bullet strings max, do NOT repeat items):\n"
        "{\n"
        '  "player_archetype": "short deck summary",\n'
        '  "phone_strengths": ["3-4 concise strengths observed in placement, baiting, spacing, elixir trades"],\n'
        '  "mistakes_or_leaks": ["1-2 risky moments or leaks if visible"],\n'
        '  "elixir_discipline_assessment": "2 sentences analyzing patience, positive elixir trades, and counter-push discipline",\n'
        '  "rules_to_teach_ai": ["4 concrete tactical rules CR_AI should copy from this phone gameplay"],\n'
        '  "sft_examples": [\n'
        '    {"elixir": 7.0, "situation": "concise arena state seen in storyboard", "best_action": "card + placement + why"},\n'
        '    {"elixir": 5.0, "situation": "concise arena state seen in storyboard", "best_action": "card + placement + why"}\n'
        '  ]\n'
        "}"
    )

    messages = [
        {"role": "system", "content": "You are an elite Clash Royale esports analyst. Output strict, concise JSON with no repeated items."},
        {"role": "user", "content": image_parts + [{"type": "text", "text": prompt}]},
    ]
    return client._post_vlm_messages(messages, max_tokens=900, temperature=0.25)


def _extract_json_from_raw(raw: str) -> Dict[str, Any]:
    """Parses JSON object from raw LLM response (handling markdown code fences)."""
    txt = (raw or "").strip()
    if not txt:
        return {}
    if "```" in txt:
        parts = txt.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") and p.endswith("}"):
                txt = p
                break
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start : end + 1])
        except Exception:
            pass
    try:
        return json.loads(txt)
    except Exception:
        return {}


def analyze_clip(
    video_path: Path,
    out_session_dir: Path,
    sample_interval_sec: float = 0.20,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_interval_sec)))

    det = ComponentDetector()
    out_session_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_session_dir / "events.jsonl"
    frames_dir = ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, Any]] = []
    play_log: List[Dict[str, Any]] = []
    prev_slot_grays: Dict[int, np.ndarray] = {}
    elixir_hist: List[Tuple[float, float]] = []  # (t, player_elx)
    last_play_t = -99.0

    sampled_keyframes: List[Tuple[np.ndarray, str, float]] = []

    frame_idx = 0
    t_proc0 = time.perf_counter()
    with open(events_path, "w", encoding="utf-8") as f_ev:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            t_sec = round(frame_idx / fps, 3)
            t0 = time.perf_counter()
            battle = is_replay_battle_frame(frame)
            h, w = frame.shape[:2]

            row: Dict[str, Any] = {
                "t": t_sec,
                "mode": "human_phone",
                "screen": "battle_arena" if battle else "non_game",
                "elixir": 0.0,
                "opp_elixir": 0.0,
                "enemies": 0,
                "enemies_our_half": 0,
                "allies": 0,
                "human_play": None,
                "latency_ms": 0.0,
            }

            if battle:
                elx, _ = read_replay_elixir(frame, top=False)
                opp_elx, _ = read_replay_elixir(frame, top=True)
                ents = extract_arena_entities_clean(det, frame)
                n_en = len(ents["enemies"])
                n_our = sum(1 for e in ents["enemies"] if e.get("in_our_half"))
                n_al = len(ents["allies"])

                row["elixir"] = elx
                row["opp_elixir"] = opp_elx
                row["enemies"] = n_en
                row["enemies_our_half"] = n_our
                row["allies"] = n_al

                # Track slot visual changes to pinpoint exact card slot used on Replay HUD
                changed_slots: List[Tuple[int, float]] = []
                cur_slots: Dict[int, np.ndarray] = {}
                for slot_id, x1, y1, x2, y2 in REPLAY_SLOTS_1080x2400:
                    sx1, sy1, sx2, sy2 = _scale_box((x1, y1, x2, y2), w, h)
                    crop = frame[sy1:sy2, sx1:sx2]
                    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 56))
                    cur_slots[slot_id] = g
                    if slot_id in prev_slot_grays:
                        diff = float(np.abs(g.astype(np.float32) - prev_slot_grays[slot_id].astype(np.float32)).mean())
                        if diff >= 16.0:
                            changed_slots.append((slot_id, diff))
                prev_slot_grays = cur_slots

                # Detect play via elixir drop over recent ~0.4s window OR clear slot swap + >=1.0 drop
                elixir_hist.append((t_sec, elx))
                recent = [e for ts, e in elixir_hist if t_sec - 0.6 <= ts < t_sec]
                peak_recent = max(recent) if recent else elx
                drop = peak_recent - elx
                if (t_sec - last_play_t) >= 0.85 and (drop >= 1.45 or (drop >= 0.95 and changed_slots)):
                    best_slot = max(changed_slots, key=lambda kv: kv[1])[0] if changed_slots else 0
                    intent = "defend" if n_our > 0 else "attack"
                    hp = {
                        "slot": best_slot,
                        "intent": intent,
                        "elixir": round(peak_recent, 2),
                        "elixir_after": round(elx, 2),
                        "est_cost": round(drop, 2),
                        "opp_elixir": round(opp_elx, 2),
                    }
                    row["human_play"] = hp
                    last_play_t = t_sec
                    play_log.append({
                        "t": t_sec,
                        "slot": best_slot,
                        "elixir_before": round(peak_recent, 2),
                        "elixir_after": round(elx, 2),
                        "est_cost": round(drop, 2),
                        "opp_elixir": round(opp_elx, 2),
                        "elixir_adv": round(peak_recent - opp_elx, 2),
                        "enemies": n_en,
                        "enemies_our_half": n_our,
                        "intent": intent,
                    })

                if not sampled_keyframes or (t_sec - sampled_keyframes[-1][2]) >= 6.5:
                    label = f"t={t_sec:.0f}s You:{elx:.1f} Opp:{opp_elx:.1f}"
                    sampled_keyframes.append((frame.copy(), label, t_sec))

            row["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            f_ev.write(json.dumps(row) + "\n")
            events.append(row)
            frame_idx += 1

    cap.release()
    elapsed = round(time.perf_counter() - t_proc0, 2)

    battle_rows = [r for r in events if r["screen"] == "battle_arena"]
    avg_elx = float(np.mean([r["elixir"] for r in battle_rows])) if battle_rows else 0.0
    avg_opp_elx = float(np.mean([r["opp_elixir"] for r in battle_rows])) if battle_rows else 0.0
    play_elxs = [p["elixir_before"] for p in play_log]
    avg_play_elx = float(np.mean(play_elxs)) if play_elxs else 0.0
    offense_cnt = sum(1 for p in play_log if p["intent"] == "attack")
    offense_pct = (100.0 * offense_cnt / len(play_log)) if play_log else 0.0
    avg_adv = float(np.mean([p["elixir_adv"] for p in play_log])) if play_log else 0.0

    # Build 2 deduplicated storyboards (up to 6 unique frames each)
    sb_paths: List[Path] = []
    if len(sampled_keyframes) >= 12:
        chosen = [sampled_keyframes[i] for i in np.linspace(0, len(sampled_keyframes) - 1, 12, dtype=int)]
        p1 = frames_dir / "pcx_storyboard_1.jpg"
        p2 = frames_dir / "pcx_storyboard_2.jpg"
        make_storyboard([(k[0], k[1]) for k in chosen[:6]], p1, cols=3)
        make_storyboard([(k[0], k[1]) for k in chosen[6:12]], p2, cols=3)
        sb_paths = [p1, p2]
    elif len(sampled_keyframes) >= 6:
        mid = len(sampled_keyframes) // 2
        p1 = frames_dir / "pcx_storyboard_1.jpg"
        p2 = frames_dir / "pcx_storyboard_2.jpg"
        make_storyboard([(k[0], k[1]) for k in sampled_keyframes[:mid]], p1, cols=3)
        make_storyboard([(k[0], k[1]) for k in sampled_keyframes[mid:]], p2, cols=3)
        sb_paths = [p1, p2]
    elif sampled_keyframes:
        p1 = frames_dir / "pcx_storyboard_1.jpg"
        make_storyboard([(k[0], k[1]) for k in sampled_keyframes[:6]], p1, cols=3)
        sb_paths = [p1]

    summary_stats = {
        "clip": video_path.name,
        "duration_sec": round(total_frames / fps, 1),
        "battle_frames": len(battle_rows),
        "detected_plays": len(play_log),
        "avg_battle_elixir": round(avg_elx, 2),
        "avg_opp_elixir": round(avg_opp_elx, 2),
        "avg_elixir_advantage_overall": round(avg_elx - avg_opp_elx, 2),
        "avg_play_elixir": round(avg_play_elx, 2),
        "avg_elixir_adv_on_play": round(avg_adv, 2),
        "offense_ratio_pct": round(offense_pct, 1),
        "processing_sec": elapsed,
    }

    qwen_ok = False
    qwen_parsed: Dict[str, Any] = {}
    qwen_raw = ""
    qwen_meta: Dict[str, Any] = {}
    if sb_paths:
        print(f"Sending {len(sb_paths)} storyboards to Qwen3-VL...", flush=True)
        qwen_ok, qwen_raw, qwen_meta = ask_qwen_analyze_phone_clip(sb_paths, summary_stats, play_log)
        qwen_parsed = _extract_json_from_raw(qwen_raw)
        print(f"Qwen ok={qwen_ok} parsed={bool(qwen_parsed)} latency_ms={round(qwen_meta.get('latency_ms', 0.0), 1)}", flush=True)

    result = {
        "summary": summary_stats,
        "plays": play_log,
        "storyboards": [str(p) for p in sb_paths],
        "qwen_ok": qwen_ok,
        "qwen_analysis": qwen_parsed,
        "qwen_raw": qwen_raw if not qwen_parsed else "",
    }
    (out_session_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Save Qwen-extracted phone SFT examples for Ryley+Human policy fine-tuning
    sft_items = qwen_parsed.get("sft_examples", []) if isinstance(qwen_parsed, dict) else []
    if sft_items:
        sft_out = ROOT / "data" / "derived" / "phone_human_sft.jsonl"
        sft_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sft_out, "w", encoding="utf-8") as f_sft:
            for item in sft_items:
                rec = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an elite Clash Royale coach trained on Ryley CR + strong human mobile gameplay.",
                        },
                        {
                            "role": "user",
                            "content": f"Elixir={item.get('elixir', '?')}. State: {item.get('situation', '')}",
                        },
                        {
                            "role": "assistant",
                            "content": str(item.get("best_action", "")),
                        },
                    ],
                    "source": "human_phone_PCxoUkeC6JA",
                }
                f_sft.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return result


def main() -> None:
    vid = ROOT / "data" / "human_clips" / "PCxoUkeC6JA.mp4"
    out = ROOT / "recordings" / "human_phone_PCxoUkeC6JA"
    res = analyze_clip(vid, out)
    print(json.dumps(res["summary"], indent=2))
    if res["qwen_analysis"]:
        print("\n=== QWEN3-VL PHONE GAMEPLAY ANALYSIS ===")
        print(json.dumps(res["qwen_analysis"], indent=2, ensure_ascii=False))
    elif res["qwen_raw"]:
        print("\n=== QWEN RAW ===")
        print(res["qwen_raw"][:1200])


if __name__ == "__main__":
    main()
