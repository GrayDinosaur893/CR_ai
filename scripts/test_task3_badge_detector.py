"""
Task 3 Verification Gate: BadgeDetector precision & recall on Recording 100928
- Tests recall and false positives against hand-labeled battle keyframes (2s..141s)
- Tests zero false-positive enemies on pre-action idle frames (0s..12s) and red banner /
  'Not enough Elixir!' / countdown text frames (no text glyphs mistaken for enemy badges)
- Generates visual overlay montage data/benchmarks/task3_badge_overlay_final.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vision.component_detector import ComponentDetector

VIDEO_PATH = ROOT / "data" / "observer_logs" / "Recording 2026-11-11 100928.mp4"

# Ground truth normalized viewport centers (cx, cy) of visible enemy badges at exact keyframes
KEYFRAME_GT = {
    2: {"enemies": [], "allies": []},
    18: {"enemies": [(0.75, 0.25)], "allies": [(0.46, 0.49)]},
    35: {"enemies": [(0.77, 0.36), (0.76, 0.46)], "allies": []},
    46: {"enemies": [(0.68, 0.47), (0.72, 0.47), (0.77, 0.47)], "allies": []},
    62: {"enemies": [(0.75, 0.58)], "allies": [(0.71, 0.54)]},
    88: {"enemies": [], "allies": []},
    110: {"enemies": [(0.68, 0.49), (0.77, 0.51), (0.67, 0.53), (0.77, 0.54)], "allies": [(0.42, 0.55), (0.62, 0.51)]},
    130: {"enemies": [(0.75, 0.53)], "allies": []},
    141: {"enemies": [(0.69, 0.49), (0.76, 0.49), (0.73, 0.51), (0.68, 0.54), (0.73, 0.54), (0.61, 0.66)], "allies": []},
}


def match_points(dets, gts, tol=0.075):
    used = set()
    tp = 0
    for gx, gy in gts:
        best_i = None
        best_d = 1e9
        for i, d in enumerate(dets):
            if i in used:
                continue
            dx, dy = d["norm_vp"][0] - gx, d["norm_vp"][1] - gy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= tol and dist < best_d:
                best_d = dist
                best_i = i
        if best_i is not None:
            used.add(best_i)
            tp += 1
    fp = len(dets) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main():
    det = ComponentDetector()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    total_tp = total_gt = total_fp = 0
    ally_tp = ally_gt = ally_fp = 0
    tiles = []

    print("=== KEYFRAME EVALUATION ===")
    for sec, gt in KEYFRAME_GT.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * 30)
        ok, frame = cap.read()
        assert ok, f"Failed frame at {sec}s"
        vp = det.detect_game_viewport(frame)
        ents = det.detect_arena_entities(frame, vp)
        enemies = ents["enemies"]
        allies = ents["allies"]

        tp_e, fp_e, fn_e = match_points(enemies, gt["enemies"])
        tp_a, fp_a, fn_a = match_points(allies, gt["allies"])
        total_tp += tp_e
        total_gt += len(gt["enemies"])
        total_fp += fp_e
        ally_tp += tp_a
        ally_gt += len(gt["allies"])
        ally_fp += fp_a

        print(
            f"t={sec:3d}s | Enemies: det={len(enemies)} gt={len(gt['enemies'])} tp={tp_e} fp={fp_e} "
            f"| Allies: det={len(allies)} gt={len(gt['allies'])} tp={tp_a} fp={fp_a}"
        )

        # Render annotated thumbnail tile
        vx1, vy1, vx2, vy2 = vp["box_px"]
        vis = frame[vy1:vy2, vx1:vx2].copy()
        vh, vw = vis.shape[:2]
        for gx, gy in gt["enemies"]:
            cv2.circle(vis, (int(gx * vw), int(gy * vh)), 14, (0, 255, 255), 2)
        for e in enemies:
            cx, cy = int(e["norm_vp"][0] * vw), int(e["norm_vp"][1] * vh)
            cv2.rectangle(vis, (cx - 12, cy - 12), (cx + 12, cy + 12), (0, 0, 255), 2)
        for a in allies:
            cx, cy = int(a["norm_vp"][0] * vw), int(a["norm_vp"][1] * vh)
            cv2.rectangle(vis, (cx - 12, cy - 12), (cx + 12, cy + 12), (255, 180, 0), 2)
        cv2.putText(
            vis,
            f"t={sec}s E:{tp_e}/{len(gt['enemies'])} fp={fp_e}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        tiles.append(cv2.resize(vis, (305, 568)))

    # Save 3x3 visual overlay grid
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
    montage = np.vstack(rows)
    out_img = ROOT / "data" / "benchmarks" / "task3_badge_overlay_final.png"
    cv2.imwrite(str(out_img), montage)

    # Opening idle + text-only banner test (t=0s..12s, plus simulated 'Not enough Elixir!' red glyph frames)
    idle_fp = 0
    idle_frames = 0
    for sec in range(0, 12):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * 30)
        ok, frame = cap.read()
        if not ok:
            continue
        idle_frames += 1
        vp = det.detect_game_viewport(frame)
        ents = det.detect_arena_entities(frame, vp)
        idle_fp += len(ents["enemies"])

    # Synthetic/composite banner stress check: take idle frame t=0s and stamp red banner text glyphs
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, clean_frame = cap.read()
    vp = det.detect_game_viewport(clean_frame)
    vx1, vy1, vx2, vy2 = vp["box_px"]
    vw, vh = vx2 - vx1, vy2 - vy1
    banner_frame = clean_frame.copy()
    cv2.putText(
        banner_frame,
        "Not enough Elixir!",
        (vx1 + int(0.24 * vw), vy1 + int(0.60 * vh)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (25, 25, 245),
        3,
    )
    cv2.putText(
        banner_frame,
        "60 Seconds Left",
        (vx1 + int(0.28 * vw), vy1 + int(0.49 * vh)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (30, 30, 240),
        3,
    )
    banner_ents = det.detect_arena_entities(banner_frame, vp)
    banner_fp = len(banner_ents["enemies"])

    enemy_recall = total_tp / float(max(total_gt, 1))
    summary = {
        "enemy_tp": total_tp,
        "enemy_gt": total_gt,
        "enemy_fp_keyframes": total_fp,
        "enemy_recall_pct": round(enemy_recall * 100.0, 2),
        "ally_tp": ally_tp,
        "ally_gt": ally_gt,
        "ally_fp_keyframes": ally_fp,
        "idle_frames_checked": idle_frames,
        "idle_fp_total": idle_fp,
        "synthetic_banner_fp": banner_fp,
        "overlay_path": str(out_img),
        "gate_pass": bool(total_fp == 0 and idle_fp == 0 and banner_fp == 0 and enemy_recall >= 0.80),
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    out_json = ROOT / "data" / "benchmarks" / "task3_gate.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.exit(0 if summary["gate_pass"] else 1)


if __name__ == "__main__":
    main()
