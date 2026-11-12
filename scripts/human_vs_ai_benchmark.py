"""
Human Threshold vs CR_AI vs Ryley CR Pro Progression & Graph Generator.

Scans all recorded matches in `recordings/session_*` (AI) and `recordings/human_*` (Human),
plus baseline `data/observer_logs/analysis_100928/video_metrics.json`, computes comparable
elixir discipline, reaction speed, offense/defense balance, and Ryley-aligned quality metrics,
saves `data/benchmarks/benchmark_history.json`, and renders PNG progression graphs in
`data/benchmarks/`.
"""

import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "data" / "benchmarks"


def _percentile(vals: List[float], pct: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = min(len(arr) - 1, max(0, int(round((pct / 100.0) * (len(arr) - 1)))))
    return float(arr[idx])


def _compute_score(
    avg_battle_elx: float,
    avg_play_elx: float,
    offense_ratio_pct: float,
    error_rate_pct: float,
    avg_rxn_ms: float,
) -> float:
    """
    0-100 Ryley-aligned Gameplay Quality Score:
      - 35 pts: Elixir discipline (avg battle elixir in 4.0-6.5 sweet spot, play elixir >= 4.2)
      - 25 pts: Zero bad/error plays (no <1.0 clicks, no unknown slot_X, no defensive wincon)
      - 25 pts: Balanced offense pressure (20%-40% of plays proactive Hog/back-roll vs passive panic)
      - 15 pts: Fast reaction loop latency (<= 40ms -> full marks)
    """
    elx_score = max(0.0, 20.0 - abs(avg_battle_elx - 5.2) * 5.0) + max(0.0, min(15.0, (avg_play_elx - 2.0) * 5.0))
    err_score = max(0.0, 25.0 - error_rate_pct * 1.25)
    if offense_ratio_pct <= 0.0:
        off_score = 0.0
    else:
        off_score = max(5.0, 25.0 - abs(offense_ratio_pct - 28.0) * 0.7)
    rxn_score = max(0.0, min(15.0, 15.0 - max(0.0, avg_rxn_ms - 30.0) * 0.15))
    return round(min(100.0, elx_score + err_score + off_score + rxn_score), 1)


def summarize_session(session_dir: pathlib.Path, kind: str) -> Optional[Dict[str, Any]]:
    ep = session_dir / "events.jsonl"
    if not ep.exists() or ep.stat().st_size == 0:
        return None
    rows = [json.loads(ln) for ln in ep.read_text(encoding="utf-8").splitlines() if ln.strip()]
    battle = [r for r in rows if r.get("screen") == "battle_arena"]
    if len(battle) < 10:
        return None

    battle_elx = [float(r.get("elixir") or 0.0) for r in battle]
    avg_battle_elx = sum(battle_elx) / len(battle_elx)
    latencies = [float(r["latency_ms"]) for r in battle if r.get("latency_ms") is not None]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    p95_lat = _percentile(latencies, 95.0)
    duration_s = max(1.0, float(battle[-1].get("t", 0.0)) - float(battle[0].get("t", 0.0)))
    loop_fps = round(len(battle) / duration_s, 2)

    if kind == "human":
        plays = [r["human_play"] for r in battle if r.get("human_play")]
        play_elx = [
            float(p.get("elixir_at_play") if p.get("elixir_at_play") is not None else (p.get("elixir") or 0.0))
            for p in plays
        ]
        offense_plays = sum(
            1 for p in plays if str(p.get("card") or "") in ("hog_rider",) or p.get("intent") == "attack"
        )
        errors = sum(1 for val in play_elx if val < 0.9)
    else:
        plays = [
            r for r in battle
            if r.get("executed") and r.get("action") not in ("click_lobby_battle", "click_post_match_ok")
        ]
        play_elx = [float(r.get("elixir") or 0.0) for r in plays]
        offense_plays = sum(
            1 for r in plays if r.get("action") in ("opposite_lane_punish", "slow_roll_back", "cycle_for_anti_air")
        )
        errors = 0
        for r in plays:
            cname = str(r.get("recommended_card") or "")
            reason = str(r.get("reasoning") or "")
            if float(r.get("elixir") or 0.0) < 1.0:
                errors += 1
            elif cname.startswith("slot_") or "using slot_" in reason or "with slot_" in reason:
                errors += 1
            elif r.get("action") == "defend" and ("using hog_rider" in reason or cname == "hog_rider"):
                errors += 1

    total_plays = len(plays)
    avg_play_elx = (sum(play_elx) / len(play_elx)) if play_elx else avg_battle_elx
    offense_ratio = (100.0 * offense_plays / total_plays) if total_plays > 0 else 0.0
    error_rate = (100.0 * errors / total_plays) if total_plays > 0 else 0.0
    leak_pct = 100.0 * sum(1 for e in battle_elx if e >= 9.8) / len(battle_elx)

    return {
        "label": session_dir.name,
        "kind": kind,
        "battle_frames": len(battle),
        "duration_s": round(duration_s, 1),
        "loop_fps": loop_fps,
        "avg_battle_elixir": round(avg_battle_elx, 2),
        "avg_play_elixir": round(avg_play_elx, 2),
        "elixir_leak_pct": round(leak_pct, 1),
        "total_plays": total_plays,
        "offense_ratio_pct": round(offense_ratio, 1),
        "error_rate_pct": round(error_rate, 1),
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "ryley_score": _compute_score(avg_battle_elx, avg_play_elx, offense_ratio, error_rate, avg_lat),
        "elixir_trace": [
            (round(float(r.get("t", 0.0)) - float(battle[0].get("t", 0.0)), 1), round(float(r.get("elixir") or 0.0), 1))
            for r in battle[:: max(1, len(battle) // 120)]
        ],
    }


def load_v1_observer_baseline() -> Optional[Dict[str, Any]]:
    vm = ROOT_DIR / "data" / "observer_logs" / "analysis_100928" / "video_metrics.json"
    if not vm.exists():
        return None
    data = json.loads(vm.read_text(encoding="utf-8"))
    summ = data.get("summary", {})
    avg_elx = float(summ.get("avg_elixir", 2.68))
    avg_lat = float(summ.get("avg_latency_ms", 77.74))
    return {
        "label": "AI_v1_100928",
        "kind": "ai",
        "battle_frames": int(summ.get("frames_sampled", 718)),
        "duration_s": 143.5,
        "loop_fps": 5.0,
        "avg_battle_elixir": round(avg_elx, 2),
        "avg_play_elixir": round(avg_elx, 2),
        "elixir_leak_pct": 0.0,
        "total_plays": 40,
        "offense_ratio_pct": 0.0,
        "error_rate_pct": 35.0,  # False enemy panics & zero Slot 4 / zero offense in baseline
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(avg_lat * 1.4, 1),
        "ryley_score": _compute_score(avg_elx, avg_elx, 0.0, 35.0, avg_lat),
        "elixir_trace": [],
    }


def generate_all_benchmarks() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_root = ROOT_DIR / "recordings"

    ai_runs: List[Dict[str, Any]] = []
    human_runs: List[Dict[str, Any]] = []

    v1 = load_v1_observer_baseline()
    if v1:
        ai_runs.append(v1)

    if rec_root.exists():
        for d in sorted(rec_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("session_"):
                s = summarize_session(d, kind="ai")
                if s and s["total_plays"] > 0:
                    ai_runs.append(s)
            elif d.name.startswith("human_"):
                s = summarize_session(d, kind="human")
                if s:
                    human_runs.append(s)

    ryley_target = {
        "label": "Ryley_Pro_Vision",
        "kind": "target",
        "avg_battle_elixir": 5.2,
        "avg_play_elixir": 5.5,
        "offense_ratio_pct": 28.0,
        "error_rate_pct": 0.0,
        "avg_latency_ms": 25.0,
        "ryley_score": 100.0,
    }

    human_threshold = max(human_runs, key=lambda r: r["ryley_score"]) if human_runs else None

    def _human_short_name(label: str) -> str:
        if "phone" in label:
            return "Phone Baseline"
        return "PC Baseline"

    # 1. Plot Progression Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("#0f172a")
    labels = [r["label"].replace("session_20261111_", "AI_") for r in ai_runs] + ["Ryley Vision"]
    x = np.arange(len(labels))

    def _style_ax(ax, title, ylabel):
        ax.set_facecolor("#1e293b")
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        ax.grid(alpha=0.2, color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Subplot 1: Elixir Discipline
    ax1 = axes[0, 0]
    _style_ax(ax1, "Elixir Conservation (Avg Battle & Play Elixir)", "Elixir (0-10)")
    b_elx = [r["avg_battle_elixir"] for r in ai_runs] + [ryley_target["avg_battle_elixir"]]
    p_elx = [r["avg_play_elixir"] for r in ai_runs] + [ryley_target["avg_play_elixir"]]
    w = 0.35
    ax1.bar(x - w / 2, b_elx, width=w, color="#38bdf8", label="Battle Avg Elixir")
    ax1.bar(x + w / 2, p_elx, width=w, color="#a855f7", label="Elixir at Card Play")
    for idx_h, hr in enumerate(human_runs):
        ls = "--" if hr is human_threshold else ":"
        clr = "#22c55e" if hr is human_threshold else "#86efac"
        ax1.axhline(hr["avg_battle_elixir"], color=clr, linestyle=ls, linewidth=1.7, label=f"{_human_short_name(hr['label'])} ({hr['avg_battle_elixir']:.2f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 8.5)
    ax1.legend(fontsize=7.5, facecolor="#0f172a", labelcolor="#f8fafc")

    # Subplot 2: Error Rate & Offense Balance
    ax2 = axes[0, 1]
    _style_ax(ax2, "Mistakes (%) vs Proactive Offense (%)", "Percent (%)")
    err_vals = [r["error_rate_pct"] for r in ai_runs] + [0.0]
    off_vals = [r["offense_ratio_pct"] for r in ai_runs] + [ryley_target["offense_ratio_pct"]]
    ax2.bar(x - w / 2, err_vals, width=w, color="#ef4444", label="Bad Plays / Error %")
    ax2.bar(x + w / 2, off_vals, width=w, color="#f59e0b", label="Offense Pressure %")
    for hr in human_runs:
        ls = "--" if hr is human_threshold else ":"
        clr = "#22c55e" if hr is human_threshold else "#86efac"
        ax2.axhline(hr["offense_ratio_pct"], color=clr, linestyle=ls, linewidth=1.5, label=f"{_human_short_name(hr['label'])} Off ({hr['offense_ratio_pct']:.1f}%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(fontsize=7.5, facecolor="#0f172a", labelcolor="#f8fafc")

    # Subplot 3: Reaction Time & Loop FPS
    ax3 = axes[1, 0]
    _style_ax(ax3, "AI Reaction Speed (Pipeline Latency ms)", "Milliseconds (Lower = Faster)")
    lat_vals = [r["avg_latency_ms"] for r in ai_runs] + [ryley_target["avg_latency_ms"]]
    bars = ax3.bar(x, lat_vals, width=0.5, color="#14b8a6")
    for b_patch, val in zip(bars, lat_vals):
        ax3.text(b_patch.get_x() + b_patch.get_width() / 2, val + 1.5, f"{val:.1f}ms", ha="center", color="#f8fafc", fontsize=8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")

    # Subplot 4: Composite Score vs Human Baseline Threshold vs Ryley Vision
    ax4 = axes[1, 1]
    _style_ax(ax4, "AI Climb: Human Baseline (Floor) -> Ryley Vision (Ceiling)", "Score (0-100)")
    scores = [r["ryley_score"] for r in ai_runs] + [100.0]
    ax4.plot(x, scores, marker="o", color="#facc15", linewidth=2.4, markersize=8, label="CR_AI Progression")
    if human_runs:
        for hr in human_runs:
            is_best = hr is human_threshold
            ax4.axhline(
                hr["ryley_score"],
                color="#22c55e" if is_best else "#86efac",
                linestyle="--" if is_best else ":",
                linewidth=2.0 if is_best else 1.5,
                label=f"{_human_short_name(hr['label'])} ({hr['ryley_score']})",
            )
    else:
        ax4.axhline(75.0, color="#22c55e", linestyle=":", linewidth=1.5, label="Record Your Match (--mode human) for Threshold")
    for xi, sc in zip(x, scores):
        ax4.text(xi, sc + 2.5, f"{sc:.1f}", ha="center", color="#facc15", fontsize=8, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha="right")
    ax4.set_ylim(0, 112)
    ax4.legend(fontsize=7.5, facecolor="#0f172a", labelcolor="#f8fafc")

    plt.tight_layout()
    prog_path = OUT_DIR / "human_vs_ai_progression.png"
    plt.savefig(prog_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Plot Elixir Trace Comparison
    fig2, ax = plt.subplots(figsize=(12, 4.5))
    fig2.patch.set_facecolor("#0f172a")
    _style_ax(ax, "Match Elixir Waveforms Over Time (Ryley Discipline Target Band: 4.0-7.5)", "Elixir")
    ax.axhspan(4.0, 7.5, color="#22c55e", alpha=0.12, label="Ryley Healthy Bank Zone (4.0-7.5)")
    palette = ["#94a3b8", "#f97316", "#38bdf8", "#a855f7", "#facc15", "#22c55e", "#34d399"]
    for idx, run in enumerate(ai_runs + human_runs):
        trace = run.get("elixir_trace") or []
        if not trace:
            continue
        ts = [pt[0] for pt in trace]
        es = [pt[1] for pt in trace]
        if run["kind"] == "human":
            lbl = f"{_human_short_name(run['label'])} ({run['label']}, avg={run['avg_battle_elixir']})"
        else:
            lbl = f"{run['label'].replace('session_20261111_', 'AI_')} (avg={run['avg_battle_elixir']})"
        ax.plot(ts, es, linewidth=2.2 if run["kind"] == "human" else 1.5, color=palette[idx % len(palette)], label=lbl)
    ax.set_xlabel("Match Elapsed Time (s)", color="#cbd5e1")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 10.2)
    ax.legend(fontsize=8, facecolor="#0f172a", labelcolor="#f8fafc", loc="upper right")
    plt.tight_layout()
    wave_path = OUT_DIR / "elixir_and_reaction_comparison.png"
    plt.savefig(wave_path, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    # Strip large traces before saving summary JSON
    summary = {
        "human_threshold": {k: v for k, v in human_threshold.items() if k != "elixir_trace"} if human_threshold else None,
        "ryley_target": ryley_target,
        "ai_runs": [{k: v for k, v in r.items() if k != "elixir_trace"} for r in ai_runs],
        "human_runs": [{k: v for k, v in r.items() if k != "elixir_trace"} for r in human_runs],
        "graphs": {
            "progression": str(prog_path),
            "elixir_waveforms": str(wave_path),
        },
    }
    json_path = OUT_DIR / "benchmark_history.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved graphs: {prog_path} & {wave_path}")
    print(f"Saved history: {json_path}")
    for r in summary["ai_runs"]:
        print(f"  [AI]    {r['label']:<24s} | Elx:{r['avg_battle_elixir']:.2f} | PlayElx:{r['avg_play_elixir']:.2f} | Off:{r['offense_ratio_pct']:4.1f}% | Err:{r['error_rate_pct']:4.1f}% | Lat:{r['avg_latency_ms']:5.1f}ms | Score:{r['ryley_score']:5.1f}")
    if summary["human_runs"]:
        for r in summary["human_runs"]:
            tag = " [BEST]" if human_threshold and r["label"] == human_threshold["label"] else ""
            print(f"  [HUMAN] {r['label']:<24s} | Elx:{r['avg_battle_elixir']:.2f} | PlayElx:{r['avg_play_elixir']:.2f} | Off:{r['offense_ratio_pct']:4.1f}% | Err:{r['error_rate_pct']:4.1f}% | Score:{r['ryley_score']:5.1f}{tag}")
    else:
        print("  [HUMAN] No human match recorded yet. Run: python scripts/run_autonomous_ai.py --mode human --duration 180")
    return summary


if __name__ == "__main__":
    generate_all_benchmarks()
