"""
CLASH IQ — Elixir Waveform Analyzer & Sawtooth Plotter
Reads video frames at 0.2s intervals, extracts the exact fill ratio
of the physical magenta elixir bar, detects drop transitions, and plots
the canonical Sawtooth Elixir Waveform.
"""

import os
import sys
import cv2
import numpy as np

def extract_elixir_waveform(video_path: str, start_sec: float = 5.0, duration_sec: float = 65.0, step_sec: float = 0.2):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    end_sec = start_sec + duration_sec
    current_sec = start_sec

    timestamps = []
    elixir_values = []
    frame_indices = []

    # Elixir bar parameters calibrated for 298x640 frame:
    # Bar Y-range: 616 to 624 (8 px high)
    # Bar X-range: 64 to 284 (220 px total = 22 px per 1.0 elixir)
    BAR_Y1, BAR_Y2 = 616, 624
    BAR_X1, BAR_X2 = 64, 284
    TOTAL_BAR_WIDTH = BAR_X2 - BAR_X1

    print(f"Analyzing {video_path} from {start_sec}s to {end_sec}s (step: {step_sec}s)...")

    while current_sec <= end_sec:
        frame_idx = int(current_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Crop horizontal bar
        bar_crop = frame[BAR_Y1:BAR_Y2, BAR_X1:BAR_X2]
        hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)

        # Purple / Magenta Elixir Mask
        mask = cv2.inRange(hsv, np.array([130, 60, 60]), np.array([175, 255, 255]))

        # Column-wise fill: at least 3 out of 8 vertical pixels must be magenta
        col_is_purple = (mask > 0).sum(axis=0) >= 3

        # Robust bar fill measurement: find the rightmost purple column in the bar
        # This completely ignores internal black/metal tick marks between elixir slots!
        purple_indices = np.where(col_is_purple)[0]
        if len(purple_indices) >= 5:
            # Bar head is the rightmost filled column
            filled_px = int(purple_indices[-1]) + 1
        else:
            filled_px = 0

        elixir_level = round((filled_px / float(TOTAL_BAR_WIDTH)) * 10.0, 2)

        timestamps.append(round(current_sec - start_sec, 2))
        elixir_values.append(elixir_level)
        frame_indices.append(frame_idx)

        current_sec += step_sec

    cap.release()
    print(f"Extracted {len(elixir_values)} data points.")
    return timestamps, elixir_values, frame_indices

def plot_waveform(timestamps, elixir_values, output_img_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    # Style: Dark aesthetic
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#111827")

    # Plot raw elixir line
    ax.plot(timestamps, elixir_values, color="#d946ef", linewidth=2.5, label="Measured Elixir (Fill %)")
    ax.fill_between(timestamps, elixir_values, color="#d946ef", alpha=0.15)

    # Detect and mark drop moments (where delta_e <= -1.2 elixir within <= 0.4s)
    drops = []
    for i in range(1, len(elixir_values)):
        delta = elixir_values[i] - elixir_values[i-1]
        dt = timestamps[i] - timestamps[i-1]
        if delta <= -1.2:
            drops.append((timestamps[i], elixir_values[i], delta))
            ax.annotate(
                f"Drop {delta:+.1f}",
                xy=(timestamps[i], elixir_values[i]),
                xytext=(timestamps[i], max(0.5, elixir_values[i] - 1.2)),
                arrowprops=dict(facecolor="#f43f5e", shrink=0.08, width=1.5, headwidth=6),
                color="#fecdd3",
                fontsize=8,
                fontweight="bold",
                ha="center"
            )

    # Reference lines for standard thresholds
    ax.axhline(10.0, color="#22c55e", linestyle="--", alpha=0.6, label="Max Cap (10)")
    ax.axhline(0.0, color="#64748b", linestyle=":", alpha=0.5)

    # Grid & Limits
    ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
    ax.set_ylim(-0.5, 10.8)
    ax.set_xlim(0, max(timestamps) if timestamps else 60)

    # Labels & Title
    ax.set_title("Clash Royale — Elixir Waveform (0.2s Sampled Sawtooth Test)", fontsize=14, fontweight="bold", color="#f8fafc", pad=12)
    ax.set_xlabel("Match Elapsed Time (Seconds)", fontsize=11, color="#94a3b8")
    ax.set_ylabel("Elixir Level (0 - 10)", fontsize=11, color="#94a3b8")
    ax.tick_params(colors="#cbd5e1")

    # Legend
    leg = ax.legend(loc="upper right", facecolor="#1f2937", edgecolor="#374151")
    for text in leg.get_texts():
        text.set_color("#f3f4f6")

    # Info summary box
    summary_text = f"Total Duration: {max(timestamps):.1f}s | Sample Points: {len(timestamps)} | Detected Drops: {len(drops)}"
    fig.text(0.13, 0.02, summary_text, color="#94a3b8", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Sawtooth graph successfully saved to: {output_img_path}")

if __name__ == "__main__":
    vid = "E:/cr_ai/media/source/match_yt_01.mp4"
    out_plot = "E:/cr_ai/frames/elixir_sawtooth_graph.png"
    times, elixirs, f_idxs = extract_elixir_waveform(vid, start_sec=8.0, duration_sec=62.0, step_sec=0.2)
    plot_waveform(times, elixirs, out_plot)
