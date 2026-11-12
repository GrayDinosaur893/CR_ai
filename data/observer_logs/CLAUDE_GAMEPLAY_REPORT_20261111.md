# CLASH IQ — Live Autonomous Gameplay & Observer Recording Report (For Claude)

**Date:** 2026-11-11  
**Target Recording:** `data/observer_logs/Recording 2026-11-11 100928.mp4` (`692x1136`, `30.00 FPS`, `4,305 frames`, `143.50s`)  
**Correlated Live Engine Log:** `task-171` (`ARMED_LIVE`, `--auto-battle`, `--record`, `10:06:29`–`10:09:32`)  
**Generated Graphs:**
- `data/observer_logs/analysis_100928/gameplay_timeline_analysis.png`
- `data/observer_logs/analysis_100928/gameplay_distribution_summary.png`
- Raw metrics JSON: `data/observer_logs/analysis_100928/video_metrics.json`

---

## 1. Executive Summary

The autonomous AI successfully launched, locked onto Clash Royale (`HWND 67380`), passed the `AccuracyGate` (`100.0%` benchmark, `mean_iou=0.880`), played an entire live match autonomously (`40` live card deployments + `1` post-match `OK` click in `task-171`), and recorded the session.

**Match Result:** **3-Crown Loss** at `t ≈ 142.6s` (`player_princess_right` fell around `t=60s`, `player_king_tower` dropped from `2315` HP at `t=110s` to `228` HP at `t=141s` and fell at `t=142.6s`; opponent `Chente` had `opponent_princess_right` at `1360` HP and `opponent_princess_left` untouched at `2534` HP).

The AI **did execute real clicks reliably**, so the execution loop works end-to-end. However, **5 concrete perception + strategy bugs** caused chronic elixir starvation (`avg elixir = 2.68`), zero offensive pushes, and eventual defeat.

---

## 2. Measured Telemetry (5.0 Hz Sampled Over 143.5s Video = 718 Frames)

| Metric | Measured Value | Notes |
|---|---:|---|
| Sampled Frames | `718` | `5.0 Hz` from `4,305` total frames (`30 FPS`) |
| Screen Classification | `battle_arena: 718` | `717` frames `>= 0.90` conf, `1` frame `< 0.90` (`0.75` at `t=143.4s`) |
| Viewport Layout | `gpg_windowed_sidebar: 716`, `full_viewport: 2` | Dynamic GPG sidebar stripping worked on `99.7%` of frames |
| Mean / p95 Perception Latency | `77.74 ms` / `114.16 ms` | Single-frame `ComponentDetector` + `TacticalActionPlanner` |
| Mean Elixir (High-Conf Battle) | `2.68` | Stuck in `0.7–3.5` range from `t=25s` to `t=118s` |
| Mean Detected Enemies / In Our Half | `3.22` / `1.49` | Includes false-positive red text/tower-carpet hits |
| Planned Decisions (`718` frames) | `wait: 574`, `defend: 144` | **`0` `opposite_lane_punish`, `0` `slow_roll_back`** |
| Live Executions (`task-171.log`) | `40` `LIVE_EXECUTED`, `1` `LIVE_MENU_CLICKED` | Slots: `S1=22`, `S2=6`, `S3=12`, **`S4=0`** |

**Top Deployment Tiles (from `task-171.log`):**  
`[11, 22]` (10x), `[12, 22]` (8x), `[9, 22]` (4x), `[10, 22]` (4x), `[12, 20]` (3x), `[13, 18]` (2x), `[13, 21]` (2x), `[11, 21]` (2x).  
All 40 plays clustered defensively in the lower-right quadrant (`cols 9–13, rows 18–22`).

---

## 3. Root-Cause Findings (With Visual Evidence)

### Finding 1 (Critical): False-Positive Enemy Detections on UI Red Text & King Carpet
- **Observed in:** `keyframes/t_002s.png`, `keyframes/t_035s.png`, `keyframes/t_141s.png`.
- **What happened:** `ComponentDetector.detect_arena_entities()` uses a bright-red HSV filter inside `[0.08..0.92] x [0.18..0.76]`. It repeatedly flagged:
  1. The red `"Not enough Elixir!"` warning banner across the center of the arena (`t_141s.png`, generating `E[8, ...]`, `E[10, ...]`, `E[11, ...]` directly on the text!).
  2. The red trim on the Opponent King Tower (`E[9, 3]` in `t_002s.png` and `t_035s.png`).
  3. Red elements inside the `"30 Seconds Left"` / `"60 Seconds Left"` countdown overlays (`t_110s.png`, `t_141s.png`).
- **Impact:** Fake `enemies` in our half (`row >= 16`) constantly triggered emergency `defend` decisions even when no troops were near the tower, fueling non-stop panic spending.

### Finding 2 (Critical): Chronic Elixir Starvation & Slot-1 Bias (`S4 = 0` plays)
- **Observed in:** `gameplay_timeline_analysis.png` (Panel 1) & `task-171.log` (`S1=22, S2=6, S3=12, S4=0`).
- **What happened:**
  1. `TacticalActionPlanner._select_playable_slot()` always picks `candidates[0]` (lowest playable slot index) when no card names match `BUILDING_HINTS` / `SPELL_HINTS` / `WINCON_HINTS`.
  2. Because `--vlm` was off (for latency), `hand_cards` were generic placeholders (`slot_1..slot_4`), so win-conditions (`Hog Rider`), buildings (`Cannon`), and spells (`Fireball`, `Arrows`) were never distinguished from 1-elixir cycle cards (`Skeletons`, `Electro Spirit`).
  3. Any cheap 1–3 elixir card in Slots 1–3 got dumped immediately on `defend`, keeping elixir pinned around `1–3` for almost 100 seconds (`t=25s..118s`) and preventing `4`-elixir cards (especially in Slot 4) from ever being played.

### Finding 3 (High): Zero Offense (`opposite_lane_punish = 0`, `slow_roll_back = 0`)
- **Observed in:** `gameplay_distribution_summary.png` (`wait: 574, defend: 144`) and `data/live_game_state.json`.
- **What happened:**
  1. `AutonomousClashPlayer` calls `self.tracker.advance_time(elapsed_sec)` without ever calling `register_card_played()` for opponent troops, so `opponent_estimated_elixir` reaches `10.0` within 14s and stays `10.0` permanently.
  2. `TacticalActionPlanner` gates `opposite_lane_punish` behind `opponent_estimated_elixir < 3.5`, which therefore **never evaluates to True**.
  3. Elixir never reached `9.0` after `t=24s`, so `slow_roll_back` also never fired. Result: 100% passive/defensive bot until overwhelmed.

### Finding 4 (Medium): `"Not enough Elixir!"` Triggers & Elixir Bar Over-Read (~0.5–0.8)
- **Observed in:** `keyframes/t_035s.png` and `keyframes/t_141s.png`.
- **What happened:**
  1. In `t_141s.png`, the droplet digit shows `2`, while `Elixir` reads `2.6` (bar ROI still slightly leads the integer droplet or picks up card-cost purple glow).
  2. During card cycle animations or right after a spend, a slot's `mean_saturation` briefly stayed `>= 62.0`, causing the AI to click an unaffordable card (`"Not enough Elixir!"` banner visible in `t_035s.png` and `t_141s.png`), which in turn created false-positive red text enemies (Finding 1).

### Finding 5 (Medium): Transition/Modal False `battle_arena (75%)` Frames
- **Observed in:** `task-144.log`, `task-171.log`, and `t=143.4s` of the recording (`conf=0.75, Elx=0.0`).
- **What happened:** Blue-bottomed popups (e.g. `King Tower Level 9`) and battle-end transition frames score `0.75` as `battle_arena`. `AccuracyGate.is_live_frame_ready(min_conf=0.90)` blocked clicks on those frames (`BLOCKED_FRAME_CONFIDENCE`), preventing misclicks, though `ComponentDetector` should classify `conf < 0.85` + `droplet=False` + `elixir=0.0` as `non_game`.

---

## 4. Prioritized Action Plan for Next Iteration

1. **Eliminate False-Positive Red Enemies (`component_detector.py`):**
   - Filter out `"Not enough Elixir!"` / `"Seconds Left"` / `"Overtime"` center text bands or require a valid compact horizontal HP-bar aspect ratio (`width > height * 2.2`, `height <= 6 px`) paired with a nearby unit/level badge rather than isolated large red letters.
   - Tighten opponent King Tower exclusion zone up to `cy_vp < 0.22` across `0.38 <= cx_vp <= 0.62`.
2. **Fast Local Card Identification (`< 5 ms`, no VLM blocking):**
   - Replace/augment slow async Qwen VLM card ID with fast OpenCV template matching / color-histogram matching on the 8-card deck crops (`Hog Rider`, `Musketeer`, `Cannon`, `Knight`, `Fireball`, `Arrows`, `Skeletons`, `Electro Spirit`) so `BUILDING_HINTS`, `SPELL_HINTS`, and `WINCON_HINTS` actually fire on every frame.
3. **Fix Elixir Starvation & Enable Offense (`action_planner.py` + `autonomous_player.py`):**
   - Do not spend `1–3` elixir cycle cards blindly on minor/distant threats when elixir `< 4.0` unless an enemy is within 3 tiles of a friendly tower (`row >= 21`) or urgent swarm threat.
   - Track approximate opponent elixir spend from newly spawned enemy units (or allow `opposite_lane_punish` whenever `current_elixir >= 7.0` and `enemies_our_half == 0` regardless of opponent estimate).
   - Rotate / score playable slots by role (defense vs win-con vs cycle) instead of always picking `candidates[0]`.
4. **Tighten Playability Check (`component_detector.py`):**
   - Require `elixir_estimate >= slot_cost` (once card/cost is known) or verify purple cost droplet brightness so `"Not enough Elixir!"` clicks drop to zero.
