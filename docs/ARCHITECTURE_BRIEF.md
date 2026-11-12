# CLASH IQ — Eye / Brain / Hand Architecture Brief

**Audience:** the coding agent (Gemini) implementing this in `e:\cr_ai`.
**Brain model:** Qwen (VLM/LLM). **Hard rule: Qwen is never in the tap path.**
**Status labels used below:** `[MEASURED]` = measured on the recorded match (`Recording 2026-11-11 100928.mp4`, 692x1136, 30 FPS) by the observe-only prototype (`perception.py`, `observer.py`). `[TARGET]` = a goal, not yet measured. `[UNVERIFIED]` = needs a test before trusting.

---

## 0. Goals, non-goals, constraints

**Goals**
1. Play full matches in real time, then run the whole game loop unattended (battle, post-match, chests, popups, later clan games).
2. Every decision is explainable from logged state (no hidden magic).
3. Real-time first: a late decision is a wrong decision.

**Non-goals (for now)**
- No training/RL. Rules + Qwen directives first. Training only after perception is solid (the dataset is 1 match).
- No reading game memory. Vision only (same info a human sees).
- No purchases, ever.

**Constraints**
- Test on an **alt account**. Botting may violate the game's terms; the enforcement level is unknown.
- UI changes with game updates: everything UI-specific lives in a versioned registry, not in code.

---

## 1. Big picture

```
            +-----------------------------------------------------------+
            |                        EYE (code, fast)                   |
 screen --> | Capture thread -> Screen Classifier -> mode switch        |
            |        |                 |                                |
            |   BATTLE perception   MENU perception                     |
            |   (30 Hz)             (2 Hz)                              |
            +--------+-----------------+--------------------------------+
                     | GameState        | ScreenState
                     v                  v
        +------------------+   +-------------------------+
        | REFLEX PLANNER   |   | NAVIGATOR (graph + goals)|
        | (code, 10-30 Hz) |   | (code, 1-2 Hz)           |
        +--------+---------+   +-----------+-------------+
                 ^  Directive               ^  Goal choice / unknown-screen help
                 |                          |
            +----+--------------------------+----+
            |   BRAIN = Qwen (async, seconds)      |
            |   strategist (battle) + planner      |
            |   (menu) + unknown-screen resolver   |
            +--------------------------------------+
                 |  Action (tap x,y + expected_result)
                 v
            +-----------+      +------------------+
            |   HAND    | ---> | SAFETY SUPERVISOR|  (allowlist, rate limit, verify-after-tap)
            +-----------+      +------------------+
                 |
            Auditor -> lessons.json -> injected into Brain prompts
```

Rules that never change:
- Eye and Reflex run without Qwen. If Qwen is slow, dead, or wrong, the bot keeps playing on a default Directive.
- Qwen outputs **intent** (JSON), never coordinates or clicks.
- Every tap goes through the Safety Supervisor.

---

## 2. EYE

### 2.1 Capture
- Dedicated daemon thread. **Single-slot "latest frame" buffer** (overwrite, never queue). Consumers read the newest frame; stale frames are dropped and counted (`dropped_frames_pct`).
- Keep the existing `CaptureEngine` (PrintWindow + mss). Do not use ADB screencap for battle (typically 100–200 ms `[UNVERIFIED]`).
- Timestamp each frame at grab time (`t_capture`). All latency is measured from this.

### 2.2 Viewport
- Strip emulator sidebar/title bar. Prototype `find_viewport()` `[MEASURED]`: x0=81, y0=45 on this recording; ~1.7 ms; re-run every ~50 frames, not every frame.

### 2.3 Screen classifier (decides BATTLE vs MENU)
- **Classical first, Qwen last.** Each known screen has anchor elements in the UI registry (template/colour/ROI). Score all anchors, pick the best screen; confidence below threshold -> `unknown`.
- `unknown` frames go to Qwen (async) for naming; navigation pauses meanwhile (see §6.6).
- Battle detection must also reject transition/popup frames. Known bug: popups and match-end frames scored `battle_arena 0.75`. Require: confidence >= 0.90 AND droplet present AND elixir bar present; otherwise not-battle.
- Cadence: classifier at 30 Hz in BATTLE (cheap check only), 2 Hz in MENU.

### 2.4 Battle perception (30 Hz) — modules and measured facts

| Module | Method | Measured |
|---|---|---|
| Elixir | purple bar fill, **droplet excluded** | 0.1 ms |
| Hand cards | template match (8 templates) + CLAHE + top1/top2 margin, **slot cache** | see 2.4.2 |
| Units | level-badge detector (solid red/blue square + white digit) | ~10 ms |
| Tower HP | digits/bars at fixed ROIs (existing code) | n/a |
| Opponent model | spawn tracking + elixir counter | cheap |

#### 2.4.1 Elixir reading (fixes the +0.5..0.8 over-read)
Root cause `[MEASURED]`: the old reader measured purple from `ex1=0.22*vw`, and the elixir **droplet** at the bar's left is permanently purple, so fill never read below ~0.8.
Fix: measure only right of the droplet, with calibrated bar ends (611 px wide viewport):
```
BAR_X0 = 189/611   # 0 elixir
BAR_X1 = 595/611   # 10 elixir
DROPLET_END = 185/611
Y band = 0.964 .. 0.992 of viewport height
elixir = clip((fill_end - X0) / (X1 - X0) * 10, 0, 10)   # contiguous run from the left
```
Validation `[MEASURED]`: regen slope 0.355/s in a clean normal-elixir segment (expected 0.357), 0.742/s in double elixir (expected 0.714). Below ~0.5 elixir the reading collapses to 0 (droplet overlap) — treat `elixir < 0.5` as "≈0".
**Must re-validate on live PrintWindow frames** (crop/scale may differ): re-run the regen-slope check.

#### 2.4.2 Hand card ID
- 8 templates in `data/card_templates/` (120x100 BGR): arrows, cannon, electro_spirit, fireball, hog_rider, knight, musketeer, skeletons.
- Matching: grayscale + CLAHE, `TM_CCOEFF_NORMED`, scales {0.8,0.9,1.0,1.1,1.2}. Return `unknown` if best < 0.45 or (best − second) < 0.04.
- Cost `[MEASURED]`: ~20 ms for 4 templates uncached (~40 ms for 8, extrapolated `[UNVERIFIED]`) -> too slow per frame.
- **Slot cache:** keep a 16x16 normalized-gray signature per slot; reuse the last ID while cosine > 0.92 (greyed-out card keeps its structure). Only cache confident IDs.
- **Cache invalidation (required, untested `[UNVERIFIED]`):** clear a slot's cache (a) immediately when the bot plays that slot, (b) whenever the slot looks empty/cooldown (flat dark blue), (c) when the signature drops below threshold. Without this a stale ID can survive a card cycle and the bot plays the wrong card.
- Playable = mean saturation >= 62 (coloured = affordable, grey = not). Additionally block any click where `cost > elixir` once the card is known (card cost table).
- Acceptance: >= 98% correct on ~40 human-labelled frames **including card-cycle animation frames**; `unknown` <= 2%.

#### 2.4.3 Units via level badge (fixes Finding 1: fake enemies from red UI text)
Every troop has a level badge: solid red (enemy) / blue (ally) square with a white digit.
Filter on the red/blue mask: connected component with bbox w 9–26, h 12–30 px (scaled by vw/611), aspect 0.55–1.1, **fill ratio >= 0.45**, **white-digit fraction 0.10–0.55**. Thin text strokes ("Not enough Elixir!", "30 Seconds Left") fail fill ratio/white test.
Search region: y from 0.09 to 0.835 of viewport (above the card tray).
**[UNVERIFIED]** The prototype's enemy counts were not checked against labelled frames. Acceptance: 0 false enemies on all frames where the red banner is visible, AND recall measured on labelled frames. If recall is poor (e.g. badges hidden/overlapped), move to YOLO (public datasets exist: ~26.8k sprites, ~12k annotated images).

#### 2.4.4 Opponent elixir counter (fixes Finding 3: stuck at 10.0)
- Start 5.0. Regen: 1 per 2.8 s normal, 1 per 1.4 s in double elixir (last minute); overtime faster `[UNVERIFIED rate]`.
- On each **new** enemy badge track (new = no existing track within ~70 px / 110 px and seen <1.5 s ago): call `register_card_played()`, subtract the card cost if known, else `ASSUMED_COST=3.5`. Save the crop to `enemy_crops/` to later build enemy templates -> known cost.
- Track `opp_cards_seen` (distinct, max 8) once identification exists.
- Unit test: with no enemy spawns the estimate may sit at 10.0; with spawns it must drop by cost. It must **not** be pinned at 10.0 after plays.

### 2.5 Menu perception (2 Hz)
Screen classification + anchor/button detection from the UI registry (§6.2). No heavy models.

---

## 3. Shared data contracts (dataclasses)

```jsonc
// GameState — produced by Eye every frame in BATTLE
{
  "t_capture": 12.345,
  "elixir": 6.4,
  "double_elixir": false,
  "clock_sec": 134,
  "hand": [{"slot":1,"card":"hog_rider","cost":4,"playable":true}, ...],
  "next_card": "fireball",
  "enemies": [{"id":7,"cx":412,"cy":305,"lane":"right","side":"enemy","age_s":1.2}],
  "allies":  [{"id":3,"cx":300,"cy":700,"lane":"left"}],
  "tower_hp": {"player_king":4008,"player_left":2534,"player_right":2425,
               "opp_king":2534,"opp_left":2534,"opp_right":1360},
  "opp_elixir_est": 4.1,
  "opp_cards_seen": ["knight","musketeer"],
  "recent_plays": [{"t":10.2,"slot":1,"card":"skeletons","tile":[11,22],"result":"ok"}]
}

// ScreenState — produced by Eye in MENU
{"t_capture": 99.1, "screen": "home", "confidence": 0.97,
 "elements": [{"id":"battle_button","bbox":[x1,y1,x2,y2],"score":0.95}],
 "popup": null}

// Directive — produced by Brain (Qwen), consumed by Reflex
{
  "mode": "defend|counterpush|bait|cycle",
  "lane_pref": "left|right|any",
  "elixir_floor": 4.0,
  "card_priority": ["hog_rider","fireball"],
  "spell_policy": {"allow": true, "min_targets": 2, "allow_tower_finish": true},
  "aggression": 0.0,
  "ttl_sec": 8,
  "reason": "short text, <= 140 chars"
}

// Goal — produced by Brain/Scheduler, consumed by Navigator
{"goal": "play_battle|collect_chest|close_popup|collect_rewards|clan_games|wait|stop",
 "params": {}, "reason": "..."}

// Action — produced by Reflex/Navigator, consumed by Hand
{"type":"play_card","slot":2,"tile":[12,21]}
{"type":"tap_element","element_id":"battle_button",
 "expect_screen":"battle_arena","timeout_s":6}
```

Default Directive (used on startup, when Qwen is late, or when TTL expires):
`mode=defend, lane_pref=any, elixir_floor=3.5, spell_policy={allow:true,min_targets:2}, aggression=0.3, ttl_sec=8`.

---

## 4. BATTLE: Reflex Planner (code, 10–30 Hz)

1. **Affordability:** never click if `card_cost > elixir` (once the card is known) or `playable == false`. Enforce again in the Hand layer. Target: `"Not enough Elixir!"` events = 0.
2. **Spell guard:** Fireball/Arrows only if >= `min_targets` enemy units inside the spell radius, or a tower can be finished (`allow_tower_finish`).
3. **Elixir floor:** do not spend a cheap card below `elixir_floor` unless an enemy is within 3 tiles of a friendly tower (`row >= 21`) or a swarm threatens it.
4. **No placement in enemy half** except valid spell targets.
5. **Cycle discipline + Role selection:** select by tactical role (`win_con`, `building`, `spell`, `defense`, `cycle`), never `candidates[0]` / Slot-1 bias.
6. **Offence must exist:** allow `counterpush`/`opposite_lane_punish` when `opp_elixir_est < 3.5` OR (`elixir >= 7.0` and `enemies_our_half == 0`).

---

## 5. BRAIN: Qwen (async) — three jobs

- **Job A — Battle strategist:** trigger every 3–5s or on events; compact JSON input (`<= ~400 tokens`), screenshot off by default; output validated `Directive` JSON; timeout/invalid keeps previous until TTL -> default Directive.
- **Job B — Menu planner:** trigger on stable MENU states when scheduler needs tie-breaking; output `Goal` from legal enum.
- **Job C — Unknown-screen resolver:** trigger when classifier returns `unknown` >= 2 consecutive menu frames; pause navigation -> classify known/new -> never invent coordinates.

---

## 6. MENUS: Navigator, UI registry, Safety

- Versioned UI registry (`config/ui_registry.json` or YAML) with screen anchors, allowlisted elements, and `denylist_regions` (`shop_tab`, `gems_buy`, `"buy"`, `"purchase"`).
- Hand verify-after-tap protocol (settle 400–800 ms, re-classify, retry max 2).
- Safety Supervisor: allowlist taps only, rate limit >= 150 ms between taps, stop conditions + `STOP` file kill switch.

---

## 7. Latency budget and gates

Log 3 timestamps per decision: `t_capture`, `t_decision`, `t_tap_sent` (+ `t_tap_done`). Report p50/p95/p99 of `capture_to_tap_ms`, `perception_ms`, `dropped_frames_pct`.
Gates: `perception_ms p95 <= 33 ms` (8 templates), `capture_to_tap_ms p95 <= 100 ms`, Qwen `sleep(10)` chaos test passes with unchanged `capture_to_tap_ms p95`.

---

## 9. Auditor and lessons

After each match compute: `spell_value`, `unaffordable_clicks`, `elixir_leak_sec` (>= 9.5), `elixir_starved_sec` (< 3), `slot_usage[1..4]`, `defend_response_ms`, `tower_damage_dealt/taken`, result -> `data/lessons.json` (max 5 short lessons, last 3 injected into Qwen Battle prompt).

---

## 11. Task order (strict pass/fail gates)

- **T0**: Latency harness baseline.
- **T1**: Capture thread + latest-frame buffer (`perception p95 <= 33 ms`, `capture_to_tap p95 <= 100 ms`).
- **T2**: Elixir calibration + Card ID merge (regen slope `0.357 / 0.714 ±10%`; card ID `>= 98%`, `unknown <= 2%` incl. cycle frames).
- **T3**: Badge detector (`0` false enemies on banners + measured recall + visual overlay review).
- **T4**: Opponent elixir counter unit tests (idle vs spawn, not stuck at 10.0).
- **T5**: `GameState` / `Directive` / Safety Supervisor + Reflex hard guards + offense check.
- **T6**: Async Qwen Job A (`Directive`) + `sleep(10)` chaos gate.
- **T7**: Post-match Auditor + `lessons.json`.
- **M1–M3**: UI Registry, Screen Classifier, Navigator + Verify-after-tap, Job C Unknown-screen resolver + `STOP` killswitch.
