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
  "hand": [{"slot":1,"card":"hog_rider","cost":4,"playable":true}, ...],  // 4 entries
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
  "aggression": 0.0,           // 0..1
  "ttl_sec": 8,
  "reason": "short text, <= 140 chars"
}

// Goal — produced by Brain/Scheduler, consumed by Navigator
{"goal": "play_battle|collect_chest|close_popup|collect_rewards|clan_games|wait|stop",
 "params": {}, "reason": "..."}

// Action — produced by Reflex/Navigator, consumed by Hand
{"type":"play_card","slot":2,"tile":[12,21]}            // battle
{"type":"tap_element","element_id":"battle_button",
 "expect_screen":"battle_arena","timeout_s":6}          // menu
```

Default Directive (used on startup, when Qwen is late, or when TTL expires):
`mode=defend, lane_pref=any, elixir_floor=3.5, spell_policy={allow:true,min_targets:2}, aggression=0.3, ttl_sec=8`.

---

## 4. BATTLE: Reflex Planner (code, 10–30 Hz)

Reflex reads the newest `GameState` + current `Directive` and returns at most one `Action` per decision tick (respect the ~1 tap/tick physical limit and a short cooldown between plays).

### 4.1 Hard guards (cannot be overridden by Qwen)
1. **Affordability:** never click if `card_cost > elixir` (once the card is known) or `playable == false`. Enforce again in the Hand layer. Target: "Not enough Elixir!" events = 0.
2. **Spell guard:** Fireball/Arrows only if >= `min_targets` enemy units inside the spell radius, or a tower can be finished (`allow_tower_finish`). A lone unit never justifies a spell. (Observed waste: Fireball on 1 unit at ~0:46, Arrows on 1 unit at ~0:35; both left the bot at 1–2 elixir with the enemy still alive.)
3. **Elixir floor:** do not spend a cheap card below `elixir_floor` unless an enemy is within 3 tiles of a friendly tower (`row >= 21`) or a swarm threatens it.
4. **No placement in enemy half** (red-tinted zone) except valid spell targets.
5. **Cycle discipline:** do not dump 1–3 elixir cards blindly; role-based selection (below).

### 4.2 Card selection by role (replaces `candidates[0]`)
Roles from a static table: `win_con` (hog_rider), `building` (cannon), `spell` (fireball, arrows), `defense` (knight, musketeer), `cycle` (skeletons, electro_spirit). Priority depends on situation: defend -> building/defense first; counterpush -> win_con + support; low elixir -> cycle.
`S4=0` plays in the last match came from always taking the lowest playable slot index; selection must consider all 4 slots.

### 4.3 Offence must exist (fixes "zero offence")
Allow `counterpush`/`opposite_lane_punish` when **either** `opp_elixir_est < 3.5` **or** (`elixir >= 7.0` and `enemies_our_half == 0`). Do not gate offence on the opponent estimate alone.

### 4.4 Placement
Tile choice per role/lane (existing tile logic is fine); building placement near bridge-side defence; win-con behind a tank/at bridge when `counterpush`.

### 4.5 Decision pacing
Decide at 10–15 Hz on the **latest** frame; perception at 30 Hz. Never block on Qwen.

---

## 5. BRAIN: Qwen (async) — three jobs

Qwen runs in a worker thread with a request queue of size 1 (latest request wins). Every call has a timeout (e.g. 8 s `[TARGET]`); on timeout/invalid output keep the previous Directive until TTL, then fall back to default.

### 5.1 Job A — Battle strategist
- **Trigger:** every 3–5 s, or on events: new enemy win-con seen, tower HP drop > 15%, elixir leak (>= 9.5 for > 3 s), double-elixir start, last 30 s.
- **Input (compact JSON, target <= ~400 tokens):** `GameState` summary (elixir, hand with roles, next card, enemy count by lane, tower HP, opp elixir est, opp cards seen, clock), last 3 plays, last 3 lessons. Screenshot optional and off by default.
- **Output:** a `Directive` JSON only.
- **Validation:** JSON-schema + clamp ranges (`elixir_floor` 0–9, `aggression` 0–1, `ttl_sec` 3–15, card names from the 8-card enum). Invalid -> discard.

Prompt skeleton (system):
```
You are the strategist for a Clash Royale bot. You do NOT click. You output ONE JSON
Directive that a fast rule engine will follow for the next few seconds.
Deck: arrows, cannon, electro_spirit, fireball, hog_rider, knight, musketeer, skeletons.
Rules: never recommend a card the hand does not contain; spells only for >=2 targets or
tower finish; keep elixir_floor high when behind on elixir; prefer counterpush when the
opponent elixir estimate is low and we have >=7 elixir. Output JSON only, no prose.
Schema: {mode, lane_pref, elixir_floor, card_priority[], spell_policy{}, aggression, ttl_sec, reason}
```

### 5.2 Job B — Menu planner
- **Trigger:** only on stable MENU states, at most every few seconds.
- **Input:** `ScreenState`, list of currently legal goals (computed by code), cooldown timers (chest ready, daily done), last 3 goals.
- **Output:** one `Goal` chosen from the **enum of legal goals only**. If a deterministic scheduler already knows the answer (e.g. "popup visible -> close_popup"), skip Qwen.

### 5.3 Job C — Unknown-screen resolver
- **Trigger:** classifier returns `unknown` for >= 2 consecutive menu frames.
- **Input:** one screenshot (downscaled) + the list of known screen names + known element ids.
- **Output:** `{"screen":"<known name or 'new'>", "suggested_close_element":"<element id or null>", "description":"..."}`.
- Navigation **pauses** until resolved. A `new` screen is saved to `unknown_screens/` for a human to add to the registry. Qwen never invents tap coordinates.

### 5.4 What Qwen must never do
- Emit coordinates or click anything.
- Run per frame.
- Be required for card identification (templates do that).
- Override the Safety Supervisor.

---

## 6. MENUS: Navigator, UI registry, Safety

### 6.1 Screen graph
Nodes = screens (`home`, `battle_arena`, `post_match`, `chest_open`, `popup_generic`, `clan_games`, ...). Edges = a tap on a registry element that leads to another screen. Navigation = shortest path on the graph (deterministic code). Each edge stores `expect_screen` and `timeout_s`.

### 6.2 UI registry (versioned YAML/JSON, one per game version/resolution)
```yaml
version: "2026-11"
resolution: [692, 1136]
screens:
  home:
    anchors: [{template: home_tab.png, roi: [..], min_score: 0.85}]
  post_match:
    anchors: [{template: ok_button.png, roi: [..], min_score: 0.85}]
elements:
  battle_button: {screen: home, template: battle.png, roi: [..], tap: center, leads_to: battle_arena}
  post_match_ok: {screen: post_match, template: ok_button.png, tap: center, leads_to: home}
  close_popup_x: {screen: popup_generic, template: x.png, tap: center, leads_to: previous}
denylist_regions: [shop_tab, gems_buy, any_text_matching: ["buy","purchase"]]
```
Adding a screen/element = adding data, not code. Templates are cropped screenshots; collect them from a manual screen recording of a human walking through the menus.

### 6.3 Goal scheduler (deterministic first)
Priority order: `close_popup` > `post_match_ok` > `collect_chest` (if ready) > `collect_rewards` > `play_battle` > `clan_games` (later) > `wait`. Timers (chest ready, daily reset) come from readable on-screen state or a local clock. Qwen (Job B) only breaks ties or handles states the scheduler cannot classify.

### 6.4 Hand protocol for every menu tap
1. Check Safety Supervisor (allowlist, not in denylist, rate limit).
2. Tap element centre.
3. **Wait** a settle time (400–800 ms), then re-classify.
4. If `expect_screen` reached -> success. Else retry (max 2), then `unknown-screen` handling or stop.

### 6.5 Safety Supervisor (applies to battle and menu)
- **Allowlist taps only** (registry elements marked safe). Anything else is refused.
- Shop, gem, purchase regions are hard-denied.
- Rate limit taps/sec; never two taps in the same 150 ms.
- **Stop conditions:** "another device connected" dialog, login/captcha/ban-style dialogs, repeated unknown screens, `N` consecutive failed verifications. On stop: write a clear log line and wait for a human; do not guess.
- Kill switch: a `STOP` file in the working directory halts all taps within one tick.

### 6.6 Unknown screens
Never tap blindly. Pause -> Job C -> if it returns a known screen/close element, proceed; otherwise stop and save the screenshot.

---

## 7. Latency budget and gates (do this FIRST — Task 0)

Log three timestamps per decision: `t_capture`, `t_decision`, `t_tap_sent`. Report p50/p95/p99 of `capture_to_tap_ms`, `perception_ms`, `dropped_frames_pct`.

| Stage | Budget | Status |
|---|---|---|
| Capture (PrintWindow) | <= 10 ms `[TARGET]` | `[UNVERIFIED]` live |
| Perception (8 templates, cached) | p95 <= 33 ms | `[MEASURED]` mean 22 ms / p95 33 ms with only 4 templates on recorded video; **re-measure with 8 + live capture** |
| Reflex decision | <= 5 ms `[TARGET]` | |
| Tap dispatch | measure | |
| **capture -> tap, end to end** | **p95 <= ~100 ms** `[TARGET]` (human reaction ≈ 200–300 ms) | |
| Qwen | not in this path | async only |

A prior live log showed one play with `latency_ms = 539.83` (others ~90–220 ms). Find the source (VLM call, blocking capture, or planner) with the Task 0 harness.

**Gate tests**
1. p95 `capture_to_tap_ms` within budget on live capture.
2. **Qwen-off test:** replace Qwen with `sleep(10)` / timeout; `capture_to_tap_ms` p95 must not change and the bot must keep playing on the default Directive.
3. Frame-drop policy: if processing exceeds the frame interval, drop (never queue); report the drop %.

---

## 8. Known failure modes (from the 143.5 s recorded loss) and where they are fixed

| # | Failure | Fix location |
|---|---|---|
| 1 | Red UI text detected as enemies | §2.4.3 badge detector |
| 2 | Chronic low elixir, S4 never played, placeholders `slot_1..4` | §2.4.2 card ID, §4.2 role selection |
| 3 | Zero offence, opp elixir stuck at 10.0 | §2.4.4, §4.3 |
| 4 | "Not enough Elixir!" clicks, bar over-read | §2.4.1, §4.1 guard 1 |
| 5 | Transition frames scored `battle_arena 0.75` | §2.3 stricter battle gate |
| 6 | Spells wasted on single units (Fireball ~0:46, Arrows ~0:35) | §4.1 guard 2 |

---

## 9. Auditor and lessons (post-match, offline, cheap)

After each match compute and store: `spell_value` (units hit per spell, wasted-spell count), `unaffordable_clicks`, `elixir_leak_sec` (time at >= 9.5), `elixir_starved_sec` (< 3), `slot_usage[1..4]`, `defend_response_ms`, `tower_damage_dealt/taken`, result. Write `lessons.json` (max 5 short lessons, e.g. "wasted Fireball on single unit at 0:46"). Inject the last 3 lessons into Qwen's battle prompt next match. No model training.

---

## 10. Mapping to the current repo (`e:\cr_ai`, as summarised by the owner)

| Brief item | Existing | Change |
|---|---|---|
| Capture thread | `CaptureEngine.process`, sync `run_live` (`--delay 0.008`) | run capture in a daemon thread, single-slot latest frame, 10–15 Hz decisions |
| Elixir + card ID | `_detect_hud_and_elixir` (`ex1=0.258*vw`), `CardTemplateMatcher` | merge calibrated elixir constants, CLAHE+margin matching, slot cache + invalidation |
| Units | `detect_arena_entities` (HSV clusters) | replace/augment with the badge detector |
| Opp elixir | `ElixirTracker`, `OpponentTracker` | ensure `register_card_played()` fires on spawns + unit test |
| Planner | `TacticalActionPlanner` | add Directive/GameState dataclasses, hard guards, spell/offence rules |
| VLM | `_maybe_trigger_vlm_card_id` (async, ~2.2 s, screenshot) | switch to compact-JSON strategist; VLM off the card-ID path |
| Auditor | offline scripts | live post-match auditor + `lessons.json` |

---

## 11. Task list (strict order, each with pass/fail)

**T0 Latency harness** (§7). Pass: numbers logged for the current system; source of the 540 ms spike identified.
**T1 Capture thread + latest-frame buffer.** Pass: perception p95 <= 33 ms with 8 templates on live capture; drop % reported.
**T2 Elixir + card ID merge.** Pass: regen-slope check live (0.357 / 0.714 per s ±10%); card ID >= 98% on ~40 labelled frames incl. cycle frames; `unknown` <= 2%.
**T3 Badge detector.** Pass: 0 fake enemies on banner frames; recall reported on labelled frames; overlay video reviewed by a human.
**T4 Opponent elixir counter.** Pass: unit tests (idle vs spawn); not pinned at 10.0 after plays.
**T5 GameState/Directive + Reflex guards.** Pass: unaffordable clicks = 0; spell only with >= min_targets or tower finish; all 4 slots can be played; offence triggers in the §4.3 conditions.
**T6 Qwen strategist (Job A) + fallback.** Pass: Qwen-off test (§7); invalid JSON ignored; Directive TTL expiry returns to default.
**T7 Auditor + lessons.** Pass: metrics saved per match; lessons injected next match.
**M1 Screen classifier + UI registry + post-match/popup handling** (screens: battle_arena, post_match, popup_generic). Pass: 20 consecutive matches loop with no manual input.
**M2 Chests/daily rewards** via the graph. Pass: verify-after-tap on every edge; zero taps outside the allowlist.
**M3 Unknown-screen resolver (Job C) + stop conditions.** Pass: forced unknown screen -> bot pauses, saves screenshot, taps nothing.
**M4 Clan games and other modes** only after M1–M3 are stable for multiple days.

---

## 12. Open items / not verified

- Live-capture numbers (capture ms, end-to-end ms) — not measured.
- 8-template perception timing — extrapolated, not measured.
- Card-cache invalidation behaviour — not tested.
- Badge detector recall — not measured; precision only eyeballed.
- Opponent regen in overtime — rate not confirmed.
- Qwen model size/host, expected latency, and JSON reliability — unknown; benchmark before relying on it.
- The enemy deck cannot be read directly from the screen; it is inferred from spawned units.
