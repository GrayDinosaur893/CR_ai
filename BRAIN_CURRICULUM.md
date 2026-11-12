# CLASH IQ — Brain Curriculum (what the Brain must learn, and what it must ignore)

**Companion to:** `ARCHITECTURE_BRIEF.md` (read that first).
**Audience:** the coding agent (Gemini). **Brain:** Qwen, async, outputs a `Directive` only.
**Tags:** `[MEASURED]` from the recorded match, `[RULE]` game rule used as fact, `[HEURISTIC]` starting value to be tuned from data (not a fact), `[UNVERIFIED]` needs a test.

---

## 0. What "training" means here

There is **no weight training** at this stage (our dataset is one match; ClashAI-style imitation needed ~15k pro replays). "Teaching the Brain" = four things:

1. **Knowledge tables** it can rely on (costs, roles, cycle rules) — static, in the prompt.
2. **Derived features** computed by code and handed to it (it never does arithmetic).
3. **A decision rubric + few-shot examples** (how to turn features into a Directive).
4. **A lessons loop** (post-match auditor -> `lessons.json` -> next prompt).

Later, once we have logs from many matches, the same inputs/outputs become the dataset for fine-tuning. So log them now (§8).

**Core principle: code computes facts, the Brain chooses intent.** LLMs are unreliable at counting. Elixir totals, leak timers, cycle position and "can the opponent afford X" are computed in code and passed as ready-made fields.

---

## 1. What the Brain must learn (relevant)

### 1.1 Elixir economy (highest priority)

**Facts**
- Cap is 10 `[RULE]`. Regen is 1 elixir per 2.8 s normally and 1 per 1.4 s in double elixir `[MEASURED]` (0.355/s and 0.742/s observed vs 0.357/0.714 expected). Double elixir was off at clock 1:11 and on by 0:53 `[MEASURED]`. Overtime rate `[UNVERIFIED]`. Start value 5 `[UNVERIFIED]`.
- Our deck costs: skeletons 1, electro_spirit 1, arrows 3, cannon 3, knight 3, fireball 4, hog_rider 4, musketeer 4. Average 2.875.

**Concepts to teach**
- **Leak:** elixir sits at ~10 and regen is wasted. Every second at 10 is lost value. `[HEURISTIC]` leak if `elixir >= 9.5` for more than 3 s.
- **Starvation:** elixir stays low so we cannot answer a threat. Observed in the recorded loss: mean 2.68, in the 0.7–3.5 band for ~100 s `[MEASURED]`.
- **Storing (banking) is deliberate, leaking is not.** Valid reasons to store: waiting for the opponent to commit before answering; saving for a double-elixir push; holding a spell for value; not having a good card in hand. Storing must have a ceiling and a timer (`bank_until`, §4) so it never turns into a leak.
- **Trade math:** a trade is positive when we spend less elixir than the opponent to neutralise a push (e.g., 1-elixir Skeletons stopping a 4-elixir push = +3). Track a running `elixir_diff = opp_spent_est - our_spent`. Prefer positive trades, avoid negative ones (e.g., a 4-elixir Fireball on one unit).
- **Never spend cheap cards just because elixir is available** when there is no threat — that caused the starvation loop. Spend to answer a threat, to cycle toward the win-con on purpose, or to avoid a leak.

### 1.2 Our hand, our next card, our cycle
- 8-card deck. 4 in hand + 1 shown as "next". A played card goes to the back of the queue and returns to hand after 4 more plays `[RULE]`.
- Code computes `hog_in_hand`, `hog_next`, `plays_until_hog` and passes them. The Brain uses them to time pushes: do not spend the elixir needed for a Hog push on cheap cards when the Hog is 1–2 plays away; cycle with cheap cards when the Hog is far.

### 1.3 Enemy cards seen and their cycle
- The screen does not show the opponent's hand. We **infer** it: each new enemy unit spawn = a card played (cost known once identified, else assumed 3.5 `[HEURISTIC]`).
- Track `opp_cards_seen` (max 8 distinct). Once 8 are known, the opponent's cycle is predictable: the card played most recently is at the back of their queue; the card they will have available soonest is the one played longest ago.
- Teach: "which of their cards counter my Hog / my push, and are those likely in their hand right now?" (`opp_counter_likely_in_hand`).

### 1.4 Opponent elixir prediction
- Code maintains `opp_elixir_est` (start 5, regen as above, minus cost of each seen play).
- Derived flags passed to the Brain:
  - `opp_tapped_out`: `opp_elixir_est < 3.5` `[HEURISTIC]` -> punish window.
  - `opp_can_punish_hog`: estimated elixir >= cost of their likeliest counter.
  - `opp_elixir_trend`: rising / falling over the last 5 s.
- Error grows over time (we may miss plays or mis-assume costs). Reset/clamp estimate when evidence contradicts it (e.g., they play two expensive cards in a row while estimate said they were tapped out).

### 1.5 Threat assessment (defence)
- Per enemy: type (when known), lane, row/distance to our tower, count, whether it targets buildings.
- Teach responses by role: building (cannon) pulls ground win-cons away from towers; cheap swarms/distractions (skeletons, electro_spirit) buy time for tower/musketeer; spells only for >=2 targets or tower finish.
- Teach **timing:** place defence when the attacker crosses the bridge, not early; do not spell before defenders commit; space a freeze/stun and a push-back apart rather than together (see playbook sources in `PLAYBOOK.md` if present).

### 1.6 Offence timing
- Counterpush when: elixir >= 7 and no enemy in our half, OR `opp_tapped_out`, OR we just won a positive trade with units still alive.
- Do not send the Hog into a likely counter when the opponent has elixir and the counter is probably in hand; cycle cheap cards instead and wait (`mode=bait|cycle`).
- Win condition awareness: crowns/towers. Prefer pressuring the tower with lower HP (`tower_hp`).

### 1.7 Game phases
Single elixir -> double elixir (last minute, observed) -> overtime `[UNVERIFIED rates]`. Brain behaviour differs: more storing and patience in single; more cycling and faster pushes in double; defend-first when a tower is low in the last seconds.

### 1.8 Spell value
Spell only if `targets_in_radius >= 2` or `tower_finish`. Observed waste: Fireball (4 elixir) and Arrows (3) each on a single unit in the last minute, leaving the bot at 1–2 elixir `[MEASURED]`. Hard-guarded in code; the Brain's `spell_policy` should agree with it, not fight it.

---

## 2. What to ignore (irrelevant to decisions)

Do **not** put these in the Brain's prompt or let them influence decisions:

| Ignore | Why |
|---|---|
| Opponent name, trophies, clan, emotes/chat | No effect on what to play now |
| Arena art, shadows, water, wall damage, tower decoration, particles | Visual noise; also caused false enemies (red trim on opponent king tower) |
| UI text overlays ("Not enough Elixir!", countdown banners, card-name labels) | They are UI, not units; treat as an error signal for the auditor only |
| Raw pixels / full screenshots (default) | Slow and unnecessary; the Brain gets structured state |
| Unit level numbers on badges | Low value for tactical choice in this prototype (note: could matter for exact trade math later) |
| Exact tile coordinates | Reflex layer chooses tiles; Brain gives intent (lane, mode) |
| Menu/shop/chest state during a battle | Navigator's job, different mode |
| More than the last ~3 plays and ~3 lessons of history | Token cost, little signal |
| Arithmetic (elixir sums, timers) | Computed in code; passed as fields |

---

## 3. Derived features the Brain receives (computed in code)

```jsonc
{
  "phase": "single|double|overtime",
  "clock_sec": 134,
  "elixir": 6.4,
  "ticks_to_leak_s": 5.0,          // (10 - elixir) * regen_period
  "leak_risk": false,               // elixir >= 9.5 for > 3 s
  "starved": false,                 // elixir < 3 for > 8 s  [HEURISTIC]
  "bank_until": null,               // if we are deliberately storing
  "hand": [
    {"slot":1,"card":"hog_rider","cost":4,"role":"win_con","playable":true},
    {"slot":2,"card":"skeletons","cost":1,"role":"cycle","playable":true},
    {"slot":3,"card":"knight","cost":3,"role":"defense","playable":true},
    {"slot":4,"card":"musketeer","cost":4,"role":"defense","playable":true}
  ],
  "next_card": "fireball",
  "plays_until_hog": 0,
  "can_afford_hog_plus_support": true,   // elixir >= 4 + cheapest support
  "enemies": {"our_half": 0, "left": 0, "right": 0, "types": []},
  "threat_level": "none|low|high",       // from lane/distance/count
  "tower_hp": {"player_king":4008,"player_left":2534,"player_right":2425,
               "opp_king":2534,"opp_left":2534,"opp_right":1360},
  "opp_elixir_est": 2.1,
  "opp_tapped_out": true,
  "opp_cards_seen": ["knight","musketeer"],
  "opp_cycle_known": false,
  "opp_counter_likely_in_hand": false,
  "elixir_diff_est": 3.0,                // opp_spent - our_spent
  "last_plays": [{"t":120.4,"card":"skeletons","result":"ok"}],
  "lessons": ["Fireball wasted on single unit at 0:46"]
}
```
Keep the whole prompt under ~400 tokens `[TARGET]`.

---

## 4. Directive fields (extends the brief's schema)

Existing: `mode`, `lane_pref`, `elixir_floor`, `card_priority[]`, `spell_policy{allow,min_targets,allow_tower_finish}`, `aggression`, `ttl_sec`, `reason`.
**Proposed additions (v1.1):**
- `bank_until`: number or null — elixir level to store up to before spending non-emergency cards (clamped 0–9; code forces a spend if held > ~6 s `[HEURISTIC]`, so storing can never become a leak).
- `avoid_cards[]`: cards the Brain wants held this window (e.g., hold Hog while a likely counter is in their hand).

Hard guards in code still win over the Brain (affordability, spell guard, floor).

---

## 5. Decision rubric (the Brain answers in this order)

1. **Leak?** If `leak_risk` -> something must be spent: prefer a counterpush if allowed, else cycle cards toward the Hog.
2. **Threat now?** If `threat_level=high` -> `mode=defend`, `elixir_floor` low (emergency), priority: building/defense, then cheap distraction.
3. **Can the opponent punish?** If `opp_tapped_out` and nothing threatens us -> `counterpush`/high aggression. If opponent has elixir and a counter is likely in hand -> `bait`/`cycle`, `avoid_cards=[hog_rider]`.
4. **Is the win-con ready?** Use `hog_in_hand/next/plays_until_hog` to pick `cycle` vs `counterpush`.
5. **Spell?** Only per the spell policy (>=2 targets or tower finish).
6. **Phase adjustment:** single -> patience/storing; double -> faster cycling; last seconds -> protect the weaker tower.
7. Output ONE Directive, `ttl_sec` 5–10, `reason` <= 140 chars.

---

## 6. Few-shot examples (state -> Directive)

**A. Leak + opponent tapped out**
State: elixir 9.6, hand [hog, skeletons, knight, musketeer], enemies 0, `opp_elixir_est` 2.1, `opp_tapped_out` true.
-> `{"mode":"counterpush","lane_pref":"any","elixir_floor":2,"card_priority":["hog_rider","musketeer"],"spell_policy":{"allow":true,"min_targets":2,"allow_tower_finish":true},"aggression":0.8,"ttl_sec":6,"reason":"Leak risk and opp tapped out: push now"}`

**B. Real threat, low elixir**
State: elixir 3.4, enemy hog right lane crossing bridge, hand [skeletons, electro_spirit, cannon, arrows].
-> `{"mode":"defend","lane_pref":"right","elixir_floor":0,"card_priority":["cannon","skeletons"],"spell_policy":{"allow":false,"min_targets":2,"allow_tower_finish":false},"aggression":0.1,"ttl_sec":5,"reason":"Hog crossing: cannon pull then skeletons"}`

**C. One unit, spell tempting**
State: elixir 5, 1 enemy unit near our tower, hand has fireball.
-> `spell_policy.allow=false` (1 target), `mode=defend`, prefer cheap card. Reason: "Single target: spell would be a negative trade".

**D. Opponent has elixir and likely counter**
State: `opp_elixir_est` 8.7, `opp_counter_likely_in_hand` true, no enemies, hog in hand, elixir 7.
-> `{"mode":"bait","avoid_cards":["hog_rider"],"card_priority":["skeletons","electro_spirit"],"bank_until":8,"aggression":0.3,"ttl_sec":8,"reason":"Opp can punish hog: cycle cheap and wait"}`

**E. Double elixir starts, nothing happening**
State: phase double, elixir 6, no threats, hog next in 2 plays.
-> `mode=cycle`, cheap cards toward Hog, `bank_until=8`, `aggression=0.5`.

---

## 7. Drills / how to test the Brain offline (no live game)

Replay recorded `GameState` logs (we have the 143 s match; add more) and check:
1. **Schema validity:** 100% parseable Directives; ranges clamped.
2. **Legality:** never references a card not in the hand; spell policy never allows a single-target spell.
3. **Leak drill:** feed states with `elixir>=9.5`; Directive must lead to a spend within its TTL.
4. **Starvation drill:** feed states with elixir <3 and no threat; Directive must not spend cheap cards (`elixir_floor` raised).
5. **Opponent-tapped-out drill:** Directive must be offensive when the flag is true and no threat.
6. **Latency drill:** Qwen timeout/garbage -> default Directive, tap latency unchanged (see brief §7).
7. **Stability:** same state twice -> materially the same Directive (low temperature).
Report pass rate per drill. A drill failing means fix the prompt/features, not hand-patch the output.

Auditor metrics to track across matches: `leak_sec`, `starved_sec`, `spell_waste_count`, `unaffordable_clicks`, `slot_usage[1..4]`, `elixir_diff_est` at 1:00 and end, tower damage dealt/taken.

---

## 8. Data to log now (for fine-tuning later)

Per decision tick (10–15 Hz) or per Directive: `GameState` (the features above), the Directive issued, the action taken, and an outcome label filled later (e.g., "positive trade", "leak", "wasted spell", "tower damage within 10 s"). Store as JSONL per match with the match result. Do not store raw video by default; keep a short clip only around flagged events. Fine-tuning is a later decision: first measure whether prompt + features + lessons already moves the metrics.

---

## 9. Curriculum order (build and verify in this order)

1. **Tables and costs** loaded from one source of truth (card cost/role table).
2. **Elixir accounting in code:** leak/starve timers, `ticks_to_leak_s`, `elixir_diff_est` — unit-tested.
3. **Cycle tracker in code:** `plays_until_hog`, own queue; opponent `opp_cards_seen` and cycle once 8 known — unit-tested.
4. **Opponent elixir + flags** (`opp_tapped_out`, `opp_can_punish_hog`).
5. **Prompt + rubric + few-shot**, then the §7 drills.
6. **Playbook macros** (combos with delays, e.g., defence sequences) — Brain picks, Reflex executes.
7. **Lessons loop** from the auditor.
Only after 1–7 are stable and logged for many matches: consider imitation/fine-tuning.

---

## 10. Not yet known (do not assume)

- Overtime elixir rate and exact double-elixir start.
- Qwen's real latency/JSON reliability on our hardware.
- How accurate `opp_elixir_est` is in practice (no ground truth yet; it drifts if a play is missed or a cost is wrong).
- Whether unit level badges matter enough to include.
- The 1-match dataset is far too small to validate any threshold marked `[HEURISTIC]`; treat them as starting values to tune with logs.
