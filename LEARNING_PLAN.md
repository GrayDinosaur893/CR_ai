# CLASH IQ — Unified Learning Plan: least-elixir answers + best placement + elixir management, learned together

**Companions:** `ARCHITECTURE_BRIEF.md`, `BRAIN_CURRICULUM.md`. Audience: coding agent (Gemini).
**Tags:** `[RULE]` game rule/source-backed, `[HEURISTIC]` starting value to tune, `[UNVERIFIED]` needs a test, `[CONFLICT]` sources disagree.

---

## 0. The idea in one paragraph

Match win/loss is far too sparse to learn from (one label per ~3 minutes). Instead, cut every match into **exchanges** (one enemy push + our response). Each exchange yields one row in an **Exchange Ledger** with: what the enemy spent, what we spent (including rescue cards), where we placed, how much tower HP we lost, and what our elixir looked like around it. From that single ledger we learn all three skills at once:

| Skill | What it reads from the ledger | What gets learned |
|---|---|---|
| Least-elixir answer | enemy_cost vs our_cost per threat type | A **counter table**: cheapest answer that actually worked per threat |
| Best placement | placement offsets + exchange score | A **placement table**: best tile offset per (card, threat, lane) |
| Elixir management | leak/starve time, reserve violations, elixir_diff | Tuned **management parameters** (reserve, bank ceiling) |

One logging schema, one scoreboard, three tables. A typical match gives roughly 10–20 exchanges `[HEURISTIC estimate]`, i.e. 10–20x more labelled data than win/loss.

Nothing here trains neural weights. It builds tables and parameters that the Reflex layer and the Brain's derived features use.

---

## 1. Research findings this plan is built on

**Trade math** `[RULE]`
- A positive trade = we spend less elixir than the opponent to neutralise the same threat; count the *whole* exchange including the rescue card (6 vs 4 = +2; add a needless 2-elixir card -> 0; need another 3 to finish -> -1).
- It is a **spending comparison, not a reading of the opponent's bar**: earlier spending and elixir regenerated during the exchange also matter. So never treat `+1` as "they have 1 less elixir than me".
- Tower HP is a resource too: spending 4 elixir to stop a lone chip attacker can be a negative trade.
- Keep a 2–3 elixir reserve outside final pushes; track opponent elixir; punish a big commitment by pushing the other lane.

**Placement** `[CONFLICT]`
- Cannon vs Hog Rider: sources give **4 tiles from the river and 2 from the tower** (4-2), others **4-3**, others "4 tiles, centre". 4-3 is quoted for Giant/Golem. Pull only works if the building-targeter has travelled <= ~2 tiles past the bridge.
- Because they disagree, placements are **priors to be tested**, not constants.

**How other AI projects reward this**
- Royale-RL suggests rewarding elixir advantage and weighting tower damage by troop value.
- A newer RL project (Naris404 issue #41) **removed** troop-trade-value, kill/loss and elixir-leak reward terms and moved to outcome reward plus potential-based crown/tower-HP shaping, because such terms can change the optimal policy. Lesson for us: elixir terms are good **diagnostics and low-level signals**, but must be validated against real outcome metrics (tower HP difference, crowns, wins) so the bot cannot "game" them (e.g., hoard elixir and never defend).

Sources are listed in §10.

---

## 2. Exchange Ledger (the core data structure)

### 2.1 Exchange lifecycle
- **Open:** first new enemy unit track (badge detector) that enters our half or crosses the bridge, or a new enemy spawn near our side. A new push in the other lane opens a separate exchange.
- **Attach:** every enemy spawn in the same lane within the window; every our-play in that lane during the window.
- **Close:** all attached enemy tracks gone, OR tower took damage and threat gone, OR 12 s timeout `[HEURISTIC]`.

### 2.2 Row schema (JSONL, one per exchange)
```jsonc
{
  "match_id":"20261111_100611", "ex_id":7, "t_open":58.2, "t_close":64.9, "lane":"right",
  "phase":"single|double|overtime",
  "enemy_cards":[{"card":"hog_rider","cost":4,"cost_known":true}],
  "enemy_cost":4, "enemy_cost_known_all":true,
  "our_plays":[{"t":59.0,"card":"cannon","cost":3,"role":"building",
                "dy_river":4, "dx_tower":2, "tile":[12,21], "slot":3},
               {"t":60.1,"card":"skeletons","cost":1,"role":"cycle","dy_river":3,"dx_tower":1}],
  "our_cost":4,
  "tower_dmg_frac":0.00,            // HP lost / max HP of that tower
  "neutralized":true,
  "elixir_before":6.8, "elixir_after":2.4,
  "reserve_violation":false,        // spent below reserve with no real threat
  "score": 0.0,                     // see 2.3
  "class":"even"                    // positive/even/negative
}
```
Placement is stored **relative to the river and the defending tower** (`dy_river`, `dx_tower` in tiles), not absolute tiles, so priors transfer. `[UNVERIFIED]` needs a calibration step to find our grid's river row and tower column (place a Cannon at a known tile and read where it lands).

### 2.3 Score
```
trade      = enemy_cost - our_cost
score      = trade - K * tower_dmg_frac           # K = 8 elixir per full tower  [HEURISTIC]
class      = positive if score > 0.5, negative if score < -0.5, else even
```
- Learn only from rows with `enemy_cost_known_all = true` at first. Enemy cost is often unknown until enemy card identification exists (see Brain brief); rows with assumed costs are logged but down-weighted `[HEURISTIC 0.3]`.
- Leak/starve are **not** in `score`; they are managed separately (§5) so one number does not hide another.

---

## 3. Skill 1 — Least-elixir answers (Counter Table)

**Structure:** `counter_table[threat_signature] -> ranked answers`, where threat_signature = (enemy card or role, lane, enemy_cost bucket) and an answer = ordered card list with delays, e.g. `[skeletons@0.0s, electro_spirit@0.4s]`.

**Seed (priors, from sources)** — examples to be tested, not trusted: Hog Rider -> Cannon (4-2/4-3 area) plus a cheap distraction; swarm -> Arrows/Fireball only if >= 2 targets; Skeletons first then a freeze/stun card against Hog (our deck has Electro Spirit instead of Ice Spirit, effect differs `[UNVERIFIED]`).

**Learning:** for each signature keep stats per answer: n, mean score, variance, tower damage. Prefer the **lowest-cost answer whose mean score is not worse than the next one by more than noise**. A cheaper answer that leaks tower damage loses automatically through `K * tower_dmg_frac`.

**Rules:**
- Minimum samples before an answer can replace the prior: **n >= 5** `[HEURISTIC]`.
- Spells never explored (too costly to be wrong); they stay behind the spell guard.

---

## 4. Skill 2 — Best placement (Placement Table)

**Arms:** for each (card, threat, lane): a small grid of offsets around the prior, e.g. for a Cannon vs Hog: `dy_river ∈ {3,4,5}` × `dx_tower ∈ {1,2,3}`. Start with **3 prior arms** (4-2, 4-3, centre-4) rather than 9 so data is not spread thin.

**Learning method: contextual bandit** (Thompson sampling on mean exchange score, per arm). Context = (card, threat type, lane). Reward = `score` of the exchange the play belonged to (use only exchanges where this card was the main answer).

**Exploration is constrained (safety):**
- Explore only when `our tower HP >= 70%` `[HEURISTIC]`, threat is low-cost (`enemy_cost <= 4`), and elixir >= card cost + reserve.
- Exploration rate starts ~0.2, decays to ~0.05 after n >= 8 per arm.
- Never explore with spells, never in the last 30 s, never when a tower is under 40% HP.
- Otherwise always play the current best arm.

**Expected pace:** with 3 arms and ~8 samples each, one threat/card pair needs ~24 relevant exchanges. If a Hog defence happens 2–4 times per match, that is on the order of **6–12 matches** for the first stable answer `[HEURISTIC estimate]`; other pairs take longer. Prior knowledge is what makes this feasible.

**Also log, do not optimise yet:** time from placement to first hit, whether the enemy retargeted the building (position change toward it).

---

## 5. Skill 3 — Elixir management (parameters + hard constraints)

**Hard constraints (always on):**
- No click when `cost > elixir` (unaffordable-click = 0).
- Reserve: keep `reserve_floor` (start 2–3) unless a real threat exists or we are in a deliberate final push. `[RULE guide]`
- Leak guard: `elixir >= 9.5` for > 3 s forces a spend (counterpush or cycle toward the win-con).
- Bank guard: deliberate banking (`bank_until`) is capped and forced to spend after ~6 s.

**Tunable parameters** (few, so tuning is possible with little data):
`reserve_floor ∈ {2,3,4}`, `bank_until ∈ {7, 8.5}`, `counterpush_elixir ∈ {6.5, 7.5}`.

**Tuning protocol:** alternate settings in blocks (A,B,A,B…) of 3–5 matches to cancel meta/opponent drift. Compare on **process metrics**: `leak_sec`, `starved_sec`, `reserve_violations`, `unaffordable_clicks`, `elixir_diff_est` at 1:00 and at the end, **and** outcome sanity metrics: tower HP difference and crowns. Win/loss alone is too noisy at this sample size; use it only as a sanity check.

---

## 6. How the three learn "simultaneously"

1. One pass per match: Eye/Reflex logs events -> **Ledger builder** closes exchanges -> writes rows.
2. After each match (batch, offline, cheap):
   - update `counter_table` stats,
   - update `placement_table` (bandit posteriors),
   - compute management metrics and log them against the active parameter setting.
3. Next match loads the updated tables. The Reflex layer reads the placement table; the Brain (Qwen) receives **derived features** like `best_known_answer[threat]`, `answer_cost`, `answer_confidence` — it does not learn tiles itself.
4. The three tables share the ledger, so a bad placement shows up as extra rescue cost (our_cost up) and tower damage — it correctly hurts all three views at once.

---

## 7. Anti-Goodhart safeguards (so the bot cannot game the score)

- Score includes tower damage, so "never defend to avoid spending" scores badly.
- Management metrics include `starved_sec` and reserve violations, so "hoard everything" is penalised.
- **Outcome check every N matches:** if exchange scores improve but tower-HP difference or crowns do not, the score definition is wrong — stop and review (this is the lesson from the RL project that dropped elixir reward terms).
- Keep the prior tables read-only as a fallback; a learned entry replaces a prior only after n >= 5 and a better mean by more than one standard error.
- Version every table; keep a `rollback` copy.

---

## 8. Phases and acceptance tests

| Phase | Work | Pass criteria |
|---|---|---|
| P0 | Tile-grid calibration (`river_row`, tower col) and placement in river/tower-relative coords | A Cannon placed at (dy=4, dx=2) lands where predicted on 5/5 tests |
| P1 | Ledger builder on the recorded match (observe-only) | Produces plausible exchanges; hand-check 10 exchanges against the video |
| P2 | Counter/placement tables seeded from priors; Reflex reads placement table | Bot plays prior placements; unaffordable clicks = 0; log complete |
| P3 | Management guards + metrics dashboard | Leak/starve/reserve metrics reported per match |
| P4 | Constrained bandit exploration on placement (alt account) | Exploration only in safe states (verified by log); no tower-HP regression vs P3 |
| P5 | Management parameter block-tuning | At least 4 alternating blocks completed; process metrics compared |
| P6 | Brain gets best-answer features | Qwen-off test still passes (latency unchanged) |
| P7 | Outcome check (§7) | Exchange-score gains confirmed by tower-HP difference |

---

## 9. Dependencies and honest limits

- **Enemy card identification is required** for trustworthy `enemy_cost`. Until built, most rows are "assumed cost" and learn slowly.
- Exchange attribution (which of our plays belonged to which push) is heuristic; start with lane + time window and review manually.
- No simulator available here: learning is **live-only**, so pace is bounded by matches played (a match plus menus is several minutes).
- Sources disagree on placement; the bandit is the arbiter. Results from one opponent skill level may not transfer.
- Small data: thresholds marked `[HEURISTIC]` are starting values, not facts.
- Test on an alt account; botting may violate the game's terms.

---

## 10. Sources consulted
- https://www.royaletools.com/blog/positive-elixir-trades-clash-royale (whole-exchange accounting, spending vs bar)
- https://clashdecks.com/guides/elixir-management-101 (trade examples, reserve)
- https://www.lootbar.com/blog/en/clash-royale-elixir-guide-prevent-running-out-during-pushes.html (2–3 elixir reserve, punish commitments)
- https://oboe.com/learn/clash-royale-strategy-essentials-bg6h2z/elixir-management-7zch2t (tower HP as a resource)
- https://theriagames.com/guide/clash-royale-cannon/ and https://www.clasher.us/clash-royale/card/Cannon (cannon placement, pull limit)
- https://clash.world/deck-building/hog-cycle-guide/ and https://clashdecks.com/guides/decks/hog-rider-decks (4-2 vs centre placement)
- https://github.com/Naris404/clash_royale_rl_agent/issues/41 (outcome reward + potential-based shaping; elixir-leak/trade terms removed)
- https://github.com/0xCyberstan/Royale-RL-A-Reinforcement-Learning-Agent (elixir-advantage reward idea)
