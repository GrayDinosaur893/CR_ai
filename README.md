# CLASH IQ — Project Foundation

**Purpose:** Build a Clash Royale gameplay-understanding and coaching system. The first milestone is not to play the game. It is to reliably understand recorded gameplay, then connect expert commentary to the decisions and game states it describes.

## Core philosophy

1. **Observe before judging.** The system must first identify what happened in the match.
2. **Separate facts from estimates.** Visible events, inferred hidden state, and uncertain predictions must never be mixed.
3. **Learn concepts, not quotes.** Expert commentary is evidence of strategic reasoning, not unquestionable ground truth.
4. **Explain every conclusion.** A coaching output must show the evidence, assumptions, alternatives, and confidence.
5. **Start with offline analysis.** Initial development uses recorded videos and manually reviewed annotations, not live-match assistance or automated gameplay.

## First milestone

Given a recorded match video, produce a timestamped event timeline containing:
- match clock and overtime state
- player/opponent identity and lane
- card deployments, troop/building/spell observations
- visible elixir values when readable
- tower damage/destruction and other important outcomes
- uncertainty and confidence for every visual claim

Second milestone: align commentary/audio segments with the corresponding game states and extract strategic concepts such as elixir counting, card cycle, predictions, counters, attack timing, and defensive efficiency.

## Documentation map

- `01_PRODUCT_REQUIREMENTS.md` — what CLASH IQ must do and must not do
- `02_DOMAIN_CONTEXT.md` — Clash Royale concepts the system must understand
- `03_SYSTEM_ARCHITECTURE.md` — proposed components and data flow
- `04_DATASET_SCHEMA.md` — event and commentary annotation formats
- `05_ANNOTATION_GUIDELINES.md` — how to label videos consistently
- `06_TRAINING_AND_EVALUATION.md` — learning stages and quality checks
- `07_MVP_ROADMAP.md` — implementation order
- `08_EXPERT_COMMENTARY_GUIDE.md` — how to use expert videos responsibly

## Important constraints

- Do not assume hidden information is visible. Opponent elixir, hand, and cycle may need to be estimated from observed events and game rules.
- Keep `observed`, `inferred`, and `predicted` values distinct.
- Every estimate must include a confidence or uncertainty range.
- Do not treat a commentator's statement as fact until it is aligned with the game state and checked where possible.
- Use videos and datasets only where you have permission and comply with applicable platform terms and copyright rules. Keep source media private; store derived annotations separately.
- The first version is an offline replay-analysis and learning tool. It should not automate gameplay or provide prohibited assistance in live competitive matches.
