# 01 — Product Requirements

## 1. Product vision

CLASH IQ is a personal Clash Royale analysis and learning system. It should understand a recorded match at the level of game events, reconstruct the strategic state as accurately as possible, learn from expert explanations, and teach the user why a decision was strong or weak.

The system is intended for a player who already understands the basics and wants advanced improvement: elixir counting, prediction, counters, card-cycle tracking, matchup decisions, and timing.

## 2. Primary user outcomes

The user should be able to:
1. Upload or select a recorded gameplay video.
2. Receive a timestamped, editable match timeline.
3. Inspect what cards were played, where, and when.
4. See visible elixir values and estimated elixir changes.
5. See the inferred card cycle and likely available responses, with uncertainty.
6. Attach expert commentary to the state it refers to.
7. Read a clear explanation of the strategic idea.
8. Review a match and identify decision points, alternatives, and lessons.
9. Correct incorrect detections so the dataset improves.

## 3. Functional requirements

### FR-1: Video ingestion
- Accept a local video file in a documented set of formats.
- Read video duration, frame rate, dimensions, and audio availability.
- Preserve original timestamps.
- Allow sampling frames at configurable intervals and around detected changes.
- Never silently alter source media.

### FR-2: Visual game understanding
Detect or estimate:
- match clock, overtime, and match end
- arena boundaries, lanes, river, bridges, towers
- card identity when a card is played or visible
- deployment side and approximate deployment location
- troops, buildings, and spells on the arena
- troop movement and combat interactions
- tower damage and destruction
- visible elixir bar/value, if legible
- UI state changes relevant to match interpretation

Every detection must include:
- timestamp or time interval
- label
- value
- confidence
- evidence reference (frame/time)
- status: `observed`, `inferred`, or `predicted`

### FR-3: Event timeline
Convert frame-level observations into match events:
- card deployed
- troop/building/spell appeared
- troop eliminated or disappeared (only when supported by evidence)
- tower attacked/damaged/destroyed
- elixir value observed
- match phase changed
- ambiguous/unknown event

Avoid duplicate events from repeated frames. Events should have start time, end time where applicable, and source evidence.

### FR-4: Game-state reconstruction
Maintain a time-indexed state:
- both players' known played cards
- known/estimated elixir
- known/possible card cycle
- active units and approximate positions
- tower health/damage state where observable
- current match phase
- uncertainty for hidden or ambiguous information

Do not claim exact opponent elixir or exact hidden hand unless directly observable from a reliable source.

### FR-5: Commentary understanding
- Transcribe audio with timestamps.
- Segment speech into meaningful statements.
- Identify whether a segment is describing an observation, prediction, explanation, evaluation, or general principle.
- Link statements to one or more game-state timestamps.
- Extract concepts such as elixir advantage, cycle advantage, counter availability, prediction, pressure, overcommitment, and punish opportunity.
- Preserve the original transcript and a normalized interpretation separately.

### FR-6: Strategic reasoning representation
For each strategic claim, store:
- game state at the decision point
- expert's stated reasoning
- recommended or chosen action, if clear
- expected opponent responses
- risks and alternatives
- observed outcome
- confidence and unresolved ambiguity

### FR-7: Replay coaching
Produce post-match lessons:
- key decision points
- missed opportunities
- inefficient trades
- overcommitments
- defensive mistakes
- cycle mistakes
- successful predictions
- alternative lines and why they may be better

The coach must distinguish hindsight from what was knowable at the time.

### FR-8: Human correction
Provide a way to correct:
- card identity
- timestamp
- player side
- lane/position
- event type
- transcript
- commentary-to-event alignment
- strategic label

Keep correction history so model errors can be measured.

## 4. Non-functional requirements

- Reliability is more important than flashy real-time output.
- All outputs must be traceable to frames, audio segments, or explicit game rules.
- Store confidence values and uncertainty.
- Support modular replacement of vision, speech, and reasoning models.
- Keep raw video, extracted frames, transcripts, and annotations organized separately.
- Make processing reproducible: record model/version/configuration used.
- Design for local/offline processing where practical.

## 5. Non-goals for the first version

- Playing matches automatically.
- Sending actions to the game client.
- Giving live competitive-match move instructions.
- Claiming perfect knowledge of hidden opponent information.
- Training a foundation model from scratch.
- Treating one creator's opinions as universally correct.

## 6. Acceptance criteria for MVP

A match is considered successfully processed when:
- a human can follow the generated timeline and correct it;
- major card deployments are detected with timestamps;
- uncertain detections are marked as uncertain rather than fabricated;
- audio transcript segments have timestamps;
- at least some commentary segments can be linked to relevant game states;
- the final coaching report cites the evidence used for each claim.
