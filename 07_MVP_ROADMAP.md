# 07 — MVP Roadmap

## Milestone 0: Project setup
Deliverables:
- organized project folders
- media manifest format
- dataset schema
- one sample video processed manually
- documented game version/mode

## Milestone 1: Video understanding prototype
Build:
- local video loader
- frame extraction with timestamps
- video preview and frame stepping
- annotation timeline
- manual event editor

Success condition: a human can annotate a complete match without losing timestamp accuracy.

## Milestone 2: Basic automatic perception
Add:
- game-screen detection
- card/deployment recognition
- arena/lane localization
- timer and visible elixir reading
- confidence and unknown labels

Success condition: automatic detections reduce annotation effort without hiding uncertainty.

## Milestone 3: Audio and commentary
Add:
- audio extraction
- timestamped speech transcription
- segment classification
- manual transcript correction

Success condition: commentary is available as timestamped segments.

## Milestone 4: Expert reasoning dataset
Add:
- commentary-to-event alignment
- strategic concept labels
- explicit expert claim vs normalized interpretation
- outcome linkage
- human review workflow

Success condition: a reviewer can inspect why the model believes a creator explained a particular decision.

## Milestone 5: Elixir and cycle engine
Add:
- versioned card metadata
- elixir accounting
- uncertainty ranges
- card-cycle hypotheses
- confidence calibration

Success condition: estimates are evaluated against manually verified examples.

## Milestone 6: Replay coach
Add:
- match summary
- key decision points
- elixir/cycle explanation
- counter and prediction analysis
- alternative decisions
- evidence-linked feedback

Success condition: strong players judge the feedback useful, grounded, and not misleading.

## Suggested folder layout

```text
CLASH_IQ/
  docs/
  data/
    manifests/
    annotations/
    transcripts/
    derived/
  media/
    source/          # private, do not commit
  frames/            # generated, do not commit
  models/
  src/
    ingest/
    vision/
    audio/
    alignment/
    game_state/
    strategy/
    coach/
  evaluations/
  scripts/
```

## First coding task

Do not start by training a neural network. Build the video ingestion + timestamped frame extraction + manual event annotation prototype. It creates the foundation and gives us trustworthy examples for later model development.
