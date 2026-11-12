# 06 — Training and Evaluation Plan

## Principle

Do not fine-tune a model on raw creator videos as the first step. First build a small, high-quality, human-reviewed dataset. Model quality depends on accurate labels and reliable state reconstruction.

## Stage A — Gameplay perception

Input: sampled frames or short video clips.

Tasks:
- identify game screen and match phase
- detect card deployments
- recognize cards
- detect player side/lane/location
- read timer and visible elixir
- track major entities and tower outcomes

Evaluation:
- card recognition accuracy
- deployment timestamp error
- player-side accuracy
- lane/location error
- elixir OCR accuracy
- false-positive and missed-event rates

Use a held-out set of matches that were not used during development.

## Stage B — Audio understanding

Input: match audio.

Tasks:
- timestamped transcription
- segmentation into commentary statements
- classify statement type
- preserve uncertainty

Evaluation:
- word error rate or human transcript agreement
- timestamp alignment error
- commentary-type classification accuracy

## Stage C — Cross-modal alignment

Input: visual events and audio segments.

Tasks:
- link statements to events/state ranges
- identify whether commentary describes, explains, predicts, or evaluates
- handle delayed references

Evaluation:
- human-rated alignment precision/recall
- percentage of claims linked to correct state
- ambiguity detection rate

## Stage D — Game-state reconstruction

Tasks:
- reconstruct card-play order
- estimate elixir ranges
- track card-cycle hypotheses
- maintain active units and tower state

Evaluation:
- compare estimates with directly visible values when available
- measure range coverage: does the true value fall within the predicted range?
- measure cycle-state accuracy on manually verified sequences
- test robustness when events are missed

## Stage E — Strategic reasoning

Tasks:
- extract expert claims and reasoning
- identify strategic concepts
- generate alternative lines
- evaluate predictions against outcomes

Evaluation:
- expert/strong-player review
- evidence-grounding score
- calibration of confidence
- factual error rate
- hindsight-bias checks
- usefulness of coaching feedback

## Data splitting

Split by complete match/video, not by adjacent frames. Otherwise near-identical frames from one match can leak into both training and test sets.

If multiple videos come from the same match, keep them in the same split. Track creator/source and game version to identify distribution bias.

## Baselines

Before training custom models, establish baselines:
- manual annotations
- pretrained vision model
- OCR/template matching for stable UI elements
- pretrained speech recognition
- rule-based elixir/cycle calculator
- language model for structured commentary extraction

Only fine-tune a component if evaluation shows a clear, measurable need and there is enough reviewed data.

## Avoid these failure modes

- Treating creator commentary as always correct.
- Training on auto-generated labels without human checks.
- Confusing correlation with causal reasoning.
- Using final match outcome as the only measure of decision quality.
- Reporting exact hidden elixir or hand information.
- Randomly splitting frames from the same video across train/test.
- Evaluating only on one creator's deck, opponent, or play style.
