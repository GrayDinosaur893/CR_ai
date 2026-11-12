# 03 — Proposed System Architecture

## Overview

Use a modular pipeline. Do not build one giant model that is expected to solve everything at once.

```text
Recorded video
   |
   +--> Video metadata + frame sampler
   |        |
   |        +--> Vision detection / tracking
   |                  |
   |                  +--> Visual observations
   |
   +--> Audio extraction
            |
            +--> Speech-to-text with timestamps
                       |
                       +--> Commentary segments

Visual observations + commentary segments
                 |
                 v
       Temporal alignment layer
                 |
                 v
       Game-state reconstruction
                 |
                 v
       Strategic concept extraction
                 |
                 v
       Reviewed dataset + replay coach
```

## Components

### A. Media ingestion
Responsibilities:
- validate file and metadata
- extract audio without changing source
- sample frames
- maintain exact timestamps
- create a processing manifest

### B. Vision pipeline
Responsibilities:
- detect game UI and arena regions
- recognize card artwork/UI icons
- detect deployments and approximate locations
- track visible entities over time
- read visible elixir and timer when possible
- emit observations with confidence and frame references

Start with a hybrid approach: image processing and pretrained vision models, plus a human correction interface. Do not assume a general vision model can reliably identify every card from tiny or animated frames.

### C. Audio pipeline
Responsibilities:
- extract audio
- transcribe speech with word/segment timestamps
- identify speaker if useful and permitted
- preserve uncertain words
- split transcript into meaningful commentary segments

### D. Temporal alignment
Responsibilities:
- connect speech segments to nearby or referenced game events
- handle commentary that describes an earlier event or predicts a future event
- allow one segment to refer to multiple events
- allow events with no commentary
- store alignment confidence and human corrections

Simple nearest-timestamp matching is only a first draft. Natural commentary may lag behind the event or discuss a broader sequence.

### E. Game-state engine
Responsibilities:
- combine observations into a timeline
- track card plays and cycle hypotheses
- estimate elixir using versioned rules
- track active units and tower state
- maintain uncertainty and competing hypotheses

### F. Strategy interpretation
Responsibilities:
- classify commentary intent: observation, prediction, explanation, evaluation, general principle
- extract concepts and causal reasoning
- map statements to game-state evidence
- separate explicit expert claims from model-generated interpretations

### G. Dataset store
Keep distinct collections/tables for:
- media manifest
- frame observations
- game events
- transcript segments
- commentary-to-event alignments
- state snapshots
- strategic annotations
- human corrections
- model evaluation results

### H. Replay coach
Responsibilities:
- summarize match
- identify decision points
- explain likely elixir/cycle situation
- show predictions and outcomes
- compare alternatives
- explain uncertainty
- produce evidence-linked lessons

## Data flow principle

Never overwrite raw observations with inferred conclusions. Keep:
- raw observation
- normalized event
- reconstructed state
- strategic interpretation
as separate layers.

## Recommended implementation sequence

1. Build media ingestion and timestamped frame extraction.
2. Build manual event annotation and timeline viewer.
3. Add automated card/deployment detection.
4. Add audio transcription.
5. Add commentary alignment.
6. Add elixir and cycle state reconstruction.
7. Add strategic concept extraction.
8. Add replay coaching and evaluation.
