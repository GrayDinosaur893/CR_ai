# 04 — Dataset Schema

Use JSONL for initial annotation exports: one JSON object per line. Keep source media outside the annotation files and refer to it by a stable media ID.

## 1. Media manifest

```json
{
  "media_id": "match_0001",
  "source_file": "private_local_file.mp4",
  "duration_seconds": 312.4,
  "fps": 60,
  "width": 1920,
  "height": 1080,
  "has_audio": true,
  "game_mode": "unknown",
  "game_version": "unknown",
  "recorded_at": null,
  "processing_version": "0.1.0"
}
```

## 2. Visual observation

```json
{
  "observation_id": "obs_000001",
  "media_id": "match_0001",
  "timestamp": 24.5,
  "frame_reference": "frames/match_0001/000024500.png",
  "type": "card_deployment",
  "value": {
    "player": "opponent",
    "card": "hog_rider",
    "lane": "right",
    "position_normalized": {"x": 0.78, "y": 0.41}
  },
  "status": "observed",
  "confidence": 0.94,
  "review_status": "unreviewed"
}
```

## 3. Game event

```json
{
  "event_id": "evt_000001",
  "media_id": "match_0001",
  "start_seconds": 24.3,
  "end_seconds": 24.8,
  "event_type": "card_deployed",
  "player": "opponent",
  "card_id": "hog_rider",
  "lane": "right",
  "position": {"x": 0.78, "y": 0.41},
  "elixir_cost": 4,
  "evidence_observation_ids": ["obs_000001"],
  "status": "observed",
  "confidence": 0.94
}
```

## 4. Transcript segment

```json
{
  "segment_id": "aud_000001",
  "media_id": "match_0001",
  "start_seconds": 25.1,
  "end_seconds": 28.7,
  "text": "He does not have his building in cycle.",
  "speaker": "commentator",
  "transcription_confidence": 0.91,
  "review_status": "unreviewed"
}
```

## 5. Commentary alignment

```json
{
  "alignment_id": "align_000001",
  "segment_id": "aud_000001",
  "related_event_ids": ["evt_000001"],
  "related_time_range": {"start": 20.0, "end": 30.0},
  "relation": "explains",
  "confidence": 0.86,
  "review_status": "unreviewed"
}
```

Allowed relation values initially:
- `describes`
- `explains`
- `predicts`
- `evaluates`
- `general_principle`
- `unclear`

## 6. Strategic annotation

```json
{
  "annotation_id": "strat_000001",
  "media_id": "match_0001",
  "decision_timestamp": 24.5,
  "concepts": ["cycle_advantage", "offensive_opportunity"],
  "expert_claim": "The opponent's building is out of cycle.",
  "normalized_reasoning": {
    "state_evidence": [
      "The opponent previously played the building.",
      "The required number of cycle plays has not yet been observed."
    ],
    "interpretation": "The building is likely unavailable at this moment.",
    "recommended_action": "Consider applying pressure with the win condition.",
    "risks": [
      "The cycle reconstruction may be incomplete.",
      "The opponent may have another defensive answer."
    ]
  },
  "outcome": {
    "observed": "unknown",
    "prediction_correct": null
  },
  "status": "inferred",
  "confidence": 0.78,
  "review_status": "needs_human_review"
}
```

## 7. State snapshot

```json
{
  "state_id": "state_000001",
  "media_id": "match_0001",
  "timestamp": 24.5,
  "player_state": {
    "visible_elixir": null,
    "estimated_elixir": {"min": 5, "max": 7},
    "known_played_cards": ["hog_rider"],
    "possible_hand": [],
    "confidence": 0.55
  },
  "opponent_state": {
    "visible_elixir": null,
    "estimated_elixir": {"min": 4, "max": 6},
    "known_played_cards": ["building_card"],
    "possible_hand": [],
    "confidence": 0.48
  },
  "active_entities": [],
  "tower_state": [],
  "assumptions": ["Illustrative example only; replace with verified game rules."]
}
```

## 8. General schema rules

- Use seconds from the start of the video as the canonical time.
- Store match clock separately from video timestamp.
- Use stable IDs; do not use array position as identity.
- Normalize arena coordinates to 0–1 and document the coordinate convention.
- `observed`, `inferred`, and `predicted` are different statuses.
- Confidence is not probability unless calibrated as such. Document what it means.
- Unknown values should be `null` or an explicit `unknown`, not guessed.
- Keep original transcript text separate from normalized paraphrase.
- Every strategic annotation should point to evidence and the relevant state snapshot.
