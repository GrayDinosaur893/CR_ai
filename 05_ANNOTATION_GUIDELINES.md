# 05 — Annotation Guidelines

## Goal

Create a consistent dataset that teaches the system what happened, what an expert said, what the expert meant, and what evidence supports the interpretation.

## 1. Label facts before interpretations

For each moment, annotate in this order:
1. visible facts
2. game events
3. reconstructed state
4. commentary segment
5. strategic interpretation
6. outcome

Do not begin by writing what you think the player should have done.

## 2. Observed vs inferred vs predicted

- `observed`: directly visible or clearly audible in the recording.
- `inferred`: derived from observations plus known game rules.
- `predicted`: a statement about a future event/action.

Examples:
- A card visibly deployed: observed.
- Opponent likely has 4–6 elixir: inferred.
- Opponent will probably use a building: predicted.

## 3. Card and troop annotation

- Use a canonical card ID from a versioned card catalogue.
- If artwork is unclear, use `unknown_card` and add candidate IDs.
- Distinguish card deployment from unit appearance if the animation obscures the exact deployment time.
- Record player side and lane only when supported.
- Do not infer a card from a unit effect alone if multiple cards could explain it.

## 4. Elixir annotation

- Read visible elixir only when the UI is sufficiently clear.
- Record the frame reference and confidence.
- For estimated elixir, record a range and assumptions.
- Include known card costs and applicable regeneration rules.
- Do not force estimates to an exact integer when the timing or initial value is uncertain.

## 5. Cycle annotation

- Record observed card plays in order.
- Track uncertainty if a card deployment is missed or misidentified.
- Distinguish “not currently in hand” from “not yet back in cycle.”
- Do not claim a counter is unavailable unless the cycle evidence supports it.

## 6. Commentary annotation

Transcribe the actual words as accurately as possible. Mark unintelligible spans rather than inventing words.

Classify each segment:
- `observation`: describes current or past visible state
- `prediction`: forecasts an opponent action or outcome
- `explanation`: gives a reason for a decision
- `evaluation`: judges a play or trade
- `general_principle`: teaches a reusable concept
- `reaction/filler`: does not convey strategic information
- `unclear`: cannot confidently classify

One sentence can have more than one function. Split segments where practical.

## 7. Align commentary to the correct moment

Commentary may:
- describe an event that just happened
- explain a decision made several seconds earlier
- predict an event that has not happened yet
- refer to a broader sequence

Use a time range and relation label. Do not simply attach every sentence to the nearest frame.

## 8. Strategic interpretation

Store the commentator's explicit claim separately from the annotator/model's interpretation.

For each interpretation ask:
- What evidence supports it?
- Which game rules are assumed?
- What alternative explanations exist?
- What information was unavailable at the time?
- What later outcome can validate or challenge the claim?

## 9. Disagreement and uncertainty

If annotators disagree:
- retain both interpretations
- mark the record as disputed
- record the reason for disagreement
- request review rather than silently choosing one

## 10. Quality checklist

Before marking an item reviewed:
- timestamp is plausible
- card identity is supported
- player side and lane are supported
- observed/inferred/predicted status is correct
- transcript has not been paraphrased as a quote
- alignment reflects meaning, not just proximity
- strategic claim includes evidence and uncertainty
