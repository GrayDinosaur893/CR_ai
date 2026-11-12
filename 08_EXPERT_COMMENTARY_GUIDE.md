# 08 — Expert Commentary as a Teaching Signal

## Why expert videos matter

A skilled creator's commentary can expose the reasoning behind decisions that are otherwise difficult to infer from pixels alone. It may explain:
- why a player chose to attack or defend
- how much elixir the opponent likely has
- which cards are out of cycle
- what response is expected
- why a counter is efficient
- when a prediction is being made
- what risk the player is accepting

## Do not train by imitation alone

The goal is not to make the model repeat a creator's phrases or copy every action. The goal is to extract reusable strategic concepts and test them against game states and outcomes.

## Commentary processing format

For each meaningful segment, capture:

1. **Verbatim transcript** — what the speaker actually said.
2. **Timestamp** — when the statement was spoken.
3. **Referenced game interval** — the event or state being discussed.
4. **Statement type** — observation, prediction, explanation, evaluation, general principle.
5. **Explicit claim** — the speaker's stated point.
6. **Reasoning** — evidence and causal link, if explicit.
7. **Strategic concepts** — elixir, cycle, counter, pressure, prediction, etc.
8. **Outcome** — what happened afterward.
9. **Review status** — unreviewed, accepted, corrected, disputed.

## Example (illustrative only)

Commentary: “They just used their building, so the win condition is safer now.”

Possible structured interpretation:
- observation: opponent played a building
- cycle claim: building may be out of hand temporarily
- strategic implication: an offensive win condition may be more difficult to stop
- risk: opponent may have another counter
- prediction: next offensive attempt may connect more reliably
- validation: inspect what the opponent actually does next

The system must not assume this claim is always correct. It should check the deck/cycle evidence and the actual response.

## Learning from multiple experts

Eventually include multiple high-level players and different archetypes. This helps avoid overfitting to:
- one player's personal style
- one deck
- one matchup
- one meta period
- one commentator's vocabulary

Keep creator/source attribution in the dataset. Do not merge conflicting expert opinions into one fake consensus.

## Media and rights

Use videos you are authorized to process. Respect platform terms, creator rights, and applicable copyright law. Do not redistribute source videos or publish a dataset containing copied media. Prefer storing timestamps, short factual annotations, and your own derived analysis. Keep attribution and provenance.
