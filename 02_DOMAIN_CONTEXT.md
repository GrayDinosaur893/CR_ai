# 02 — Clash Royale Domain Context

This document is the initial knowledge base for any model or coding agent working on CLASH IQ. It defines concepts; it is not a substitute for a verified, version-specific game-rules database.

## 1. Match structure

A match contains two opposing players, each with:
- a deck of cards
- a rotating hand/cycle
- an elixir resource
- arena units and structures
- towers with health

The arena has two lanes connected by bridges and separated by a river. Units move, target, attack, defend, and interact according to their card/unit mechanics.

The match has a timed regulation phase and may enter overtime or other special match phases depending on the current game mode/rules. The system must identify the mode and avoid hard-coding assumptions that may change.

## 2. Cards and deployments

A card can represent a troop, building, or spell. Playing a card generally spends elixir and causes an effect at a location or on a target.

A deployment event should record:
- which player deployed it
- card identity
- deployment timestamp
- approximate location/lane
- card category (troop/building/spell)
- elixir cost, from a versioned card metadata source
- confidence and visual evidence

A card shown in the hand is not necessarily played. A card played is not necessarily still alive later. These are separate facts.

## 3. Elixir

Elixir is the resource used to play cards. The system should understand:
- current visible elixir, when readable
- elixir spent on a known card play
- elixir regenerated over time according to the applicable match rules
- possible elixir cap
- possible special game modes or modifiers

Do not hard-code one regeneration rate for every mode or update. Keep game rules versioned/configurable.

### Elixir accounting

A simplified estimate can be represented as:

`estimated_elixir(t) = previous_elixir + regenerated_elixir - known_card_costs`

The estimate must respect the game's cap and applicable mode rules. If the initial elixir or exact play timestamp is unknown, represent a range rather than a false exact number.

Maintain separate fields:
- `visible_elixir`: directly read from the UI
- `estimated_elixir`: calculated from prior state and known spends
- `confidence`: reliability of estimate
- `assumptions`: rules and observations used

Opponent elixir is usually not directly visible to the viewer. It must be estimated from observed card plays and elapsed time, so it should never be presented as certain unless independently observed.

## 4. Card cycle

A player's deck has a defined card rotation. Playing a card advances the cycle; the next card becomes available according to the game's current cycle rules.

The system should track:
- cards known to be in the deck, if available
- cards played and their order
- cards currently visible in hand, if visually readable
- cards likely out of cycle
- cards likely back in cycle
- uncertainty when a play or hand card is not recognized

A card being out of cycle can create a temporary opportunity, but only if the system has enough evidence to infer the current cycle.

## 5. Counters and defensive interactions

A counter is not simply a card that can defeat another card. Counter effectiveness depends on:
- elixir cost and trade
- placement and timing
- troop/building/spell mechanics
- support cards
- lane and tower involvement
- opponent's available responses
- resulting surviving units and counterpush potential

Store the context of a counter interaction. Avoid universal statements such as “Card A always counters Card B.”

## 6. Predictions

A prediction is a hypothesis about a future opponent action, based on current state and opponent tendencies.

Examples of prediction classes:
- likely defensive card
- likely spell response
- likely lane switch or opposite-lane pressure
- likely support card
- likely punish after an overcommitment

Every prediction must record:
- prediction timestamp
- candidate action(s)
- probability/confidence or qualitative likelihood
- evidence and assumptions
- what actually happened
- whether it was correct

Do not label a prediction as correct merely because one outcome happened; evaluate whether the reasoning was justified using information available at the time.

## 7. Strategic concepts

The model should learn these concepts as distinct labels:
- positive/negative/neutral elixir trade
- elixir advantage/disadvantage
- cycle advantage/disadvantage
- counter availability
- pressure
- split-lane pressure
- punish opportunity
- overcommitment
- defensive efficiency
- spell value
- tower damage trade-off
- counterpush
- tempo
- matchup-specific win condition
- prediction
- baiting
- resource conservation
- risk/reward

Each label needs a definition and examples. Labels may co-occur.

## 8. Decision quality

Evaluate a decision using the state at the time, not only the final outcome.

A decision review should consider:
1. What information was available?
2. What did the player know or reasonably infer?
3. What options were legal and plausible?
4. What were the expected costs and risks?
5. What happened afterward?
6. Was the decision good given the information at that moment?

A good decision can have a bad outcome; a bad decision can get lucky.

## 9. Unknowns and versioning

Game balance, card mechanics, modes, UI, and rules can change. Store:
- game version/date if known
- mode
- card metadata version
- source for rule assumptions
- unresolved mechanics questions

Never silently apply current card costs or mechanics to old footage without checking whether they match that footage's era.
