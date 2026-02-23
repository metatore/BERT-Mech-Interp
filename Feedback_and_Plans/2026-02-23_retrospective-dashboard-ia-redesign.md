# Dashboard IA Redesign Retrospective (Phase 2 UX Pass)

Date: 2026-02-23

## Why This Pass Happened

The dashboard had strong technical depth and drilldown power, but it became hard to follow in practice:
- too many top-level sections
- non-linear reading order
- jargon-heavy framing
- unclear "so what" for novice readers

This pass focused on redesigning the dashboard from first principles with progressive disclosure while preserving the strongest existing behaviors.

## What We Kept (and Why)

- Drilldown capability (aggregate -> category -> query -> pair)
- Interactive model-score-by-ground-truth plot
- Highlighted text diffs for counterfactual swaps
- Observational vs causal distinction

These are core to making the dashboard useful as a debugging tool instead of static explainability output.

## What Changed

## 1) Information architecture was simplified into a clear narrative

Top-level flow is now:
1. Health Snapshot
2. Focus Finder
3. Drilldown Workspace

This replaced the previous "many cards at the same level" structure that forced users to guess what to read first.

## 2) Progressive disclosure is more intentional

- Seed-set context is collapsed by default
- Token attribution and attention are separated from causal evidence
- High-detail diagnostics remain available but no longer compete with the main story

## 3) Beginner mode toggle was removed

Instead of dual modes, the UI now uses one plain-language, scannable layout by default.

## 4) Selected Pair readout was rebuilt around reading order

The readout now prioritizes:
- orientation (query + candidate item + GT label)
- model-output threshold check
- pairwise comparison (separate card)
- token attribution (for the selected pair)
- causal tests

This reduced context switching and made the model-output interpretation much easier.

## 5) Pairwise Check became an explicit comparison UI

The pairwise section now shows:
- pass/fail badge
- query (restated)
- candidate vs comparison item (side-by-side)
- GT labels (both)
- model scores (both)
- expected top-ranked item
- actual top-ranked item

This solved a recurring confusion point: users could not infer pairwise correctness without seeing the other item clearly.

## 6) Causal tests were simplified from dense table to card-based scan

Causal examples are now rendered as cards emphasizing:
- what changed
- score reaction
- verdict

This improved readability and made per-edit reasoning more obvious.

## Bugs / UX issues discovered during the redesign

## ESCI shorthand normalization bug (important)

The UI treated `E` as "not Exact" in the threshold check because the client-side logic expected `"Exact"` only.

Impact:
- wrong threshold rule text
- wrong pass/fail threshold verdict
- confusing `E (E)` label rendering

Fix:
- client-side ESCI normalization (`E/S/C/I` -> `Exact/Substitute/Complement/Irrelevant`)
- label display standardized to `Exact (E)` style

## What worked well in the collaboration loop

- Iterating directly on screenshot-driven feedback was highly effective
- Reframing UI problems as "reading order" and "scanability" produced better decisions than tweaking styles in place
- Separating "selected pair" and "pairwise comparison" reduced cognitive load significantly

## What remains / next UX opportunities

- Further tighten spacing/visual rhythm in the Pairwise Check card
- Add inline tooltip/help copy for ESCI labels (`E/S/C/I`) in the pairwise card
- Add a compact "why failed" summary chip when pairwise and threshold checks disagree
- Consider a small "Compare runs" placeholder section once backend support lands

## Takeaway

The dashboard is now much closer to an interactive debugging workflow:
- first understand health,
- then choose where to investigate,
- then inspect one pair deeply with the right context.

The biggest UX improvement was not adding new analysis; it was reordering the same information so the user can reason about it in the correct sequence.
