# Phase 2D Retrospective: ESCI-Aware Counterfactual Evaluation + UX/Runtime Hardening

Date: 2026-02-23

## Why We Changed the Causal Evaluation
The previous causal pass/fail logic was too sensitive to small score changes:

- sign-only checks (`up`/`down`) over-flagged tiny margin wiggles as "wrong-direction"
- `neutral` effectively required exact zero delta
- drilldowns made it hard to isolate meaningful failures
- causal evaluation was not aligned tightly enough to the actual ranking/calibration contract (`E > S > C > I`, `Exact > 0.9`)

The goal of this phase was to move from "delta-sign debugging" to an ESCI-aware causal evaluation that better reflects the ranking problem we care about.

## What We Shipped

### 1) ESCI-aware LLM causal labels (expected edited label, not just direction)
- OpenAI causal judge now predicts `expected_edited_esci_label` (`E/S/C/I`, with `unknown` fallback)
- Expected movement is derived from the label transition when possible (for example `S->E` implies `up`, `E->C` implies `down`)
- Existing `expected_delta_direction` is preserved for backward compatibility

This gives a clearer causal trace:
- original label -> original score -> text change -> expected edited label -> edited score -> expected vs actual movement -> result

### 2) Causal result v2: split failures by type
- Added ESCI-aware causal verdicts:
  - `pass`
  - `fail_order`
  - `fail_threshold`
  - `fail_both`
  - `marginal`
  - `ambiguous`
- Threshold policy enforced in causal evaluation:
  - expected edited label `E` => edited score should be `> 0.9`
  - expected edited label non-`E` => edited score should be `< 0.9`

### 3) Ranking-first checks in counterfactual rows
- Added group-relative rank checks against existing query-group peers:
  - `original_rank_in_group`
  - `edited_rank_in_group`
  - `rank_delta_in_group`
  - `rank_movement_check`
- Added ESCI pairwise order check against peers:
  - `pairwise_esci_order_check`
  - pass/fail counts vs peers in the group

This makes failure classification more faithful to the actual ranking task.

### 4) Dashboard usability upgrades for causal triage
- Failure summary now supports causal v2 breakdowns by edit type:
  - pass / fail-any / order-fail / threshold-fail / both
- Edit-type drilldown now supports result filtering:
  - failures only
  - order failures
  - threshold failures
  - both
  - marginal / pass / ambiguous
- Drilldown result filter styled to match existing dashboard controls

### 5) OpenAI causal pipeline operational improvements
- OpenAI edits now default to `auto`:
  - if API key present -> use OpenAI edits
  - otherwise -> heuristic fallback
- Automatic `certifi` SSL context usage for OpenAI requests (reduces local macOS/Python cert failures)
- Bounded parallelism for OpenAI causal labeling (`--openai-label-workers`) for faster cache-miss-heavy runs

## Key Design Decisions

### Use the LLM for semantic relabeling only
We rely on the LLM to infer the **expected edited ESCI label** because heuristics cannot robustly determine whether a counterfactual edit changes the pair from `Exact` to `Substitute` / `Complement` / `Irrelevant`.

We still keep the cross-encoder as the system under test for the actual score/rank behavior.

### Keep score deltas as diagnostics, not primary truth
`delta_margin` and `delta_prob` remain useful debugging signals, but causal verdicts should be primarily driven by:

- ESCI label transition expectation
- rank/order movement
- threshold policy (`Exact > 0.9`, non-Exact `< 0.9`)

### Backward compatibility during transition
Existing `expected_delta_direction` and `sign_consistent` fields are preserved so older artifacts and dashboard views still render.

## Remaining Gaps / Next Logical Steps
- Parallelize OpenAI edit generation (bounded concurrency) in addition to labeling
- Optional partial checkpoint writes (`counterfactual_results.partial.csv`) during long runs
- Dashboard tooltips clarifying `order-fail` vs `threshold-fail` vs `marginal`
- Add a per-query/group causal summary showing which ESCI transitions are most error-prone
- Consider configurable threshold policy (default remains `0.9`)

## Operational Notes
- `python src/generate_counterfactual_dataset.py` now defaults to `--edit-generator auto`
- To force OpenAI edits: `--edit-generator openai`
- To force heuristics: `--edit-generator heuristic`
- To speed up OpenAI labeling: `--openai-label-workers 4` (or higher, depending on API/network behavior)
- On local macOS Python setups, `certifi` is used automatically when installed to reduce SSL trust issues
