# BERT Cross-Encoder Mech-Interp Prototype (ESCI + Edge Cases)

This prototype explains **how** a cross-encoder relevance model behaves for e-commerce query-item pairs.

Model default:
- `cross-encoder/ms-marco-MiniLM-L12-v2`

Primary workflow:
1. Build a probe dataset (`ESCI-core + handcrafted edge cases`).
2. Score query-item pairs.
3. Run token attribution and attention summaries.
4. Run causal counterfactual edits and delta checks.
5. Produce question-driven scorecards and failure triage outputs.
6. Build a progressive-disclosure dashboard for interactive debugging.

## Project layout
- `src/inference.py`: model load + scoring
- `src/attribution.py`: gradient-based token attribution
- `src/attention.py`: layer/head attention summaries
- `src/probes.py`: ESCI load/tag/pair utilities
- `src/curate_dataset.py`: build `probe_set_v1.csv`
- `src/reporting.py`: directional checks + absolute score checks + artifact export
- `src/generate_attributions_dataset.py`: per-probe attribution cache for dashboard drill-down
- `src/generate_attention_dataset.py`: per-probe attention cache for dashboard drill-down
- `src/generate_counterfactual_dataset.py`: per-probe causal counterfactual deltas
- `src/build_dashboard.py`: static HTML dashboard generator
- `data/handcrafted_seed.csv`: guaranteed edge-case coverage
- `notebooks/cross_encoder_mech_interp.ipynb`: end-to-end runnable notebook

## Local quickstart (CPU is fine)
```bash
cd /Users/salvatoretornatore/Dev-Sandbox/BERT-Mech-Interp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/curate_dataset.py
jupyter notebook notebooks/cross_encoder_mech_interp.ipynb
```

## Outputs
The notebook writes artifacts to `outputs/`:
- `scored_pairs.csv`
- `question_scorecard.csv`
- `failure_triage.csv`
- `absolute_scorecard.csv`
- `label_score_summary.csv`
- `absolute_violations.csv`
- `token_attributions.csv`
- `attention_summary.csv`
- `counterfactual_results.csv`
- `brief.md`

Generate a review dashboard (progressive drill-down):
```bash
cd /Users/salvatoretornatore/Dev-Sandbox/BERT-Mech-Interp
source .venv/bin/activate
# Optional: attribution method
# python src/generate_attributions_dataset.py --method integrated_gradients --ig-steps 20
python src/generate_attributions_dataset.py
python src/generate_attention_dataset.py
python src/generate_counterfactual_dataset.py
python src/build_dashboard.py
```
Then open:
- `outputs/dashboard.html`

Additional dashboard caches:
- `outputs/attributions_by_probe.csv`
- `outputs/attention_by_probe.csv`
- `outputs/counterfactual_results.csv`

## Dashboard navigation
The dashboard uses progressive disclosure:
1. `What Was Analyzed` (aggregate metrics + assumptions)
2. `Score Calibration Snapshot` (Exact-vs-Non-Exact absolute checks + score-by-label chart)
3. `Failure Buckets Summary` (sign-consistency by edit type + wrong-direction examples)
4. `Choose Category` (`brand_match`, `attribute_match`, `negation`, `bundle_vs_canonical`)
5. `Choose Query` (query-level ranking performance)
6. `Choose Query-Item Pair` (ground truth vs model score/result)
7. `Inspect Model Behavior`:
   - `Observational`: token attribution + attention summary
   - `Causal`: per-edit before/after deltas, wrong-direction highlighting, and drilldown examples

Why query-level first:
- This is a ranking task, so the natural unit for diagnosis is query-level ordering, then candidate-level drill-down.

## Causal tests (Phase 2)
Counterfactual edits are deterministic text edits currently implemented for:
- `brand_swap`
- `size_swap`
- `color_swap`
- `negation_flip`
- `category_swap`

For each edit, the pipeline records:
- original and edited relevance signals,
- `delta_margin` / `delta_prob`,
- expected direction and sign-consistency.

Note: this is a heuristic text-edit engine (toy-friendly, not a full product-attribute parser).

## Usability mode (Phase 2C)
Dashboard now includes:
- `Beginner Mode` toggle (default ON),
- top-3 curated examples,
- plain-language "What this means" summary,
- edit-type drilldown with original vs edited text highlights.

`Advanced Mode` (toggle OFF) preserves detailed numeric views.

## Dashboard at a glance
Overview and top-level navigation:

![Dashboard overview](docs/images/dashboard_overview.png)

Category -> query -> query-item drill-down:

![Dashboard drilldown](docs/images/dashboard_drilldown.png)

Token attribution + attention diagnostics for the selected query-item pair:

![Dashboard diagnostics](docs/images/dashboard_diagnostics.png)

## Ground truth and scoring semantics
Ground-truth mapping from ESCI:
- `Exact` = 3
- `Substitute` = 2
- `Complement` = 1
- `Irrelevant` = 0

Model output:
- Cross-encoder output is normalized to portable fields:
  - `relevance_margin` (primary ranking signal)
  - `relevance_prob` (probability proxy when definable)
  - `score` (backward-compatible alias)

Evaluation:
- Primary pass/fail is directional ranking within each `pair_group_id` (not direct score regression).
- Secondary score-separation checks evaluate `Exact` vs `Non-Exact` with thresholded metrics and guardrails for high-scoring `Irrelevant` pairs.

## ESCI label usage
The model is used as a **single-score ranker**. ESCI labels are used for ordered checks:
- `Exact > Substitute > Complement > Irrelevant`

## Optional Google Colab path
1. Open the notebook in Colab.
2. Runtime:
   - CPU for quick runs.
   - Optional T4/A100 GPU for larger probe sizes.
3. Install dependencies in first cell:
```python
!pip -q install torch transformers pandas numpy datasets scikit-learn
```
4. Run the same notebook steps and export CSV artifacts.

## Notes
- Attention is supporting evidence, not standalone explanation.
- `handcrafted_seed.csv` ensures negation and bundle cases are always represented.

## Tests
Run the lightweight test suite:
```bash
cd /Users/salvatoretornatore/Dev-Sandbox/BERT-Mech-Interp
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
```
