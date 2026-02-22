# BERT Relevance Debugger (Mechanistic-Interp Prototype)

This project is a beginner-friendly dashboard for understanding **why** a BERT cross-encoder ranks e-commerce items the way it does.

It is designed for questions like:
- "Is the model generally behaving well?"
- "Where does it fail?"
- "What changed when we edited brand/size/negation?"

Default model:
- `cross-encoder/ms-marco-MiniLM-L12-v2`

## What You Get
- A curated probe set (`ESCI + handcrafted edge cases`)
- Ranking and calibration checks
- Token attribution + attention summaries (observational evidence)
- Counterfactual edit tests (causal evidence)
- An interactive dashboard with beginner mode and drilldowns

## Quickstart
```bash
cd /Users/salvatoretornatore/Dev-Sandbox/BERT-Mech-Interp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/curate_dataset.py
python src/generate_attributions_dataset.py
python src/generate_attention_dataset.py
python src/generate_counterfactual_dataset.py
python src/build_dashboard.py
```

Open:
- `outputs/dashboard.html`

## Dashboard Walkthrough
1. Start at **Top 3 Examples** for quick intuition.
2. Use **Failure Buckets Summary** to see which edit types fail most.
3. Click an edit type to drill into **individual query/item examples**.
4. Click an example row to jump to full detail.
5. In **Causal Tests**, compare before/after score changes and check whether reaction was expected.
6. Use **Beginner Mode** for plain-language interpretation, or turn it off for full numeric detail.

## Screenshots
Overview:

![Dashboard overview](docs/images/dashboard_overview.png)

Category/query/pair drilldown:

![Dashboard drilldown](docs/images/dashboard_drilldown.png)

Observational diagnostics (token attribution + attention):

![Dashboard diagnostics](docs/images/dashboard_diagnostics.png)

Causal summary (failure buckets + edit-type outcomes):

![Dashboard causal summary](docs/images/dashboard_causal_summary.png)

Causal drilldown (per-example score changes + text edits):

![Dashboard causal drilldown](docs/images/dashboard_causal_drilldown.png)

## Core Concepts (Plain English)
- **Observational evidence**: what the model seemed to focus on (attribution, attention).
- **Causal evidence**: what actually changed in score when we edited text.
- **Wrong-direction case**: an edit that should decrease relevance but increases score (or vice versa).

## Counterfactual Edit Types
Current deterministic edits:
- `brand_swap`
- `size_swap`
- `color_swap`
- `negation_flip`
- `category_swap`

Each row stores:
- original vs edited text,
- before/after model outputs,
- delta,
- expected direction,
- pass/fail sign consistency.

## Output Artifacts
Main files in `outputs/`:
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
- `dashboard.html`

## Project Structure
- `src/inference.py`: scoring + relevance signal extraction
- `src/attribution.py`: token attribution (fast + IG modes)
- `src/attention.py`: attention summaries
- `src/causal.py`: counterfactual edit engine
- `src/reporting.py`: evaluation + artifact export
- `src/build_dashboard.py`: static dashboard generation
- `src/curate_dataset.py`: probe set build

## Tests
```bash
cd /Users/salvatoretornatore/Dev-Sandbox/BERT-Mech-Interp
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_*.py'
```

## Scope Note
This repo is a **toy but practical prototype** for relevance debugging. Counterfactual edits are currently heuristic text edits (not a full production metadata parser).
