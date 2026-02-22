from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from causal import generate_counterfactual_results
from inference import load_cross_encoder


def generate_counterfactual_for_scored_pairs(
    scored_csv: Path,
    out_csv: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2",
    max_rows: int | None = None,
) -> Path:
    scored = pd.read_csv(scored_csv)
    if max_rows is not None:
        scored = scored.head(max_rows).copy()

    bundle = load_cross_encoder(model_name=model_name)
    causal_df = generate_counterfactual_results(scored, bundle)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    causal_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal counterfactual deltas for each scored query-item pair.")
    parser.add_argument("--scored-csv", type=Path, default=Path("outputs/scored_pairs.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/counterfactual_results.csv"))
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out = generate_counterfactual_for_scored_pairs(
        scored_csv=args.scored_csv,
        out_csv=args.out_csv,
        model_name=args.model_name,
        max_rows=args.max_rows,
    )
    print(f"Wrote counterfactual dataset to {out}")


if __name__ == "__main__":
    main()
