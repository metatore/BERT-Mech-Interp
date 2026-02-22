from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from attribution import token_gradient_attribution
from inference import load_cross_encoder


def generate_attributions_for_scored_pairs(
    scored_csv: Path,
    out_csv: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2",
    max_rows: int | None = None,
    method: Literal["grad_x_embed_saliency", "integrated_gradients"] = "grad_x_embed_saliency",
    ig_steps: int = 20,
) -> Path:
    scored = pd.read_csv(scored_csv)
    if max_rows is not None:
        scored = scored.head(max_rows).copy()

    required = {"probe_id", "query", "item_text"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Missing required columns in {scored_csv}: {sorted(missing)}")

    bundle = load_cross_encoder(model_name=model_name)

    rows = []
    for _, row in scored.iterrows():
        probe_id = str(row["probe_id"])
        query = str(row["query"])
        item_text = str(row["item_text"])
        attr = token_gradient_attribution(bundle, query, item_text, method=method, ig_steps=ig_steps)
        attr["probe_id"] = probe_id
        attr["query"] = query
        attr["item_text"] = item_text
        rows.append(attr)

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token attributions for each scored query-item pair.")
    parser.add_argument("--scored-csv", type=Path, default=Path("outputs/scored_pairs.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/attributions_by_probe.csv"))
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default="grad_x_embed_saliency",
        choices=["grad_x_embed_saliency", "integrated_gradients"],
    )
    parser.add_argument("--ig-steps", type=int, default=20)
    args = parser.parse_args()

    out = generate_attributions_for_scored_pairs(
        scored_csv=args.scored_csv,
        out_csv=args.out_csv,
        model_name=args.model_name,
        max_rows=args.max_rows,
        method=args.method,
        ig_steps=args.ig_steps,
    )
    print(f"Wrote attribution dataset to {out}")


if __name__ == "__main__":
    main()
