from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import pandas as pd

from causal import OpenAICausalLabeler, OpenAICounterfactualEditor, generate_counterfactual_results
from inference import load_cross_encoder


def generate_counterfactual_for_scored_pairs(
    scored_csv: Path,
    out_csv: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2",
    max_rows: int | None = None,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-5-mini",
    edit_generator: str = "auto",
    openai_edit_cache: Path = Path("outputs/openai_edit_cache.jsonl"),
    openai_label_workers: int = 4,
) -> Path:
    scored = pd.read_csv(scored_csv)
    if max_rows is not None:
        scored = scored.head(max_rows).copy()

    bundle = load_cross_encoder(model_name=model_name)
    labeler = OpenAICausalLabeler(api_key=openai_api_key, model=openai_model) if openai_api_key else None
    resolved_edit_generator = edit_generator
    if resolved_edit_generator == "auto":
        resolved_edit_generator = "openai" if openai_api_key else "heuristic"

    editor = None
    if resolved_edit_generator == "openai":
        if not openai_api_key:
            raise RuntimeError("OpenAI edit generator requires API key (set OPENAI_API_KEY or use --prompt-openai-api-key).")
        editor = OpenAICounterfactualEditor(api_key=openai_api_key, model=openai_model, cache_path=openai_edit_cache)
    causal_df = generate_counterfactual_results(
        scored,
        bundle,
        labeler=labeler,
        editor=editor,
        label_max_workers=(int(openai_label_workers) if openai_api_key else 1),
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    causal_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate causal counterfactual deltas for each scored query-item pair.")
    parser.add_argument("--scored-csv", type=Path, default=Path("outputs/scored_pairs.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/counterfactual_results.csv"))
    parser.add_argument("--model-name", type=str, default="cross-encoder/ms-marco-MiniLM-L12-v2")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--openai-model", type=str, default="gpt-5-mini")
    parser.add_argument(
        "--edit-generator",
        type=str,
        default="auto",
        choices=["auto", "heuristic", "openai"],
        help="Counterfactual edit generator: auto uses OpenAI when API key is available, else heuristics.",
    )
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai-edit-cache", type=Path, default=Path("outputs/openai_edit_cache.jsonl"))
    parser.add_argument(
        "--openai-label-workers",
        type=int,
        default=4,
        help="Max concurrent OpenAI causal labeling requests (bounded parallelism). Use 1 to disable.",
    )
    parser.add_argument(
        "--prompt-openai-api-key",
        action="store_true",
        help="Prompt securely (hidden input) for an OpenAI API key to enable causal expected-direction labeling.",
    )
    args = parser.parse_args()

    openai_api_key = os.environ.get(args.openai_api_key_env)
    if args.prompt_openai_api_key and not openai_api_key:
        openai_api_key = getpass.getpass("OpenAI API key (input hidden, optional): ").strip() or None
    if openai_api_key:
        print(f"OpenAI causal labeling: enabled via {args.openai_api_key_env if os.environ.get(args.openai_api_key_env) else 'hidden prompt'}")
        print(f"OpenAI causal labeling parallelism: max_workers={max(int(args.openai_label_workers), 1)}")
    else:
        print("OpenAI causal labeling: disabled (no API key provided). Causal labels/result pass-fail columns will be left unlabeled.")
    resolved_edit_generator = args.edit_generator
    if resolved_edit_generator == "auto":
        resolved_edit_generator = "openai" if openai_api_key else "heuristic"

    if resolved_edit_generator == "openai":
        if not openai_api_key:
            raise SystemExit("OpenAI edit generator requested but no API key provided. Set OPENAI_API_KEY or use --prompt-openai-api-key.")
        auto_note = " (auto)" if args.edit_generator == "auto" else ""
        print(f"OpenAI counterfactual edit generation: enabled{auto_note} ({args.openai_model}); cache={args.openai_edit_cache}")
    else:
        if args.edit_generator == "auto":
            print("Counterfactual edit generation: heuristic rules (auto fallback; no API key)")
        else:
            print("Counterfactual edit generation: heuristic rules")

    out = generate_counterfactual_for_scored_pairs(
        scored_csv=args.scored_csv,
        out_csv=args.out_csv,
        model_name=args.model_name,
        max_rows=args.max_rows,
        openai_api_key=openai_api_key,
        openai_model=args.openai_model,
        edit_generator=resolved_edit_generator,
        openai_edit_cache=args.openai_edit_cache,
        openai_label_workers=args.openai_label_workers,
    )
    print(f"Wrote counterfactual dataset to {out}")


if __name__ == "__main__":
    main()
