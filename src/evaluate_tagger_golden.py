from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import pandas as pd

from probes import OpenAIQuestionTagger, TAG_LABELS


def evaluate_golden(
    golden_csv: Path,
    out_csv: Path | None,
    api_key: str,
    model: str,
    cache_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(golden_csv)
    required = {"query", "item_text", "expected_question_tag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Golden CSV missing columns: {sorted(missing)}")

    tagger = OpenAIQuestionTagger(api_key=api_key, model=model, cache_path=cache_path)
    rows = []
    for _, r in df.iterrows():
        query = str(r["query"])
        item_text = str(r["item_text"])
        expected = str(r["expected_question_tag"]).strip()
        decision = tagger.label(query, item_text)
        rows.append(
            {
                "query": query,
                "item_text": item_text,
                "expected_question_tag": expected,
                "predicted_question_tag": decision.question_tag,
                "tag_confidence": decision.tag_confidence,
                "tag_source": decision.tag_source,
                "correct": bool(decision.question_tag == expected),
            }
        )

    out = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    return out


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows.")
        return
    acc = float(df["correct"].mean())
    print(f"Rows: {len(df)}")
    print(f"Accuracy: {acc:.3f}")
    print("")
    print("By expected tag:")
    by_tag = df.groupby("expected_question_tag")["correct"].agg(count="count", accuracy="mean").reset_index()
    print(by_tag.to_string(index=False))
    print("")
    print("Confusion counts:")
    conf = pd.crosstab(df["expected_question_tag"], df["predicted_question_tag"])
    print(conf.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenAI probe tagger against a golden set.")
    parser.add_argument("--golden-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--openai-model", type=str, default="gpt-5-mini")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--prompt-openai-api-key", action="store_true")
    parser.add_argument("--tag-cache", type=Path, default=Path("outputs/openai_tag_cache.jsonl"))
    args = parser.parse_args()

    api_key = os.environ.get(args.openai_api_key_env)
    if args.prompt_openai_api_key and not api_key:
        api_key = getpass.getpass("OpenAI API key (input hidden): ").strip() or None
    if not api_key:
        raise SystemExit("No OpenAI API key provided.")

    print(f"Allowed tags: {', '.join(TAG_LABELS)}")
    print(f"Model: {args.openai_model}")
    print(f"Cache: {args.tag_cache}")
    df = evaluate_golden(
        golden_csv=args.golden_csv,
        out_csv=args.out_csv,
        api_key=api_key,
        model=args.openai_model,
        cache_path=args.tag_cache,
    )
    print_summary(df)


if __name__ == "__main__":
    main()
