from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

import pandas as pd

from probes import OpenAIQuestionTagger, ProbeConfig, apply_openai_tags, load_esci_from_hf, pairwise_directional_subset


def load_handcrafted(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {
        "probe_id",
        "source",
        "question_tag",
        "query",
        "item_text",
        "esci_label",
        "relevance_score",
        "pair_group_id",
        "expected_direction",
        "target_tokens_query",
        "target_tokens_item",
        "notes",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Handcrafted CSV missing: {sorted(missing)}")
    if "tag_reason" not in df.columns:
        df["tag_reason"] = "manual"
    if "tag_confidence" not in df.columns:
        df["tag_confidence"] = "high"
    if "tag_source" not in df.columns:
        df["tag_source"] = "manual_seed"
    return df


def build_probe_set(
    handcrafted_csv: str | Path,
    out_csv: str | Path,
    target_size: int = 80,
    tagger: OpenAIQuestionTagger | None = None,
    allow_esci_fallback: bool = True,
) -> pd.DataFrame:
    hand = load_handcrafted(handcrafted_csv)
    print(f"[CURATE] Handcrafted seed rows: {len(hand)}", flush=True)

    remaining = max(target_size - len(hand), 0)
    if remaining > 0:
        print(f"[CURATE] Target size={target_size}; attempting ESCI augmentation for up to {remaining} additional rows", flush=True)
        try:
            # Pair first using cheap heuristic tags, then apply OpenAI tags only to the selected rows.
            # Lower oversampling now that pair builder has a backfill pass (reduces HF streaming time a lot).
            esci_raw = load_esci_from_hf(ProbeConfig(max_rows=max(remaining * 4, 120)), tagger=None)
            print(f"[CURATE] ESCI raw heuristic-tagged rows: {len(esci_raw)}", flush=True)
            esci_pairs = pairwise_directional_subset(
                esci_raw,
                max_pairs_per_tag=max(remaining // 8, 2),
                target_rows=remaining,
                max_pairs_per_query=4,
            )
            print(f"[CURATE] ESCI pairwise rows before truncation: {len(esci_pairs)}", flush=True)
            esci_pairs = esci_pairs.head(remaining)
            print(f"[CURATE] ESCI rows after truncation to remaining budget: {len(esci_pairs)}", flush=True)
            if tagger is not None and len(esci_pairs) > 0:
                esci_pairs = apply_openai_tags(esci_pairs, tagger, max_workers=8)
                print(f"[CURATE] ESCI rows re-tagged with OpenAI: {len(esci_pairs)}", flush=True)
        except Exception as exc:
            if not allow_esci_fallback:
                raise RuntimeError(
                    f"ESCI augmentation failed and fallback is disabled. Requested target_size={target_size}, "
                    f"handcrafted_rows={len(hand)}. Root cause: {exc}"
                ) from exc
            print(f"Warning: ESCI load failed, using handcrafted-only probe set. Reason: {exc}")
            esci_pairs = pd.DataFrame(columns=hand.columns)
    else:
        esci_pairs = pd.DataFrame(columns=hand.columns)

    for col in hand.columns:
        if col not in esci_pairs.columns:
            esci_pairs[col] = ""

    merged = pd.concat([hand, esci_pairs[hand.columns]], ignore_index=True)
    merged = merged.head(target_size).reset_index(drop=True)
    print(f"[CURATE] Final merged rows: {len(merged)}", flush=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build probe_set_v1.csv from handcrafted seeds + sampled ESCI pairs.")
    parser.add_argument("--target-size", type=int, default=120, help="Total number of query-item pairs (QIPs) to include.")
    parser.add_argument("--handcrafted-csv", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--tagger", type=str, default="heuristic", choices=["heuristic", "openai"])
    parser.add_argument("--openai-model", type=str, default="gpt-5-mini")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--prompt-openai-api-key", action="store_true")
    parser.add_argument("--tag-cache", type=Path, default=Path("outputs/openai_tag_cache.jsonl"))
    parser.add_argument(
        "--allow-esci-fallback",
        action="store_true",
        help="If ESCI augmentation fails, continue with handcrafted-only probes (default is fail-fast when --tagger openai).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    hand_path = args.handcrafted_csv or (root / "data" / "handcrafted_seed.csv")
    out_path = args.out_csv or (root / "data" / "probe_set_v1.csv")

    tagger = None
    if args.tagger == "openai":
        api_key = os.environ.get(args.openai_api_key_env)
        if args.prompt_openai_api_key and not api_key:
            api_key = getpass.getpass("OpenAI API key (input hidden): ").strip() or None
        if not api_key:
            raise SystemExit(
                "OpenAI tagger requested but no API key provided. Set OPENAI_API_KEY or use --prompt-openai-api-key."
            )
        tagger = OpenAIQuestionTagger(api_key=api_key, model=args.openai_model, cache_path=args.tag_cache)
        print(f"OpenAI probe tagging enabled ({args.openai_model}); cache={args.tag_cache}")
    else:
        print("Heuristic probe tagging enabled.")

    allow_esci_fallback = args.allow_esci_fallback if args.tagger == "openai" else True
    df = build_probe_set(
        hand_path,
        out_path,
        target_size=int(args.target_size),
        tagger=tagger,
        allow_esci_fallback=allow_esci_fallback,
    )
    print(f"Wrote {len(df)} rows to {out_path}")
