from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


NEGATION_TERMS = {"not", "without", "exclude", "excluding", "no "}
BUNDLE_TERMS = {"bundle", "pack", "set", "2-pack", "3-pack", "kit"}
SPEC_PATTERN = re.compile(r"\b(\d+\s?(gb|tb|oz|inch|in|mm|cm|mah|w))\b", re.IGNORECASE)


@dataclass
class ProbeConfig:
    locale: str = "us"
    use_small_version: bool = True
    max_rows: int = 50


def _contains_any(text: str, terms: set[str]) -> bool:
    t = text.lower()
    return any(term in t for term in terms)


def tag_question(query: str, item_text: str) -> str:
    q = str(query).lower()
    i = str(item_text).lower()
    if _contains_any(q, NEGATION_TERMS):
        return "negation"
    if _contains_any(q + " " + i, BUNDLE_TERMS):
        return "bundle_vs_canonical"
    if SPEC_PATTERN.search(q) or SPEC_PATTERN.search(i):
        return "attribute_match"
    return "brand_match"


def map_esci_to_score(label: str) -> int:
    mapping = {"E": 3, "S": 2, "C": 1, "I": 0, "exact": 3, "substitute": 2, "complement": 1, "irrelevant": 0}
    key = str(label).strip().lower()
    if key in mapping:
        return mapping[key]
    key_u = str(label).strip().upper()
    if key_u in mapping:
        return mapping[key_u]
    raise ValueError(f"Unknown ESCI label: {label}")


def load_esci_from_hf(config: ProbeConfig) -> pd.DataFrame:
    from datasets import load_dataset

    # Streaming avoids materializing the full ~2M-row train split in memory.
    ds = load_dataset("tasksource/esci", split="train", streaming=True)

    rows: list[dict[str, Any]] = []
    # Oversample to retain diversity before deterministic downsample.
    target = max(config.max_rows * 6, 120)

    for ex in ds:
        locale = str(ex.get("product_locale", "")).lower()
        if locale and locale != config.locale.lower():
            continue

        if config.use_small_version:
            try:
                if int(ex.get("small_version", 0)) != 1:
                    continue
            except Exception:
                continue

        query = ex.get("query")
        title = ex.get("product_title")
        label = ex.get("esci_label")
        if query is None or title is None or label is None:
            continue

        try:
            relevance = map_esci_to_score(label)
        except Exception:
            continue

        rows.append(
            {
                "query": str(query),
                "item_text": str(title),
                "esci_label": str(label),
                "relevance_score": relevance,
            }
        )

        if len(rows) >= target:
            break

    if not rows:
        raise ValueError("No ESCI rows matched current filters (locale/slice).")

    out = pd.DataFrame(rows).dropna()
    out["question_tag"] = [tag_question(q, t) for q, t in zip(out["query"], out["item_text"])]
    out = out.sample(min(config.max_rows, len(out)), random_state=7).reset_index(drop=True)
    out["source"] = "esci"
    out["probe_id"] = [f"esci_{i:04d}" for i in range(len(out))]
    out["notes"] = "esci_sample"
    return out


def pairwise_directional_subset(df: pd.DataFrame, max_pairs_per_tag: int = 8) -> pd.DataFrame:
    rows = []
    group_counter = 0

    for tag, part in df.groupby("question_tag"):
        made = 0
        for query, qpart in part.groupby("query"):
            if made >= max_pairs_per_tag:
                break
            qpart = qpart.sort_values("relevance_score", ascending=False)
            if len(qpart) < 2:
                continue
            hi = qpart.iloc[0]
            lo = qpart.iloc[-1]
            if hi["relevance_score"] == lo["relevance_score"]:
                continue

            gid = f"grp_{group_counter:04d}"
            group_counter += 1

            hi_row = hi.to_dict()
            lo_row = lo.to_dict()
            hi_row["pair_group_id"] = gid
            lo_row["pair_group_id"] = gid
            hi_row["expected_direction"] = "should_rank_higher"
            lo_row["expected_direction"] = "should_rank_lower"
            hi_row["target_tokens_query"] = ""
            hi_row["target_tokens_item"] = ""
            lo_row["target_tokens_query"] = ""
            lo_row["target_tokens_item"] = ""
            rows.extend([hi_row, lo_row])
            made += 1

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["pair_group_id", "expected_direction", "target_tokens_query", "target_tokens_item"])

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out
