from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch

from inference import ModelBundle


@dataclass
class CounterfactualEdit:
    edit_type: str
    edited_text: str
    expected_delta_direction: str


COLOR_SWAP = {
    "black": "white",
    "white": "black",
    "blue": "red",
    "red": "blue",
    "green": "purple",
    "purple": "green",
}

BRAND_SWAP = {
    "nike": "adidas",
    "adidas": "nike",
    "apple": "samsung",
    "samsung": "apple",
    "sony": "bose",
    "bose": "sony",
}

CATEGORY_SWAP = {
    "shoes": "headphones",
    "headphones": "shoes",
    "laptop": "coffee maker",
    "coffee maker": "laptop",
    "phone case": "water bottle",
    "water bottle": "phone case",
}

SIZE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s?(oz|inch|in|mm|cm|gb|tb|mah|w)\b", re.IGNORECASE)
NEGATION_PATTERN = re.compile(r"\b(without|exclude|excluding|not)\b", re.IGNORECASE)


def _swap_first(text: str, mapping: dict[str, str]) -> str:
    lower = text.lower()
    for src, dst in mapping.items():
        idx = lower.find(src)
        if idx >= 0:
            return text[:idx] + dst + text[idx + len(src) :]
    return text


def _size_swap(text: str) -> str:
    m = SIZE_PATTERN.search(text)
    if not m:
        return text
    raw_num = m.group(1)
    try:
        num = float(raw_num)
    except Exception:
        return text
    new_num = num * 2 if num < 100 else max(num / 2, 1.0)
    if new_num.is_integer():
        new_num_text = str(int(new_num))
    else:
        new_num_text = f"{new_num:.1f}"
    return text[: m.start(1)] + new_num_text + text[m.end(1) :]


def _negation_flip(text: str) -> tuple[str, str]:
    if NEGATION_PATTERN.search(text):
        edited = NEGATION_PATTERN.sub("with", text, count=1)
        return edited, "up"
    return f"without {text}", "down"


def _score_pair(bundle: ModelBundle, query: str, item_text: str) -> tuple[float, float]:
    enc = bundle.adapter.tokenize_pairs(
        [query],
        [item_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(bundle.adapter.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = bundle.adapter.forward_logits(enc)
        signal = bundle.adapter.extract_relevance_signal(logits)
    return float(signal.relevance_margin[0]), float(signal.relevance_prob[0])


def _edit_candidates(item_text: str) -> list[CounterfactualEdit]:
    edits: list[CounterfactualEdit] = []
    rules: list[tuple[str, Callable[[str], str], str]] = [
        ("brand_swap", lambda t: _swap_first(t, BRAND_SWAP), "down"),
        ("size_swap", _size_swap, "down"),
        ("color_swap", lambda t: _swap_first(t, COLOR_SWAP), "down"),
        ("category_swap", lambda t: _swap_first(t, CATEGORY_SWAP), "down"),
    ]
    for edit_type, fn, expected in rules:
        edited = fn(item_text)
        if edited != item_text:
            edits.append(CounterfactualEdit(edit_type=edit_type, edited_text=edited, expected_delta_direction=expected))

    negated, neg_expected = _negation_flip(item_text)
    if negated != item_text:
        edits.append(CounterfactualEdit(edit_type="negation_flip", edited_text=negated, expected_delta_direction=neg_expected))
    return edits


def generate_counterfactual_results(scored_df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    required = {"probe_id", "query", "item_text"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for causal generation: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in scored_df.iterrows():
        probe_id = str(row["probe_id"])
        query = str(row["query"])
        item_text = str(row["item_text"])
        pair_group_id = str(row.get("pair_group_id", ""))

        original_margin = float(row["relevance_margin"]) if "relevance_margin" in row else None
        original_prob = float(row["relevance_prob"]) if "relevance_prob" in row else None
        if (
            original_margin is None
            or original_prob is None
            or pd.isna(original_margin)
            or pd.isna(original_prob)
        ):
            original_margin, original_prob = _score_pair(bundle, query, item_text)

        for edit in _edit_candidates(item_text):
            edited_margin, edited_prob = _score_pair(bundle, query, edit.edited_text)
            delta_margin = edited_margin - original_margin
            delta_prob = edited_prob - original_prob
            if edit.expected_delta_direction == "down":
                sign_consistent = delta_margin < 0
            elif edit.expected_delta_direction == "up":
                sign_consistent = delta_margin > 0
            else:
                sign_consistent = abs(delta_margin) < 1e-6

            rows.append(
                {
                    "probe_id": probe_id,
                    "pair_group_id": pair_group_id,
                    "query": query,
                    "item_text": item_text,
                    "edit_type": edit.edit_type,
                    "original_text": item_text,
                    "edited_text": edit.edited_text,
                    "expected_delta_direction": edit.expected_delta_direction,
                    "original_margin": original_margin,
                    "edited_margin": edited_margin,
                    "delta_margin": delta_margin,
                    "original_prob": original_prob,
                    "edited_prob": edited_prob,
                    "delta_prob": delta_prob,
                    "sign_consistent": bool(sign_consistent),
                    "artifact_kind": "causal",
                }
            )

    return pd.DataFrame(rows)
