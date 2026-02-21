from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from inference import ModelBundle


def _mean_masked(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    return float(values[mask].mean())


def attention_summary(
    bundle: ModelBundle,
    query: str,
    item_text: str,
    max_length: int = 256,
) -> pd.DataFrame:
    tok = bundle.tokenizer(
        query,
        item_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = tok["input_ids"].to(bundle.device)
    token_type_ids = tok.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(bundle.device)
    attention_mask = tok["attention_mask"].to(bundle.device)

    with torch.no_grad():
        out = bundle.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    tti = token_type_ids[0].detach().cpu().numpy()
    valid = attention_mask[0].detach().cpu().numpy().astype(bool)
    q_mask = (tti == 0) & valid
    i_mask = (tti == 1) & valid

    rows: list[dict[str, Any]] = []
    attentions = out.attentions
    for layer_idx, layer_attn in enumerate(attentions):
        # [batch, heads, seq, seq]
        arr = layer_attn[0].detach().cpu().numpy()
        num_heads = arr.shape[0]
        for head_idx in range(num_heads):
            head = arr[head_idx]
            cls_row = head[0]
            cls_to_query = _mean_masked(cls_row, q_mask)
            cls_to_item = _mean_masked(cls_row, i_mask)

            qi_mask = np.outer(q_mask, i_mask)
            q_to_i = _mean_masked(head, qi_mask)

            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "cls_to_query_mean": cls_to_query,
                    "cls_to_item_mean": cls_to_item,
                    "query_to_item_mean": q_to_i,
                }
            )

    return pd.DataFrame(rows)
