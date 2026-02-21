from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from inference import ModelBundle


def _segment(token_type_id: int, token: str) -> str:
    if token in {"[CLS]", "[SEP]", "[PAD]"}:
        return "special"
    return "query" if token_type_id == 0 else "item"


def token_gradient_attribution(
    bundle: ModelBundle,
    query: str,
    item_text: str,
    max_length: int = 256,
) -> pd.DataFrame:
    model = bundle.model
    tokenizer = bundle.tokenizer

    enc = tokenizer(
        query,
        item_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(bundle.device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        # RoBERTa-like models may not have token_type_ids.
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(bundle.device)
    attention_mask = enc["attention_mask"].to(bundle.device)

    embeddings = model.get_input_embeddings()(input_ids)
    embeddings = embeddings.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    logit = outputs.logits.squeeze(-1)
    logit.backward()

    grads = embeddings.grad.detach().cpu().numpy()[0]
    embs = embeddings.detach().cpu().numpy()[0]

    # Signed saliency proxy: grad dot embedding.
    signed = np.sum(grads * embs, axis=1)
    abs_mag = np.abs(signed)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    tti = token_type_ids[0].detach().cpu().tolist()

    rows: list[dict[str, Any]] = []
    for i, (tok, seg_id, s, a) in enumerate(zip(tokens, tti, signed, abs_mag)):
        rows.append(
            {
                "position": i,
                "token": tok,
                "segment": _segment(seg_id, tok),
                "signed_attr": float(s),
                "abs_attr": float(a),
            }
        )

    df = pd.DataFrame(rows)
    # Normalized attribution is easier to compare across examples.
    denom = float(df["abs_attr"].sum()) if len(df) else 1.0
    df["norm_abs_attr"] = df["abs_attr"] / max(denom, 1e-8)
    return df
