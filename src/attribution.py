from __future__ import annotations

from typing import Literal
from typing import Any

import numpy as np
import pandas as pd
import torch

from inference import ModelBundle


def _segment(token_type_id: int, token: str) -> str:
    if token in {"[CLS]", "[SEP]", "[PAD]"}:
        return "special"
    return "query" if token_type_id == 0 else "item"


def _margin_scalar_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape[1] == 1:
        return logits[:, 0].sum()
    if logits.shape[1] == 2:
        return (logits[:, 1] - logits[:, 0]).sum()
    raise ValueError(f"Unsupported logits shape for attribution: {tuple(logits.shape)}")


def token_gradient_attribution(
    bundle: ModelBundle,
    query: str,
    item_text: str,
    max_length: int = 256,
    method: Literal["grad_x_embed_saliency", "integrated_gradients"] = "grad_x_embed_saliency",
    ig_steps: int = 20,
) -> pd.DataFrame:
    model = bundle.adapter.model
    tokenizer = bundle.adapter.tokenizer

    enc = tokenizer(
        query,
        item_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    input_ids = enc["input_ids"].to(bundle.adapter.device)
    token_type_ids = enc.get("token_type_ids")
    if token_type_ids is None:
        # RoBERTa-like models may not have token_type_ids.
        token_type_ids = torch.zeros_like(input_ids)
    token_type_ids = token_type_ids.to(bundle.adapter.device)
    attention_mask = enc["attention_mask"].to(bundle.adapter.device)

    embeddings = bundle.adapter.get_input_embeddings()(input_ids)
    embeddings = embeddings.detach().clone()

    if method == "grad_x_embed_saliency":
        run_emb = embeddings.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(
            inputs_embeds=run_emb,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        scalar = _margin_scalar_from_logits(outputs.logits)
        scalar.backward()
        grads = run_emb.grad.detach().cpu().numpy()[0]
        embs = run_emb.detach().cpu().numpy()[0]
    elif method == "integrated_gradients":
        baseline = torch.zeros_like(embeddings)
        total_grads = torch.zeros_like(embeddings)
        ig_steps = max(int(ig_steps), 2)
        for alpha in np.linspace(0.0, 1.0, ig_steps):
            step_emb = (baseline + alpha * (embeddings - baseline)).detach().clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            outputs = model(
                inputs_embeds=step_emb,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            scalar = _margin_scalar_from_logits(outputs.logits)
            scalar.backward()
            total_grads += step_emb.grad.detach()

        avg_grads = total_grads / ig_steps
        grads = avg_grads.detach().cpu().numpy()[0]
        embs = (embeddings - baseline).detach().cpu().numpy()[0]
    else:
        raise ValueError(f"Unsupported attribution method: {method}")

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
                "attribution_method": method,
                "attribution_kind": "observational",
            }
        )

    df = pd.DataFrame(rows)
    # Normalized attribution is easier to compare across examples.
    denom = float(df["abs_attr"].sum()) if len(df) else 1.0
    df["norm_abs_attr"] = df["abs_attr"] / max(denom, 1e-8)
    return df
