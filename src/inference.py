from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class ModelBundle:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2") -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return ModelBundle(tokenizer=tokenizer, model=model, device=device)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_pairs(bundle: ModelBundle, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    required = {"probe_id", "query", "item_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    queries = df["query"].tolist()
    items = df["item_text"].tolist()

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            q_batch = queries[i : i + batch_size]
            t_batch = items[i : i + batch_size]
            enc = bundle.tokenizer(
                q_batch,
                t_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(bundle.device) for k, v in enc.items()}
            logits = bundle.model(**enc).logits.squeeze(-1)
            logits_np = logits.detach().cpu().numpy()
            probs_np = _sigmoid(logits_np)
            for j, (logit, prob) in enumerate(zip(logits_np, probs_np)):
                rows.append(
                    {
                        "row_idx": i + j,
                        "logit": float(logit),
                        "score": float(prob),
                    }
                )

    out = df.reset_index(drop=True).copy()
    score_df = pd.DataFrame(rows).set_index("row_idx")
    out = out.join(score_df)

    if "pair_group_id" in out.columns:
        # Some rows can have missing group ids; keep nullable integer ranks.
        ranks = out.groupby("pair_group_id", dropna=True)["score"].rank(ascending=False, method="first")
        out["rank_in_group"] = ranks.astype("Int64")

    return out


def topk_by_query(scored_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    if "query" not in scored_df.columns:
        raise ValueError("Expected query column")
    return (
        scored_df.sort_values(["query", "score"], ascending=[True, False])
        .groupby("query")
        .head(k)
        .reset_index(drop=True)
    )
