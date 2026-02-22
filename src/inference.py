from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RelevanceSignal:
    relevance_margin: np.ndarray
    relevance_prob: np.ndarray
    raw_logit: np.ndarray
    logit_pos: np.ndarray
    logit_neg: np.ndarray


class BaseCrossEncoderAdapter(ABC):
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def tokenize_pairs(self, queries: Sequence[str], items: Sequence[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def extract_relevance_signal(self, logits: torch.Tensor) -> RelevanceSignal:
        raise NotImplementedError

    @abstractmethod
    def get_input_embeddings(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def supports_attentions(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_hidden_states(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def supports_token_type_ids(self) -> bool:
        raise NotImplementedError


@dataclass
class HFCrossEncoderAdapter(BaseCrossEncoderAdapter):
    tokenizer: Any
    model: Any
    device: torch.device
    name: str

    def model_id(self) -> str:
        return self.name

    def tokenize_pairs(self, queries: Sequence[str], items: Sequence[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.tokenizer(queries, items, **kwargs)

    def forward_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**batch).logits

    def extract_relevance_signal(self, logits: torch.Tensor) -> RelevanceSignal:
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, C] or [B], got {tuple(logits.shape)}")

        batch_size = logits.shape[0]
        margin = np.zeros(batch_size, dtype=np.float32)
        prob = np.zeros(batch_size, dtype=np.float32)
        raw_logit = np.full(batch_size, np.nan, dtype=np.float32)
        logit_pos = np.full(batch_size, np.nan, dtype=np.float32)
        logit_neg = np.full(batch_size, np.nan, dtype=np.float32)

        if logits.shape[1] == 1:
            raw = logits[:, 0]
            margin = raw.detach().cpu().numpy().astype(np.float32)
            prob = torch.sigmoid(raw).detach().cpu().numpy().astype(np.float32)
            raw_logit = margin.copy()
        elif logits.shape[1] == 2:
            neg = logits[:, 0]
            pos = logits[:, 1]
            margin = (pos - neg).detach().cpu().numpy().astype(np.float32)
            prob = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)
            logit_pos = pos.detach().cpu().numpy().astype(np.float32)
            logit_neg = neg.detach().cpu().numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported logits shape {tuple(logits.shape)}; expected [B,1] or [B,2].")

        return RelevanceSignal(
            relevance_margin=margin,
            relevance_prob=prob,
            raw_logit=raw_logit,
            logit_pos=logit_pos,
            logit_neg=logit_neg,
        )

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def supports_attentions(self) -> bool:
        return True

    def supports_hidden_states(self) -> bool:
        return True

    def supports_token_type_ids(self) -> bool:
        return getattr(self.tokenizer, "model_input_names", None) is not None and "token_type_ids" in self.tokenizer.model_input_names


@dataclass
class ModelBundle:
    adapter: BaseCrossEncoderAdapter


def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L12-v2") -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    adapter = HFCrossEncoderAdapter(tokenizer=tokenizer, model=model, device=device, name=model_name)
    return ModelBundle(adapter=adapter)


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
            enc = bundle.adapter.tokenize_pairs(
                q_batch,
                t_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(bundle.adapter.device) for k, v in enc.items()}
            logits = bundle.adapter.forward_logits(enc)
            signal = bundle.adapter.extract_relevance_signal(logits)
            probs_np = signal.relevance_prob
            margins_np = signal.relevance_margin
            for j in range(len(margins_np)):
                score = probs_np[j] if not np.isnan(probs_np[j]) else _sigmoid(np.array([margins_np[j]], dtype=np.float32))[0]
                rows.append(
                    {
                        "row_idx": i + j,
                        "relevance_margin": float(margins_np[j]),
                        "relevance_prob": float(probs_np[j]),
                        "score": float(score),
                        "raw_logit": None if np.isnan(signal.raw_logit[j]) else float(signal.raw_logit[j]),
                        "logit_pos": None if np.isnan(signal.logit_pos[j]) else float(signal.logit_pos[j]),
                        "logit_neg": None if np.isnan(signal.logit_neg[j]) else float(signal.logit_neg[j]),
                    }
                )

    out = df.reset_index(drop=True).copy()
    score_df = pd.DataFrame(rows).set_index("row_idx")
    out = out.join(score_df)

    if "pair_group_id" in out.columns:
        # Some rows can have missing group ids; keep nullable integer ranks.
        ranks = out.groupby("pair_group_id", dropna=True)["relevance_margin"].rank(ascending=False, method="first")
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
