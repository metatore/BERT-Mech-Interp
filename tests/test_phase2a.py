from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attribution import token_gradient_attribution
from causal import CausalLabel, generate_counterfactual_results
from inference import BaseCrossEncoderAdapter, HFCrossEncoderAdapter, ModelBundle, score_pairs
from probes import tag_question


class DummyTokenizer:
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __call__(self, queries, items, **kwargs):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(items, str):
            items = [items]
        input_ids = []
        token_type_ids = []
        attention_mask = []
        for q, i in zip(queries, items):
            q_id = 10 + (len(q) % 50)
            i_id = 20 + (len(i) % 50)
            ids = [101, q_id, 102, i_id, 102]
            tti = [0, 0, 0, 1, 1]
            mask = [1, 1, 1, 1, 1]
            input_ids.append(ids)
            token_type_ids.append(tti)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        out = []
        for x in ids:
            if x == 101:
                out.append("[CLS]")
            elif x == 102:
                out.append("[SEP]")
            elif x == 0:
                out.append("[PAD]")
            else:
                out.append(f"tok{x}")
        return out


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim: int = 1):
        super().__init__()
        self.emb = torch.nn.Embedding(512, 16)
        self.proj = torch.nn.Linear(16, out_dim)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, token_type_ids=None, output_attentions=False):
        if inputs_embeds is None:
            x = self.emb(input_ids)
        else:
            x = inputs_embeds
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)
        logits = self.proj(pooled)
        if output_attentions:
            bsz, seq, _ = x.shape
            attn = torch.ones((bsz, 2, seq, seq), dtype=x.dtype, device=x.device)
            attn = attn / seq
            return SimpleNamespace(logits=logits, attentions=(attn, attn))
        return SimpleNamespace(logits=logits)


class ReverseProbAdapter(BaseCrossEncoderAdapter):
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = DummyTokenizer()
        self.model = DummyModel(out_dim=1).eval()

    def model_id(self) -> str:
        return "reverse-prob-adapter"

    def tokenize_pairs(self, queries, items, **kwargs):
        return self.tokenizer(queries, items, **kwargs)

    def forward_logits(self, batch):
        # Build logits from item token id so margins are deterministic.
        item_ids = batch["input_ids"][:, 3].float()
        return item_ids.unsqueeze(-1)

    def extract_relevance_signal(self, logits):
        margins = logits[:, 0].detach().cpu().numpy()
        probs = 1.0 - (1.0 / (1.0 + np.exp(-margins)))
        n = len(margins)
        return SimpleNamespace(
            relevance_margin=margins.astype(np.float32),
            relevance_prob=probs.astype(np.float32),
            raw_logit=margins.astype(np.float32),
            logit_pos=np.full(n, np.nan, dtype=np.float32),
            logit_neg=np.full(n, np.nan, dtype=np.float32),
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def supports_attentions(self) -> bool:
        return True

    def supports_hidden_states(self) -> bool:
        return True

    def supports_token_type_ids(self) -> bool:
        return True


class Phase2ATests(unittest.TestCase):
    def test_extract_relevance_signal_single_and_two_class(self):
        adapter = HFCrossEncoderAdapter(
            tokenizer=DummyTokenizer(),
            model=DummyModel(out_dim=1).eval(),
            device=torch.device("cpu"),
            name="dummy",
        )
        single = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
        single_sig = adapter.extract_relevance_signal(single)
        self.assertEqual(single_sig.relevance_margin.shape[0], 2)
        self.assertAlmostEqual(float(single_sig.relevance_prob[0]), 0.5, places=5)
        self.assertTrue(np.isfinite(single_sig.raw_logit).all())

        two_logits = torch.tensor([[0.0, 1.0], [3.0, 1.0]], dtype=torch.float32)
        two_sig = adapter.extract_relevance_signal(two_logits)
        self.assertTrue(np.isfinite(two_sig.logit_pos).all())
        self.assertTrue(np.isfinite(two_sig.logit_neg).all())
        self.assertAlmostEqual(float(two_sig.relevance_margin[0]), 1.0, places=5)

    def test_score_pairs_ranks_by_margin_not_prob(self):
        bundle = ModelBundle(adapter=ReverseProbAdapter())
        df = pd.DataFrame(
            [
                {
                    "probe_id": "a",
                    "query": "usb cable",
                    "item_text": "short",
                    "pair_group_id": "g1",
                    "expected_direction": "should_rank_higher",
                },
                {
                    "probe_id": "b",
                    "query": "usb cable",
                    "item_text": "very very long item text",
                    "pair_group_id": "g1",
                    "expected_direction": "should_rank_lower",
                },
            ]
        )
        out = score_pairs(bundle, df, batch_size=2)
        top = out.sort_values("rank_in_group").iloc[0]
        self.assertEqual(top["probe_id"], "b")

    def test_attribution_supports_both_methods(self):
        bundle = ModelBundle(
            adapter=HFCrossEncoderAdapter(
                tokenizer=DummyTokenizer(),
                model=DummyModel(out_dim=2).eval(),
                device=torch.device("cpu"),
                name="dummy-2class",
            )
        )
        grad_df = token_gradient_attribution(bundle, "nike shoes", "red running shoes", method="grad_x_embed_saliency")
        ig_df = token_gradient_attribution(bundle, "nike shoes", "red running shoes", method="integrated_gradients", ig_steps=8)
        self.assertGreater(len(grad_df), 0)
        self.assertGreater(len(ig_df), 0)
        self.assertIn("attribution_method", grad_df.columns)
        self.assertEqual(grad_df["attribution_kind"].iloc[0], "observational")
        self.assertEqual(ig_df["attribution_method"].iloc[0], "integrated_gradients")

    def test_counterfactual_generation_and_schema(self):
        class DummyLabeler:
            def label(self, **kwargs):
                return CausalLabel(
                    expected_delta_direction="down",
                    expected_reason="test",
                    expected_confidence="high",
                    label_source="test_judge",
                    expected_edited_esci_label="S",
                )

        bundle = ModelBundle(
            adapter=HFCrossEncoderAdapter(
                tokenizer=DummyTokenizer(),
                model=DummyModel(out_dim=2).eval(),
                device=torch.device("cpu"),
                name="dummy-2class",
            )
        )
        scored = pd.DataFrame(
            [
                {
                    "probe_id": "p1",
                    "pair_group_id": "g1",
                    "query": "nike black shoes",
                    "item_text": "nike black shoes 10 oz",
                }
            ]
        )
        out = generate_counterfactual_results(scored, bundle, labeler=DummyLabeler())
        self.assertGreater(len(out), 0)
        self.assertTrue((out["artifact_kind"] == "causal").all())
        self.assertTrue(out["edited_text"].map(lambda x: isinstance(x, str) and len(x) > 0).all())
        self.assertTrue(out["expected_delta_direction"].isin({"up", "down", "neutral"}).all())
        self.assertTrue(out["sign_consistent"].map(lambda x: isinstance(x, (bool, np.bool_))).all())
        self.assertTrue((out["label_source"] == "test_judge").all())
        self.assertIn("expected_edited_esci_label", out.columns)
        self.assertIn("threshold_check", out.columns)
        self.assertIn("causal_result_v2", out.columns)
        self.assertTrue(out["expected_edited_esci_label"].isin({"E", "S", "C", "I"}).all())

    def test_counterfactual_labels_disabled_without_labeler(self):
        bundle = ModelBundle(
            adapter=HFCrossEncoderAdapter(
                tokenizer=DummyTokenizer(),
                model=DummyModel(out_dim=2).eval(),
                device=torch.device("cpu"),
                name="dummy-2class",
            )
        )
        scored = pd.DataFrame([{"probe_id": "p1", "pair_group_id": "g1", "query": "nike shoes", "item_text": "nike shoes"}])
        out = generate_counterfactual_results(scored, bundle, labeler=None)
        self.assertGreater(len(out), 0)
        self.assertTrue(out["expected_delta_direction"].isna().all())
        self.assertTrue(out["sign_consistent"].isna().all())

    def test_tagging_allows_unclassified(self):
        decision = tag_question("random search phrase", "generic product listing")
        self.assertEqual(decision.question_tag, "unclassified")
        self.assertIn(decision.tag_reason, {"manual", "regex", "fallback"})
        self.assertIn(decision.tag_confidence, {"high", "heuristic", "low"})


if __name__ == "__main__":
    unittest.main()
