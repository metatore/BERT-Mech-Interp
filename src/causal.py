from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import torch

from inference import ModelBundle


@dataclass
class CounterfactualEdit:
    edit_type: str
    edited_text: str


@dataclass
class CausalLabel:
    expected_delta_direction: str | None
    expected_reason: str | None
    expected_confidence: str | None
    label_source: str


class CausalLabeler(Protocol):
    def label(self, *, query: str, original_text: str, edited_text: str, edit_type: str, esci_label: str | None = None) -> CausalLabel:
        ...


class CounterfactualEditor(Protocol):
    def generate_edits(self, *, query: str, item_text: str) -> list["CounterfactualEdit"]:
        ...


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
ALLOWED_EDIT_TYPES = ("brand_swap", "size_swap", "color_swap", "negation_flip", "category_swap")


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


def _normalize_expected_direction(value: object) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"up", "down", "neutral", "unknown"}:
        return v
    return None


def _compute_sign_consistency(expected_delta_direction: str | None, delta_margin: float) -> bool | None:
    if expected_delta_direction == "down":
        return bool(delta_margin < 0)
    if expected_delta_direction == "up":
        return bool(delta_margin > 0)
    if expected_delta_direction == "neutral":
        return bool(abs(delta_margin) < 1e-6)
    return None


class OpenAICausalLabeler:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        timeout_s: float = 30.0,
        max_retries: int = 4,
        cache_path: str | Path = "outputs/openai_causal_label_cache.jsonl",
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.cache_path = Path(cache_path)
        self.prompt_version = "causal_labeler_v2_neutral_preferred"
        self._cache: dict[str, CausalLabel] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0
        self._api_total_s = 0.0
        self._load_cache()

    def _cache_key(self, query: str, original_text: str, edited_text: str, edit_type: str, esci_label: str | None) -> str:
        s = json.dumps(
            {
                "q": query,
                "o": original_text,
                "e": edited_text,
                "t": edit_type,
                "l": esci_label,
                "m": self.model,
                "pv": self.prompt_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = str(rec.get("cache_key", "")).strip()
                if not key:
                    continue
                self._cache[key] = CausalLabel(
                    expected_delta_direction=_normalize_expected_direction(rec.get("expected_delta_direction")),
                    expected_reason=None if rec.get("expected_reason") is None else str(rec.get("expected_reason")),
                    expected_confidence=(str(rec.get("expected_confidence")).lower() if rec.get("expected_confidence") is not None else None),
                    label_source=str(rec.get("label_source", "openai_judge")),
                )
        except Exception:
            self._cache = {}

    def _append_cache(
        self,
        key: str,
        *,
        query: str,
        original_text: str,
        edited_text: str,
        edit_type: str,
        esci_label: str | None,
        label: CausalLabel,
    ) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "cache_key": key,
            "query": query,
            "original_text": original_text,
            "edited_text": edited_text,
            "edit_type": edit_type,
            "esci_label": esci_label,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "expected_delta_direction": label.expected_delta_direction,
            "expected_reason": label.expected_reason,
            "expected_confidence": label.expected_confidence,
            "label_source": label.label_source,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def stats_snapshot(self) -> dict[str, float]:
        avg_api = (self._api_total_s / self._api_calls) if self._api_calls else 0.0
        return {
            "cache_hits": float(self._cache_hits),
            "cache_misses": float(self._cache_misses),
            "api_calls": float(self._api_calls),
            "avg_api_latency_s": avg_api,
        }

    def _request_json(self, payload: dict[str, object]) -> dict[str, object]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                self._api_calls += 1
                self._api_total_s += max(time.time() - t0, 0.0)
                return parsed
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = "<unable to read error body>"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
                last_exc = RuntimeError(f"OpenAI HTTP {exc.code}: {body}")
                sleep_s = min(2 ** attempt, 8)
                print(
                    f"[CAUSAL] OpenAI transient HTTP {exc.code}; retry {attempt + 1}/{self.max_retries} in {sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(2 ** attempt, 8)
                print(
                    f"[CAUSAL] OpenAI timeout/network error; retry {attempt + 1}/{self.max_retries} in {sleep_s}s: {exc}",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after retries: {last_exc}")

    def label(self, *, query: str, original_text: str, edited_text: str, edit_type: str, esci_label: str | None = None) -> CausalLabel:
        key = self._cache_key(query, original_text, edited_text, edit_type, esci_label)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        sys_prompt = (
            "You are labeling expected relevance direction for a search ranking counterfactual. "
            "Judge whether the EDITED item should be more relevant, less relevant, or uncertain relative to the ORIGINAL item for the given QUERY. "
            "Use 'neutral' when no meaningful relevance change is expected for the query. Use 'unknown' only for genuinely ambiguous/contradictory cases. "
            "Return strict JSON only."
        )
        user_prompt = {
            "task": "counterfactual_expected_direction",
            "query": query,
            "original_item_text": original_text,
            "edited_item_text": edited_text,
            "edit_type": edit_type,
            "original_esci_label_if_known": esci_label or None,
            "instructions": [
                "Decide expected_delta_direction as one of: up, down, neutral, unknown.",
                "Prefer neutral when the changed attribute is not relevant to the query intent and no meaningful score change is expected.",
                "Use unknown only when the effect is genuinely ambiguous, contradictory, or insufficiently grounded.",
                "Provide a short plain-English reason.",
                "Provide confidence as one of: high, medium, low.",
            ],
            "json_schema": {
                "expected_delta_direction": "up|down|neutral|unknown",
                "expected_reason": "string",
                "expected_confidence": "high|medium|low",
            },
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
        }
        try:
            resp = self._request_json(payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, RuntimeError) as exc:
            raise RuntimeError(f"OpenAI causal labeling request failed: {exc}") from exc

        direction = _normalize_expected_direction(parsed.get("expected_delta_direction"))
        reason = parsed.get("expected_reason")
        conf = parsed.get("expected_confidence")
        if isinstance(reason, str):
            reason = reason.strip()
        else:
            reason = None
        if conf is not None:
            conf = str(conf).strip().lower()
        if conf not in {"high", "medium", "low"}:
            conf = None
        out = CausalLabel(
            expected_delta_direction=direction,
            expected_reason=reason,
            expected_confidence=conf,
            label_source="openai_judge",
        )
        self._cache[key] = out
        self._append_cache(
            key,
            query=query,
            original_text=original_text,
            edited_text=edited_text,
            edit_type=edit_type,
            esci_label=esci_label,
            label=out,
        )
        return out


class OpenAICounterfactualEditor:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        timeout_s: float = 45.0,
        max_retries: int = 4,
        cache_path: str | Path = "outputs/openai_edit_cache.jsonl",
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.cache_path = Path(cache_path)
        self.prompt_version = "openai_editor_v1"
        self._cache: dict[str, list[CounterfactualEdit]] = {}
        self._load_cache()

    def _key(self, query: str, item_text: str) -> str:
        s = json.dumps({"q": query, "i": item_text, "m": self.model, "pv": self.prompt_version}, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = str(rec.get("cache_key", "")).strip()
                if not key:
                    continue
                item_text = str(rec.get("item_text", ""))
                self._cache[key] = self._parse_edits(rec.get("edits", []), item_text)
        except Exception:
            self._cache = {}

    def _append_cache(self, key: str, query: str, item_text: str, edits: list[CounterfactualEdit]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "cache_key": key,
            "query": query,
            "item_text": item_text,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "edits": [{"edit_type": e.edit_type, "edited_text": e.edited_text} for e in edits],
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _request_json(self, payload: dict[str, object]) -> dict[str, object]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = "<unable to read error body>"
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
                last_exc = RuntimeError(f"OpenAI HTTP {exc.code}: {body}")
                sleep_s = min(2 ** attempt, 8)
                print(f"[CAUSAL] OpenAI edit transient HTTP {exc.code}; retry {attempt + 1}/{self.max_retries} in {sleep_s}s", flush=True)
                time.sleep(sleep_s)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(2 ** attempt, 8)
                print(f"[CAUSAL] OpenAI edit timeout/network error; retry {attempt + 1}/{self.max_retries} in {sleep_s}s: {exc}", flush=True)
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI edit request failed after retries: {last_exc}")

    def _parse_edits(self, raw: object, original_text: str) -> list[CounterfactualEdit]:
        edits: list[CounterfactualEdit] = []
        seen_types: set[str] = set()
        if not isinstance(raw, list):
            return edits
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            edit_type = str(rec.get("edit_type", "")).strip().lower()
            edited_text = str(rec.get("edited_text", "")).strip()
            if edit_type not in ALLOWED_EDIT_TYPES or edit_type in seen_types:
                continue
            if not edited_text or edited_text == original_text:
                continue
            seen_types.add(edit_type)
            edits.append(CounterfactualEdit(edit_type=edit_type, edited_text=edited_text))
        return edits

    def generate_edits(self, *, query: str, item_text: str) -> list[CounterfactualEdit]:
        key = self._key(query, item_text)
        if key in self._cache:
            return self._cache[key]
        sys_prompt = (
            "You generate realistic e-commerce item-title counterfactual edits for model stress testing. "
            "Return strict JSON only. Keep edits minimal and grammatical. Preserve word order where possible."
        )
        user_payload = {
            "task": "generate_counterfactual_edits",
            "query": query,
            "item_text": item_text,
            "allowed_edit_types": list(ALLOWED_EDIT_TYPES),
            "instructions": [
                "Generate at most one edited_text per edit_type and omit non-applicable types.",
                "Edits must remain natural product titles, not prompts or fragments.",
                "Prefer edits on attributes likely relevant to the query intent.",
                "For negation_flip, prefer in-place edits like with->without, not prefixing the whole title.",
                "Do not output explanations.",
            ],
            "example_bad_negation": {
                "item_text": "Laptop sleeve with heavy duty zipper",
                "bad_edit": "without Laptop sleeve with heavy duty zipper",
                "good_edit": "Laptop sleeve without heavy duty zipper",
            },
            "json_schema": {"edits": [{"edit_type": "one of allowed_edit_types", "edited_text": "string"}]},
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }
        try:
            resp = self._request_json(payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, RuntimeError) as exc:
            raise RuntimeError(f"OpenAI counterfactual edit generation failed: {exc}") from exc
        edits = self._parse_edits(parsed.get("edits", []), item_text)
        self._cache[key] = edits
        self._append_cache(key, query, item_text, edits)
        return edits


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
    rules: list[tuple[str, Callable[[str], str]]] = [
        ("brand_swap", lambda t: _swap_first(t, BRAND_SWAP)),
        ("size_swap", _size_swap),
        ("color_swap", lambda t: _swap_first(t, COLOR_SWAP)),
        ("category_swap", lambda t: _swap_first(t, CATEGORY_SWAP)),
    ]
    for edit_type, fn in rules:
        edited = fn(item_text)
        if edited != item_text:
            edits.append(CounterfactualEdit(edit_type=edit_type, edited_text=edited))

    negated, _neg_expected = _negation_flip(item_text)
    if negated != item_text:
        edits.append(CounterfactualEdit(edit_type="negation_flip", edited_text=negated))
    return edits


def generate_counterfactual_results(
    scored_df: pd.DataFrame,
    bundle: ModelBundle,
    labeler: CausalLabeler | None = None,
    editor: CounterfactualEditor | None = None,
) -> pd.DataFrame:
    required = {"probe_id", "query", "item_text"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for causal generation: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    total_qips = len(scored_df)
    start = time.time()
    processed_qips = 0
    total_edits = 0
    print(f"[CAUSAL] Generating counterfactuals for {total_qips} scored rows", flush=True)
    for _, row in scored_df.iterrows():
        probe_id = str(row["probe_id"])
        query = str(row["query"])
        item_text = str(row["item_text"])
        pair_group_id = str(row.get("pair_group_id", ""))
        esci_label = None if pd.isna(row.get("esci_label")) else str(row.get("esci_label", ""))

        original_margin = float(row["relevance_margin"]) if "relevance_margin" in row else None
        original_prob = float(row["relevance_prob"]) if "relevance_prob" in row else None
        if (
            original_margin is None
            or original_prob is None
            or pd.isna(original_margin)
            or pd.isna(original_prob)
        ):
            original_margin, original_prob = _score_pair(bundle, query, item_text)

        edits = editor.generate_edits(query=query, item_text=item_text) if editor is not None else _edit_candidates(item_text)
        total_edits += len(edits)
        for edit in edits:
            edited_margin, edited_prob = _score_pair(bundle, query, edit.edited_text)
            delta_margin = edited_margin - original_margin
            delta_prob = edited_prob - original_prob
            if labeler is not None:
                judgment = labeler.label(
                    query=query,
                    original_text=item_text,
                    edited_text=edit.edited_text,
                    edit_type=edit.edit_type,
                    esci_label=esci_label,
                )
            else:
                judgment = CausalLabel(
                    expected_delta_direction=None,
                    expected_reason=None,
                    expected_confidence=None,
                    label_source="disabled_no_api_key",
                )
            sign_consistent = _compute_sign_consistency(judgment.expected_delta_direction, delta_margin)

            rows.append(
                {
                    "probe_id": probe_id,
                    "pair_group_id": pair_group_id,
                    "query": query,
                    "item_text": item_text,
                    "edit_type": edit.edit_type,
                    "original_text": item_text,
                    "edited_text": edit.edited_text,
                    "expected_delta_direction": judgment.expected_delta_direction,
                    "expected_reason": judgment.expected_reason,
                    "expected_confidence": judgment.expected_confidence,
                    "label_source": judgment.label_source,
                    "original_margin": original_margin,
                    "edited_margin": edited_margin,
                    "delta_margin": delta_margin,
                    "original_prob": original_prob,
                    "edited_prob": edited_prob,
                    "delta_prob": delta_prob,
                    "sign_consistent": sign_consistent,
                    "artifact_kind": "causal",
                }
            )
        processed_qips += 1
        if processed_qips % 10 == 0 or processed_qips == total_qips:
            elapsed = max(time.time() - start, 1e-6)
            msg = (
                f"[CAUSAL] Processed {processed_qips}/{total_qips} rows "
                f"({processed_qips/elapsed:.2f} rows/s), generated_edits={total_edits}, output_rows={len(rows)}"
            )
            if labeler is not None and hasattr(labeler, "stats_snapshot"):
                try:
                    s = labeler.stats_snapshot()
                    msg += (
                        f" | label cache hits={int(s['cache_hits'])} misses={int(s['cache_misses'])}"
                        f" | api calls={int(s['api_calls'])} avg_api={s['avg_api_latency_s']:.2f}s"
                    )
                except Exception:
                    pass
            print(msg, flush=True)

    return pd.DataFrame(rows)
