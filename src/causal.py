from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import re
import socket
import ssl
import threading
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
    expected_edited_esci_label: str | None = None


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
ESCI_ORDER = {"I": 0, "C": 1, "S": 2, "E": 3}
EXACT_PROB_THRESHOLD = 0.9

CAUSAL_LABELER_SYSTEM_PROMPT = (
    "You are labeling the expected ESCI relevance label of an EDITED query-item pair for a search ranking counterfactual. "
    "Infer how the EDITED item should be labeled for the QUERY using ESCI categories (Exact, Substitute, Complement, Irrelevant). "
    "Use 'unknown' only when genuinely ambiguous or insufficiently grounded. "
    "Return strict JSON only."
)
CAUSAL_LABELER_INSTRUCTIONS = [
    "Decide expected_edited_esci_label as one of: E, S, C, I, unknown.",
    "Interpret E/S/C/I as Exact/Substitute/Complement/Irrelevant for the QUERY.",
    "Provide expected_delta_direction relative to the ORIGINAL item as one of: up, down, neutral, unknown.",
    "If original_esci_label_if_known is missing or unclear, infer the edited label anyway and use unknown for direction when needed.",
    "Provide a short plain-English reason.",
    "Provide confidence as one of: high, medium, low.",
]
CAUSAL_LABELER_JSON_SCHEMA = {
    "expected_edited_esci_label": "E|S|C|I|unknown",
    "expected_delta_direction": "up|down|neutral|unknown",
    "expected_reason": "string",
    "expected_confidence": "high|medium|low",
}
# Include post-processing semantics here because cache validity depends on both
# prompt text and how we interpret model outputs.
CAUSAL_LABELER_CACHE_SEMANTICS = {
    "direction_resolution": "preserve_same_label_judge_direction_enforce_cross_label_ordinal",
    "threshold_causal_mapping": "transition_based_inherited_fail_not_counted_as_introduced",
}


def _causal_labeler_prompt_contract_spec() -> dict[str, object]:
    return {
        "system_prompt": CAUSAL_LABELER_SYSTEM_PROMPT,
        "instructions": CAUSAL_LABELER_INSTRUCTIONS,
        "json_schema": CAUSAL_LABELER_JSON_SCHEMA,
        "cache_semantics": CAUSAL_LABELER_CACHE_SEMANTICS,
    }


def _causal_labeler_prompt_version() -> str:
    spec = _causal_labeler_prompt_contract_spec()
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"causal_labeler_contract_{digest}"


def _build_ssl_context() -> ssl.SSLContext | None:
    # Prefer certifi automatically on local macOS/Python installs where system trust can be misconfigured.
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


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


def _normalize_esci_label(value: object) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    mapping = {
        "e": "E",
        "exact": "E",
        "s": "S",
        "substitute": "S",
        "c": "C",
        "complement": "C",
        "i": "I",
        "irrelevant": "I",
        "unknown": None,
    }
    return mapping.get(v, None)


def _derive_expected_direction_from_transition(original_esci_label: str | None, expected_edited_esci_label: str | None) -> str | None:
    orig = _normalize_esci_label(original_esci_label)
    new = _normalize_esci_label(expected_edited_esci_label)
    if orig is None or new is None:
        return None
    if ESCI_ORDER[new] > ESCI_ORDER[orig]:
        return "up"
    if ESCI_ORDER[new] < ESCI_ORDER[orig]:
        return "down"
    return "neutral"


def _resolve_expected_direction(
    original_esci_label: str | None,
    expected_edited_esci_label: str | None,
    parsed_expected_direction: object,
) -> str | None:
    """Reconcile coarse ESCI transition with the judge's directional expectation.

    ESCI transitions are ordinal and coarse. A same-label transition (e.g., I->I)
    does not imply zero score change, so preserve the judge's up/down/neutral
    signal in that case. For cross-label transitions, enforce the ordinal
    direction implied by the transition.
    """
    derived = _derive_expected_direction_from_transition(original_esci_label, expected_edited_esci_label)
    parsed = _normalize_expected_direction(parsed_expected_direction)
    if parsed == "unknown":
        parsed = None

    if derived in {"up", "down"}:
        return derived
    if derived == "neutral":
        return parsed if parsed in {"up", "down", "neutral"} else "neutral"
    return parsed


def _compute_sign_consistency(expected_delta_direction: str | None, delta_margin: float) -> bool | None:
    if expected_delta_direction == "down":
        return bool(delta_margin < 0)
    if expected_delta_direction == "up":
        return bool(delta_margin > 0)
    if expected_delta_direction == "neutral":
        return bool(abs(delta_margin) < 1e-6)
    return None


def _actual_rank_against_group(edited_margin: float, peer_margins: list[float]) -> int:
    return 1 + sum(1 for m in peer_margins if m > edited_margin)


def _rank_movement_label(rank_delta: int | None) -> str | None:
    if rank_delta is None:
        return None
    if rank_delta < 0:
        return "up"
    if rank_delta > 0:
        return "down"
    return "neutral"


def _compute_rank_movement_check(expected_delta_direction: str | None, rank_delta: int | None) -> str | None:
    if expected_delta_direction is None or rank_delta is None:
        return None
    actual = _rank_movement_label(rank_delta)
    if actual is None:
        return None
    if expected_delta_direction == "neutral":
        return "pass" if actual == "neutral" else "fail"
    if actual == "neutral":
        return "marginal"
    return "pass" if actual == expected_delta_direction else "fail"


def _compute_pairwise_esci_order_check(
    expected_edited_esci_label: str | None,
    edited_margin: float,
    peers: list[dict[str, object]],
    self_probe_id: str,
) -> tuple[str | None, int, int]:
    expected_lbl = _normalize_esci_label(expected_edited_esci_label)
    if expected_lbl is None:
        return None, 0, 0

    passes = 0
    fails = 0
    for peer in peers:
        if str(peer.get("probe_id", "")) == self_probe_id:
            continue
        peer_lbl = _normalize_esci_label(peer.get("esci_label"))
        if peer_lbl is None:
            continue
        if peer_lbl == expected_lbl:
            continue
        peer_margin = peer.get("relevance_margin")
        if peer_margin is None or pd.isna(peer_margin):
            continue
        peer_margin_f = float(peer_margin)
        if ESCI_ORDER[expected_lbl] > ESCI_ORDER[peer_lbl]:
            if edited_margin > peer_margin_f:
                passes += 1
            else:
                fails += 1
        else:
            if edited_margin < peer_margin_f:
                passes += 1
            else:
                fails += 1

    if passes == 0 and fails == 0:
        return None, 0, 0
    if fails > 0:
        return "fail", passes, fails
    return "pass", passes, fails


def _compute_threshold_check(expected_edited_esci_label: str | None, edited_prob: float, threshold: float = EXACT_PROB_THRESHOLD) -> str | None:
    edited_lbl = _normalize_esci_label(expected_edited_esci_label)
    if edited_lbl is None:
        return None
    if edited_lbl == "E":
        return "pass" if edited_prob > threshold else "fail"
    return "pass" if edited_prob < threshold else "fail"


def _compute_threshold_transition_status(
    original_esci_label: str | None,
    expected_edited_esci_label: str | None,
    original_prob: float,
    edited_prob: float,
    threshold: float = EXACT_PROB_THRESHOLD,
) -> tuple[str | None, str | None, str | None]:
    """Return (original_abs_check, edited_abs_check, transition_status).

    transition_status is causal-facing and distinguishes inherited baseline
    failures from violations introduced by the edit.
    """
    orig_lbl = _normalize_esci_label(original_esci_label)
    edited_lbl = _normalize_esci_label(expected_edited_esci_label)
    original_check = _compute_threshold_check(orig_lbl, original_prob, threshold=threshold) if orig_lbl is not None else None
    edited_check = _compute_threshold_check(edited_lbl, edited_prob, threshold=threshold) if edited_lbl is not None else None
    if original_check is None and edited_check is None:
        return original_check, edited_check, None
    if original_check is None:
        return original_check, edited_check, ("introduced_fail" if edited_check == "fail" else "edited_only_pass")
    if edited_check is None:
        return original_check, edited_check, "original_only_context"

    if original_check == "pass" and edited_check == "pass":
        return original_check, edited_check, "unchanged_pass"
    if original_check == "pass" and edited_check == "fail":
        return original_check, edited_check, "introduced_fail"
    if original_check == "fail" and edited_check == "pass":
        return original_check, edited_check, "resolved_fail"

    # Both fail. If the exactness bucket (E vs non-E) is unchanged, this is
    # clearly inherited baseline miscalibration rather than a new causal error.
    same_exactness_bucket = (orig_lbl == "E") == (edited_lbl == "E")
    if same_exactness_bucket:
        return original_check, edited_check, "inherited_fail"
    return original_check, edited_check, "persistent_fail_across_transition"


def _compute_threshold_check_regression_aware(
    original_esci_label: str | None,
    expected_edited_esci_label: str | None,
    original_prob: float,
    edited_prob: float,
    threshold: float = EXACT_PROB_THRESHOLD,
) -> str | None:
    """Avoid attributing inherited absolute-calibration failures to a same-class edit.

    Example: original expected I and edited expected I, but both probabilities are
    above the Exact threshold. The edit did not introduce the threshold violation;
    the model was already miscalibrated on the original pair.
    """
    _orig_check, edited_status, transition_status = _compute_threshold_transition_status(
        original_esci_label,
        expected_edited_esci_label,
        original_prob,
        edited_prob,
        threshold=threshold,
    )
    if edited_status is None:
        return None
    if transition_status == "introduced_fail":
        return "fail"
    if transition_status in {"inherited_fail", "persistent_fail_across_transition"}:
        return "inherited_fail"
    return edited_status


def _compose_causal_result_v2(
    rank_movement_check: str | None,
    pairwise_order_check: str | None,
    threshold_check: str | None,
) -> tuple[str, str]:
    failures: list[str] = []
    if rank_movement_check == "fail" or pairwise_order_check == "fail":
        failures.append("order")
    if threshold_check == "fail":
        failures.append("threshold")

    if len(failures) == 2:
        return "fail_both", "ESCI ordering/rank movement and E/non-E threshold checks failed"
    if failures == ["order"]:
        return "fail_order", "ESCI ordering/rank movement did not match expected label transition"
    if failures == ["threshold"]:
        return "fail_threshold", f"Edited score violated threshold policy (E>{EXACT_PROB_THRESHOLD:.1f}, non-E<{EXACT_PROB_THRESHOLD:.1f})"

    if rank_movement_check == "marginal":
        return "marginal", "No rank movement despite expected label transition"

    positive_checks = [x for x in (rank_movement_check, pairwise_order_check, threshold_check) if x == "pass"]
    judged_checks = [x for x in (rank_movement_check, pairwise_order_check, threshold_check) if x is not None]
    if positive_checks and not failures:
        return "pass", "Edited score/rank behavior matched expected ESCI label transition"
    if judged_checks and not failures:
        return "marginal", "No clear order impact; threshold/order checks did not indicate failure"
    return "ambiguous", "Insufficient group context or unknown expected edited ESCI label"


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
        self.prompt_version = _causal_labeler_prompt_version()
        self._cache: dict[str, CausalLabel] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0
        self._api_total_s = 0.0
        self._ssl_context = _build_ssl_context()
        self._lock = threading.Lock()
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
                    expected_edited_esci_label=_normalize_esci_label(rec.get("expected_edited_esci_label")),
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
            "expected_edited_esci_label": label.expected_edited_esci_label,
            "expected_reason": label.expected_reason,
            "expected_confidence": label.expected_confidence,
            "label_source": label.label_source,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def stats_snapshot(self) -> dict[str, float]:
        with self._lock:
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
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=self._ssl_context) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                with self._lock:
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
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        sys_prompt = CAUSAL_LABELER_SYSTEM_PROMPT
        user_prompt = {
            "task": "counterfactual_expected_edited_esci_label",
            "query": query,
            "original_item_text": original_text,
            "edited_item_text": edited_text,
            "edit_type": edit_type,
            "original_esci_label_if_known": esci_label or None,
            "instructions": CAUSAL_LABELER_INSTRUCTIONS,
            "json_schema": CAUSAL_LABELER_JSON_SCHEMA,
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

        expected_edited_esci_label = _normalize_esci_label(parsed.get("expected_edited_esci_label"))
        direction = _resolve_expected_direction(
            esci_label,
            expected_edited_esci_label,
            parsed.get("expected_delta_direction"),
        )
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
            expected_edited_esci_label=expected_edited_esci_label,
        )
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                return existing
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
        self._ssl_context = _build_ssl_context()
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
                with urllib.request.urlopen(req, timeout=self.timeout_s, context=self._ssl_context) as resp:
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
    label_max_workers: int = 1,
) -> pd.DataFrame:
    required = {"probe_id", "query", "item_text"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for causal generation: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    group_rows: dict[str, list[dict[str, object]]] = {}
    if "pair_group_id" in scored_df.columns:
        for gid, part in scored_df.groupby("pair_group_id", dropna=False):
            key = "" if pd.isna(gid) else str(gid)
            group_rows[key] = part.to_dict(orient="records")
    total_qips = len(scored_df)
    start = time.time()
    processed_qips = 0
    total_edits = 0
    use_parallel_labeling = bool(labeler is not None and int(label_max_workers) > 1)
    label_workers = max(int(label_max_workers), 1)
    print(f"[CAUSAL] Generating counterfactuals for {total_qips} scored rows", flush=True)
    if use_parallel_labeling:
        print(f"[CAUSAL] OpenAI label parallelism enabled (max_workers={label_workers})", flush=True)
    label_executor = ThreadPoolExecutor(max_workers=label_workers) if use_parallel_labeling else None
    try:
        for _, row in scored_df.iterrows():
            probe_id = str(row["probe_id"])
            query = str(row["query"])
            item_text = str(row["item_text"])
            raw_pair_group_id = row.get("pair_group_id", "")
            pair_group_id = "" if pd.isna(raw_pair_group_id) else str(raw_pair_group_id)
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
            scored_edits: list[tuple[CounterfactualEdit, float, float, float, float]] = []
            for edit in edits:
                edited_margin, edited_prob = _score_pair(bundle, query, edit.edited_text)
                delta_margin = edited_margin - original_margin
                delta_prob = edited_prob - original_prob
                scored_edits.append((edit, edited_margin, edited_prob, delta_margin, delta_prob))

            judgments_by_idx: list[CausalLabel] = []
            if labeler is None:
                judgments_by_idx = [
                    CausalLabel(
                        expected_delta_direction=None,
                        expected_reason=None,
                        expected_confidence=None,
                        label_source="disabled_no_api_key",
                        expected_edited_esci_label=None,
                    )
                    for _ in scored_edits
                ]
            elif label_executor is None or len(scored_edits) <= 1:
                judgments_by_idx = [
                    labeler.label(
                        query=query,
                        original_text=item_text,
                        edited_text=edit.edited_text,
                        edit_type=edit.edit_type,
                        esci_label=esci_label,
                    )
                    for (edit, _, _, _, _) in scored_edits
                ]
            else:
                futures: list[Future[CausalLabel]] = []
                for edit, _, _, _, _ in scored_edits:
                    fut = label_executor.submit(
                        labeler.label,
                        query=query,
                        original_text=item_text,
                        edited_text=edit.edited_text,
                        edit_type=edit.edit_type,
                        esci_label=esci_label,
                    )
                    futures.append(fut)
                judgments_by_idx = [f.result() for f in futures]

            for (edit, edited_margin, edited_prob, delta_margin, delta_prob), judgment in zip(scored_edits, judgments_by_idx):
                sign_consistent = _compute_sign_consistency(judgment.expected_delta_direction, delta_margin)
                peers = group_rows.get(pair_group_id, [])
                peer_margins = [
                    float(p.get("relevance_margin"))
                    for p in peers
                    if str(p.get("probe_id", "")) != probe_id and p.get("relevance_margin") is not None and not pd.isna(p.get("relevance_margin"))
                ]
                original_rank_in_group = _actual_rank_against_group(original_margin, peer_margins) if peer_margins else None
                edited_rank_in_group = _actual_rank_against_group(edited_margin, peer_margins) if peer_margins else None
                rank_delta = (
                    int(edited_rank_in_group - original_rank_in_group)
                    if original_rank_in_group is not None and edited_rank_in_group is not None
                    else None
                )
                actual_rank_movement = _rank_movement_label(rank_delta)
                rank_movement_check = _compute_rank_movement_check(judgment.expected_delta_direction, rank_delta)
                pairwise_order_check, pairwise_order_passes, pairwise_order_fails = _compute_pairwise_esci_order_check(
                    judgment.expected_edited_esci_label,
                    edited_margin,
                    peers,
                    probe_id,
                )
                original_threshold_check, edited_threshold_check, threshold_change_status = _compute_threshold_transition_status(
                    esci_label,
                    judgment.expected_edited_esci_label,
                    original_prob,
                    edited_prob,
                )
                threshold_check = _compute_threshold_check_regression_aware(
                    esci_label,
                    judgment.expected_edited_esci_label,
                    original_prob,
                    edited_prob,
                )
                causal_result_v2, causal_result_reason = _compose_causal_result_v2(
                    rank_movement_check=rank_movement_check,
                    pairwise_order_check=pairwise_order_check,
                    threshold_check=threshold_check,
                )
                original_esci_norm = _normalize_esci_label(esci_label)
                edited_esci_norm = _normalize_esci_label(judgment.expected_edited_esci_label)
                expected_label_transition = (
                    f"{original_esci_norm}->{edited_esci_norm}"
                    if original_esci_norm is not None and edited_esci_norm is not None
                    else None
                )

                rows.append(
                    {
                        "probe_id": probe_id,
                        "pair_group_id": pair_group_id,
                        "query": query,
                        "item_text": item_text,
                        "edit_type": edit.edit_type,
                        "original_text": item_text,
                        "edited_text": edit.edited_text,
                        "original_esci_label": original_esci_norm,
                        "expected_edited_esci_label": judgment.expected_edited_esci_label,
                        "expected_label_transition": expected_label_transition,
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
                        "original_rank_in_group": original_rank_in_group,
                        "edited_rank_in_group": edited_rank_in_group,
                        "rank_delta_in_group": rank_delta,
                        "actual_rank_movement": actual_rank_movement,
                        "rank_movement_check": rank_movement_check,
                        "pairwise_esci_order_check": pairwise_order_check,
                        "pairwise_esci_order_passes": pairwise_order_passes,
                        "pairwise_esci_order_fails": pairwise_order_fails,
                        "original_threshold_check": original_threshold_check,
                        "edited_threshold_check": edited_threshold_check,
                        "threshold_change_status": threshold_change_status,
                        "threshold_check": threshold_check,
                        "threshold_policy": f"E>{EXACT_PROB_THRESHOLD:.1f}, non-E<{EXACT_PROB_THRESHOLD:.1f}",
                        "causal_result_v2": causal_result_v2,
                        "causal_result_reason": causal_result_reason,
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
    finally:
        if label_executor is not None:
            label_executor.shutdown(wait=True)

    return pd.DataFrame(rows)
