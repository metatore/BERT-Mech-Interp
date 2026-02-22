from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


HEURISTIC_TAGGER_VERSION = "heuristic_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEURISTIC_DIR = PROJECT_ROOT / "data" / "heuristics"

DEFAULT_NEGATION_TERMS = {"not", "without", "exclude", "excluding", "except", "non", "free of", "no"}
DEFAULT_BUNDLE_TERMS = {"bundle", "pack", "set", "kit", "count"}
DEFAULT_SPEC_UNITS = {"gb", "tb", "mb", "oz", "fl oz", "inch", "in", "mm", "cm", "mah", "w", "hz", "v", "ft", "m", "ml", "l"}
DEFAULT_COLOR_WORDS = {
    "black", "white", "blue", "red", "green", "yellow", "pink", "purple", "gray", "grey",
    "silver", "gold", "brown", "orange", "beige",
}
DEFAULT_BRAND_TERMS = {
    "nike", "adidas", "puma", "reebok", "new balance", "asics", "under armour",
    "samsung", "apple", "lg", "sony", "bose", "jbl", "beats", "anker", "sandisk",
    "dell", "hp", "lenovo", "asus", "acer", "canon", "nikon", "panasonic", "philips",
    "kitchenaid", "keurig", "dyson", "shark", "instant pot", "ninja",
}
DEFAULT_STOPWORDS = {
    "the", "a", "an", "for", "with", "and", "or", "to", "of", "on", "in", "by", "from",
    "men", "mens", "women", "womens", "kid", "kids", "new", "best", "smart", "tv",
    "shoe", "shoes", "running", "wireless", "portable", "pro", "plus",
}


def _read_lexicon_txt(path: Path) -> set[str]:
    vals: set[str] = set()
    if not path.exists():
        return vals
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        vals.add(line.lower())
    return vals


def _load_lexicon(name: str, default: set[str]) -> set[str]:
    vals = _read_lexicon_txt(HEURISTIC_DIR / f"{name}.txt")
    return vals if vals else set(default)


def _phrase_set_pattern(terms: set[str], *, suffix_nonword: bool = False) -> re.Pattern[str]:
    parts = sorted({t.strip().lower() for t in terms if str(t).strip()}, key=len, reverse=True)
    if not parts:
        return re.compile(r"(?!x)x")
    escaped = "|".join(re.escape(t) for t in parts)
    if suffix_nonword:
        return re.compile(rf"\b(?:{escaped})(?=\b|\W|$)", re.IGNORECASE)
    return re.compile(rf"\b(?:{escaped})\b", re.IGNORECASE)


NEGATION_TERMS = _load_lexicon("negation_terms", DEFAULT_NEGATION_TERMS)
BUNDLE_TERMS = _load_lexicon("bundle_terms", DEFAULT_BUNDLE_TERMS)
SPEC_UNITS = _load_lexicon("spec_units", DEFAULT_SPEC_UNITS)
COLOR_WORDS = _load_lexicon("colors", DEFAULT_COLOR_WORDS)
BRAND_TERMS = _load_lexicon("brands", DEFAULT_BRAND_TERMS)
STOPWORDS = _load_lexicon("stopwords", DEFAULT_STOPWORDS)

NEGATION_PATTERN = _phrase_set_pattern(NEGATION_TERMS, suffix_nonword=True)
BUNDLE_WORD_PATTERN = _phrase_set_pattern(BUNDLE_TERMS)
PACK_COUNT_PATTERN = re.compile(
    r"\b(?:pack of\s*\d+|\d+\s*[- ]?pack|\d+\s*ct|\d+\s*count)\b",
    re.IGNORECASE,
)
SPEC_PATTERN = re.compile(
    r"\b("
    rf"\d+(?:\.\d+)?\s?(?:{'|'.join(re.escape(u) for u in sorted(SPEC_UNITS, key=len, reverse=True))})"
    r"|(?:\d+\s?x\s?\d+)"
    r"|(?:xl|xxl|xxxl|xs|s|m|l)\b"
    r")",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class ProbeConfig:
    locale: str = "us"
    use_small_version: bool = True
    max_rows: int = 50


@dataclass
class TagDecision:
    question_tag: str
    tag_reason: Literal["manual", "regex", "fallback", "openai_judge"]
    tag_confidence: Literal["high", "heuristic", "low"]
    tag_source: str = "heuristic_v1"


TAG_LABELS = (
    "negation",
    "bundle_vs_canonical",
    "attribute_match",
    "brand_match",
    "other_lexical",
    "unclassified",
)


def _normalize_text(text: str) -> str:
    t = str(text).lower()
    t = t.replace('"', " inch ")
    t = re.sub(r"\b(in\.?|inches)\b", " inch ", t)
    t = re.sub(r"\b(gbs?)\b", "gb", t)
    t = re.sub(r"\b(tbs?)\b", "tb", t)
    t = re.sub(r"\b(pack of)\s*(\d+)\b", r"\2 pack", t)
    t = re.sub(r"(\d+)\s*-\s*pack\b", r"\1 pack", t)
    t = re.sub(r"(\d+)\s*ct\b", r"\1 count", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _find_brands(text: str) -> set[str]:
    t = _normalize_text(text)
    found: set[str] = set()
    for brand in BRAND_TERMS:
        if re.search(rf"\b{re.escape(brand)}\b", t):
            found.add(brand)
    return found


def _bundle_signal(query_norm: str, item_norm: str) -> bool:
    if PACK_COUNT_PATTERN.search(query_norm) or PACK_COUNT_PATTERN.search(item_norm):
        return True
    return bool(BUNDLE_WORD_PATTERN.search(query_norm) or BUNDLE_WORD_PATTERN.search(item_norm))


def _attribute_signal(query_norm: str, item_norm: str) -> bool:
    if SPEC_PATTERN.search(query_norm) or SPEC_PATTERN.search(item_norm):
        return True
    q_tokens = set(_tokenize(query_norm))
    i_tokens = set(_tokenize(item_norm))
    # Color is treated as an attribute signal, but only if the query names a color.
    if q_tokens & COLOR_WORDS and i_tokens & COLOR_WORDS:
        return True
    return False


def _brand_signal(query_norm: str, item_norm: str) -> bool:
    q_brands = _find_brands(query_norm)
    i_brands = _find_brands(item_norm)
    return bool(q_brands or i_brands) and bool(q_brands != i_brands or (q_brands and i_brands))


def _other_lexical_signal(query_norm: str, item_norm: str) -> bool:
    q_tokens = [t for t in _tokenize(query_norm) if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
    i_tokens = [t for t in _tokenize(item_norm) if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()]
    if len(q_tokens) < 2 or len(i_tokens) < 2:
        return False
    overlap = set(q_tokens) & set(i_tokens)
    # Treat meaningful lexical overlap (not explained by prior rules) as "other_lexical".
    return len(overlap) >= 1


def tag_question(query: str, item_text: str) -> TagDecision:
    q = _normalize_text(str(query))
    i = _normalize_text(str(item_text))
    # Deterministic precedence: negation > bundle > attribute > brand > other_lexical > unclassified
    if NEGATION_PATTERN.search(q):
        return TagDecision(question_tag="negation", tag_reason="regex", tag_confidence="high", tag_source=HEURISTIC_TAGGER_VERSION)
    if _bundle_signal(q, i):
        return TagDecision(question_tag="bundle_vs_canonical", tag_reason="regex", tag_confidence="heuristic", tag_source=HEURISTIC_TAGGER_VERSION)
    if _attribute_signal(q, i):
        return TagDecision(question_tag="attribute_match", tag_reason="regex", tag_confidence="heuristic", tag_source=HEURISTIC_TAGGER_VERSION)
    if _brand_signal(q, i):
        return TagDecision(question_tag="brand_match", tag_reason="regex", tag_confidence="heuristic", tag_source=HEURISTIC_TAGGER_VERSION)
    if _other_lexical_signal(q, i):
        return TagDecision(question_tag="other_lexical", tag_reason="regex", tag_confidence="low", tag_source=HEURISTIC_TAGGER_VERSION)
    return TagDecision(question_tag="unclassified", tag_reason="fallback", tag_confidence="low", tag_source=HEURISTIC_TAGGER_VERSION)


def _normalize_tag_label(value: object) -> str:
    v = str(value).strip().lower()
    aliases = {
        "brand": "brand_match",
        "brand_match": "brand_match",
        "attribute": "attribute_match",
        "attribute_match": "attribute_match",
        "spec": "attribute_match",
        "spec_match": "attribute_match",
        "negation": "negation",
        "bundle": "bundle_vs_canonical",
        "bundle_vs_canonical": "bundle_vs_canonical",
        "other": "other_lexical",
        "other_lexical": "other_lexical",
        "unknown": "unclassified",
        "unclassified": "unclassified",
    }
    return aliases.get(v, "unclassified")


def _normalize_tag_confidence(value: object) -> Literal["high", "heuristic", "low"]:
    v = str(value).strip().lower()
    if v == "high":
        return "high"
    if v in {"medium", "heuristic"}:
        return "heuristic"
    return "low"


def _cache_key(query: str, item_text: str, prompt_version: str, model: str) -> str:
    s = json.dumps({"q": query, "i": item_text, "pv": prompt_version, "m": model}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class OpenAIQuestionTagger:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        cache_path: str | Path = "outputs/openai_tag_cache.jsonl",
        timeout_s: float = 30.0,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.prompt_version = "tagger_v1_closed_set"
        self.cache_path = Path(cache_path)
        self._cache: dict[str, TagDecision] = {}
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._api_calls = 0
        self._api_total_s = 0.0
        self._api_prompt_bytes = 0
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = str(rec.get("cache_key", ""))
                if not key:
                    continue
                self._cache[key] = TagDecision(
                    question_tag=_normalize_tag_label(rec.get("question_tag")),
                    tag_reason="openai_judge",
                    tag_confidence=_normalize_tag_confidence(rec.get("tag_confidence")),
                    tag_source=str(rec.get("tag_source", "openai_tagger_v1")),
                )
        except Exception:
            # Cache corruption should not block tagging.
            self._cache = {}

    def _append_cache(self, key: str, query: str, item_text: str, decision: TagDecision, reason_short: str) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "cache_key": key,
            "query": query,
            "item_text": item_text,
            "question_tag": decision.question_tag,
            "tag_confidence": decision.tag_confidence,
            "reason_short": reason_short[:200],
            "tag_source": decision.tag_source,
            "prompt_version": self.prompt_version,
            "model": self.model,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _request_json(self, payload: dict[str, object]) -> dict[str, object]:
        payload_bytes = len(json.dumps(payload).encode("utf-8"))
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
                self._api_prompt_bytes += payload_bytes
                return parsed
            except urllib.error.HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    body = "<unable to read error body>"
                # Retry only transient classes; surface deterministic 4xx immediately.
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
                last_exc = RuntimeError(f"OpenAI HTTP {exc.code}: {body}")
                sleep_s = min(2 ** attempt, 8)
                print(
                    f"[ESCI] OpenAI tagger transient HTTP {exc.code}; retry {attempt + 1}/{self.max_retries} in {sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                sleep_s = min(2 ** attempt, 8)
                print(
                    f"[ESCI] OpenAI tagger timeout/network error; retry {attempt + 1}/{self.max_retries} in {sleep_s}s: {exc}",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after retries: {last_exc}")

    def stats_snapshot(self) -> dict[str, float]:
        avg_api_s = (self._api_total_s / self._api_calls) if self._api_calls else 0.0
        avg_prompt_bytes = (self._api_prompt_bytes / self._api_calls) if self._api_calls else 0.0
        return {
            "cache_hits": float(self._cache_hits),
            "cache_misses": float(self._cache_misses),
            "api_calls": float(self._api_calls),
            "avg_api_latency_s": avg_api_s,
            "avg_prompt_bytes": avg_prompt_bytes,
        }

    def _prompt_messages(self, query: str, item_text: str) -> list[dict[str, str]]:
        sys_prompt = (
            "You are a deterministic classifier for search-debugging probe tags. "
            "Choose exactly one tag from a closed label set and return strict JSON only. "
            "Prefer consistency over creativity."
        )
        user_payload = {
            "task": "classify_probe_tag",
            "prompt_version": self.prompt_version,
            "allowed_tags": list(TAG_LABELS),
            "tag_definitions": {
                "negation": "Query contains a negation or exclusion constraint that changes relevance (e.g., not/without/excluding).",
                "bundle_vs_canonical": "Bundle/count/pack mismatch or single-item vs pack distinction is central.",
                "attribute_match": "Specs/attributes (size/capacity/dimension/model variant) drive the distinction.",
                "brand_match": "Brand alignment or mismatch is central.",
                "other_lexical": "Lexical phrasing mismatch matters, but not primarily brand/spec/bundle/negation.",
                "unclassified": "Ambiguous or insufficient evidence to confidently place in another bucket.",
            },
            "tie_break_rules": [
                "If negation is present in the query, choose negation.",
                "If pack/bundle/count mismatch is central, choose bundle_vs_canonical.",
                "If numeric/spec attributes drive the distinction, choose attribute_match.",
                "If brand tokens are the main distinction, choose brand_match.",
                "If multiple apply and none is clearly dominant, choose unclassified.",
            ],
            "few_shot_examples": [
                {"query": "nike running shoes", "item_text": "Adidas men's running shoes", "question_tag": "brand_match"},
                {"query": "usb c cable 2m", "item_text": "USB-C cable 1m braided", "question_tag": "attribute_match"},
                {"query": "coffee beans not decaf", "item_text": "Whole bean coffee decaf medium roast", "question_tag": "negation"},
                {"query": "aa batteries pack of 24", "item_text": "AA alkaline batteries 4-pack", "question_tag": "bundle_vs_canonical"},
            ],
            "input": {"query": query, "item_text": item_text},
            "output_schema": {
                "question_tag": "one of allowed_tags",
                "tag_confidence": "high|medium|low",
                "reason_short": "short string <= 25 words",
            },
        }
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

    def label(self, query: str, item_text: str) -> TagDecision:
        key = _cache_key(query, item_text, self.prompt_version, self.model)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": self._prompt_messages(query, item_text),
        }
        try:
            resp = self._request_json(payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError, urllib.error.URLError, RuntimeError) as exc:
            raise RuntimeError(f"OpenAI tagger request failed: {exc}") from exc

        decision = TagDecision(
            question_tag=_normalize_tag_label(parsed.get("question_tag")),
            tag_reason="openai_judge",
            tag_confidence=_normalize_tag_confidence(parsed.get("tag_confidence")),
            tag_source=f"openai_tagger:{self.model}:{self.prompt_version}",
        )
        reason_short = str(parsed.get("reason_short", "")).strip()
        with self._lock:
            # Another worker may have populated the same key while this request was in flight.
            existing = self._cache.get(key)
            if existing is not None:
                return existing
            self._cache[key] = decision
            self._append_cache(key, query, item_text, decision, reason_short)
        return decision


def apply_openai_tags(
    df: pd.DataFrame,
    tagger: OpenAIQuestionTagger,
    *,
    max_workers: int = 8,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    total = len(out)
    print(f"[ESCI] OpenAI re-tagging selected rows: {total} (workers={max_workers})", flush=True)
    decisions: dict[int, TagDecision] = {}
    start = time.time()

    def _task(ix: int, q: str, t: str) -> tuple[int, TagDecision]:
        return ix, tagger.label(q, t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        futures = [
            pool.submit(_task, idx, str(row["query"]), str(row["item_text"]))
            for idx, row in out.iterrows()
        ]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            ix, dec = fut.result()
            decisions[ix] = dec
            done += 1
            if done % 10 == 0 or done == total:
                elapsed = max(time.time() - start, 1e-6)
                s = tagger.stats_snapshot()
                print(
                    f"[ESCI] OpenAI re-tagged {done}/{total} ({done/elapsed:.1f} rows/s)"
                    f" | cache hits={int(s['cache_hits'])} misses={int(s['cache_misses'])}"
                    f" | api calls={int(s['api_calls'])} avg_api={s['avg_api_latency_s']:.2f}s",
                    flush=True,
                )

    ordered = [decisions[idx] for idx in out.index]
    out["question_tag"] = [d.question_tag for d in ordered]
    out["tag_reason"] = [d.tag_reason for d in ordered]
    out["tag_confidence"] = [d.tag_confidence for d in ordered]
    out["tag_source"] = [d.tag_source for d in ordered]
    return out


def map_esci_to_score(label: str) -> int:
    mapping = {"E": 3, "S": 2, "C": 1, "I": 0, "exact": 3, "substitute": 2, "complement": 1, "irrelevant": 0}
    key = str(label).strip().lower()
    if key in mapping:
        return mapping[key]
    key_u = str(label).strip().upper()
    if key_u in mapping:
        return mapping[key_u]
    raise ValueError(f"Unknown ESCI label: {label}")


def load_esci_from_hf(config: ProbeConfig, tagger: OpenAIQuestionTagger | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    # Streaming avoids materializing the full ~2M-row train split in memory.
    print("[ESCI] Connecting to Hugging Face ESCI stream (this can take ~10-60s before first rows)...", flush=True)
    ds = load_dataset("tasksource/esci", split="train", streaming=True)

    rows: list[dict[str, Any]] = []
    # Oversample to retain diversity before deterministic downsample.
    target = max(config.max_rows * 3, 120)
    print(
        f"[ESCI] Streaming candidates (target_raw={target}, locale={config.locale}, small_version={config.use_small_version})",
        flush=True,
    )
    seen = 0

    for ex in ds:
        seen += 1
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
        if len(rows) % 25 == 0:
            print(f"[ESCI] Collected {len(rows)}/{target} candidates (streamed {seen} rows)", flush=True)

        if len(rows) >= target:
            break

    if not rows:
        raise ValueError("No ESCI rows matched current filters (locale/slice).")

    out = pd.DataFrame(rows).dropna()
    print(f"[ESCI] Tagging {len(out)} candidates using {'OpenAI' if tagger is not None else 'heuristics'}", flush=True)
    tags = []
    tag_start = time.time()
    total = len(out)
    for idx, (q, t) in enumerate(zip(out["query"], out["item_text"]), start=1):
        tags.append(tagger.label(q, t) if tagger is not None else tag_question(q, t))
        if idx % 25 == 0 or idx == total:
            elapsed = max(time.time() - tag_start, 1e-6)
            msg = f"[ESCI] Tagged {idx}/{total} ({idx/elapsed:.1f} rows/s)"
            if tagger is not None:
                s = tagger.stats_snapshot()
                msg += (
                    f" | cache hits={int(s['cache_hits'])} misses={int(s['cache_misses'])}"
                    f" | api calls={int(s['api_calls'])} avg_api={s['avg_api_latency_s']:.2f}s"
                    f" | avg_payload={s['avg_prompt_bytes']:.0f}B"
                )
            print(msg, flush=True)
    out["question_tag"] = [t.question_tag for t in tags]
    out["tag_reason"] = [t.tag_reason for t in tags]
    out["tag_confidence"] = [t.tag_confidence for t in tags]
    out["tag_source"] = [t.tag_source for t in tags]
    out = out.sample(min(config.max_rows, len(out)), random_state=7).reset_index(drop=True)
    print(f"[ESCI] Downsampled to {len(out)} rows (max_rows={config.max_rows})", flush=True)
    out["source"] = "esci"
    out["probe_id"] = [f"esci_{i:04d}" for i in range(len(out))]
    out["notes"] = "esci_sample"
    return out


def pairwise_directional_subset(
    df: pd.DataFrame,
    max_pairs_per_tag: int = 8,
    *,
    target_rows: int | None = None,
    max_pairs_per_query: int = 3,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["pair_group_id", "expected_direction", "target_tokens_query", "target_tokens_item"])

    work = df.reset_index(drop=True).copy()
    work["_row_id"] = range(len(work))
    rows: list[dict[str, Any]] = []
    group_counter = 0
    used_pair_ids: set[tuple[int, int]] = set()

    def append_pair(hi_row: pd.Series, lo_row: pd.Series) -> bool:
        nonlocal group_counter
        hi_id = int(hi_row["_row_id"])
        lo_id = int(lo_row["_row_id"])
        pair_id = (hi_id, lo_id)
        if hi_id == lo_id or pair_id in used_pair_ids:
            return False
        if float(hi_row["relevance_score"]) <= float(lo_row["relevance_score"]):
            return False
        used_pair_ids.add(pair_id)
        gid = f"grp_{group_counter:04d}"
        group_counter += 1

        hi_dict = {k: v for k, v in hi_row.to_dict().items() if k != "_row_id"}
        lo_dict = {k: v for k, v in lo_row.to_dict().items() if k != "_row_id"}
        hi_dict["pair_group_id"] = gid
        lo_dict["pair_group_id"] = gid
        hi_dict["expected_direction"] = "should_rank_higher"
        lo_dict["expected_direction"] = "should_rank_lower"
        hi_dict["target_tokens_query"] = ""
        hi_dict["target_tokens_item"] = ""
        lo_dict["target_tokens_query"] = ""
        lo_dict["target_tokens_item"] = ""
        rows.extend([hi_dict, lo_dict])
        return True

    def query_candidate_pairs(qpart: pd.DataFrame, limit: int) -> list[tuple[pd.Series, pd.Series]]:
        """Generate multiple hi/lo pairs for one query without exploding combinatorially."""
        if len(qpart) < 2:
            return []
        qpart = qpart.sort_values("relevance_score", ascending=False).reset_index(drop=True)
        if float(qpart.iloc[0]["relevance_score"]) == float(qpart.iloc[-1]["relevance_score"]):
            return []
        out_pairs: list[tuple[pd.Series, pd.Series]] = []
        # Score buckets let us produce diverse high-vs-lower combinations.
        by_score = {score: part.reset_index(drop=True) for score, part in qpart.groupby("relevance_score", sort=False)}
        score_levels = sorted(by_score.keys(), reverse=True)
        hi_levels = score_levels[:-1]
        lo_levels = score_levels[1:]
        # Prioritize widest score-gap combinations first.
        for hi_score in hi_levels:
            for lo_score in reversed([s for s in lo_levels if s < hi_score]):
                hi_part = by_score[hi_score]
                lo_part = by_score[lo_score]
                for hi_idx in range(min(2, len(hi_part))):
                    for lo_idx in range(min(2, len(lo_part))):
                        out_pairs.append((hi_part.iloc[hi_idx], lo_part.iloc[lo_idx]))
                        if len(out_pairs) >= limit:
                            return out_pairs
        return out_pairs

    # Pass 1: balanced by tag (one pair per query, capped per tag).
    balanced_pairs = 0
    for tag, part in work.groupby("question_tag", sort=False):
        made = 0
        for query, qpart in part.groupby("query", sort=False):
            if made >= max_pairs_per_tag:
                break
            cands = query_candidate_pairs(qpart, limit=1)
            if not cands:
                continue
            hi, lo = cands[0]
            if append_pair(hi, lo):
                made += 1
                balanced_pairs += 1
                if target_rows is not None and len(rows) >= target_rows:
                    out = pd.DataFrame(rows).reset_index(drop=True)
                    print(f"[PAIR] Balanced pass reached target with {balanced_pairs} pairs", flush=True)
                    return out

    # Pass 2: relaxed backfill (multiple pairs/query, no per-tag cap) to approach target.
    backfill_pairs = 0
    for query, qpart in work.groupby("query", sort=False):
        cands = query_candidate_pairs(qpart, limit=max_pairs_per_query)
        for hi, lo in cands:
            if append_pair(hi, lo):
                backfill_pairs += 1
                if target_rows is not None and len(rows) >= target_rows:
                    out = pd.DataFrame(rows).reset_index(drop=True)
                    print(
                        f"[PAIR] Built {balanced_pairs + backfill_pairs} pairs "
                        f"(balanced={balanced_pairs}, backfill={backfill_pairs})",
                        flush=True,
                    )
                    return out

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["pair_group_id", "expected_direction", "target_tokens_query", "target_tokens_item"])

    out = pd.DataFrame(rows).reset_index(drop=True)
    print(
        f"[PAIR] Built {len(out)//2} pairs total (balanced={balanced_pairs}, backfill={backfill_pairs})",
        flush=True,
    )
    return out
