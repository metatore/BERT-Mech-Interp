from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


HF_BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"
DEFAULT_CATEGORIES = [
    "meta_Amazon_Fashion",
    "meta_Clothing_Shoes_and_Jewelry",
    "meta_Sports_and_Outdoors",
    "meta_Electronics",
    "meta_Home_and_Kitchen",
]

# Obvious non-brand / low-value store names to exclude from a bootstrap brand lexicon.
BLOCKLIST = {
    "generic",
    "unknown",
    "unbranded",
    "no brand",
    "none",
    "amazon",
    "amazon basics",
    "amazonbasics",
}

# Normalize common brand aliases / formatting variants.
ALIASES = {
    "underarmor": "under armour",
    "under armor": "under armour",
    "newbalance": "new balance",
    "hewlett packard": "hp",
    "lg electronics": "lg",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_brand(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"[/|]+", " ", s)
    s = re.sub(r"[\(\)\[\]\{\}:;,+*!?]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = ALIASES.get(s, s)
    return s


def looks_like_brand(s: str) -> bool:
    if not s:
        return False
    if s in BLOCKLIST:
        return False
    if len(s) < 2 or len(s) > 40:
        return False
    toks = TOKEN_RE.findall(s)
    if not toks:
        return False
    if len(toks) > 4:
        return False
    if all(t.isdigit() for t in toks):
        return False
    # Reject strings that are mostly generic product words.
    generic = {"pack", "set", "kit", "accessories", "store", "product", "products"}
    if set(toks).issubset(generic):
        return False
    return True


def iter_jsonl_via_curl(url: str, max_rows: int | None = None) -> Iterable[dict]:
    proc = subprocess.Popen(
        ["curl", "-L", "-s", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    count = 0
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    finally:
        proc.kill()
        proc.wait(timeout=5)


def extract_brand_candidate(rec: dict) -> tuple[str, str] | None:
    details = rec.get("details") or {}
    if isinstance(details, dict):
        for key in ("Brand", "brand"):
            val = details.get(key)
            if val:
                b = normalize_brand(str(val))
                if looks_like_brand(b):
                    return b, f"details.{key}"
    store = normalize_brand(str(rec.get("store") or ""))
    if looks_like_brand(store):
        return store, "store"
    return None


def build_brand_lexicon(categories: list[str], rows_per_category: int) -> tuple[Counter[str], dict[str, set[str]], dict[str, set[str]]]:
    counts: Counter[str] = Counter()
    sources_by_brand: dict[str, set[str]] = defaultdict(set)
    categories_by_brand: dict[str, set[str]] = defaultdict(set)
    for cat in categories:
        url = f"{HF_BASE}/{cat}.jsonl"
        print(f"[BRANDS] Scanning {cat} (up to {rows_per_category} rows)", flush=True)
        seen_rows = 0
        found_rows = 0
        for rec in iter_jsonl_via_curl(url, max_rows=rows_per_category):
            seen_rows += 1
            extracted = extract_brand_candidate(rec)
            if extracted is None:
                continue
            brand, source = extracted
            counts[brand] += 1
            sources_by_brand[brand].add(source)
            categories_by_brand[brand].add(cat)
            found_rows += 1
        print(
            f"[BRANDS] {cat}: scanned={seen_rows} candidate_rows={found_rows} unique_brands_so_far={len(counts)}",
            flush=True,
        )
    return counts, sources_by_brand, categories_by_brand


def main() -> None:
    parser = argparse.ArgumentParser(description="Build heuristic brand lexicon from Amazon Reviews'23 metadata (details.Brand/store).")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, help="Metadata category files without .jsonl suffix.")
    parser.add_argument("--rows-per-category", type=int, default=20000, help="Max rows to scan from each category for a fast bootstrap.")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum observed count to keep a brand.")
    parser.add_argument("--out-txt", type=Path, default=Path("data/heuristics/brands.txt"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/brand_lexicon_candidates.csv"))
    args = parser.parse_args()

    counts, sources_by_brand, categories_by_brand = build_brand_lexicon(args.categories, args.rows_per_category)
    kept = [(b, c) for b, c in counts.items() if c >= args.min_count]
    kept.sort(key=lambda x: (-x[1], x[0]))

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.write_text(
        "\n".join(brand for brand, _ in kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["brand", "count", "sources", "num_categories", "categories"])
        for brand, count in kept:
            cats = sorted(categories_by_brand.get(brand, set()))
            srcs = sorted(sources_by_brand.get(brand, set()))
            writer.writerow([brand, count, "|".join(srcs), len(cats), "|".join(cats)])

    print(f"[BRANDS] Wrote {len(kept)} brands to {args.out_txt}", flush=True)
    print(f"[BRANDS] Wrote candidate stats to {args.out_csv}", flush=True)
    if kept:
        print("[BRANDS] Top sample:", flush=True)
        for brand, count in kept[:15]:
            print(f"  {brand} ({count})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        if exc.filename == "curl":
            raise SystemExit("curl is required for this script (used to avoid local Python SSL issues).") from exc
        raise
