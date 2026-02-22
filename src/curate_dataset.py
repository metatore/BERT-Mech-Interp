from __future__ import annotations

from pathlib import Path

import pandas as pd

from probes import ProbeConfig, load_esci_from_hf, pairwise_directional_subset


def load_handcrafted(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {
        "probe_id",
        "source",
        "question_tag",
        "query",
        "item_text",
        "esci_label",
        "relevance_score",
        "pair_group_id",
        "expected_direction",
        "target_tokens_query",
        "target_tokens_item",
        "notes",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Handcrafted CSV missing: {sorted(missing)}")
    if "tag_reason" not in df.columns:
        df["tag_reason"] = "manual"
    if "tag_confidence" not in df.columns:
        df["tag_confidence"] = "high"
    return df


def build_probe_set(
    handcrafted_csv: str | Path,
    out_csv: str | Path,
    target_size: int = 80,
) -> pd.DataFrame:
    hand = load_handcrafted(handcrafted_csv)

    remaining = max(target_size - len(hand), 0)
    if remaining > 0:
        try:
            esci_raw = load_esci_from_hf(ProbeConfig(max_rows=max(remaining * 4, 60)))
            esci_pairs = pairwise_directional_subset(esci_raw, max_pairs_per_tag=max(remaining // 8, 2))
            esci_pairs = esci_pairs.head(remaining)
        except Exception as exc:
            print(f"Warning: ESCI load failed, using handcrafted-only probe set. Reason: {exc}")
            esci_pairs = pd.DataFrame(columns=hand.columns)
    else:
        esci_pairs = pd.DataFrame(columns=hand.columns)

    for col in hand.columns:
        if col not in esci_pairs.columns:
            esci_pairs[col] = ""

    merged = pd.concat([hand, esci_pairs[hand.columns]], ignore_index=True)
    merged = merged.head(target_size).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    hand_path = root / "data" / "handcrafted_seed.csv"
    out_path = root / "data" / "probe_set_v1.csv"
    df = build_probe_set(hand_path, out_path, target_size=80)
    print(f"Wrote {len(df)} rows to {out_path}")
