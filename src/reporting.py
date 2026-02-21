from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def evaluate_directional_checks(scored_df: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_group_id", "score", "expected_direction"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for gid, part in scored_df.groupby("pair_group_id"):
        part = part.sort_values("score", ascending=False)
        top = part.iloc[0]
        passed = top["expected_direction"] == "should_rank_higher"
        qtag = top["question_tag"] if "question_tag" in part.columns else "unknown"
        rows.append(
            {
                "pair_group_id": gid,
                "question_tag": qtag,
                "passed": bool(passed),
                "top_probe_id": top.get("probe_id", ""),
                "top_score": float(top["score"]),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        summary = (
            out.groupby("question_tag")["passed"]
            .agg(pass_rate="mean", num_groups="count")
            .reset_index()
        )
        out = out.merge(summary, on="question_tag", how="left")
    return out


def make_failure_triage(scored_df: pd.DataFrame, checks_df: pd.DataFrame) -> pd.DataFrame:
    failed_groups = set(checks_df.loc[~checks_df["passed"], "pair_group_id"].tolist())
    if not failed_groups:
        return pd.DataFrame(
            columns=[
                "pair_group_id",
                "question_tag",
                "query",
                "item_text",
                "esci_label",
                "relevance_score",
                "expected_direction",
                "model_pred_direction",
                "score",
                "group_score_margin",
                "suspected_failure_type",
            ]
        )

    failed = scored_df[scored_df["pair_group_id"].isin(failed_groups)].copy()

    # Model's realized direction from ranking position.
    failed["is_model_top"] = failed["rank_in_group"] == 1 if "rank_in_group" in failed.columns else False
    failed["model_pred_direction"] = failed["is_model_top"].map(
        {True: "predicted_higher", False: "predicted_lower"}
    )

    # Margin helps separate ambiguous close calls from clearer misses.
    margins = []
    for gid, part in failed.groupby("pair_group_id"):
        part_sorted = part.sort_values("score", ascending=False)
        if len(part_sorted) >= 2:
            margin = float(part_sorted.iloc[0]["score"] - part_sorted.iloc[1]["score"])
        else:
            margin = 0.0
        margins.append({"pair_group_id": gid, "group_score_margin": margin})
    margin_df = pd.DataFrame(margins)
    failed = failed.merge(margin_df, on="pair_group_id", how="left")

    def classify_failure(row: pd.Series) -> str:
        qtag = str(row.get("question_tag", ""))
        margin = float(row.get("group_score_margin", 0.0))
        if qtag == "negation":
            return "negation-handling"
        if margin < 0.05:
            return "close-call-ambiguous"
        return "token-salience-or-interaction"

    failed["suspected_failure_type"] = failed.apply(classify_failure, axis=1)

    cols = [
        "pair_group_id",
        "question_tag",
        "query",
        "item_text",
        "esci_label",
        "relevance_score",
        "expected_direction",
        "model_pred_direction",
        "score",
        "group_score_margin",
        "suspected_failure_type",
    ]
    cols = [c for c in cols if c in failed.columns]
    return failed[cols].sort_values(["pair_group_id", "score"], ascending=[True, False])


def export_artifacts(
    out_dir: str | Path,
    scored_df: pd.DataFrame,
    checks_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    token_attr_df: pd.DataFrame,
    attention_df: pd.DataFrame,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(out / "scored_pairs.csv", index=False)
    checks_df.to_csv(out / "question_scorecard.csv", index=False)
    failure_df.to_csv(out / "failure_triage.csv", index=False)
    token_attr_df.to_csv(out / "token_attributions.csv", index=False)
    attention_df.to_csv(out / "attention_summary.csv", index=False)

    brief_lines = [
        "# Mech-Interp Prototype Brief",
        "",
        f"Total pairs scored: {len(scored_df)}",
        f"Directional groups: {checks_df['pair_group_id'].nunique() if len(checks_df) else 0}",
        f"Directional pass rate: {checks_df['passed'].mean():.2f}" if len(checks_df) else "Directional pass rate: n/a",
        "",
        "## Pass Rate by Question",
    ]

    if len(checks_df):
        for qtag, part in checks_df.groupby("question_tag"):
            brief_lines.append(f"- {qtag}: {part['passed'].mean():.2f} ({len(part)} groups)")
    else:
        brief_lines.append("- n/a")

    (out / "brief.md").write_text("\n".join(brief_lines))
