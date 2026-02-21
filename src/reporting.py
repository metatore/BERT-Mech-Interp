from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_esci_label(label: object) -> str:
    mapping = {
        "e": "Exact",
        "s": "Substitute",
        "c": "Complement",
        "i": "Irrelevant",
        "exact": "Exact",
        "substitute": "Substitute",
        "complement": "Complement",
        "irrelevant": "Irrelevant",
    }
    key = str(label).strip().lower()
    return mapping.get(key, str(label).strip())


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


def evaluate_absolute_checks(
    scored_df: pd.DataFrame,
    exact_threshold: float = 0.9,
    irrelevant_high_threshold: float = 0.9,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"score", "esci_label"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = scored_df.copy()
    out["label_norm"] = out["esci_label"].map(_normalize_esci_label)
    out["is_exact"] = out["label_norm"] == "Exact"
    out["is_non_exact"] = ~out["is_exact"]
    out["pred_exact"] = out["score"] >= float(exact_threshold)

    tp = int((out["pred_exact"] & out["is_exact"]).sum())
    fn = int((~out["pred_exact"] & out["is_exact"]).sum())
    fp = int((out["pred_exact"] & out["is_non_exact"]).sum())
    tn = int((~out["pred_exact"] & out["is_non_exact"]).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    accuracy = float((tp + tn) / len(out)) if len(out) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    irr = out[out["label_norm"] == "Irrelevant"].copy()
    irr_count = int(len(irr))
    irr_high = irr[irr["score"] >= float(irrelevant_high_threshold)].copy()
    irr_high_count = int(len(irr_high))
    irr_high_rate = float(irr_high_count / irr_count) if irr_count else 0.0

    summary = pd.DataFrame(
        [
            {
                "exact_threshold": float(exact_threshold),
                "irrelevant_high_threshold": float(irrelevant_high_threshold),
                "total_pairs": int(len(out)),
                "exact_pairs": int(out["is_exact"].sum()),
                "non_exact_pairs": int(out["is_non_exact"].sum()),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "precision_exact": precision,
                "recall_exact": recall,
                "specificity_non_exact": specificity,
                "f1_exact": f1,
                "accuracy": accuracy,
                "irrelevant_pairs": irr_count,
                "irrelevant_high_score_count": irr_high_count,
                "irrelevant_high_score_rate": irr_high_rate,
            }
        ]
    )

    per_label = (
        out.groupby("label_norm")["score"]
        .agg(
            count="count",
            mean="mean",
            min="min",
            p10=lambda x: x.quantile(0.10),
            p25=lambda x: x.quantile(0.25),
            p50=lambda x: x.quantile(0.50),
            p75=lambda x: x.quantile(0.75),
            p90=lambda x: x.quantile(0.90),
            max="max",
        )
        .reset_index()
        .sort_values("label_norm")
    )

    violations = out[
        ((out["label_norm"] == "Exact") & (out["score"] < float(exact_threshold)))
        | ((out["label_norm"] != "Exact") & (out["score"] >= float(exact_threshold)))
        | ((out["label_norm"] == "Irrelevant") & (out["score"] >= float(irrelevant_high_threshold)))
    ].copy()
    keep = [
        "probe_id",
        "pair_group_id",
        "question_tag",
        "query",
        "item_text",
        "esci_label",
        "label_norm",
        "relevance_score",
        "expected_direction",
        "score",
    ]
    keep = [c for c in keep if c in violations.columns]
    violations = violations[keep].sort_values("score", ascending=False)

    return summary, per_label, violations


def export_artifacts(
    out_dir: str | Path,
    scored_df: pd.DataFrame,
    checks_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    token_attr_df: pd.DataFrame,
    attention_df: pd.DataFrame,
    absolute_summary_df: pd.DataFrame | None = None,
    label_score_summary_df: pd.DataFrame | None = None,
    absolute_violations_df: pd.DataFrame | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(out / "scored_pairs.csv", index=False)
    checks_df.to_csv(out / "question_scorecard.csv", index=False)
    failure_df.to_csv(out / "failure_triage.csv", index=False)
    token_attr_df.to_csv(out / "token_attributions.csv", index=False)
    attention_df.to_csv(out / "attention_summary.csv", index=False)
    if absolute_summary_df is None or label_score_summary_df is None or absolute_violations_df is None:
        absolute_summary_df, label_score_summary_df, absolute_violations_df = evaluate_absolute_checks(scored_df)
    absolute_summary_df.to_csv(out / "absolute_scorecard.csv", index=False)
    label_score_summary_df.to_csv(out / "label_score_summary.csv", index=False)
    absolute_violations_df.to_csv(out / "absolute_violations.csv", index=False)

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
