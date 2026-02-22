from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_nullable_bool(x: object) -> bool | None:
    if pd.isna(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1"}:
        return True
    if s in {"false", "0"}:
        return False
    return None


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


def _build_seed_overview(project_root: Path) -> dict[str, object]:
    seed_path = project_root / "data" / "handcrafted_seed.csv"
    seed = _safe_read_csv(seed_path)
    if seed.empty:
        return {"available": False, "rows": 0, "pair_groups": 0, "by_tag": [], "example_pairs": []}

    out: dict[str, object] = {
        "available": True,
        "rows": int(len(seed)),
        "pair_groups": int(seed["pair_group_id"].nunique()) if "pair_group_id" in seed.columns else 0,
        "by_tag": [],
        "example_pairs": [],
    }
    if "question_tag" in seed.columns:
        by_tag = []
        for tag, part in seed.groupby("question_tag", sort=False):
            by_tag.append({"question_tag": str(tag), "rows": int(len(part)), "queries": int(part["query"].nunique())})
        out["by_tag"] = sorted(by_tag, key=lambda r: (-int(r["rows"]), str(r["question_tag"])))

    # Show a few representative handcrafted pairs by tag for quick visual inspection.
    if {"pair_group_id", "query", "item_text"}.issubset(seed.columns):
        examples: list[dict[str, object]] = []
        for tag, part in seed.groupby("question_tag", sort=False):
            shown = 0
            for gid, g in part.groupby("pair_group_id", sort=False):
                if shown >= 2:
                    break
                g = g.sort_values("relevance_score", ascending=False) if "relevance_score" in g.columns else g
                hi = g.iloc[0]
                lo = g.iloc[-1] if len(g) > 1 else g.iloc[0]
                examples.append(
                    {
                        "question_tag": str(tag),
                        "pair_group_id": str(gid),
                        "query": str(hi.get("query", "")),
                        "high_item_text": str(hi.get("item_text", "")),
                        "high_label": str(hi.get("esci_label", "")),
                        "low_item_text": str(lo.get("item_text", "")),
                        "low_label": str(lo.get("esci_label", "")),
                        "notes": str(hi.get("notes", "") or lo.get("notes", "")),
                    }
                )
                shown += 1
        out["example_pairs"] = examples[:12]
    return out


def _compute_absolute_metrics(scored: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if scored.empty:
        return {}, [], []

    exact_threshold = 0.9
    irrelevant_high_threshold = 0.9

    tmp = scored.copy()
    tmp["label_norm"] = tmp["esci_label"].map(_normalize_esci_label)
    tmp["score"] = tmp["score"].map(_safe_float)

    is_exact = tmp["label_norm"] == "Exact"
    pred_exact = tmp["score"] >= exact_threshold
    is_non_exact = ~is_exact

    tp = int((pred_exact & is_exact).sum())
    fn = int((~pred_exact & is_exact).sum())
    fp = int((pred_exact & is_non_exact).sum())
    tn = int((~pred_exact & is_non_exact).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    irr = tmp[tmp["label_norm"] == "Irrelevant"]
    irr_count = int(len(irr))
    irr_high_count = int((irr["score"] >= irrelevant_high_threshold).sum())
    irr_high_rate = float(irr_high_count / irr_count) if irr_count else 0.0

    summary = {
        "exact_threshold": exact_threshold,
        "irrelevant_high_threshold": irrelevant_high_threshold,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision_exact": precision,
        "recall_exact": recall,
        "specificity_non_exact": specificity,
        "f1_exact": f1,
        "irrelevant_high_score_count": irr_high_count,
        "irrelevant_high_score_rate": irr_high_rate,
    }

    order = {"Exact": 0, "Substitute": 1, "Complement": 2, "Irrelevant": 3}
    rows = []
    for lbl, part in tmp.groupby("label_norm"):
        rows.append(
            {
                "label_norm": lbl,
                "count": int(len(part)),
                "mean": float(part["score"].mean()),
                "min": float(part["score"].min()),
                "p25": float(part["score"].quantile(0.25)),
                "p50": float(part["score"].quantile(0.50)),
                "p75": float(part["score"].quantile(0.75)),
                "p90": float(part["score"].quantile(0.90)),
                "max": float(part["score"].max()),
            }
        )
    label_summary = sorted(rows, key=lambda r: order.get(str(r["label_norm"]), 999))
    points = []
    for _, r in tmp.iterrows():
        points.append(
            {
                "label": str(r["label_norm"]),
                "score": float(r["score"]),
                "probe_id": str(r.get("probe_id", "")),
                "query": str(r.get("query", "")),
                "question_tag": str(r.get("question_tag", "")),
            }
        )
    return summary, label_summary, points


def _build_payload(outputs_dir: Path) -> dict[str, object]:
    project_root = outputs_dir.parent
    seed_overview = _build_seed_overview(project_root)
    scored = _safe_read_csv(outputs_dir / "scored_pairs.csv")
    scorecard = _safe_read_csv(outputs_dir / "question_scorecard.csv")
    attrs = _safe_read_csv(outputs_dir / "attributions_by_probe.csv")
    attn = _safe_read_csv(outputs_dir / "attention_by_probe.csv")
    causal = _safe_read_csv(outputs_dir / "counterfactual_results.csv")
    absolute_scorecard = _safe_read_csv(outputs_dir / "absolute_scorecard.csv")
    label_score_summary = _safe_read_csv(outputs_dir / "label_score_summary.csv")

    if scored.empty:
        return {
            "overview": {},
            "categories": [],
            "queries": {},
            "items_by_query": {},
            "attrs_by_probe": {},
            "attn_by_probe": {},
            "causal_by_probe": {},
            "failure_buckets": {"by_edit_type": [], "wrong_direction_examples": [], "edit_examples_by_type": {}},
            "causal_labeling_status": "no_causal_data",
            "top_examples": [],
            "seed_overview": seed_overview,
            "absolute": {},
            "label_score_summary": [],
            "score_points": [],
            "esci_map": {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0},
        }

    # Normalize fields used by UI.
    scored = scored.copy()
    probe_lookup: dict[str, dict[str, object]] = {}
    if "probe_id" in scored.columns:
        for _, r in scored.iterrows():
            probe_lookup[str(r.get("probe_id", ""))] = {
                "query": str(r.get("query", "")),
                "question_tag": str(r.get("question_tag", "")),
                "item_text": str(r.get("item_text", "")),
            }
    if "rank_in_group" in scored.columns:
        scored["model_pred_direction"] = scored["rank_in_group"].apply(
            lambda x: "predicted_higher" if _safe_float(x, 9999.0) == 1.0 else "predicted_lower"
        )
    else:
        scored["model_pred_direction"] = "predicted_lower"

    # Pair-group pass/fail lookup.
    group_pass = {}
    if not scorecard.empty and {"pair_group_id", "passed"}.issubset(scorecard.columns):
        group_pass = {
            str(r["pair_group_id"]): bool(r["passed"]) for _, r in scorecard[["pair_group_id", "passed"]].drop_duplicates().iterrows()
        }
    scored["group_passed"] = scored["pair_group_id"].astype(str).map(group_pass).fillna(False)

    # Overview metrics.
    total_pairs = int(len(scored))
    total_groups = int(scored["pair_group_id"].nunique()) if "pair_group_id" in scored.columns else 0
    pass_rate = float(pd.Series(group_pass.values()).mean()) if group_pass else 0.0

    # Category summary.
    cat_rows = []
    for cat, part in scored.groupby("question_tag"):
        g = part[["pair_group_id", "group_passed"]].drop_duplicates()
        cat_rows.append(
            {
                "question_tag": str(cat),
                "num_queries": int(part["query"].nunique()),
                "num_pairs": int(len(part)),
                "num_groups": int(g["pair_group_id"].nunique()),
                "pass_rate": float(g["group_passed"].mean()) if len(g) else 0.0,
            }
        )
    categories = sorted(cat_rows, key=lambda x: (-x["pass_rate"], x["question_tag"]))

    # Query-level summary per category + item rows per query.
    queries: dict[str, list[dict[str, object]]] = {}
    items_by_query: dict[str, list[dict[str, object]]] = {}

    for cat, part_cat in scored.groupby("question_tag"):
        qrows = []
        for q, part_q in part_cat.groupby("query"):
            g = part_q[["pair_group_id", "group_passed"]].drop_duplicates()
            qrows.append(
                {
                    "query": str(q),
                    "num_pairs": int(len(part_q)),
                    "num_groups": int(g["pair_group_id"].nunique()),
                    "pass_rate": float(g["group_passed"].mean()) if len(g) else 0.0,
                }
            )

            key = str(q)
            items = []
            for _, r in part_q.sort_values("score", ascending=False).iterrows():
                items.append(
                    {
                        "probe_id": str(r.get("probe_id", "")),
                        "pair_group_id": str(r.get("pair_group_id", "")),
                        "item_text": str(r.get("item_text", "")),
                        "esci_label": str(r.get("esci_label", "")),
                        "relevance_score": _safe_float(r.get("relevance_score", 0.0)),
                        "expected_direction": str(r.get("expected_direction", "")),
                        "model_score": _safe_float(r.get("score", 0.0)),
                        "model_pred_direction": str(r.get("model_pred_direction", "")),
                        "rank_in_group": int(_safe_float(r.get("rank_in_group", 0.0), 0.0)) if str(r.get("rank_in_group", "")) != "" else None,
                        "group_passed": bool(r.get("group_passed", False)),
                        "question_tag": str(r.get("question_tag", "")),
                        "tag_reason": str(r.get("tag_reason", "")),
                        "tag_confidence": str(r.get("tag_confidence", "")),
                        "query": str(r.get("query", "")),
                    }
                )
            items_by_query[key] = items

        queries[str(cat)] = sorted(qrows, key=lambda x: (-x["pass_rate"], x["query"]))

    # Attribution payload keyed by probe_id.
    attrs_by_probe: dict[str, list[dict[str, object]]] = {}
    if not attrs.empty and "probe_id" in attrs.columns:
        keep = [c for c in ["position", "token", "segment", "signed_attr", "abs_attr", "norm_abs_attr"] if c in attrs.columns]
        for pid, part in attrs.groupby("probe_id"):
            attrs_by_probe[str(pid)] = part[keep].to_dict(orient="records")

    # Attention payload keyed by probe_id.
    attn_by_probe: dict[str, list[dict[str, object]]] = {}
    if not attn.empty and "probe_id" in attn.columns:
        keep = [c for c in ["layer", "head", "cls_to_query_mean", "cls_to_item_mean", "query_to_item_mean"] if c in attn.columns]
        for pid, part in attn.groupby("probe_id"):
            layer_mean = (
                part.groupby("layer", as_index=False)[["cls_to_query_mean", "cls_to_item_mean", "query_to_item_mean"]]
                .mean()
                .sort_values("layer")
            )
            attn_by_probe[str(pid)] = layer_mean.to_dict(orient="records")

    # Causal payload keyed by probe_id.
    causal_by_probe: dict[str, list[dict[str, object]]] = {}
    failure_by_edit_type: list[dict[str, object]] = []
    wrong_direction_examples: list[dict[str, object]] = []
    edit_examples_by_type: dict[str, list[dict[str, object]]] = {}
    causal_labeling_status = "no_causal_data"
    top_examples: list[dict[str, object]] = []
    if not causal.empty and "probe_id" in causal.columns:
        causal_labeling_status = "present_unlabeled"
        keep = [
            c
            for c in [
                "edit_type",
                "expected_delta_direction",
                "original_margin",
                "edited_margin",
                "delta_margin",
                "original_prob",
                "edited_prob",
                "delta_prob",
                "sign_consistent",
                "expected_reason",
                "expected_confidence",
                "label_source",
                "original_text",
                "edited_text",
            ]
            if c in causal.columns
        ]
        if "delta_margin" in causal.columns:
            causal["abs_delta_margin"] = causal["delta_margin"].map(lambda x: abs(_safe_float(x)))
        else:
            causal["abs_delta_margin"] = 0.0
        for pid, part in causal.groupby("probe_id"):
            part = part.sort_values("abs_delta_margin", ascending=False)
            causal_by_probe[str(pid)] = part[keep + ["abs_delta_margin"]].to_dict(orient="records")

        if "label_source" in causal.columns:
            sources = {str(x) for x in causal["label_source"].dropna().tolist() if str(x)}
            if any(s == "openai_judge" for s in sources):
                causal_labeling_status = "enabled_openai"
            elif any(s == "disabled_no_api_key" for s in sources):
                causal_labeling_status = "disabled_no_api_key"
            elif sources:
                causal_labeling_status = "other_label_source"

        # Drilldown examples grouped by edit type.
        if "edit_type" in causal.columns:
            for edit_type, part in causal.groupby("edit_type"):
                part = part.sort_values("abs_delta_margin", ascending=False).head(80)
                ex_rows: list[dict[str, object]] = []
                for _, r in part.iterrows():
                    pid = str(r.get("probe_id", ""))
                    meta = probe_lookup.get(pid, {})
                    ex_rows.append(
                        {
                            "probe_id": pid,
                            "query": str(r.get("query", "")) or str(meta.get("query", "")),
                            "item_text": str(r.get("item_text", "")) or str(meta.get("item_text", "")),
                            "original_text": str(r.get("original_text", "")),
                            "edited_text": str(r.get("edited_text", "")),
                            "question_tag": str(meta.get("question_tag", "")),
                            "expected_delta_direction": "" if pd.isna(r.get("expected_delta_direction")) else str(r.get("expected_delta_direction", "")),
                            "expected_reason": "" if pd.isna(r.get("expected_reason")) else str(r.get("expected_reason", "")),
                            "expected_confidence": "" if pd.isna(r.get("expected_confidence")) else str(r.get("expected_confidence", "")),
                            "label_source": "" if pd.isna(r.get("label_source")) else str(r.get("label_source", "")),
                            "original_prob": _safe_float(r.get("original_prob", 0.0)),
                            "edited_prob": _safe_float(r.get("edited_prob", 0.0)),
                            "delta_prob": _safe_float(r.get("delta_prob", 0.0)),
                            "delta_margin": _safe_float(r.get("delta_margin", 0.0)),
                            "sign_consistent": _safe_nullable_bool(r.get("sign_consistent")),
                        }
                    )
                edit_examples_by_type[str(edit_type)] = ex_rows

        labeled_causal = causal[causal["sign_consistent"].notna()].copy() if "sign_consistent" in causal.columns else pd.DataFrame()
        if {"edit_type", "sign_consistent"}.issubset(labeled_causal.columns) and len(labeled_causal):
            by_edit = (
                labeled_causal.groupby("edit_type")["sign_consistent"]
                .agg(num_tests="count", sign_consistency="mean")
                .reset_index()
                .sort_values(["sign_consistency", "num_tests"], ascending=[True, False])
            )
            for _, r in by_edit.iterrows():
                failure_by_edit_type.append(
                    {
                        "edit_type": str(r["edit_type"]),
                        "num_tests": int(r["num_tests"]),
                        "sign_consistency": float(r["sign_consistency"]),
                        "failure_rate": float(1.0 - float(r["sign_consistency"])),
                    }
                )

        if {"sign_consistent", "delta_margin", "probe_id", "query", "edit_type"}.issubset(causal.columns):
            wrong = causal[causal["sign_consistent"] == False].copy()  # noqa: E712
            if len(wrong):
                wrong["abs_delta_margin"] = wrong["delta_margin"].map(lambda x: abs(_safe_float(x)))
                wrong = wrong.sort_values("abs_delta_margin", ascending=False).head(15)
                wrong_direction_examples = [
                    {
                        "probe_id": str(r.get("probe_id", "")),
                        "query": str(r.get("query", "")),
                        "edit_type": str(r.get("edit_type", "")),
                        "expected_delta_direction": "" if pd.isna(r.get("expected_delta_direction")) else str(r.get("expected_delta_direction", "")),
                        "delta_margin": _safe_float(r.get("delta_margin", 0.0)),
                        "sign_consistent": _safe_nullable_bool(r.get("sign_consistent")),
                    }
                    for _, r in wrong.iterrows()
                ]

        # Curated examples for non-technical quick entry.
        if "delta_margin" in causal.columns:
            causal["abs_delta_margin"] = causal["delta_margin"].map(lambda x: abs(_safe_float(x)))
            if "sign_consistent" in causal.columns:
                stable = causal[causal["sign_consistent"] == True].copy()  # noqa: E712
                wrong = causal[causal["sign_consistent"] == False].copy()  # noqa: E712
            else:
                stable = pd.DataFrame(columns=causal.columns)
                wrong = pd.DataFrame(columns=causal.columns)

            if len(wrong):
                r = wrong.sort_values("abs_delta_margin", ascending=False).iloc[0]
                pid = str(r.get("probe_id", ""))
                meta = probe_lookup.get(pid, {})
                top_examples.append(
                    {
                        "kind": "most_concerning_failure",
                        "title": "Most Concerning Failure",
                        "probe_id": pid,
                        "query": str(r.get("query", "")) or str(meta.get("query", "")),
                        "question_tag": str(meta.get("question_tag", "")),
                        "edit_type": str(r.get("edit_type", "")),
                        "delta_margin": _safe_float(r.get("delta_margin", 0.0)),
                        "summary": "Model moved in the wrong direction with a large change.",
                    }
                )
            if len(stable):
                r = stable.sort_values("abs_delta_margin", ascending=False).iloc[0]
                pid = str(r.get("probe_id", ""))
                meta = probe_lookup.get(pid, {})
                top_examples.append(
                    {
                        "kind": "strongest_stable_behavior",
                        "title": "Strongest Stable Behavior",
                        "probe_id": pid,
                        "query": str(r.get("query", "")) or str(meta.get("query", "")),
                        "question_tag": str(meta.get("question_tag", "")),
                        "edit_type": str(r.get("edit_type", "")),
                        "delta_margin": _safe_float(r.get("delta_margin", 0.0)),
                        "summary": "Model changed strongly and in the expected direction.",
                    }
                )
            if len(wrong):
                r = wrong.sort_values("abs_delta_margin", ascending=True).iloc[0]
                pid = str(r.get("probe_id", ""))
                meta = probe_lookup.get(pid, {})
                top_examples.append(
                    {
                        "kind": "most_ambiguous_case",
                        "title": "Most Ambiguous Case",
                        "probe_id": pid,
                        "query": str(r.get("query", "")) or str(meta.get("query", "")),
                        "question_tag": str(meta.get("question_tag", "")),
                        "edit_type": str(r.get("edit_type", "")),
                        "delta_margin": _safe_float(r.get("delta_margin", 0.0)),
                        "summary": "Model still moved the wrong way, but by a smaller amount.",
                    }
                )

    absolute_summary, fallback_label_summary, score_points = _compute_absolute_metrics(scored)
    if not absolute_scorecard.empty:
        absolute_summary = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in absolute_scorecard.iloc[0].to_dict().items()}
    if not label_score_summary.empty:
        label_score_rows = label_score_summary.to_dict(orient="records")
    else:
        label_score_rows = fallback_label_summary

    return {
        "overview": {
            "total_pairs": total_pairs,
            "total_groups": total_groups,
            "pass_rate": pass_rate,
            "failed_groups": int(total_groups - round(pass_rate * total_groups)),
        },
        "categories": categories,
        "queries": queries,
        "items_by_query": items_by_query,
        "attrs_by_probe": attrs_by_probe,
        "attn_by_probe": attn_by_probe,
        "causal_by_probe": causal_by_probe,
        "failure_buckets": {
            "by_edit_type": failure_by_edit_type,
            "wrong_direction_examples": wrong_direction_examples,
            "edit_examples_by_type": edit_examples_by_type,
        },
        "causal_labeling_status": causal_labeling_status,
        "top_examples": top_examples,
        "seed_overview": seed_overview,
        "absolute": absolute_summary,
        "label_score_summary": label_score_rows,
        "score_points": score_points,
        "esci_map": {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0},
    }


def build_dashboard(outputs_dir: Path, out_html: Path) -> Path:
    payload = _build_payload(outputs_dir)
    data_json = json.dumps(payload)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Mech-Interp Dashboard</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --line: #dbe2ea;
      --text: #17212e;
      --muted: #64748b;
      --teal: #0f766e;
      --teal-soft: #e6fffb;
    }}
    body {{ margin:0; font-family: "Avenir Next", "Segoe UI", sans-serif; background: var(--bg); color:var(--text); }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .title {{ margin:0 0 6px 0; font-size:30px; }}
    .subtitle {{ margin:0 0 18px 0; color:var(--muted); }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:12px; }}
    .section-title {{ margin:0 0 10px 0; font-size:19px; }}
    .explain {{ background:#f8fafc; border-left:4px solid var(--teal); border-radius:8px; padding:10px 12px; color:#334155; font-size:14px; line-height:1.5; }}
    .grid {{ display:grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap:10px; }}
    .k {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }}
    .v {{ font-size:28px; font-weight:700; margin-top:4px; }}
    .layout {{ display:grid; grid-template-columns: 1fr 1.2fr; gap:12px; }}
    .pane {{ min-height: 260px; }}
    .table-wrap {{ overflow:auto; max-height: 360px; border:1px solid var(--line); border-radius:10px; }}
    table {{ width:100%; border-collapse: collapse; font-size:13px; }}
    th, td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
    th {{ background:#f8fafc; position:sticky; top:0; z-index:1; }}
    tr.clickable {{ cursor:pointer; }}
    tr.clickable:hover {{ background:#f8fbff; }}
    .pill {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:#fff; }}
    .ok {{ color:#065f46; background:#d1fae5; border-color:#a7f3d0; }}
    .bad {{ color:#991b1b; background:#fee2e2; border-color:#fecaca; }}
    .meta-grid {{ display:grid; grid-template-columns: repeat(4, minmax(140px,1fr)); gap:8px; margin:8px 0; }}
    .meta {{ border:1px solid var(--line); border-radius:8px; padding:8px; background:#fff; }}
    .meta .k {{ font-size:10px; }}
    .meta .v {{ font-size:15px; font-weight:600; margin-top:4px; }}
    .pair-text {{ border:1px dashed var(--line); border-radius:8px; background:#fcfdff; padding:8px; margin:6px 0; font-size:14px; line-height:1.5; }}
    .two-col {{ display:grid; grid-template-columns: 1fr 1fr; gap:10px; }}
    .token-cloud {{ line-height:2.0; font-size:15px; }}
    .tok {{ border:1px solid #dbe2ea; border-radius:6px; padding:2px 6px; margin-right:4px; display:inline-block; }}
    .insight-list {{ margin: 8px 0 0 18px; padding: 0; }}
    .attn-bars {{ display:grid; gap:8px; margin-top:8px; }}
    .attn-row {{ display:grid; grid-template-columns: 96px 1fr 84px; gap:10px; align-items:center; min-height:22px; }}
    .attn-row > div:first-child {{ white-space: nowrap; font-size: 13px; }}
    .attn-row > div:last-child {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; font-size: 13px; }}
    .attn-track {{ height:10px; background:#eef2f7; border-radius:8px; overflow:hidden; }}
    .attn-fill {{ height:10px; background:linear-gradient(90deg,#0ea5e9,#14b8a6); }}
    .muted {{ color:var(--muted); }}
    .chart-wrap {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:#fcfdff; }}
    .calib-grid {{ display:grid; grid-template-columns: 1.3fr 1fr; gap:10px; }}
    .score-dot {{ fill:#0f766e; fill-opacity:0.55; cursor:pointer; }}
    .score-dot:hover {{ stroke:#0f172a; stroke-width:1.2; fill-opacity:0.9; }}
    .label-chip {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line); margin-right:6px; }}
    .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; letter-spacing:0.02em; border:1px solid var(--line); margin-left:6px; }}
    .badge.obs {{ color:#0c4a6e; background:#e0f2fe; border-color:#bae6fd; }}
    .badge.causal {{ color:#365314; background:#ecfccb; border-color:#d9f99d; }}
    .controls {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:6px 0 10px 0; }}
    .controls label {{ font-size:12px; color:var(--muted); }}
    .controls select {{ border:1px solid var(--line); border-radius:8px; padding:4px 8px; background:#fff; }}
    tr.warnrow td {{ background:#fff1f2; }}
    .small-note {{ font-size:12px; color:var(--muted); }}
    .mode-toggle {{ display:flex; align-items:center; gap:8px; margin: 0 0 12px 0; }}
    .mode-toggle input {{ transform: scale(1.05); }}
    .examples-strip {{ display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:10px; }}
    .example-card {{ border:1px solid var(--line); border-radius:10px; background:#fcfdff; padding:10px; cursor:pointer; }}
    .example-card:hover {{ background:#f8fbff; }}
    .example-title {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }}
    .example-query {{ font-weight:600; margin:4px 0; }}
    .example-summary {{ font-size:13px; color:#334155; }}
    .seed-grid {{ display:grid; grid-template-columns: 1fr 1.5fr; gap:10px; }}
    .seed-pair-card {{ border:1px solid var(--line); border-radius:10px; padding:10px; background:#fcfdff; }}
    .seed-pair-card .seed-tag {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; }}
    .seed-pair-card .seed-query {{ font-weight:600; margin:4px 0 8px 0; }}
    .seed-row {{ font-size:13px; margin:3px 0; }}
    .seed-row strong {{ font-size:12px; }}
    .score-compare {{ display:grid; gap:4px; min-width:180px; }}
    .score-row {{ display:grid; grid-template-columns: 52px 1fr 56px; gap:8px; align-items:center; font-size:12px; }}
    .score-track {{ height:8px; border-radius:999px; background:#e2e8f0; overflow:hidden; }}
    .score-fill-before {{ height:8px; background:#94a3b8; }}
    .score-fill-after {{ height:8px; background:#0f766e; }}
    .story-card {{ border:1px solid var(--line); border-radius:10px; background:#f8fafc; padding:10px; margin-bottom:10px; }}
    .banner {{ border:1px solid #fcd34d; background:#fffbeb; color:#78350f; border-radius:10px; padding:10px 12px; margin: 0 0 12px 0; }}
    .banner code {{ background:#fff7d6; padding:1px 4px; border-radius:4px; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .meta-grid {{ grid-template-columns: repeat(2, minmax(140px,1fr)); }}
      .calib-grid {{ grid-template-columns: 1fr; }}
      .examples-strip {{ grid-template-columns: 1fr; }}
      .seed-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1 class=\"title\">E-commerce Relevance Mech-Interp Dashboard</h1>
    <p class=\"subtitle\">Progressive drill-down: Overview → Category → Query → Query-item diagnostics</p>
    <div class=\"mode-toggle\">
      <input id=\"mode-toggle\" type=\"checkbox\" checked />
      <label for=\"mode-toggle\"><strong>Beginner Mode</strong> (plain language + visual explanations)</label>
    </div>
    <div id=\"causal-labeling-banner\"></div>

    <div class=\"card\">
      <h2 class=\"section-title\">Start Here: Top 3 Examples</h2>
      <div class=\"explain\">
        Click one of these examples to jump straight to an important case.
      </div>
      <div style=\"height:10px\"></div>
      <div class=\"examples-strip\" id=\"top-examples\"></div>
    </div>

    <div class=\"card\">
      <h2 class=\"section-title\">Handcrafted Seed Set Overview</h2>
      <div class=\"explain\">
        These are the manually designed examples used as the core sanity-check set. They are intentionally small and balanced across failure modes.
      </div>
      <div style=\"height:10px\"></div>
      <div class=\"seed-grid\">
        <div>
          <div class=\"grid\" id=\"seed-overview-cards\"></div>
          <div style=\"height:10px\"></div>
          <div class=\"muted\" style=\"margin-bottom:8px\">Seed rows by category</div>
          <div class=\"table-wrap\" id=\"seed-tag-table\"></div>
        </div>
        <div>
          <div class=\"muted\" style=\"margin-bottom:8px\">Example handcrafted pairs</div>
          <div id=\"seed-example-pairs\"></div>
        </div>
      </div>
    </div>

    <div class=\"card\">
      <h2 class=\"section-title\">What Was Analyzed</h2>
      <div class=\"explain\">
        This is a ranking evaluation. For each query, the model scores candidate items. A group passes when the item expected to rank higher actually receives a higher model score.
        Ground truth labels are mapped as: Exact=3, Substitute=2, Complement=1, Irrelevant=0.
      </div>
      <div style=\"height:10px\"></div>
      <div class=\"grid\" id=\"overview-cards\"></div>
    </div>

    <div class=\"card\">
      <h2 class=\"section-title\">Score Calibration Snapshot</h2>
      <div class=\"explain\">
        Secondary evaluation checks absolute separation. Target policy: Exact should score above threshold, Non-Exact should score below.
      </div>
      <div style=\"height:10px\"></div>
      <div class=\"grid\" id=\"absolute-cards\"></div>
      <div style=\"height:10px\"></div>
      <div class=\"calib-grid\">
        <div class=\"chart-wrap\">
          <div class=\"muted\" style=\"margin-bottom:8px\">Model score by ground-truth label (x-axis: score 0→1)</div>
          <svg id=\"label-score-chart\" viewBox=\"0 0 760 250\" style=\"width:100%; height:auto;\"></svg>
        </div>
        <div>
          <div class=\"muted\" style=\"margin-bottom:8px\">Label score summary</div>
          <div class=\"table-wrap\" id=\"label-summary-table\"></div>
        </div>
      </div>
    </div>

    <div class=\"card\">
      <h2 class=\"section-title\">Failure Buckets Summary <span class=\"badge causal\">Causal</span></h2>
      <div class=\"explain\">
        This section summarizes counterfactual stress tests by edit type and highlights biggest wrong-direction deltas.
      </div>
      <div style=\"height:10px\"></div>
      <div class=\"two-col\">
        <div>
          <div class=\"muted\" style=\"margin-bottom:8px\">Sign-consistency by edit type</div>
          <div class=\"table-wrap\" id=\"failure-by-edit\"></div>
        </div>
        <div>
          <div class=\"muted\" style=\"margin-bottom:8px\">Top wrong-direction examples</div>
          <div class=\"table-wrap\" id=\"failure-wrong\"></div>
        </div>
      </div>
      <div style=\"height:10px\"></div>
      <div>
        <div class=\"muted\" id=\"failure-drilldown-label\">Edit type drilldown: click a row above to see individual examples</div>
        <div class=\"table-wrap\" id=\"failure-edit-drilldown\"></div>
      </div>
    </div>

    <div class=\"card\">
      <h2 class=\"section-title\">Step 1: Choose Category</h2>
      <div class=\"table-wrap\"><table id=\"categories-table\"></table></div>
    </div>

    <div class=\"layout\">
      <div class=\"card pane\">
        <h2 class=\"section-title\">Step 2: Choose Query</h2>
        <div id=\"selected-category\" class=\"muted\"></div>
        <div style=\"height:8px\"></div>
        <div class=\"table-wrap\"><table id=\"queries-table\"></table></div>
      </div>

      <div class=\"card pane\">
        <h2 class=\"section-title\">Step 3: Choose Query-Item Pair</h2>
        <div id=\"selected-query\" class=\"muted\"></div>
        <div style=\"height:8px\"></div>
        <div class=\"table-wrap\"><table id=\"items-table\"></table></div>
      </div>
    </div>

    <div class=\"card\" id=\"detail-card\">
      <h2 class=\"section-title\">Step 4: Inspect Model Behavior</h2>
      <div class=\"explain\">Green tokens pushed relevance up. Red tokens pushed down. Darker color means stronger influence. Attention is supporting evidence; causal edits are stronger behavior checks.</div>
      <div class=\"meta-grid\">
        <div class=\"meta\"><div class=\"k\">Ground Truth Label</div><div class=\"v\" id=\"meta-label\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Ground Truth Score</div><div class=\"v\" id=\"meta-rel\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Model Output Score</div><div class=\"v\" id=\"meta-model\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Expected Direction</div><div class=\"v\" id=\"meta-exp\">-</div></div>
      </div>
      <div class=\"small-note\" id=\"tag-provenance\"></div>
      <div id=\"pair-query\" class=\"pair-text\"></div>
      <div id=\"pair-item\" class=\"pair-text\"></div>
      <div id=\"what-this-means\" class=\"story-card\"></div>

      <div class=\"two-col\">
        <div class=\"card\">
          <h3 style=\"margin-top:0\">Query Token Attribution <span class=\"badge obs\">Observational</span></h3>
          <div id=\"query-cloud\"></div>
          <div class=\"two-col\" style=\"margin-top:8px\">
            <div><strong>Helpful query tokens</strong><div id=\"q-pos\" class=\"table-wrap\"></div></div>
            <div><strong>Harmful query tokens</strong><div id=\"q-neg\" class=\"table-wrap\"></div></div>
          </div>
        </div>
        <div class=\"card\">
          <h3 style=\"margin-top:0\">Item Token Attribution <span class=\"badge obs\">Observational</span></h3>
          <div id=\"item-cloud\"></div>
          <div class=\"two-col\" style=\"margin-top:8px\">
            <div><strong>Helpful item tokens</strong><div id=\"i-pos\" class=\"table-wrap\"></div></div>
            <div><strong>Harmful item tokens</strong><div id=\"i-neg\" class=\"table-wrap\"></div></div>
          </div>
        </div>
      </div>

      <div class=\"card\">
        <h3 style=\"margin-top:0\">Causal Tests <span class=\"badge causal\">Causal</span></h3>
        <div class=\"explain\">
          Counterfactual edits perturb one property at a time. Wrong-direction deltas indicate behavior that conflicts with expected relevance changes.
        </div>
        <div class=\"controls\">
          <label for=\"causal-edit-filter\">Edit type</label>
          <select id=\"causal-edit-filter\"></select>
        </div>
        <div class=\"table-wrap\" id=\"causal-table\"></div>
      </div>

      <div class=\"card\">
        <h3 style=\"margin-top:0\">Attention Summary by Layer <span class=\"badge obs\">Observational</span></h3>
        <div class=\"explain\">
          First-principles read: this tells you where the model is focusing its computation.
          If attention on query terms is weak when it should be high, or cross-segment attention is low, that points to retrieval-matching issues.
        </div>
        <div id=\"attn-insights\" class=\"explain\" style=\"margin-top:8px;\"></div>
        <div id=\"attn-bars\" class=\"attn-bars\"></div>
        <div class=\"table-wrap\" id=\"attn-table\" style=\"margin-top:8px\"></div>
      </div>
    </div>
  </div>

  <script>
    const data = {data_json};

    function pct(v) {{ return `${{(100*Number(v||0)).toFixed(1)}}%`; }}
    function esc(s) {{ return String(s ?? ""); }}
    function escAttr(s) {{
      return String(s ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/'/g, "&#39;");
    }}
    function passPill(v) {{
      const ok = Number(v||0) >= 0.7;
      return `<span class=\"pill ${{ok ? 'ok' : 'bad'}}\">${{pct(v)}}</span>`;
    }}

    function renderOverview() {{
      const o = data.overview || {{}};
      const cards = [
        ["Total Pairs", o.total_pairs ?? 0],
        ["Directional Groups", o.total_groups ?? 0],
        ["Pass Rate", pct(o.pass_rate ?? 0)],
        ["Failed Groups", o.failed_groups ?? 0],
      ];
      document.getElementById("overview-cards").innerHTML = cards.map(([k,v]) =>
        `<div class=\"meta\"><div class=\"k\">${{k}}</div><div class=\"v\">${{v}}</div></div>`
      ).join("");
    }}

    function renderSeedOverview() {{
      const s = data.seed_overview || {{}};
      const cardsHost = document.getElementById("seed-overview-cards");
      const tagHost = document.getElementById("seed-tag-table");
      const exHost = document.getElementById("seed-example-pairs");
      if (!s.available) {{
        cardsHost.innerHTML = "";
        tagHost.innerHTML = "<p class='muted' style='padding:8px'>No handcrafted seed file found.</p>";
        exHost.innerHTML = "";
        return;
      }}
      const cards = [
        ["Seed Rows", Number(s.rows || 0)],
        ["Pair Groups", Number(s.pair_groups || 0)],
        ["Categories", Number((s.by_tag || []).length)],
        ["Example Pairs Shown", Number((s.example_pairs || []).length)],
      ];
      cardsHost.innerHTML = cards.map(([k,v]) =>
        `<div class=\"meta\"><div class=\"k\">${{esc(k)}}</div><div class=\"v\">${{v}}</div></div>`
      ).join("");

      const tagRows = (s.by_tag || []).map(r => ({{
        category: r.question_tag,
        rows: Number(r.rows || 0),
        queries: Number(r.queries || 0),
      }}));
      tagHost.innerHTML = renderSimpleTable(tagRows);

      const exs = s.example_pairs || [];
      if (!exs.length) {{
        exHost.innerHTML = "<p class='muted'>No example handcrafted pairs available.</p>";
        return;
      }}
      exHost.innerHTML = exs.map(r => `
        <div class="seed-pair-card">
          <div class="seed-tag">${{esc(r.question_tag || "")}} · ${{esc(r.pair_group_id || "")}}</div>
          <div class="seed-query">${{esc(r.query || "")}}</div>
          <div class="seed-row"><strong>Higher relevance example:</strong> [${{esc(r.high_label || "-")}}] ${{esc(r.high_item_text || "")}}</div>
          <div class="seed-row"><strong>Lower relevance example:</strong> [${{esc(r.low_label || "-")}}] ${{esc(r.low_item_text || "")}}</div>
          ${{r.notes ? `<div class=\"small-note\" style=\"margin-top:6px\">${{esc(r.notes)}}</div>` : ""}}
        </div>
      `).join("");
    }}

    function renderCausalLabelingBanner() {{
      const host = document.getElementById("causal-labeling-banner");
      const status = String(data.causal_labeling_status || "");
      if (status === "enabled_openai") {{
        host.innerHTML = `<div class="banner" style="border-color:#86efac;background:#f0fdf4;color:#14532d;">
          Causal expected-direction labels are enabled via OpenAI judge. Failure buckets and wrong-direction badges are based on model-judged expectations.
        </div>`;
        return;
      }}
      if (status === "disabled_no_api_key" || status === "present_unlabeled" || status === "other_label_source") {{
        host.innerHTML = `<div class="banner">
          Causal labels are disabled or unavailable. Score changes are still shown, but <strong>expected vs wrong-direction</strong> judgments may be unjudged.<br>
          To enable labels, regenerate counterfactuals with a key:
          <code>python src/generate_counterfactual_dataset.py --prompt-openai-api-key --openai-model gpt-5-mini</code>
        </div>`;
        return;
      }}
      host.innerHTML = "";
    }}

    function f4(v) {{ return Number(v || 0).toFixed(4); }}

    function renderAbsolute() {{
      const a = data.absolute || {{}};
      const t = Number(a.exact_threshold ?? 0.9);
      const cards = [
        {{
          label: `Exact Recall @${{t.toFixed(2)}}`,
          value: pct(a.recall_exact ?? 0),
          tip: "Of all Exact items, the share scoring at or above threshold. Higher means fewer missed Exact results."
        }},
        {{
          label: `Non-Exact Specificity @${{t.toFixed(2)}}`,
          value: pct(a.specificity_non_exact ?? 0),
          tip: "Of all Non-Exact items, the share scoring below threshold. Higher means fewer false Exact-like scores."
        }},
        {{
          label: "Exact F1",
          value: f4(a.f1_exact ?? 0),
          tip: "Harmonic mean of Exact precision and recall at threshold. Useful single-number balance metric."
        }},
        {{
          label: "Irrelevant >= 0.90",
          value: `${{a.irrelevant_high_score_count ?? 0}} (${{pct(a.irrelevant_high_score_rate ?? 0)}})`,
          tip: "Count and rate of Irrelevant items with very high model scores. This is a high-risk calibration violation."
        }},
      ];
      document.getElementById("absolute-cards").innerHTML = cards.map(c =>
        `<div class=\"meta\"><div class=\"k\" title=\"${{escAttr(c.tip)}}\">${{esc(c.label)}}</div><div class=\"v\">${{c.value}}</div></div>`
      ).join("");

      const lrows = data.label_score_summary || [];
      const tableRows = lrows.map(r => ({{
        label: r.label_norm,
        n: Number(r.count ?? 0),
        mean: f4(r.mean),
        p50: f4(r.p50),
        p90: f4(r.p90),
      }}));
      document.getElementById("label-summary-table").innerHTML = renderSimpleTable(tableRows);

      renderLabelScoreChart(data.score_points || []);
    }}

    function renderLabelScoreChart(points) {{
      const svg = document.getElementById("label-score-chart");
      const labels = ["Exact", "Substitute", "Complement", "Irrelevant"];
      const W = 760, H = 250, L = 120, R = 20, T = 18, B = 26;
      const innerW = W - L - R;
      const innerH = H - T - B;
      const rowGap = innerH / Math.max(labels.length, 1);

      let out = [];
      out.push(`<rect x="0" y="0" width="${{W}}" height="${{H}}" fill="#fcfdff"/>`);

      for (let tick = 0; tick <= 10; tick++) {{
        const x = L + (innerW * tick / 10.0);
        out.push(`<line x1="${{x}}" y1="${{T}}" x2="${{x}}" y2="${{H-B}}" stroke="#e2e8f0" stroke-width="1"/>`);
        out.push(`<text x="${{x}}" y="${{H-8}}" text-anchor="middle" fill="#64748b" font-size="11">${{(tick/10).toFixed(1)}}</text>`);
      }}

      labels.forEach((lbl, i) => {{
        const y = T + rowGap * (i + 0.5);
        out.push(`<line x1="${{L}}" y1="${{y}}" x2="${{W-R}}" y2="${{y}}" stroke="#edf2f7" stroke-width="1"/>`);
        out.push(`<text x="${{L-10}}" y="${{y+4}}" text-anchor="end" fill="#334155" font-size="12">${{esc(lbl)}}</text>`);
      }});

      (points || []).forEach((p, idx) => {{
        const label = String(p.label || "");
        const li = labels.indexOf(label);
        if (li < 0) return;
        const score = Math.max(0, Math.min(1, Number(p.score || 0)));
        const x = L + innerW * score;
        const jitter = (((idx * 37) % 100) / 100.0 - 0.5) * (rowGap * 0.55);
        const y = T + rowGap * (li + 0.5) + jitter;
        const color = label === "Exact" ? "#0f766e" : (label === "Irrelevant" ? "#b91c1c" : "#1d4ed8");
        const pid = escAttr(p.probe_id || "");
        const q = escAttr(p.query || "");
        const cat = escAttr(p.question_tag || "");
        const tip = escAttr(`Probe: ${{p.probe_id || ""}} | Label: ${{label}} | Score: ${{score.toFixed(4)}}`);
        out.push(`<circle class="score-dot" cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="3.2" fill="${{color}}" fill-opacity="0.58" title="${{tip}}" data-probe="${{pid}}" data-query="${{q}}" data-cat="${{cat}}" />`);
      }});

      out.push(`<line x1="${{L}}" y1="${{H-B}}" x2="${{W-R}}" y2="${{H-B}}" stroke="#94a3b8" stroke-width="1.2"/>`);
      out.push(`<text x="${{L + innerW/2}}" y="${{H-2}}" text-anchor="middle" fill="#64748b" font-size="12">Model output score</text>`);
      svg.innerHTML = out.join("");
      svg.querySelectorAll("circle.score-dot").forEach(dot => {{
        dot.addEventListener("click", () => {{
          const cat = dot.getAttribute("data-cat") || "";
          const query = dot.getAttribute("data-query") || "";
          const probe = dot.getAttribute("data-probe") || "";
          if (cat) {{
            selectCategory(cat);
          }}
          if (query) {{
            selectQuery(query);
          }}
          if (probe) {{
            selectProbe(probe);
          }}
          document.getElementById("detail-card")?.scrollIntoView({{ behavior: "smooth", block: "start" }});
        }});
      }});
    }}

    function renderFailureBuckets() {{
      const fb = data.failure_buckets || {{}};
      const byEdit = (fb.by_edit_type || []).map(r => ({{
        edit_type: String(r.edit_type || ""),
        num_tests: Number(r.num_tests || 0),
        sign_consistency: Number(r.sign_consistency || 0),
        failure_rate: Number(r.failure_rate || 0),
      }}));
      const wrong = (fb.wrong_direction_examples || []).map(r => ({{
        probe_id: r.probe_id,
        edit_type: r.edit_type,
        expected_delta_direction: r.expected_delta_direction,
        delta_margin: Number(r.delta_margin || 0).toFixed(4),
        query: r.query,
      }}));
      document.getElementById("failure-wrong").innerHTML = renderSimpleTable(wrong);

      const byEditHost = document.getElementById("failure-by-edit");
      if (!byEdit.length) {{
        byEditHost.innerHTML = "<p class='muted' style='padding:8px'>No causal label summary available. Provide an OpenAI API key when generating counterfactuals to enable expected-direction labels.</p>";
        document.getElementById("failure-edit-drilldown").innerHTML = "<p class='muted' style='padding:8px'>No edit-type examples available.</p>";
        return;
      }}

      const head = `<thead><tr><th>Edit Type</th><th>Tests</th><th>Sign Consistency</th><th>Failure Rate</th></tr></thead>`;
      const body = byEdit.map(r => `
        <tr class="clickable" data-edit-type="${{escAttr(r.edit_type)}}">
          <td>${{esc(r.edit_type)}}</td>
          <td>${{r.num_tests}}</td>
          <td>${{pct(r.sign_consistency)}}</td>
          <td>${{pct(r.failure_rate)}}</td>
        </tr>
      `).join("");
      byEditHost.innerHTML = `<table>${{head}}<tbody>${{body}}</tbody></table>`;
      byEditHost.querySelectorAll("tr[data-edit-type]").forEach(tr => {{
        tr.addEventListener("click", () => {{
          selectedFailureEditType = tr.getAttribute("data-edit-type") || "";
          renderFailureEditDrilldown(selectedFailureEditType);
        }});
      }});

      selectedFailureEditType = byEdit[0].edit_type;
      renderFailureEditDrilldown(selectedFailureEditType);
    }}

    function renderFailureEditDrilldown(editType) {{
      const fb = data.failure_buckets || {{}};
      const host = document.getElementById("failure-edit-drilldown");
      const label = document.getElementById("failure-drilldown-label");
      const rows = fb.edit_examples_by_type?.[editType] || [];
      label.textContent = editType
        ? `Edit type drilldown: ${{editType}}`
        : "Edit type drilldown: click a row above to see individual examples";
      if (!rows.length) {{
        host.innerHTML = "<p class='muted' style='padding:8px'>No examples for this edit type.</p>";
        return;
      }}
      const head = `
        <thead>
          <tr>
            <th>Query</th>
            <th>Item</th>
            <th>Text Change</th>
            <th>Expected</th>
            <th>Before</th>
            <th>After</th>
            <th>Delta</th>
            <th>Result</th>
          </tr>
        </thead>
      `;
      const body = rows.map((r, idx) => {{
        const verdict = causalVerdict(r.sign_consistent);
        return `
          <tr class="clickable ${{verdict.state === 'fail' ? 'warnrow' : ''}}" data-drill-idx="${{idx}}" data-edit-type="${{escAttr(editType)}}">
            <td>${{esc(r.query || "")}}</td>
            <td>${{esc(r.item_text || "")}}</td>
            <td>${{renderTextChange(r.original_text, r.edited_text)}}</td>
            <td>${{esc(r.expected_delta_direction || "unknown")}}</td>
            <td>${{Number(r.original_prob || 0).toFixed(3)}}</td>
            <td>${{Number(r.edited_prob || 0).toFixed(3)}}</td>
            <td>${{Number(r.delta_margin || 0).toFixed(3)}}</td>
            <td><span class="pill ${{verdict.pillClass}}">${{verdict.label}}</span></td>
          </tr>
        `;
      }}).join("");
      host.innerHTML = `<table>${{head}}<tbody>${{body}}</tbody></table>`;
      host.querySelectorAll("tr[data-drill-idx]").forEach(tr => {{
        tr.addEventListener("click", () => {{
          const i = Number(tr.getAttribute("data-drill-idx") || 0);
          const chosen = rows[i] || {{}};
          jumpToProbe(chosen);
        }});
      }});
    }}

    function findCategoryByQuery(query) {{
      const cats = data.categories || [];
      for (const c of cats) {{
        const qs = data.queries?.[c.question_tag] || [];
        if (qs.some(x => String(x.query) === String(query))) return c.question_tag;
      }}
      return "";
    }}

    function jumpToProbe(example) {{
      const cat = example.question_tag || findCategoryByQuery(example.query || "");
      if (cat) selectCategory(cat);
      if (example.query) selectQuery(example.query);
      if (example.probe_id) selectProbe(example.probe_id);
      document.getElementById("detail-card")?.scrollIntoView({{ behavior: "smooth", block: "start" }});
    }}

    function renderTopExamples() {{
      const host = document.getElementById("top-examples");
      const rows = data.top_examples || [];
      if (!rows.length) {{
        host.innerHTML = "<p class='muted'>No curated examples yet. Generate counterfactual results to populate this area.</p>";
        return;
      }}
      host.innerHTML = rows.map((r, idx) => `
        <div class="example-card" data-ex-idx="${{idx}}">
          <div class="example-title">${{esc(r.title || "Example")}}</div>
          <div class="example-query">${{esc(r.query || "")}}</div>
          <div class="example-summary">${{esc(r.summary || "")}}</div>
          <div class="small-note">Edit: ${{esc(r.edit_type || "-")}} | Delta: ${{Number(r.delta_margin || 0).toFixed(3)}}</div>
        </div>
      `).join("");
      host.querySelectorAll(".example-card").forEach(card => {{
        card.addEventListener("click", () => {{
          const idx = Number(card.getAttribute("data-ex-idx") || 0);
          jumpToProbe(rows[idx] || {{}});
        }});
      }});
    }}

    function scoreCompareCell(beforeProb, afterProb) {{
      const b = Math.max(0, Math.min(1, Number(beforeProb || 0)));
      const a = Math.max(0, Math.min(1, Number(afterProb || 0)));
      return `
        <div class="score-compare">
          <div class="score-row">
            <div>Before</div>
            <div class="score-track"><div class="score-fill-before" style="width:${{(b*100).toFixed(1)}}%"></div></div>
            <div>${{b.toFixed(3)}}</div>
          </div>
          <div class="score-row">
            <div>After</div>
            <div class="score-track"><div class="score-fill-after" style="width:${{(a*100).toFixed(1)}}%"></div></div>
            <div>${{a.toFixed(3)}}</div>
          </div>
        </div>
      `;
    }}

    function causalVerdict(signConsistent) {{
      if (signConsistent === true) return {{ state: "pass", label: "as expected", pillClass: "ok" }};
      if (signConsistent === false) return {{ state: "fail", label: "wrong-direction", pillClass: "bad" }};
      return {{ state: "unknown", label: "unjudged", pillClass: "" }};
    }}

    function renderTextChange(originalText, editedText) {{
      const o = String(originalText || "");
      const e = String(editedText || "");
      if (!o && !e) return "<span class='muted'>n/a</span>";
      if (!o || !e || o === e) return `<div class="small-note">${{esc(e || o)}}</div>`;

      const ot = o.split(/\\s+/).filter(Boolean);
      const et = e.split(/\\s+/).filter(Boolean);
      const oSet = new Set(ot.map(t => t.toLowerCase()));
      const eSet = new Set(et.map(t => t.toLowerCase()));
      const markStyle = "background:#fde68a; border-radius:4px; padding:0 2px;";
      const om = ot.map(t => eSet.has(t.toLowerCase()) ? esc(t) : `<span style="${{markStyle}}">${{esc(t)}}</span>`).join(" ");
      const em = et.map(t => oSet.has(t.toLowerCase()) ? esc(t) : `<span style="${{markStyle}}">${{esc(t)}}</span>`).join(" ");
      return `
        <div class="small-note"><strong>Original:</strong> ${{om}}</div>
        <div class="small-note"><strong>Edited:</strong> ${{em}}</div>
      `;
    }}

    function renderWhatThisMeans(probeId) {{
      const host = document.getElementById("what-this-means");
      const rows = data.causal_by_probe?.[probeId] || [];
      if (!rows.length) {{
        host.innerHTML = "<strong>What this means:</strong> No causal test rows for this example yet.";
        return;
      }}
      const labeled = rows.filter(r => r.sign_consistent === true || r.sign_consistent === false);
      if (!labeled.length) {{
        host.innerHTML = "<strong>What this means:</strong> Score changes are shown, but expected-direction labels are disabled. Regenerate counterfactuals with an OpenAI API key to mark edits as expected vs wrong-direction.";
        return;
      }}
      const wrong = labeled.filter(r => r.sign_consistent === false);
      const best = labeled.slice().sort((a,b) => Number(b.abs_delta_margin || 0) - Number(a.abs_delta_margin || 0))[0];
      if (wrong.length) {{
        const worst = wrong.slice().sort((a,b) => Number(b.abs_delta_margin || 0) - Number(a.abs_delta_margin || 0))[0];
        host.innerHTML = `
          <strong>What this means:</strong> This example has a concern. After a <strong>${{esc(worst.edit_type)}}</strong> change, the model moved in the wrong direction.<br>
          <span class="muted">Expected: ${{esc(worst.expected_delta_direction)}} | Actual change: ${{Number(worst.delta_margin || 0).toFixed(3)}} margin.</span>
        `;
      }} else {{
        host.innerHTML = `
          <strong>What this means:</strong> This example looks stable. The strongest edit (<strong>${{esc(best.edit_type)}}</strong>) moved the model in the expected direction.<br>
          <span class="muted">Largest change: ${{Number(best.delta_margin || 0).toFixed(3)}} margin.</span>
        `;
      }}
    }}

    function renderCategories() {{
      const rows = data.categories || [];
      const head = `<thead><tr><th>Category</th><th>Queries</th><th>Pairs</th><th>Groups</th><th>Pass Rate</th></tr></thead>`;
      const body = rows.map(r =>
        `<tr class=\"clickable\" data-cat=\"${{esc(r.question_tag)}}\"><td>${{esc(r.question_tag)}}</td><td>${{r.num_queries}}</td><td>${{r.num_pairs}}</td><td>${{r.num_groups}}</td><td>${{passPill(r.pass_rate)}}</td></tr>`
      ).join("");
      const tbl = document.getElementById("categories-table");
      tbl.innerHTML = head + `<tbody>${{body}}</tbody>`;
      tbl.querySelectorAll("tr[data-cat]").forEach(tr => tr.addEventListener("click", () => selectCategory(tr.dataset.cat)));
    }}

    function renderSimpleTable(rows) {{
      if (!rows.length) return "<p class='muted' style='padding:8px'>No data available.</p>";
      const cols = Object.keys(rows[0]);
      const th = `<thead><tr>${{cols.map(c=>`<th>${{esc(c)}}</th>`).join("")}}</tr></thead>`;
      const tb = `<tbody>${{rows.map(r=>`<tr>${{cols.map(c=>`<td>${{esc(r[c])}}</td>`).join("")}}</tr>`).join("")}}</tbody>`;
      return `<table>${{th+tb}}</table>`;
    }}

    function normalizeTokens(rows) {{
      const out = [];
      for (const r of rows || []) {{
        const tok = String(r.token ?? "");
        const seg = String(r.segment ?? "");
        if (["[CLS]","[SEP]","[PAD]"].includes(tok)) continue;
        if (!["query","item"].includes(seg)) continue;
        const signed = Number(r.signed_attr ?? 0);
        if (tok.startsWith("##") && out.length) {{
          out[out.length - 1].token += tok.slice(2);
          out[out.length - 1].signed_attr += signed;
          out[out.length - 1].abs_attr = Math.abs(out[out.length - 1].signed_attr);
        }} else {{
          out.push({{ token: tok, segment: seg, signed_attr: signed, abs_attr: Math.abs(signed) }});
        }}
      }}
      return out;
    }}

    function cloud(tokens, seg) {{
      const xs = tokens.filter(t => t.segment === seg);
      if (!xs.length) return "<p class='muted'>No tokens available.</p>";
      const maxAbs = Math.max(...xs.map(x => Math.abs(x.abs_attr)), 1e-8);
      return `<div class='token-cloud'>${{xs.map(t => {{
        const a = (0.15 + 0.45 * Math.min(Math.abs(t.signed_attr)/maxAbs, 1)).toFixed(3);
        const bg = t.signed_attr >= 0 ? `rgba(16,185,129,${{a}})` : `rgba(239,68,68,${{a}})`;
        return `<span class='tok' style='background:${{bg}}'>${{esc(t.token)}}</span>`;
      }}).join(" ")}}</div>`;
    }}

    function topTokens(tokens, seg, pos, k=5) {{
      const xs = tokens.filter(t => t.segment === seg && (pos ? t.signed_attr > 0 : t.signed_attr < 0));
      xs.sort((a,b) => pos ? (b.signed_attr - a.signed_attr) : (a.signed_attr - b.signed_attr));
      return xs.slice(0,k).map(t => ({{ token: t.token, impact: t.signed_attr.toFixed(4) }}));
    }}

    let selectedCategory = null;
    let selectedQuery = null;
    let selectedProbe = null;
    let selectedCausalFilter = "all";
    let selectedFailureEditType = "";
    let beginnerMode = true;

    function renderCausalTests(probeId) {{
      const filter = document.getElementById("causal-edit-filter");
      const table = document.getElementById("causal-table");
      const rows = data.causal_by_probe?.[probeId] || [];
      const editTypes = Array.from(new Set(rows.map(r => String(r.edit_type || "")).filter(Boolean))).sort();
      const opts = ["all", ...editTypes];
      filter.innerHTML = opts.map(o => `<option value="${{escAttr(o)}}">${{esc(o)}}</option>`).join("");
      filter.value = opts.includes(selectedCausalFilter) ? selectedCausalFilter : "all";

      const shown = rows
        .filter(r => filter.value === "all" ? true : String(r.edit_type || "") === filter.value)
        .slice()
        .sort((a,b) => Number(b.abs_delta_margin || 0) - Number(a.abs_delta_margin || 0));

      if (!shown.length) {{
        table.innerHTML = "<p class='muted' style='padding:8px'>No causal rows available for this probe.</p>";
        return;
      }}

      const head = beginnerMode
        ? `
          <thead>
            <tr>
              <th>Edit Type</th>
              <th>Expected Reaction</th>
              <th>Before vs After Score</th>
              <th>Model Reaction</th>
              <th>Result</th>
            </tr>
          </thead>
        `
        : `
          <thead>
            <tr>
              <th>Edit Type</th>
              <th>Expected</th>
              <th>Original Margin</th>
              <th>Edited Margin</th>
              <th>Delta Margin</th>
              <th>Delta Prob</th>
              <th>Result</th>
            </tr>
          </thead>
        `;
      const body = shown.map(r => {{
        const verdict = causalVerdict(r.sign_consistent);
        if (beginnerMode) {{
          const dm = Number(r.delta_margin || 0);
          const reaction = dm > 0 ? "Score went up" : (dm < 0 ? "Score went down" : "No change");
          return `
            <tr class="${{verdict.state === 'fail' ? 'warnrow' : ''}}">
              <td>${{esc(r.edit_type)}}</td>
              <td>${{esc(r.expected_delta_direction || 'unjudged')}}</td>
              <td>${{scoreCompareCell(r.original_prob, r.edited_prob)}}</td>
              <td>${{reaction}} (${{dm.toFixed(3)}})</td>
              <td><span class="pill ${{verdict.pillClass}}">${{verdict.label}}</span></td>
            </tr>
          `;
        }}
        return `
          <tr class="${{verdict.state === 'fail' ? 'warnrow' : ''}}">
            <td>${{esc(r.edit_type)}}</td>
            <td>${{esc(r.expected_delta_direction || 'unjudged')}}</td>
            <td>${{Number(r.original_margin || 0).toFixed(4)}}</td>
            <td>${{Number(r.edited_margin || 0).toFixed(4)}}</td>
            <td>${{Number(r.delta_margin || 0).toFixed(4)}}</td>
            <td>${{Number(r.delta_prob || 0).toFixed(4)}}</td>
            <td><span class="pill ${{verdict.pillClass}}">${{verdict.label}}</span></td>
          </tr>
        `;
      }}).join("");
      table.innerHTML = `<table>${{head}}<tbody>${{body}}</tbody></table>`;
    }}

    function selectCategory(cat) {{
      selectedCategory = cat;
      document.getElementById("selected-category").innerHTML = `<strong>Category:</strong> ${{esc(cat)}}`;
      const qs = data.queries?.[cat] || [];
      const head = `<thead><tr><th>Query</th><th>Pairs</th><th>Groups</th><th>Pass Rate</th></tr></thead>`;
      const body = qs.map(q => `<tr class=\"clickable\" data-query=\"${{esc(q.query)}}\"><td>${{esc(q.query)}}</td><td>${{q.num_pairs}}</td><td>${{q.num_groups}}</td><td>${{passPill(q.pass_rate)}}</td></tr>`).join("");
      const tbl = document.getElementById("queries-table");
      tbl.innerHTML = head + `<tbody>${{body}}</tbody>`;
      tbl.querySelectorAll("tr[data-query]").forEach(tr => tr.addEventListener("click", () => selectQuery(tr.dataset.query)));

      if (qs.length) selectQuery(qs[0].query);
      else {{ document.getElementById("items-table").innerHTML = ""; clearDetail(); }}
    }}

    function selectQuery(query) {{
      selectedQuery = query;
      document.getElementById("selected-query").innerHTML = `<strong>Query:</strong> ${{esc(query)}}`;
      const items = data.items_by_query?.[query] || [];
      const head = `<thead><tr><th>Item</th><th>GT Label</th><th>GT Score</th><th>Model Score</th><th>Expected</th><th>Model Pred</th><th>Group Result</th></tr></thead>`;
      const body = items.map(it => `
        <tr class=\"clickable\" data-probe=\"${{esc(it.probe_id)}}\">\
          <td>${{esc(it.item_text)}}</td>\
          <td>${{esc(it.esci_label)}}</td>\
          <td>${{Number(it.relevance_score).toFixed(0)}}</td>\
          <td>${{Number(it.model_score).toFixed(4)}}</td>\
          <td>${{esc(it.expected_direction)}}</td>\
          <td>${{esc(it.model_pred_direction)}}</td>\
          <td><span class=\"pill ${{it.group_passed ? 'ok' : 'bad'}}\">${{it.group_passed ? 'pass' : 'fail'}}</span></td>\
        </tr>
      `).join("");
      const tbl = document.getElementById("items-table");
      tbl.innerHTML = head + `<tbody>${{body}}</tbody>`;
      tbl.querySelectorAll("tr[data-probe]").forEach(tr => tr.addEventListener("click", () => selectProbe(tr.dataset.probe)));

      if (items.length) selectProbe(items[0].probe_id);
      else clearDetail();
    }}

    function clearDetail() {{
      ["meta-label","meta-rel","meta-model","meta-exp"].forEach(id => document.getElementById(id).textContent = "-");
      document.getElementById("tag-provenance").textContent = "";
      document.getElementById("pair-query").innerHTML = "";
      document.getElementById("pair-item").innerHTML = "";
      document.getElementById("what-this-means").innerHTML = "";
      document.getElementById("query-cloud").innerHTML = "";
      document.getElementById("item-cloud").innerHTML = "";
      document.getElementById("q-pos").innerHTML = "";
      document.getElementById("q-neg").innerHTML = "";
      document.getElementById("i-pos").innerHTML = "";
      document.getElementById("i-neg").innerHTML = "";
      document.getElementById("causal-edit-filter").innerHTML = "";
      document.getElementById("causal-table").innerHTML = "";
      document.getElementById("attn-table").innerHTML = "";
      document.getElementById("attn-insights").innerHTML = "";
      document.getElementById("attn-bars").innerHTML = "";
    }}

    function renderAttentionInsights(arows) {{
      if (!arows.length) {{
        document.getElementById("attn-insights").innerHTML = "No attention data available for this probe.";
        document.getElementById("attn-bars").innerHTML = "";
        return;
      }}

      const avg = (name) => arows.reduce((s, r) => s + Number(r[name] || 0), 0) / arows.length;
      const qAvg = avg("cls_to_query_mean");
      const iAvg = avg("cls_to_item_mean");
      const qiAvg = avg("query_to_item_mean");

      const layers = arows.map(r => Number(r.layer));
      const maxLayer = Math.max(...layers);
      const final = arows.filter(r => Number(r.layer) === maxLayer)[0] || arows[arows.length - 1];
      const qFinal = Number(final.cls_to_query_mean || 0);
      const iFinal = Number(final.cls_to_item_mean || 0);
      const qiFinal = Number(final.query_to_item_mean || 0);

      const dominantAvg = qAvg > iAvg * 1.15 ? "query-focused" : (iAvg > qAvg * 1.15 ? "item-focused" : "balanced");
      const dominantFinal = qFinal > iFinal * 1.15 ? "query-focused" : (iFinal > qFinal * 1.15 ? "item-focused" : "balanced");
      const crossLevel = qiFinal >= 0.06 ? "strong" : (qiFinal >= 0.03 ? "moderate" : "weak");

      const actions = [];
      if (dominantFinal === "query-focused") {{
        actions.push("Final layer is query-heavy. Check whether important item specs/brands are being underused.");
      }} else if (dominantFinal === "item-focused") {{
        actions.push("Final layer is item-heavy. Check if query constraints (especially negation/spec tokens) are being ignored.");
      }} else {{
        actions.push("Final layer is balanced between query and item, which is usually healthy for matching.");
      }}
      if (crossLevel === "weak") {{
        actions.push("Cross query→item attention is weak. Expect weaker precise matching; inspect failure cases first.");
      }} else if (crossLevel === "moderate") {{
        actions.push("Cross query→item attention is moderate. Verify if critical tokens (brand/spec) still drive ranking correctly.");
      }} else {{
        actions.push("Cross query→item attention is strong. Model is likely connecting query terms to item evidence.");
      }}
      if (dominantAvg !== dominantFinal) {{
        actions.push(`Focus shifts across layers (overall ${{dominantAvg}}, final ${{dominantFinal}}). This can indicate late-stage reweighting.`);
      }}

      document.getElementById("attn-insights").innerHTML = `
        <strong>Attention verdict:</strong> final layer is <strong>${{dominantFinal}}</strong>, cross-linking is <strong>${{crossLevel}}</strong>.<br>
        <span class="muted">Context: overall (all layers) focus is ${{dominantAvg}}.</span><br>
        <ul class="insight-list">${{actions.map(a => `<li>${{a}}</li>`).join("")}}</ul>
      `;

      const maxV = Math.max(...arows.flatMap(r => [Number(r.cls_to_query_mean||0), Number(r.cls_to_item_mean||0), Number(r.query_to_item_mean||0)]), 1e-8);
      const bars = [
        ["[CLS]→Query", qFinal],
        ["[CLS]→Item", iFinal],
        ["Query→Item", qiFinal],
      ];
      document.getElementById("attn-bars").innerHTML = `
        <div class="muted">Final-layer snapshot (Layer ${{maxLayer}})</div>
        ${{bars.map(([k,v]) => `
          <div class="attn-row">
            <div>${{k}}</div>
            <div class="attn-track"><div class="attn-fill" style="width:${{Math.min(100, (v/maxV)*100).toFixed(1)}}%"></div></div>
            <div>${{v.toFixed(4)}}</div>
          </div>
        `).join("")}}
      `;
    }}

    function selectProbe(probeId) {{
      selectedProbe = probeId;
      const item = (data.items_by_query?.[selectedQuery] || []).find(x => x.probe_id === probeId) || {{}};

      document.getElementById("meta-label").textContent = esc(item.esci_label || "-");
      document.getElementById("meta-rel").textContent = item.relevance_score != null ? String(item.relevance_score) : "-";
      document.getElementById("meta-model").textContent = item.model_score != null ? Number(item.model_score).toFixed(4) : "-";
      document.getElementById("meta-exp").textContent = esc(item.expected_direction || "-");
      document.getElementById("tag-provenance").textContent =
        (item.tag_reason || item.tag_confidence)
          ? `Tag provenance: tag=${{item.question_tag || "-"}}, reason=${{item.tag_reason || "-"}}, confidence=${{item.tag_confidence || "-"}}`
          : "";
      document.getElementById("pair-query").innerHTML = `<b>Query:</b> ${{esc(item.query || selectedQuery || "")}}`;
      document.getElementById("pair-item").innerHTML = `<b>Item:</b> ${{esc(item.item_text || "")}}`;

      const trows = data.attrs_by_probe?.[probeId] || [];
      const toks = normalizeTokens(trows);
      document.getElementById("query-cloud").innerHTML = cloud(toks, "query");
      document.getElementById("item-cloud").innerHTML = cloud(toks, "item");
      document.getElementById("q-pos").innerHTML = renderSimpleTable(topTokens(toks, "query", true));
      document.getElementById("q-neg").innerHTML = renderSimpleTable(topTokens(toks, "query", false));
      document.getElementById("i-pos").innerHTML = renderSimpleTable(topTokens(toks, "item", true));
      document.getElementById("i-neg").innerHTML = renderSimpleTable(topTokens(toks, "item", false));

      const arows = data.attn_by_probe?.[probeId] || [];
      renderAttentionInsights(arows);
      document.getElementById("attn-table").innerHTML = renderSimpleTable(arows.map(r => ({{
        layer: r.layer,
        cls_to_query_mean: Number(r.cls_to_query_mean).toFixed(4),
        cls_to_item_mean: Number(r.cls_to_item_mean).toFixed(4),
        query_to_item_mean: Number(r.query_to_item_mean).toFixed(4),
      }})));

      renderCausalTests(probeId);
      renderWhatThisMeans(probeId);
    }}

    renderOverview();
    renderSeedOverview();
    renderCausalLabelingBanner();
    renderTopExamples();
    renderAbsolute();
    renderFailureBuckets();
    renderCategories();
    document.getElementById("mode-toggle").addEventListener("change", (e) => {{
      beginnerMode = Boolean(e.target.checked);
      if (selectedProbe) renderCausalTests(selectedProbe);
    }});
    document.getElementById("causal-edit-filter").addEventListener("change", (e) => {{
      selectedCausalFilter = e.target.value || "all";
      if (selectedProbe) renderCausalTests(selectedProbe);
    }});
    if ((data.categories || []).length) {{
      selectCategory(data.categories[0].question_tag);
    }}
  </script>
</body>
</html>
"""

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static progressive-disclosure HTML dashboard.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/dashboard.html"))
    args = parser.parse_args()

    out = build_dashboard(args.outputs_dir, args.out_html)
    print(f"Wrote dashboard to {out}")


if __name__ == "__main__":
    main()
