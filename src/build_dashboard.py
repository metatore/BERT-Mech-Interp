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


def _build_payload(outputs_dir: Path) -> dict[str, object]:
    scored = _safe_read_csv(outputs_dir / "scored_pairs.csv")
    scorecard = _safe_read_csv(outputs_dir / "question_scorecard.csv")
    attrs = _safe_read_csv(outputs_dir / "attributions_by_probe.csv")
    attn = _safe_read_csv(outputs_dir / "attention_by_probe.csv")

    if scored.empty:
        return {
            "overview": {},
            "categories": [],
            "queries": {},
            "items_by_query": {},
            "attrs_by_probe": {},
            "attn_by_probe": {},
            "esci_map": {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0},
        }

    # Normalize fields used by UI.
    scored = scored.copy()
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
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }}
      .layout {{ grid-template-columns: 1fr; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .meta-grid {{ grid-template-columns: repeat(2, minmax(140px,1fr)); }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1 class=\"title\">E-commerce Relevance Mech-Interp Dashboard</h1>
    <p class=\"subtitle\">Progressive drill-down: Overview → Category → Query → Query-item diagnostics</p>

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
      <div class=\"explain\">Green tokens pushed relevance up. Red tokens pushed down. Darker color means stronger influence. Attention by layer is supporting evidence, not a full explanation.</div>
      <div class=\"meta-grid\">
        <div class=\"meta\"><div class=\"k\">Ground Truth Label</div><div class=\"v\" id=\"meta-label\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Ground Truth Score</div><div class=\"v\" id=\"meta-rel\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Model Output Score</div><div class=\"v\" id=\"meta-model\">-</div></div>
        <div class=\"meta\"><div class=\"k\">Expected Direction</div><div class=\"v\" id=\"meta-exp\">-</div></div>
      </div>
      <div id=\"pair-query\" class=\"pair-text\"></div>
      <div id=\"pair-item\" class=\"pair-text\"></div>

      <div class=\"two-col\">
        <div class=\"card\">
          <h3 style=\"margin-top:0\">Query Token Attribution</h3>
          <div id=\"query-cloud\"></div>
          <div class=\"two-col\" style=\"margin-top:8px\">
            <div><strong>Helpful query tokens</strong><div id=\"q-pos\" class=\"table-wrap\"></div></div>
            <div><strong>Harmful query tokens</strong><div id=\"q-neg\" class=\"table-wrap\"></div></div>
          </div>
        </div>
        <div class=\"card\">
          <h3 style=\"margin-top:0\">Item Token Attribution</h3>
          <div id=\"item-cloud\"></div>
          <div class=\"two-col\" style=\"margin-top:8px\">
            <div><strong>Helpful item tokens</strong><div id=\"i-pos\" class=\"table-wrap\"></div></div>
            <div><strong>Harmful item tokens</strong><div id=\"i-neg\" class=\"table-wrap\"></div></div>
          </div>
        </div>
      </div>

      <div class=\"card\">
        <h3 style=\"margin-top:0\">Attention Summary by Layer</h3>
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
      document.getElementById("pair-query").innerHTML = "";
      document.getElementById("pair-item").innerHTML = "";
      document.getElementById("query-cloud").innerHTML = "";
      document.getElementById("item-cloud").innerHTML = "";
      document.getElementById("q-pos").innerHTML = "";
      document.getElementById("q-neg").innerHTML = "";
      document.getElementById("i-pos").innerHTML = "";
      document.getElementById("i-neg").innerHTML = "";
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
    }}

    renderOverview();
    renderCategories();
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
