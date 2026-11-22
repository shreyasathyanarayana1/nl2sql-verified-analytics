"""Agent 10 — Explanation. Writes the long-form answer: what the data shows, how it was worked out,
how far to trust it, and what to keep in mind.

Every sentence is computed from the verified result and the pipeline state — nothing is generated
freely, so the explanation cannot claim a number the query did not return.
"""
from typing import Any, Dict, List, Optional

from app.agents.visualization import _cap, _fmt, _unit, band_label
from app.datasets import Dataset
from app.semantic import Semantic

SMALL_GROUP = 30  # groups smaller than this get a reliability caveat

KIND_PHRASE = {
    "aggregate": "a summary calculation",
    "ranking": "a ranking",
    "distribution": "a distribution — how values are spread out",
    "correlation": "a relationship between two measurements",
    "drivers": "a comparison of every available factor against one outcome",
    "list": "a filtered list of records",
    "count_having": "a count of groups meeting a condition",
}


def _pct(part: float, whole: float) -> str:
    return f"{round(100 * part / whole)}%" if whole else "—"


def _section(key: str, title: str, paragraphs: List[str], bullets: Optional[List[str]] = None, after: int = 1) -> Dict[str, Any]:
    """`after` is the number of paragraphs that come before the bullet list (the one that introduces it ends with a colon)."""
    return {"key": key, "title": title, "paragraphs": [p for p in paragraphs if p], "bullets": bullets or [], "bullets_after": after}


def _join(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build(ds: Dataset, sem: Semantic, state: Dict[str, Any]) -> Dict[str, Any]:
    intent, result, plan, viz = state["intent"], state["result"], state["plan"], state["visualization"]
    scope = [f["label"] for f in intent["filters"]]
    scope_txt = f" ({', '.join(scope)})" if scope else ""
    findings = _findings(ds, sem, intent, result, plan, viz, scope_txt)
    sections = [
        _section("findings", "What the data shows", findings["paragraphs"], findings["bullets"], findings["after"]),
        _method(ds, state),
        _trust(state),
        _caveats(ds, intent, result, findings["small"], state, scope),
    ]
    return {"sections": [s for s in sections if s["paragraphs"] or s["bullets"]]}


# --------------------------------------------------------------------------- findings

def _findings(ds, sem, intent, result, plan, viz, scope_txt) -> Dict[str, Any]:
    out = {"paragraphs": [], "bullets": [], "small": [], "after": 1}
    cols, rows = result["columns"], result["rows"]
    kind = intent["kind"]
    if not rows:
        out["paragraphs"].append(f"No {ds.plural} matched{scope_txt}, so there is nothing to summarise. "
                                 "Removing or loosening a filter is the usual fix.")
        return out
    if kind == "drivers":
        return _drivers(ds, viz, out)
    if kind == "distribution":
        return _distribution(ds, plan, rows, scope_txt, out)
    if kind == "correlation":
        return _correlation(ds, plan, cols, rows, viz, scope_txt, out)
    metric = plan["spec"].get("metric_alias")
    mi = cols.index(metric) if metric in cols else None
    if kind == "list" or (kind == "ranking" and intent["ranking"]["target"] == "rows"):
        return _row_ranking(ds, intent, cols, rows, mi, result, scope_txt, out)
    if not intent["group_by"]:
        return _single(ds, sem, intent, rows, mi, scope_txt, out)
    return _groups(ds, sem, intent, cols, rows, mi, viz, scope_txt, out)


def _groups(ds, sem, intent, cols, rows, mi, viz, scope_txt, out):
    measure = intent["measures"][0]
    gcol = ds.columns[intent["group_by"][0]["col"]]
    labels = list(viz["labels"]) + ["(missing)"] * (len(rows) - len(viz["labels"]))
    count_alias = sem.count_alias()
    ni = cols.index(count_alias) if count_alias in cols and measure["agg"] != "COUNT" else None
    label = sem.measure_label(measure)
    is_rate = measure["agg"] == "RATE"
    unit = "%" if is_rate else "" if measure["agg"] == "COUNT" else _unit(ds.columns[measure["col"]].unit)
    fmt = lambda v: f"{_fmt(v)}{unit}"

    groups = [{"label": labels[i], "value": r[mi], "n": r[ni] if ni is not None else None}
              for i, r in enumerate(rows) if isinstance(r[mi], (int, float))]
    known = [g for g in groups if g["label"] != "(missing)"]
    if not known:
        return out

    if intent.get("share"):
        si = cols.index("share_pct")
        total = sum(r[mi] for r in rows)
        ranked = sorted(zip(labels, rows), key=lambda p: -(p[1][si] or 0))
        out["paragraphs"].append(f"The {_fmt(total)} {ds.plural}{scope_txt} split into {len(ranked)} {gcol.label} groups:")
        out["bullets"] = [f"{lbl} — {_fmt(r[si])}% ({_fmt(r[mi])} {ds.plural})" for lbl, r in ranked]
        if len(ranked) >= 3:
            top2 = ranked[0][1][si] + ranked[1][1][si]
            out["paragraphs"].append(f"{ranked[0][0]} and {ranked[1][0]} together account for {_fmt(round(top2, 1))}% of {ds.plural}, "
                                     f"while {ranked[-1][0]} is the smallest group at {_fmt(ranked[-1][1][si])}%.")
        out["small"] = [(lbl, r[mi]) for lbl, r in ranked if r[mi] < SMALL_GROUP]
        return out

    ordered = bool(gcol.order) or bool(intent["group_by"][0]["bin"])
    listing = known if ordered else sorted(known, key=lambda g: -g["value"])
    out["paragraphs"].append(
        f"{len(known)} {gcol.label} groups were compared{scope_txt}"
        + (", listed in their natural order:" if ordered else ", from highest to lowest:"))
    out["bullets"] = [f"{g['label']} — {fmt(g['value'])}" + (f" ({_fmt(g['n'])} {ds.plural})" if g["n"] is not None else "")
                      for g in listing]

    hi = max(known, key=lambda g: g["value"])
    lo = min(known, key=lambda g: g["value"])
    if hi["value"] == lo["value"]:
        out["paragraphs"].append(f"Every group has the same {label}, so {gcol.label} makes no difference here.")
        return out

    if ni is not None and measure["agg"] in ("AVG", "RATE"):
        weighted = [(g["value"], g["n"]) for g in groups if g["n"]]
        total_n = sum(n for _, n in weighted)
        overall = sum(v * n for v, n in weighted) / total_n if total_n else None
        if overall is not None:
            above = [g["label"] for g in known if g["value"] > overall]
            out["paragraphs"].append(
                f"Taking all {_fmt(total_n)} {ds.plural} together, the {label} is {fmt(round(overall, 2))}. "
                f"{len(above)} of the {len(known)} groups {'sits' if len(above) == 1 else 'sit'} above that figure"
                + (f" ({', '.join(above)})" if 0 < len(above) <= 3 else "") + ".")
            gap = hi["value"] - lo["value"]
            if is_rate:
                size = "large" if gap >= 15 else "moderate" if gap >= 5 else "small"
                rel = (f" — {hi['label']} is {hi['value'] / lo['value']:.1f} times as high as {lo['label']}"
                       if lo["value"] > 0 and hi["value"] / lo["value"] >= 1.5 else "")
                out["paragraphs"].append(f"The difference between the highest group ({hi['label']}) and the lowest ({lo['label']}) is "
                                         f"{_fmt(round(gap, 1))} percentage points, which is {size}{rel}.")
            elif overall:
                share = 100 * gap / abs(overall)
                size = "large" if share >= 10 else "moderate" if share >= 3 else "small"
                out["paragraphs"].append(f"The highest group ({hi['label']}) is {fmt(round(gap, 2))} above the lowest ({lo['label']}). "
                                         f"That is about {share:.1f}% of the overall figure — a {size} difference in practical terms.")

    if ordered and len(known) >= 3:
        steps = [b["value"] - a["value"] for a, b in zip(known, known[1:])]
        ups, downs = sum(s > 0 for s in steps), sum(s < 0 for s in steps)
        first, last = known[0], known[-1]
        if ups == len(steps):
            trend = f"rises at every step from {first['label']} to {last['label']}"
        elif downs == len(steps):
            trend = f"falls at every step from {first['label']} to {last['label']}"
        else:
            trend = (f"moves from {fmt(first['value'])} at {first['label']} to {fmt(last['value'])} at {last['label']}, "
                     f"but not steadily — it goes up in {ups} of {len(steps)} steps and down in {downs}")
        out["paragraphs"].append(f"Because {gcol.label} has a natural order, the pattern matters: the {label} {trend}.")

    out["small"] = [(g["label"], g["n"]) for g in known if g["n"] is not None and g["n"] < SMALL_GROUP]
    return out


def _single(ds, sem, intent, rows, mi, scope_txt, out):
    measure = intent["measures"][0]
    value = rows[0][mi] if mi is not None else None
    if value is None:
        return out
    if measure["agg"] == "COUNT":
        text = f"There {'is' if value == 1 else 'are'} {_fmt(value)} {ds.plural if value != 1 else ds.singular}" + \
            (f" {_join([f['label'] for f in intent['filters']])}" if intent["filters"] else "")
        if intent["filters"] and ds.row_count:
            text += f", which is {_pct(value, ds.row_count)} of all {_fmt(ds.row_count)} {ds.plural} in the dataset"
        out["paragraphs"].append(text + ".")
    elif measure["agg"] == "RATE":
        col = ds.columns[measure["col"]]
        out["paragraphs"].append(f"{_fmt(value)}% of {ds.plural}{scope_txt} have {col.label}. Records with no recorded value for "
                                 f"{col.label} are left out of both the top and bottom of that fraction.")
    else:
        col = ds.columns[measure["col"]]
        out["paragraphs"].append(f"The {sem.measure_label(measure)} is {_fmt(value)}{_unit(col.unit)}, computed over every "
                                 f"{ds.singular}{scope_txt} with a recorded {col.label}.")
    if not intent["filters"] and measure["agg"] != "COUNT":
        out["paragraphs"].append(f"This covers the whole dataset of {_fmt(ds.row_count)} {ds.plural}. "
                                 f"Breaking it down by a category (for example “… by {_example_group(ds)}”) shows where it varies.")
    return out


def _example_group(ds) -> str:
    cats = [c for c in ds.columns.values() if c.role in ("category", "flag")]
    return cats[0].label if cats else "group"


def _row_ranking(ds, intent, cols, rows, mi, result, scope_txt, out):
    if intent["kind"] == "list" or mi is None:
        n = result["row_count"]
        out["paragraphs"].append(f"{_fmt(n)} {ds.plural if n != 1 else ds.singular}{scope_txt} matched. "
                                 "Each row in the table is one record, and clicking a row shows its full details.")
        return out
    ii = cols.index(ds.id_column)
    col = ds.columns[cols[mi]]
    unit = _unit(col.unit)
    values = [r[mi] for r in rows if isinstance(r[mi], (int, float))]
    word = "highest" if intent["ranking"]["direction"] == "DESC" else "lowest"
    out["paragraphs"].append(
        f"These are the {len(rows)} {ds.plural}{scope_txt} with the {word} {col.label}. "
        f"Values run from {_fmt(values[0])}{unit} down to {_fmt(values[-1])}{unit}"
        + (f", and the average across them is {_fmt(round(sum(values) / len(values), 2))}{unit}." if values else "."))
    out["bullets"] = [f"{ds.entity['id_label']} {r[ii]} — {_fmt(r[mi])}{unit}" for r in rows[:5]]
    if len(values) >= 2 and values[0] != values[1]:
        out["paragraphs"].append(f"The top {ds.singular} stands {_fmt(round(abs(values[0] - values[1]), 2))}{unit} clear of the second.")
    return out


def _distribution(ds, plan, rows, scope_txt, out):
    b = plan["bucket"]
    col = ds.columns[b["column"]]
    bands = [(band_label(r[0], b["width"], b["integer"], b["unit"]), r[1]) for r in rows if r[0] is not None]
    total = sum(n for _, n in bands)
    if not total:
        return out
    peak = max(bands, key=lambda p: p[1])
    running, median_band = 0, bands[0][0]
    for lbl, n in bands:
        running += n
        if running >= total / 2:
            median_band = lbl
            break
    out["paragraphs"].append(
        f"{_fmt(total)} {ds.plural}{scope_txt} are spread across {len(bands)} {col.label} bands, from {bands[0][0]} up to {bands[-1][0]}. "
        f"The busiest band is {peak[0]} with {_fmt(peak[1])} {ds.plural} ({_pct(peak[1], total)}), and the middle {ds.singular} "
        f"falls in the {median_band} band.")
    top3 = sorted(bands, key=lambda p: -p[1])[:3]
    out["paragraphs"].append(f"The three largest bands hold {_pct(sum(n for _, n in top3), total)} of all {ds.plural} between them:")
    out["bullets"] = [f"{lbl} — {_fmt(n)} {ds.plural} ({_pct(n, total)})" for lbl, n in top3]
    out["after"] = 2
    tails = [(lbl, n) for lbl, n in (bands[0], bands[-1]) if n < 0.02 * total]
    if tails:
        out["paragraphs"].append("The outermost bands are very thin (" + ", ".join(f"{lbl}: {_fmt(n)}" for lbl, n in tails) +
                                 "), so those are rare or unusual values rather than a typical pattern.")
    return out


def _correlation(ds, plan, cols, rows, viz, scope_txt, out):
    x, y = ds.columns[plan["axes"]["x"]], ds.columns[plan["axes"]["y"]]
    xi, yi = cols.index(x.name), cols.index(y.name)
    pts = [(r[xi], r[yi]) for r in rows if isinstance(r[xi], (int, float)) and isinstance(r[yi], (int, float))]
    r_val = (viz.get("stats") or {}).get("pearson_r")
    if r_val is None or len(pts) < 3:
        return out
    n = len(pts)
    mx, my = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
    var_x = sum((p[0] - mx) ** 2 for p in pts)
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / var_x if var_x else 0
    strength = "very weak" if abs(r_val) < 0.1 else "weak" if abs(r_val) < 0.3 else "moderate" if abs(r_val) < 0.5 else "strong"
    out["paragraphs"].append(
        f"Across {_fmt(n)} {ds.plural}{scope_txt} with both values recorded, the correlation between {x.label} and {y.label} is "
        f"r = {r_val:.2f}, a {strength} {'positive' if r_val > 0 else 'negative'} relationship.")
    out["paragraphs"].append(
        f"Squaring r gives about {round(100 * r_val ** 2)}%: that is roughly how much of the variation in {y.label} lines up with "
        f"{x.label}. The remaining {100 - round(100 * r_val ** 2)}% comes from other things this chart does not show.")
    per = (x.unit[:-1] if x.unit.endswith("s") else x.unit) if x.unit and x.unit != "%" else "point"
    out["paragraphs"].append(
        f"On average, {y.label} changes by {slope:+.2f}{_unit(y.unit)} per {per} of {x.label}. "
        f"The typical {ds.singular} has {x.label} {_fmt(round(mx, 1))}{_unit(x.unit)} and {y.label} {_fmt(round(my, 1))}{_unit(y.unit)}.")
    return out


def _drivers(ds, viz, out):
    d = viz.get("drivers")
    if not d or not d["factors"]:
        return out
    unit = d["unit"]
    pts = " percentage points" if unit == "%" else unit
    factors = d["factors"]
    out["paragraphs"].append(
        f"{len(factors)} factors were tested against {d['outcome']}. For each one, {ds.plural} were split into its levels and the "
        f"best and worst level were compared; levels with fewer than {d['min_support']} {ds.plural} were ignored so a handful of "
        "records cannot top the ranking. The factors with the largest gaps:")
    out["bullets"] = [
        f"{_cap(f['label'])} — {f['best']['level']}: {_fmt(f['best']['value'])}{unit} vs {f['worst']['level']}: "
        f"{_fmt(f['worst']['value'])}{unit} (gap {_fmt(f['spread'])}{pts})"
        for f in factors[:5]]
    weak = [f["label"] for f in factors if f["spread"] < factors[0]["spread"] * 0.1]
    if weak:
        out["paragraphs"].append(f"{len(weak)} factor{'s' if len(weak) != 1 else ''} made almost no difference "
                                 f"({', '.join(weak[:4])}{'…' if len(weak) > 4 else ''}).")
    return out


# --------------------------------------------------------------------------- method / trust / caveats

def _method(ds, state) -> Dict[str, Any]:
    intent, ctx, plan, attempts, result = state["intent"], state["schema_context"], state["plan"], state["attempts"], state["result"]
    paras = []
    reading = f"The question was read as {KIND_PHRASE.get(intent['kind'], intent['kind'])}."
    ml = intent.get("ml_intent") or {}
    if ml.get("enabled"):
        agrees = ml["prediction"] == intent["kind"]
        reading += (f" The intent classifier independently predicted “{ml['prediction']}” with {ml['confidence']:.0%} confidence"
                    + (", agreeing with the rule-based parser." if agrees else "; the rule-based parser's reading was kept."))
    paras.append(reading)
    if ctx:
        if ctx["total_tables"] == 1:
            paras.append(f"The dataset has a single table, {ctx['tables'][0]}, which the schema retriever passed to the SQL generator. The plan:")
        else:
            paras.append(f"The schema retriever ranked all {ctx['total_tables']} tables by relevance and gave the SQL generator only "
                         f"{_join(ctx['tables'])}, so it could not reach for anything else. The plan:")
    first = attempts[0] if attempts else None
    if first:
        rejected = [a for a in attempts if a["validation"] and not a["validation"]["passed"]]
        verify = ("The SQL was checked for schema, meaning and join errors and screened by a hallucination classifier before it was "
                  "allowed to run.")
        if not rejected:
            verify += " The first draft passed every check, so it ran unchanged."
        else:
            types = sorted({t for a in rejected for t in a["validation"]["hallucination_types"]})
            fixes = [s for a in attempts if a["repair"] for s in a["repair"]["strategies"]]
            verify += (f" {len(rejected)} draft{'s were' if len(rejected) != 1 else ' was'} rejected ({', '.join(types)} errors); "
                       f"the repair agent applied {', '.join(fixes) or 'a fix'} and draft {len(attempts)} passed.")
        paras.append(verify)
    if result:
        paras.append(f"The final query ran read-only and returned {_fmt(result['row_count'])} row{'s' if result['row_count'] != 1 else ''} "
                     f"in {result['exec_ms']} ms.")
    # the plan steps sit between the retrieval paragraph and the verification paragraph
    return _section("method", "How the answer was worked out", paras, plan["steps"] if plan else [], after=2 if ctx else 1)


def _trust(state) -> Dict[str, Any]:
    c, rc = state["confidence"], state["result_checks"]
    paras, bullets = [], []
    if c:
        paras.append(f"Confidence is {c['score']}/100 ({c['label'].lower()}). It combines the SQL checks, the result checks and the data "
                     "quality findings below with a learned confidence model.")
    applicable = [x for x in (rc or {}).get("checks", []) if x["applicable"]]
    if applicable:
        paras.append(f"After the query ran, {len(applicable)} independent check{'s were' if len(applicable) != 1 else ' was'} run on the answer itself:")
        bullets = [f"{'Passed' if x['passed'] else 'Failed'}: {x['title']} — {x['detail']}" for x in applicable]
    lowered = [r["text"] for r in (c or {}).get("reasons", []) if r["impact"] < 0]
    if lowered:
        paras.append("The score was lowered by: " + "; ".join(t.rstrip(".") for t in lowered) + ".")
    return _section("trust", "How far to trust it", paras, bullets, after=2 if c else 1)


def _caveats(ds, intent, result, small, state, scope) -> Dict[str, Any]:
    paras = []
    if scope:
        paras.append(f"Only {ds.plural} {_join(scope)} are included; everyone else was filtered out before any figure was computed.")
    for lbl, n in small[:3]:
        paras.append(f"{lbl} has only {_fmt(n)} {ds.plural if n != 1 else ds.singular}, so its figure could shift noticeably with a few more records.")
    if intent["kind"] == "correlation":
        paras.append("A correlation shows that two measurements move together in this dataset, not that one causes the other.")
    elif intent["kind"] == "drivers" or (intent["group_by"] and not intent.get("share")):
        paras.append("These are associations in this dataset. They show which groups differ, not that one thing causes another.")
    if result and result["row_count"] > 500:
        paras.append("Only the first 500 rows are shown in the table; the figures above use every row.")
    return _section("caveats", "Keep in mind", paras)
