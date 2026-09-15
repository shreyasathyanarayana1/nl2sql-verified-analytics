"""Agent 9 — Visualization. Picks a chart for the result shape and explains it in plain English."""
from typing import Any, Dict, List, Optional

from app.agents.data_quality import pearson
from app.datasets import Dataset
from app.semantic import Semantic

DRIVER_MIN_SUPPORT = 0.02  # ignore levels holding less than 2% of rows when ranking factors


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:,}" if isinstance(v, int) else str(v)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def band_label(start: Any, width: float, integer: bool, unit: str) -> str:
    if start is None:
        return "(missing)"
    if integer and width == 1:
        return f"{_fmt(start)}{'%' if unit == '%' else ''}"
    end = start + width - (1 if integer and float(width).is_integer() else 0)
    suffix = "%" if unit == "%" else ""
    return f"{_fmt(start)}–{_fmt(end)}{suffix}"


def _base(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {"chart_type": "table", "label_column": None, "key_column": None, "value_column": None, "value_label": None,
            "labels": [], "values": [], "points": [], "horizontal": False, "x_label": None, "y_label": None, "stats": None,
            "drivers": None, "explanation": "", "sql_in_plain_english": plan["steps"]}


def build(ds: Dataset, sem: Semantic, intent: Dict[str, Any], result: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    viz = _base(plan)
    cols, rows = result["columns"], result["rows"]
    kind = intent["kind"]
    scope = [f["label"] for f in intent["filters"]]
    scope_txt = f" ({', '.join(scope)})" if scope else ""

    if not rows:
        viz["chart_type"] = "empty"
        viz["explanation"] = f"No {ds.plural} matched{scope_txt}. Try removing a filter."
        return viz

    if kind == "drivers":
        return _drivers(ds, sem, intent, result, plan, viz, scope_txt)

    if kind == "distribution":
        b = plan["bucket"]
        col = ds.columns[b["column"]]
        labels = [band_label(r[0], b["width"], b["integer"], b["unit"]) for r in rows]
        values = [r[1] for r in rows]
        total = sum(values)
        peak = max(range(len(rows)), key=lambda i: values[i])
        viz.update(chart_type="histogram", label_column="bucket_start", key_column="bucket_start", value_column=sem.count_alias(),
                   value_label=f"number of {ds.plural}", labels=labels, values=values,
                   x_label=f"{col.label}" + (f" ({col.unit})" if col.unit and col.unit != "%" else ""))
        viz["explanation"] = (f"Across {_fmt(total)} {ds.plural}{scope_txt}, the most common {col.label} band is {labels[peak]}, "
                              f"with {_fmt(values[peak])} {ds.plural} ({round(100 * values[peak] / total)}%).")
        return viz

    if kind == "correlation":
        x, y = ds.columns[plan["axes"]["x"]], ds.columns[plan["axes"]["y"]]
        xi, yi, ii = cols.index(x.name), cols.index(y.name), cols.index(ds.id_column)
        r_val = pearson([r[xi] for r in rows], [r[yi] for r in rows])
        step = max(1, len(rows) // 1500)  # plot at most ~1,500 points; r uses every row
        points = [{"x": r[xi], "y": r[yi], "label": f"{ds.entity['id_label']} {r[ii]}"} for r in rows[::step]]
        viz.update(chart_type="scatter", points=points, x_label=sem.column_labels()[x.name], y_label=sem.column_labels()[y.name],
                   stats={"pearson_r": round(r_val, 3) if r_val is not None else None, "n": len(rows), "plotted": len(points)})
        viz["explanation"] = _explain_correlation(ds, r_val, x.label, y.label, len(rows), scope_txt)
        return viz

    labels_map = sem.column_labels()
    measure = intent["measures"][0] if intent["measures"] else None
    metric_col = plan["spec"].get("metric_alias")
    mi = cols.index(metric_col) if metric_col in cols else None

    if kind == "list" or (kind == "ranking" and intent["ranking"]["target"] == "rows"):
        if kind == "ranking" and mi is not None:
            ii = cols.index(ds.id_column)
            viz.update(chart_type="bar", horizontal=True, label_column=ds.id_column, key_column=None, value_column=metric_col,
                       value_label=labels_map.get(metric_col, metric_col),
                       labels=[f"{ds.entity['id_label']} {r[ii]}" for r in rows], values=[r[mi] for r in rows])
            word = "highest" if intent["ranking"]["direction"] == "DESC" else "lowest"
            viz["explanation"] = (f"These are the {len(rows)} {ds.plural} with the {word} {ds.columns[metric_col].label}{scope_txt}, "
                                  f"from {_fmt(rows[0][mi])} to {_fmt(rows[-1][mi])}.")
        else:
            trunc = " (showing the first 200)" if result["row_count"] >= 200 else ""
            viz["explanation"] = f"Found {_fmt(result['row_count'])} {ds.plural if result['row_count'] != 1 else ds.singular}{scope_txt}{trunc}."
        return viz

    groups = intent["group_by"]
    if not groups:
        value = rows[0][mi] if mi is not None else None
        viz.update(chart_type="kpi", value_column=metric_col, value_label=labels_map.get(metric_col, metric_col))
        if measure["agg"] == "COUNT":
            viz["explanation"] = f"{_fmt(value)} {ds.plural if value != 1 else ds.singular}{scope_txt}."
        elif measure["agg"] == "RATE":
            viz["explanation"] = f"{_fmt(value)}% of {ds.plural}{scope_txt} have {ds.columns[measure['col']].label}."
        else:
            col = ds.columns[measure["col"]]
            viz["explanation"] = f"The {sem.measure_label(measure)}{scope_txt} is {_fmt(value)}{_unit(col.unit)}."
        return viz

    g = groups[0]
    gcol = ds.columns[g["col"]]
    key = f"{gcol.name}_band" if g["bin"] else gcol.name
    gi = cols.index(key)
    def group_label(v: Any) -> str:
        if g["bin"]:
            return band_label(v, g["bin"], gcol.type == "INTEGER", gcol.unit)
        if v is None:
            return "(missing)"
        if gcol.role == "flag":
            return f"{'With' if v == 'Yes' else 'Without'} {gcol.label}"
        return str(v)
    labels = [group_label(r[gi]) for r in rows]
    values = [r[mi] for r in rows]
    ordered = bool(g["bin"]) or bool(gcol.order)
    chart = "bar"
    if intent.get("share"):
        chart = "pie"
        values = [r[cols.index("share_pct")] for r in rows]
    elif ordered and len(rows) >= 4 and g["bin"]:
        chart = "line"
    elif len(rows) > 40:
        chart = "table"
    chart_labels, chart_values = labels, values
    if chart in ("line", "bar") and labels and labels[-1] == "(missing)":
        # the "missing" group sorts last; keep it in the table but not on a chart axis it doesn't belong to
        chart_labels, chart_values = labels[:-1], values[:-1]
    viz.update(chart_type=chart, label_column=key, key_column=key, value_column=metric_col,
               value_label="share (%)" if intent.get("share") else labels_map.get(metric_col, metric_col),
               labels=chart_labels, values=chart_values, horizontal=chart == "bar" and len(rows) > 8,
               x_label=labels_map.get(key, key))
    viz["explanation"] = _explain_groups(ds, sem, intent, rows, cols, labels, mi, scope_txt, gcol)
    return viz


def _unit(unit: str) -> str:
    return "%" if unit == "%" else (f" {unit}" if unit and unit != "points" else "")


def _explain_groups(ds, sem, intent, rows, cols, labels, mi, scope_txt, gcol) -> str:
    measure = intent["measures"][0]
    pct = "%" if measure["agg"] in ("RATE",) else ""
    unit = "" if measure["agg"] in ("RATE", "COUNT") else _unit(ds.columns[measure["col"]].unit) if measure.get("col") else ""
    fmt = lambda v: f"{_fmt(v)}{pct or unit}"
    pairs = [(labels[i], r[mi]) for i, r in enumerate(rows) if isinstance(r[mi], (int, float)) and labels[i] != "(missing)"]
    label = sem.measure_label(measure)
    if intent.get("share"):
        si = cols.index("share_pct")
        top = max(range(len(rows)), key=lambda i: rows[i][si] or 0)
        return f"{labels[top]} is the largest group{scope_txt} at {_fmt(rows[top][si])}% of {ds.plural} ({_fmt(rows[top][mi])})."
    if intent["kind"] == "ranking":
        word = "highest" if intent["ranking"]["direction"] == "DESC" else "lowest"
        if len(pairs) == 1:
            return f"{pairs[0][0]} has the {word} {label}{scope_txt}: {fmt(pairs[0][1])}."
        return f"These are the {len(pairs)} {gcol.label} groups with the {word} {label}{scope_txt}; {pairs[0][0]} leads with {fmt(pairs[0][1])}."
    if not pairs:
        return f"{_cap(label)} by {gcol.label}{scope_txt}."
    hi = max(pairs, key=lambda p: p[1])
    lo = min(pairs, key=lambda p: p[1])
    if hi[1] == lo[1]:
        return f"{_cap(label)} is the same across {gcol.label} groups{scope_txt}: {fmt(hi[1])}."
    gap = hi[1] - lo[1]
    if gcol.order or intent["group_by"][0]["bin"]:
        first, last = pairs[0], pairs[-1]
        trend = "rises" if last[1] > first[1] else "falls"
        return (f"{_cap(label)}{scope_txt} {trend} from {fmt(first[1])} ({first[0]}) to {fmt(last[1])} ({last[0]}); "
                f"highest is {hi[0]} ({fmt(hi[1])}), lowest is {lo[0]} ({fmt(lo[1])}).")
    return (f"{_cap(label)} by {gcol.label}{scope_txt}: highest is {hi[0]} ({fmt(hi[1])}), lowest is {lo[0]} ({fmt(lo[1])}) — "
            f"a gap of {_fmt(round(gap, 2))}{pct or unit}.")


def _explain_correlation(ds: Dataset, r: Optional[float], x: str, y: str, n: int, scope_txt: str) -> str:
    if r is None:
        return f"There isn't enough variation across {n} {ds.plural}{scope_txt} to measure a relationship."
    strength = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong"
    if abs(r) < 0.1:
        return f"There is essentially no linear relationship between {x} and {y} across {_fmt(n)} {ds.plural}{scope_txt} (r = {r:.2f})."
    direction = "higher" if r > 0 else "lower"
    return (f"{_cap(ds.plural)} with higher {x} tend to have {direction} {y} — a {strength} {'positive' if r > 0 else 'negative'} "
            f"relationship across {_fmt(n)} {ds.plural}{scope_txt} (r = {r:.2f}). This shows association, not proof of cause.")


def _drivers(ds, sem, intent, result, plan, viz, scope_txt) -> Dict[str, Any]:
    cols, rows = result["columns"], result["rows"]
    alias = plan["spec"]["metric_alias"]
    fi, li, mi, ni = cols.index("factor"), cols.index("level"), cols.index(alias), cols.index("n")
    measure = intent["measures"][0]
    outcome = ds.columns[measure["col"]]
    total = sum(r[ni] for r in rows if r[fi] == rows[0][fi])
    min_support = max(5, int(DRIVER_MIN_SUPPORT * total))
    unit = "%" if measure["agg"] == "RATE" else _unit(outcome.unit)
    factors = []
    for name, label in plan["factors"].items():
        col = ds.columns[name]
        levels = [r for r in rows if r[fi] == name]
        if col.role == "measure":
            levels.sort(key=lambda r: float(r[li]))
        elif col.order:
            levels.sort(key=lambda r: col.order.index(r[li]) if r[li] in col.order else 99)
        shown = []
        for r in levels:
            lvl = band_label(int(float(r[li])) if col.type == "INTEGER" else float(r[li]), col.bin_width or 10, col.type == "INTEGER", col.unit) \
                if col.role == "measure" else str(r[li])
            shown.append({"level": lvl, "value": r[mi], "n": r[ni], "supported": r[ni] >= min_support})
        supported = [lv for lv in shown if lv["supported"] and lv["value"] is not None]
        if len(supported) < 2:
            continue
        best = max(supported, key=lambda lv: lv["value"])
        worst = min(supported, key=lambda lv: lv["value"])
        factors.append({"factor": name, "label": label, "spread": round(best["value"] - worst["value"], 2),
                        "best": best, "worst": worst, "levels": shown})
    factors.sort(key=lambda f: f["spread"], reverse=True)
    top = factors[:10]
    viz.update(chart_type="drivers", horizontal=True, labels=[_cap(f["label"]) for f in top], values=[f["spread"] for f in top],
               value_label=f"gap between best and worst level ({'percentage points' if unit == '%' else outcome.unit or 'points'})",
               drivers={"outcome": sem.measure_label(measure), "unit": unit, "min_support": min_support, "factors": factors})
    if not factors:
        viz["explanation"] = f"No factor showed a clear difference in {sem.measure_label(measure)}{scope_txt}."
        return viz
    lead = factors[0]
    parts = [f"{_cap(lead['label'])} shows the biggest difference: {lead['best']['level']} averages "
             f"{_fmt(lead['best']['value'])}{unit} vs {_fmt(lead['worst']['value'])}{unit} for {lead['worst']['level']}"]
    if len(factors) > 2:
        gap = " pts" if unit == "%" else ""
        parts.append(f"followed by {factors[1]['label']} ({_fmt(factors[1]['spread'])}{gap}) and {factors[2]['label']} ({_fmt(factors[2]['spread'])}{gap})")
    weakest = factors[-1]
    viz["explanation"] = (f"Of {len(factors)} factors, " + "; ".join(parts) + f". {_cap(weakest['label'])} makes the least difference "
                          f"({_fmt(weakest['spread'])}{' pts' if unit == '%' else ''}). These are associations in this data, not proof of cause.")
    if measure["agg"] == "RATE":
        viz["explanation"] = viz["explanation"].replace(" averages ", " has a rate of ")
    return viz
