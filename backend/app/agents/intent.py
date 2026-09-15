"""Agent 1 — Intent Understanding (dataset-agnostic).

Reads the question against the active dataset's semantic layer — column names, labels,
synonyms and category values from its manifest — and produces a structured IntentObject:

    kind        aggregate | ranking | list | distribution | correlation | drivers | unknown
    measures    [{col, agg}]  agg ∈ AVG SUM MIN MAX COUNT RATE (RATE = share of a yes/no flag)
    group_by    [{col, bin}]  bin = band width for numeric columns
    filters     [{col, op, value, label}]
    ranking     {direction, limit, target: groups | rows}

Nothing here knows about students or patients: every word it understands comes from the
manifest. Words it can't place are reported (unknown_terms) instead of being guessed at.
The structured parser remains the authoritative extractor; a trained local intent classifier
provides an independent learned intent-family signal.
"""
import difflib
import re
from typing import Any, Dict, List, Optional, Tuple

from app.datasets import ColumnMeta, Dataset
from app import local_ml

OPS: List[Tuple[str, str]] = [
    ("greater than or equal to", ">="), ("less than or equal to", "<="), ("no less than", ">="), ("no more than", "<="),
    ("at least", ">="), ("at most", "<="), ("up to", "<="), ("more than", ">"), ("greater than", ">"), ("higher than", ">"),
    ("older than", ">"), ("younger than", "<"), ("less than", "<"), ("fewer than", "<"), ("lower than", "<"),
    ("exceeding", ">"), ("above", ">"), ("over", ">"), ("below", "<"), ("under", "<"), ("equal to", "="), ("exactly", "="),
    (">=", ">="), ("<=", "<="), (">", ">"), ("<", "<"), ("=", "="),
]
OP_WORDS = {">": "above", ">=": "at least", "<": "below", "<=": "at most", "=": "equal to"}
UNIT_RE = r"(?:\s*(?:%|percent|hours?|hrs?|years?|yrs?|mg/dl|mm hg|bpm|points?|marks|sessions?|mm|vessels?))?"
NUM = r"(\d+(?:\.\d+)?)"

GROUP_BEFORE = re.compile(r"(?:\bby|\bper|\bfor each|\bfor every|\beach|\bacross(?: all| different)?|\bamong|\bgrouped by|\bsplit by|"
                          r"\bbroken down by|\bbased on|\bin each|\bfor different|\bof each|\bvary by|\bvaries by|\bdiffer by)"
                          r"\s+(?:the\s+|their\s+|different\s+|a\s+)?$")
GROUP_AFTER = re.compile(r"^\s*-?\s*(?:wise|groups?|bands?|brackets?|categories)\b")
RATE_BEFORE = re.compile(r"(?:rate of|percentage of|percent of|proportion of|share of|% of|prevalence of|how common is|"
                         r"likelihood of|chance of|risk of|incidence of)\s+$")
RATE_AFTER = re.compile(r"^\s*(?:rates?|percentage|prevalence|incidence|proportion)\b")
NEG_BEFORE = re.compile(r"(?:without|\bno|\bnot|don't have|dont have|do not have|doesn't have|lacking|who lack|don't|do not)\s+(?:any\s+)?$")
POS_BEFORE = re.compile(r"(?:\bwith|\bhave|\bhas|\bhaving|\bwho|\bthat|\btake|\btaking|\btakes|\bdo|\bdoing|\bin|\bget|\bgetting|"
                        r"\bshowing|\bwhose)\s+(?:any\s+|some\s+|an?\s+)?$")
WITH_AND_WITHOUT = re.compile(r"(?:with and without|with or without|with vs\.? without|and without)\s+$")

DRIVER_RE = re.compile(r"\b(?:what|which) (?:factors?|things|variables|features|attributes)\b|\bwhat (?:affects|influences|drives|"
                       r"impacts|predicts|determines|explains|matters)\b|\bdrivers? of\b|\bfactors? (?:that )?(?:affect|influence|"
                       r"drive|impact)|\bbiggest (?:impact|effect)|\bmost important factors?\b|\bkey factors?\b|\bwhat is linked to\b")
RELATION_RE = re.compile(r"\b(?:vs\.?|versus|against|affects?|affecting|impacts?|relat\w*|correlat\w*|depends?|influenc\w*|"
                         r"linked|link|associated|association|connection|connected)\b")
DISTRIBUTION_RE = re.compile(r"\b(?:distribution|distributed|spread|histogram|bands|range of)\b")
SHARE_RE = re.compile(r"\b(?:share|proportion|breakdown|split|composition|makeup|make up)\b|\b(?:percentage|percent|%) of (?:all\s+)?\w+\b")
COUNT_RE = re.compile(r"\b(?:how many|number of|count of|count|total number of|no\. of)\b")
LIST_RE = re.compile(r"\b(?:show|list|find|display|give me|which|who|get)\b")
HIGH_RE = re.compile(r"\b(?:highest|maximum|max|most|best|top|greatest|largest|biggest|strongest)\b")
LOW_RE = re.compile(r"\b(?:lowest|minimum|min|least|worst|bottom|smallest|weakest|poorest|fewest)\b")
AVG_RE = re.compile(r"\b(?:average|avg|mean|typical)\b")
SUM_RE = re.compile(r"\b(?:total|sum)\b")
COMPARE_RE = re.compile(r"\b(?:compare|comparing|comparison|vs\.?|versus|between|difference)\b")
TOP_N_RE = re.compile(r"\b(?:top|bottom|best|worst|first|last|highest|lowest)\s+(\d+)\b")

STOPWORDS = set("""
a an the of for to in on at by per with without and or vs versus is are was were be been being do does did doing done have has had
having what which who whom whose how many much show me list find display give get tell please all any some each every their
there these those this that than then also only just more most less least very really can could would should will i we you it
its they them our your my average avg mean typical total sum count number no not top bottom best worst highest lowest maximum
minimum max min greatest largest smallest biggest group groups grouped band bands split broken down based across among between
compare comparing comparison difference distribution distributed spread histogram share proportion breakdown percentage percent
rate rates overall affect affects affecting impact impacts influence influences relate related relationship correlation correlate
correlated depend depends link linked associated association drivers driver factors factor things variables features attributes matter
matters important key predict predicts determines explain explains where when why wise level levels type types kind value values
over under above below exactly equal older younger than up from into about like data dataset records record rows row
people person vary varies differ likely likelihood chance risk prevalence incidence common hours years percent points
""".split())


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("%", " % ")
    return re.sub(r"\s+", " ", text).strip()


class Mention:
    __slots__ = ("start", "end", "text", "targets", "consumed")

    def __init__(self, start: int, end: int, text: str, targets: List[Tuple[str, str, Any]]):
        self.start, self.end, self.text, self.targets, self.consumed = start, end, text, targets, False

    @property
    def cols(self) -> List[str]:
        return list(dict.fromkeys(t[1] for t in self.targets if t[0] == "col"))

    @property
    def is_value(self) -> bool:
        return any(t[0] == "val" for t in self.targets)

    def value_cols(self) -> List[str]:
        return list(dict.fromkeys(t[1] for t in self.targets if t[0] == "val"))

    def value_for(self, col: str) -> Any:
        return next(t[2] for t in self.targets if t[0] == "val" and t[1] == col)


class IntentParser:
    def __init__(self, ds: Dataset):
        self.ds = ds
        index: Dict[str, List[Tuple[str, str, Any]]] = {}
        for c in ds.columns.values():
            for phrase in {c.name.replace("_", " "), c.label.lower(), *(s.lower() for s in c.synonyms)}:
                index.setdefault(phrase, []).append(("col", c.name, None))
            if c.role == "category":
                for v in c.values:
                    index.setdefault(str(v).lower(), []).append(("val", c.name, v))
                for v, syns in c.value_synonyms.items():
                    for s in syns:
                        entry = ("val", c.name, v)
                        if entry not in index.setdefault(s.lower(), []):
                            index[s.lower()].append(entry)
        self.index = index
        self.phrases = sorted(index, key=len, reverse=True)
        self.patterns = {p: re.compile(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])") for p in self.phrases}
        ent = ds.entity
        self.entity_re = re.compile(r"\b(?:%s|%s|people|records|cases)\b" % (re.escape(ent["plural"]), re.escape(ent["singular"])))
        self.vocabulary = set()
        for p in self.phrases:
            self.vocabulary |= set(re.findall(r"[a-z]+", p))

    # ------------------------------------------------------------------ helpers

    def _mentions(self, q: str) -> List[Mention]:
        taken: List[Tuple[int, int]] = []
        found: List[Mention] = []
        for phrase in self.phrases:
            for m in self.patterns[phrase].finditer(q):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append(Mention(m.start(), m.end(), phrase, self.index[phrase]))
        return sorted(found, key=lambda x: x.start)

    def _col(self, name: str) -> ColumnMeta:
        return self.ds.columns[name]

    @staticmethod
    def _words_between(q: str, a: int, b: int) -> int:
        return len(re.findall(r"[a-z0-9]+", q[a:b])) if b > a else 0

    def _label(self, col: ColumnMeta, op: str, value: Any) -> str:
        if op == "in":
            return f"{col.label}: {' / '.join(str(v) for v in value)}"
        if col.role == "flag":
            return f"{'with' if value == col.positive else 'without'} {col.label}"
        if col.role == "category":
            return f"{col.label} {value}" if op == "=" else f"{col.label} {op} {value}"
        shown = f"{value:g}" if isinstance(value, float) else str(value)
        return f"{col.label} {OP_WORDS.get(op, op)} {shown}{col.unit_suffix() if col.unit == '%' else ''}"

    # ------------------------------------------------------------------ parse

    def parse(self, question: str, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = question.strip()
        q = _normalize(raw)
        mentions = self._mentions(q)
        chars = list(q)
        for m in mentions:
            chars[m.start:m.end] = " " * (m.end - m.start)
        work = "".join(chars)

        intent: Dict[str, Any] = {
            "raw": raw, "dataset": self.ds.id, "kind": None, "measures": [], "group_by": [], "filters": [], "ranking": None,
            "share": False, "list_columns": [], "notes": [], "follow_up": False, "unknown_terms": [], "ambiguous_values": [],
            "needs_measure": False, "entity": bool(self.entity_re.search(work)),
        }
        filters, group_by, measures, notes = intent["filters"], intent["group_by"], intent["measures"], intent["notes"]

        def add_filter(col: ColumnMeta, op: str, value: Any) -> None:
            for f in filters:
                if f["col"] == col.name and op == "=" and f["op"] in ("=", "in") and col.role == "category":
                    values = f["value"] if f["op"] == "in" else [f["value"]]
                    if value not in values:
                        values.append(value)
                    f.update(op="in", value=values, label=self._label(col, "in", values))
                    return
            filters.append({"col": col.name, "op": op, "value": value, "label": self._label(col, op, value)})

        def add_group(col: ColumnMeta) -> None:
            if any(g["col"] == col.name for g in group_by):
                return
            entry = {"col": col.name, "bin": None}
            if col.role == "measure":
                entry["bin"] = col.bin_width or _nice_width(col)
                notes.append(f"Grouped {col.label} into bands of {entry['bin']:g}{col.unit_suffix()}.")
            group_by.append(entry)

        col_mentions = [m for m in mentions if m.cols]
        numeric_mentions = [m for m in col_mentions if any(self._col(c).role == "measure" for c in m.cols)]

        # 1. numeric comparisons: "attendance below 65", "studied more than 25 hours", "older than 55"
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"between\s+" + NUM + r"\s+and\s+" + NUM + UNIT_RE, q):
            target = self._nearest(numeric_mentions, m.start(), m.end(), q)
            if target:
                col = self._col(next(c for c in target.cols if self._col(c).role == "measure"))
                add_filter(col, ">=", _num(m.group(1)))
                add_filter(col, "<=", _num(m.group(2)))
                target.consumed = True
                spans.append((m.start(), m.end()))
        for word, op in OPS:
            for m in re.finditer(r"(?<![a-z<>=])" + re.escape(word) + r"\s*" + NUM + UNIT_RE, q):
                if any(s <= m.start() < e or s < m.end() <= e for s, e in spans):
                    continue
                if word in ("older than", "younger than"):
                    target = next((x for x in numeric_mentions if x.start <= m.start() < x.end), None)
                else:
                    target = self._nearest(numeric_mentions, m.start(), m.end(), q)
                if not target:
                    continue
                col = self._col(next(c for c in target.cols if self._col(c).role == "measure"))
                add_filter(col, op, _num(m.group(1)))
                target.consumed = True
                spans.append((m.start(), m.end()))
                for x in numeric_mentions:  # the unit word ("25 hours") names the same column
                    if col.name in x.cols and m.start() <= x.start and x.end <= m.end():
                        x.consumed = True

        # 2. category values: "public schools", "female", "high motivation"
        by_col: Dict[str, List[Mention]] = {}
        for vm in [m for m in mentions if m.is_value]:
            cands = vm.value_cols()
            bound = cands[0] if len(cands) == 1 else None
            if bound is None:
                for cm in col_mentions:
                    if cm is vm or not set(cm.cols) & set(cands):
                        continue
                    gap = self._words_between(q, vm.end, cm.start) if cm.start >= vm.end else self._words_between(q, cm.end, vm.start)
                    if gap <= 3:
                        bound = next(c for c in cm.cols if c in cands)
                        cm.consumed = True
                        break
            vm.consumed = True
            if bound is None:
                intent["ambiguous_values"].append({"phrase": vm.text, "start": vm.start, "end": vm.end, "candidates": cands})
                continue
            by_col.setdefault(bound, []).append(vm)
            for cm in col_mentions:
                if cm is not vm and bound in cm.cols and self._words_between(q, min(cm.end, vm.end), max(cm.start, vm.start)) <= 3:
                    cm.consumed = True
        for col_name, vms in by_col.items():
            col = self._col(col_name)
            values = list(dict.fromkeys(vm.value_for(col_name) for vm in vms))
            if len(values) > 1 and (COMPARE_RE.search(q) or " and " in q):
                add_group(col)
            add_filter(col, "in" if len(values) > 1 else "=", values if len(values) > 1 else values[0])

        # 3. yes/no flags: "with extracurricular activities", "heart disease rate", "with and without heart disease"
        pending_flags: List[Tuple[Mention, ColumnMeta]] = []
        for cm in col_mentions:
            if cm.consumed:
                continue
            flag = next((self._col(c) for c in cm.cols if self._col(c).role == "flag"), None)
            if not flag:
                continue
            before, after = q[max(0, cm.start - 40):cm.start], q[cm.end:cm.end + 20]
            if WITH_AND_WITHOUT.search(before):
                add_group(flag)
            elif RATE_BEFORE.search(before) or RATE_AFTER.search(after) or "rate" in cm.text:
                measures.append({"col": flag.name, "agg": "RATE"})
            elif GROUP_BEFORE.search(before):
                add_group(flag)
            elif NEG_BEFORE.search(before):
                add_filter(flag, "=", flag.negative)
            elif POS_BEFORE.search(before):
                add_filter(flag, "=", flag.positive)
            else:
                pending_flags.append((cm, flag))
                continue
            cm.consumed = True

        # 4. explicit grouping: "by school type", "per age group", "which chest pain type ..."
        for cm in col_mentions:
            if cm.consumed:
                continue
            before, after = q[max(0, cm.start - 40):cm.start], q[cm.end:cm.end + 15]
            is_which = re.search(r"\b(?:which|what)\s+$", before)
            col = self._col(cm.cols[0])
            if col.role == "measure" and GROUP_BEFORE.search(before):
                # "top 10 students by exam score": a lone numeric column after "by" is what to rank by, not a grouping
                others = [x for x in col_mentions if x is not cm and not x.consumed
                          and any(self._col(c).role in ("measure", "flag") for c in x.cols)]
                if not others and not (COUNT_RE.search(work) or SHARE_RE.search(q) or DISTRIBUTION_RE.search(work)
                                       or any(mm.get("col") for mm in measures)):
                    continue
            if (GROUP_BEFORE.search(before) or GROUP_AFTER.search(after)) or (is_which and col.role != "measure"):
                add_group(col)
                cm.consumed = True

        remaining = [m for m in col_mentions if not m.consumed and not any(m is p for p, _ in pending_flags)]
        rem_cols = [self._col(m.cols[0]) for m in remaining]
        pending_cols = [f for _, f in pending_flags]

        high, low = HIGH_RE.search(work), LOW_RE.search(work)
        direction = None
        if high and (not low or high.start() < low.start()):
            direction = "DESC"
        elif low:
            direction = "ASC"
        agg_word = "AVG" if AVG_RE.search(work) else "SUM" if SUM_RE.search(work) else None

        # 5. analysis type
        relation_cols = self._ordered_pair(remaining, pending_flags, group_by, measures)
        if DRIVER_RE.search(q):
            outcome = next((c for c in rem_cols + pending_cols if c.role in ("measure", "flag")), None)
            outcome = outcome or next((self._col(m["col"]) for m in measures if m.get("col")), None) or self.ds.primary_measure
            measures[:] = [{"col": outcome.name, "agg": "RATE" if outcome.role == "flag" else "AVG"}]
            group_by.clear()
            intent["kind"] = "drivers"
            pending_flags = []
            for m in remaining:
                m.consumed = True
        elif RELATION_RE.search(work) and len(relation_cols) >= 2:
            self._relation(intent, relation_cols, add_group)
            pending_flags = []
            for m in remaining:
                m.consumed = True
        elif DISTRIBUTION_RE.search(work) and (rem_cols or group_by):
            target = rem_cols[0] if rem_cols else self._col(group_by[0]["col"])
            if target.role == "measure":
                measures[:] = [{"col": target.name, "agg": None}]
                group_by.clear()
                intent["kind"] = "distribution"
            else:
                add_group(target)
                measures[:] = [{"col": None, "agg": "COUNT"}]
                intent["share"] = True
            for m in remaining:
                m.consumed = True

        if intent["kind"] is None:
            for m, col in zip(remaining, rem_cols):
                if not m.consumed and col.role == "measure":
                    measures.append({"col": col.name, "agg": self._agg_near(q, m.start) or agg_word})
                    m.consumed = True
            if COUNT_RE.search(work) and not any(x["agg"] in ("RATE", "AVG") for x in measures):
                column_measures = [x for x in measures if x.get("col")]
                if column_measures and not intent["entity"]:
                    for x in column_measures:
                        x["agg"] = x["agg"] or "SUM"
                else:
                    measures[:] = [{"col": None, "agg": "COUNT"}]
            for cm, flag in pending_flags:
                if not measures and (flag.outcome or agg_word or direction):
                    measures.append({"col": flag.name, "agg": "RATE"})
                else:
                    add_filter(flag, "=", flag.positive)
                cm.consumed = True
            if SHARE_RE.search(q) and group_by and not any(x["agg"] not in (None, "COUNT") for x in measures):
                measures[:] = [{"col": None, "agg": "COUNT"}]
                intent["share"] = True
            for m in [x for x in col_mentions if not x.consumed]:
                col = self._col(m.cols[0])
                if measures and col.role != "measure":
                    add_group(col)
                else:
                    intent["list_columns"].append(col.name)
                m.consumed = True

            self._ranking(intent, work, direction)

            if not intent["kind"]:
                if measures:
                    for x in measures:
                        x["agg"] = x["agg"] or "AVG"
                    intent["kind"] = "aggregate"
                elif filters or intent["list_columns"] or (intent["entity"] and LIST_RE.search(work)):
                    intent["kind"] = "list"
                elif agg_word or direction or group_by:
                    intent["kind"] = "aggregate"
                    intent["needs_measure"] = True
                else:
                    intent["kind"] = "unknown"

        # 6. words nothing matched
        leftovers = [w for w in re.findall(r"[a-z][a-z'-]{2,}", work) if w not in STOPWORDS and w not in self.vocabulary
                     and not self.entity_re.fullmatch(w)]
        intent["unknown_terms"] = list(dict.fromkeys(leftovers))

        follow_up_phrase = re.match(r"^(?:only|just|and|now|what about|how about|same|but|excluding|except|then)\b", q)
        if previous and previous.get("dataset") == self.ds.id and intent["kind"] in ("list", "unknown", "aggregate") \
                and not intent["measures"] and not intent["list_columns"] and (filters or group_by) \
                and (not intent["entity"] or follow_up_phrase):
            intent = merge_follow_up(previous, intent)

        if intent["unknown_terms"] and intent["kind"] != "unknown" and not intent["needs_measure"]:
            notes.append(f"Ignored words that don't match anything in this dataset: {', '.join(intent['unknown_terms'])}.")
        ml_kind = local_ml.predict_intent_kind(raw)
        intent["ml_intent"] = ml_kind
        if ml_kind.get("enabled") and ml_kind.get("confidence", 0) >= 0.80 and ml_kind.get("prediction") != intent["kind"]:
            intent["notes"].append(f"Local ML intent model predicted {ml_kind['prediction']} with {ml_kind['confidence']:.0%}; deterministic parser retained as the authoritative structured intent.")
        return intent

    # ------------------------------------------------------------------ sub-steps

    def _nearest(self, candidates: List[Mention], start: int, end: int, q: str) -> Optional[Mention]:
        before = [m for m in candidates if m.end <= start and self._words_between(q, m.end, start) <= 5]
        after = [m for m in candidates if m.start >= end and self._words_between(q, end, m.start) <= 3]
        if before:
            return max(before, key=lambda m: m.end)
        if after:
            return min(after, key=lambda m: m.start)
        return None

    @staticmethod
    def _agg_near(q: str, pos: int) -> Optional[str]:
        window = q[max(0, pos - 30):pos]
        if AVG_RE.search(window):
            return "AVG"
        if re.search(r"\b(?:total|sum)\b\s*(?:of\s+)?(?:the\s+)?$", window):
            return "SUM"
        return None

    def _ordered_pair(self, remaining, pending_flags, group_by, measures) -> List[ColumnMeta]:
        cols = [(m.start, self._col(m.cols[0])) for m in remaining] + [(cm.start, f) for cm, f in pending_flags]
        ordered = [c for _, c in sorted(cols, key=lambda x: x[0])]
        ordered += [self._col(g["col"]) for g in group_by] + [self._col(m["col"]) for m in measures if m.get("col")]
        unique: List[ColumnMeta] = []
        for c in ordered:
            if c.name not in [u.name for u in unique]:
                unique.append(c)
        return unique

    def _relation(self, intent: Dict[str, Any], cols: List[ColumnMeta], add_group) -> None:
        primary = self.ds.primary_measure.name
        x, y = cols[0], cols[1]
        if x.name == primary or (x.outcome and not y.outcome):
            x, y = y, x
        measures, group_by = intent["measures"], intent["group_by"]
        measures.clear()
        group_by.clear()
        if x.role == "measure" and y.role == "measure":
            intent["kind"] = "correlation"
            measures += [{"col": x.name, "agg": None}, {"col": y.name, "agg": None}]
            return
        if y.role not in ("measure", "flag") and x.role in ("measure", "flag"):
            x, y = y, x
        add_group(x)
        if y.role in ("measure", "flag"):
            measures.append({"col": y.name, "agg": "RATE" if y.role == "flag" else "AVG"})
            intent["notes"].append(f"Compared {y.label} across {x.label} to see whether they are related.")
        else:
            add_group(y)
            measures.append({"col": None, "agg": "COUNT"})
        intent["kind"] = "aggregate"

    def _ranking(self, intent: Dict[str, Any], work: str, direction: Optional[str]) -> None:
        if not direction:
            return
        measures, group_by = intent["measures"], intent["group_by"]
        m = TOP_N_RE.search(work) or re.search(r"\b(\d+)\s+(?:%s)\b" % re.escape(self.ds.plural), work)
        limit = int(m.group(1)) if m else None
        column_measures = [x for x in measures if x.get("col")]
        if group_by:
            if not measures:
                intent["needs_measure"] = True
            singular = bool(re.search(r"\b(?:which|what)\b", work)) and limit is None
            intent["ranking"] = {"direction": direction, "limit": limit or (1 if singular else 10), "target": "groups"}
            intent["kind"] = "ranking"
            for x in measures:
                x["agg"] = x["agg"] or "AVG"
        elif intent["entity"] or limit:
            intent["ranking"] = {"direction": direction, "limit": limit or 10, "target": "rows"}
            intent["kind"] = "ranking"
            numeric = [x for x in column_measures if self._col(x["col"]).role == "measure"]
            if numeric:
                measures[:] = [{"col": numeric[0]["col"], "agg": None}]
            else:
                measures.clear()
                intent["needs_measure"] = True
        elif column_measures and all(x["agg"] is None for x in column_measures):
            for x in column_measures:
                x["agg"] = "MAX" if direction == "DESC" else "MIN"
        elif not measures:
            intent["needs_measure"] = True

    def suggest_columns(self, term: str, roles=("measure", "flag"), n: int = 3, cutoff: float = 0.6) -> List[ColumnMeta]:
        pool: Dict[str, ColumnMeta] = {}
        for c in self.ds.columns.values():
            if c.role in roles:
                for p in [c.label, *c.synonyms]:
                    pool[p.lower()] = c
        out: List[ColumnMeta] = []
        for hit in difflib.get_close_matches(term, list(pool), n=n * 3, cutoff=cutoff):
            if pool[hit].name not in [c.name for c in out]:
                out.append(pool[hit])
        return out[:n]


def _num(text: str) -> Any:
    value = float(text)
    return int(value) if value.is_integer() else value


def _nice_width(col: ColumnMeta) -> float:
    span = (col.maximum or 0) - (col.minimum or 0)
    for w in (1, 2, 5, 10, 20, 25, 50, 100, 250, 500, 1000):
        if span / w <= 8:
            return w
    return round(span / 8)


def merge_follow_up(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = {k: (v.copy() if isinstance(v, (list, dict)) else v) for k, v in previous.items()}
    merged["measures"] = [dict(m) for m in previous["measures"]]
    merged["raw"] = current["raw"]
    new_cols = {f["col"] for f in current["filters"]}
    merged["filters"] = [f for f in previous["filters"] if f["col"] not in new_cols] + current["filters"]
    if current["group_by"]:
        merged["group_by"] = current["group_by"]
    merged["follow_up"] = True
    merged["notes"] = [f"Follow-up: reused the measure and grouping from “{previous['raw']}”."] + \
                      [n for n in current["notes"] if n.startswith("Grouped")]
    merged["ambiguous_values"] = current["ambiguous_values"]
    merged["unknown_terms"] = []
    merged["needs_measure"] = False
    return merged


_parsers: Dict[str, IntentParser] = {}


def parser_for(ds: Dataset) -> IntentParser:
    parser = _parsers.get(ds.id)
    if parser is None or parser.ds is not ds:
        parser = _parsers[ds.id] = IntentParser(ds)
    return parser


def extract_intent(ds: Dataset, question: str, previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return parser_for(ds).parse(question, previous)
