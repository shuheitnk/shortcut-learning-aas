"""Check the numbers the documents quote against the report roots' files.

Every number docs/PAPER_SOURCES.md and docs/SETTINGS.md show from a result is
registered here with its file, rows and column. The check rounds each value from
full precision, as the paper does, and confirms the document's section says
exactly that; an unregistered row in PAPER_SOURCES.md's numbers section fails too.

``--structure`` checks only where each quote is and its shape (no outputs needed;
the test suite runs it). ``--render`` prints the tables with their values, and
``--tex`` lists the paper's numbers that no registered value rounds to.

    uv run python scripts/check_document_numbers.py --root outputs
    uv run python scripts/check_document_numbers.py --structure
    uv run python scripts/check_document_numbers.py --root outputs --render
    uv run python scripts/check_document_numbers.py --root outputs --tex <paper>
"""

from __future__ import annotations

import argparse
import math
import re
import string
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]

# Where `just reproduce` writes the report roots and their store.
DEFAULT_ROOT = Path("outputs")

# A value in a quote when no report root is read: a grouped number, a number or a
# name.
NUMBER = r"(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|[\w.-]+)"

# A value written as the file's own text, which may be any table cell.
TEXT = r"[^|\n]+?"

# Decimals, a format specification such as ".3g" (p-values span orders of
# magnitude), or None for the file's own text.
Digits = int | str | None


HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


# A count, a statistic of ``column``, or the ``label`` where ``column`` is largest.
REDUCTIONS = ("count", "median", "q1", "q3", "max", "min", "mean", "argmax")

# Which selected rows a reduction keeps, by the sign of ``column``.
KEEPS = ("all", "positive", "zero")


@dataclass(frozen=True)
class Source:
    """One value of one report file: the row ``where`` selects, in ``column``.

    With ``reduce``, ``where`` may select several rows and the value is their
    reduction, after ``keep`` narrows them by the sign of ``column``.
    """

    path: str
    where: Mapping[str, str]
    column: str
    reduce: str | None = None
    keep: str = "all"
    # The column ``argmax`` returns the value of.
    label: str = ""
    # A value subtracted from this one, for a difference the text quotes.
    minus: Source | None = None


@dataclass(frozen=True)
class Quote:
    """Values a document writes, and the text it writes them in.

    ``text`` holds one ``{}`` per source (``str.format`` rules) and must appear
    verbatim, values filled in, in the section under ``section``.
    """

    document: str
    section: str
    text: str
    sources: tuple[Source, ...]
    digits: Digits = 3
    # Why the quote is there, for a reader of a failure.
    what: str = ""
    # Extra decimals for some sources, by position.
    digits_at: Mapping[int, Digits] = field(default_factory=dict)
    # Whether the values are written with their thousands grouped ("14,400").
    grouped: bool = False

    def digits_of(self, index: int) -> Digits:
        """How the value at ``index`` is written."""
        return self.digits_at.get(index, self.digits)


def headings(lines: Sequence[str]) -> list[re.Match[str] | None]:
    """Each line's heading match, or None; a ``#`` inside fenced code is not one."""
    found: list[re.Match[str] | None] = []
    fenced = False
    for line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            found.append(None)
            continue
        found.append(None if fenced else HEADING.match(line))
    return found


def section_text(text: str, heading: str) -> str:
    """The body under the first heading whose title starts with ``heading``."""
    lines = text.splitlines()
    matches = headings(lines)
    for index, match in enumerate(matches):
        if match is None or not match.group(2).startswith(heading):
            continue
        level = len(match.group(1))
        end = len(lines)
        for following in range(index + 1, len(lines)):
            inner = matches[following]
            if inner is not None and len(inner.group(1)) <= level:
                end = following
                break
        return "\n".join(lines[index + 1 : end])
    raise KeyError(heading)


def write_value(value: object, digits: Digits, grouped: bool = False) -> str:
    """Write one value as a document does: fixed decimals, no negative zero."""
    if digits is None:
        return str(value)
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return "nan"
    if isinstance(digits, str):
        written = format(number, digits)
    else:
        written = f"{number:,.{digits}f}" if grouped else f"{number:.{digits}f}"
    if float(written.replace(",", "")) == 0.0:
        written = written.lstrip("-")
    return written


def read_table(path: Path) -> pd.DataFrame:
    """Read a report file, a CSV as text so a value keeps the file's own spelling."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# Further directories a report file is looked for in when ``--root`` lacks it,
# such as an appendix's own root (``--also``).
ALSO_ROOTS: list[Path] = []


def locate(root: Path, path: str) -> Path | None:
    """The file at ``path`` under ``root`` or, failing that, under ``--also``."""
    for base in (root, *ALSO_ROOTS):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return None


def read_value(root: Path, source: Source) -> object:
    """The one value ``source`` names."""
    found = locate(root, source.path)
    if found is None:
        raise LookupError(f"{root} does not hold {source.path}")
    table = read_table(found)
    selected = pd.Series(True, index=table.index)
    for column, value in source.where.items():
        selected &= table[column].astype(str) == value
    rows = table.loc[selected]
    if source.reduce is None:
        if len(rows) != 1:
            raise LookupError(
                f"{source.path}: {dict(source.where)} selects {len(rows)} rows"
            )
        value = rows.iloc[0][source.column]
    else:
        value = reduce_rows(rows, source)
    if source.minus is None:
        return value
    return float(value) - float(read_value(root, source.minus))  # type: ignore[arg-type]


def reduce_rows(rows: pd.DataFrame, source: Source) -> object:
    """Reduce the rows a source selects, as its ``keep`` and ``reduce`` say."""
    if source.reduce not in REDUCTIONS or source.keep not in KEEPS:
        raise LookupError(
            f"{source.path}: no reduction {source.reduce!r}/{source.keep!r}"
        )
    values = pd.Series(rows[source.column], dtype=float)
    if source.keep == "positive":
        values = values.loc[values > 0]
    elif source.keep == "zero":
        values = values.loc[values == 0]
    if source.reduce == "count":
        return len(values)
    if values.empty:
        raise LookupError(f"{source.path}: {dict(source.where)} selects nothing")
    if source.reduce == "argmax":
        return rows.loc[values.idxmax(), source.label]
    quantile = {"q1": 0.25, "median": 0.5, "q3": 0.75}.get(source.reduce)
    if quantile is not None:
        return float(values.quantile(quantile))
    return float(getattr(values, source.reduce)())


def quote_parts(text: str) -> list[tuple[str, bool]]:
    """The text's literal runs, each with whether a value follows it."""
    return [
        (literal, field is not None)
        for literal, field, _, _ in string.Formatter().parse(text)
    ]


def quote_pattern(quote: Quote) -> re.Pattern[str]:
    """The quote's text as a pattern, with a placeholder for each value."""
    pieces = []
    index = 0
    for literal, value in quote_parts(quote.text):
        pieces.append(re.escape(literal))
        if value:
            pieces.append(TEXT if quote.digits_of(index) is None else NUMBER)
            index += 1
    return re.compile("".join(pieces))


def check_structure(quote: Quote, repository: Path = REPOSITORY) -> str | None:
    """What is wrong with the quote's place or shape, or None."""
    document = repository / quote.document
    if not document.is_file():
        return f"{quote.document} does not exist"
    try:
        body = section_text(document.read_text(encoding="utf-8"), quote.section)
    except KeyError:
        return f"{quote.document}: no heading starts with {quote.section!r}"
    places = sum(value for _, value in quote_parts(quote.text))
    if places != len(quote.sources):
        return f"{quote.text!r} has {places} places for {len(quote.sources)} values"
    if quote_pattern(quote).search(body) is None:
        return (
            f"{quote.document} § {quote.section!r}: nothing reads like {quote.text!r}"
        )
    return None


def check_values(quote: Quote, root: Path, repository: Path = REPOSITORY) -> str | None:
    """What differs between the document and the report files, or None."""
    problem = check_structure(quote, repository)
    if problem is not None:
        return problem
    try:
        values = [
            write_value(read_value(root, source), quote.digits_of(index), quote.grouped)
            for index, source in enumerate(quote.sources)
        ]
    except LookupError as missing:
        return str(missing)
    expected = quote.text.format(*values)
    body = section_text(
        (repository / quote.document).read_text(encoding="utf-8"), quote.section
    )
    if expected in body:
        return None
    return f"{quote.document} § {quote.section!r}: the files say {expected!r}"


# ---------------------------------------------------------------------------
# The quotes, one builder per document
# ---------------------------------------------------------------------------


SHIFT = "preanalysis/metrics/shift_summary_bbob_10d.parquet"

# The preanalysis reports the mean over problems, standardized by the population
# deviation.
REPORTED_SHIFT = {"statistic": "mean", "standardization": "population"}

PAIRS = {
    "G1": ("BIPOP-CMA-ES / Adaptive Nelder-Mead", "bipop-cmaes__adaptive-nelder-mead"),
    "G2": ("SHADE / L-SHADE", "shade__l-shade"),
    "G3": ("BOBYQA / Dual Annealing", "bobyqa__dual-annealing"),
}


# How far each problem's map departs from the identity, on the held-out block and
# for the augmentation on the training block.
TEST_STRENGTH = "preanalysis/warp_strength.csv"
TRAINING_STRENGTH = "experiment4/warp_strength.csv"


def strength(path: str, statistic: str) -> Source:
    """A statistic of a block's map strengths."""
    return Source(path, {}, "strength", reduce=statistic)


PREDICTOR_LEVELS = "experiment1/predictor_levels.csv"
PREDICTOR_TESTS = "experiment1/predictor_tests.csv"
CALIBRATION = "experiment2/superiority_calibration.csv"
PAIR_TUNING = "experiment2/pair_tuning.csv"

# The member each pair's calibration tunes.
TUNED_MEMBERS = {"G1": "Adaptive Nelder-Mead", "G2": "SHADE", "G3": "Dual Annealing"}


def tuning(group: str, side: str, column: str) -> Source:
    """One pair's default or selected configuration, as its tuning scored it."""
    return Source(PAIR_TUNING, {"group": group, "configuration": side}, column)


# The premise checks: Experiment 1's on the control block, Experiment 4's on the
# training block and every family's held-out block.
HELD_OUT_PREMISES = "experiment1/premises.csv"
EXPERIMENT4_PREMISES = "experiment4/premises.csv"
CONTROL_BLOCK = "spline/warpheld"
TRAINING_BLOCK = "training block"


def anchor(path: str, block: str) -> Source:
    """How many untransformed feature values of a block agreed with training's."""
    return Source(path, {"check": "anchor", "block": block}, "n_compared")


CALIBRATION_DECILES = "experiment2/superiority_calibration_deciles.csv"

# The representations' order down a table of Experiments 2 and 4.
VARIANTS = ("value", "rank", "concat")

# The three pairwise comparisons of the representations, as the tests name them.
COMPARISONS = (("concat", "rank"), ("concat", "value"), ("rank", "value"))


SPLINE_SCOPE = "spline only"


# ---------------------------------------------------------------------------
# The two documents under docs/
# ---------------------------------------------------------------------------
#
# Each table is declared once below; one row is one quote.

SETTINGS_DOCUMENT = "docs/SETTINGS.md"
SOURCES_DOCUMENT = "docs/PAPER_SOURCES.md"

# How each kind of number is written in the two documents.
LEVEL = 3
P_VALUE = ".3g"
COUNT = 0
WORD = None

UNSEEN_SCOPE = "unseen-family mean"
GROUPS = ("G1", "G2", "G3")
FEATURE_SETS = (("ela", "", "ELA"), ("deep-ela", "_deep-ela", "Deep-ELA"))
STATISTICS = (
    ("smallest", "min"),
    ("first quartile", "q1"),
    ("median", "median"),
    ("third quartile", "q3"),
    ("largest", "max"),
)
# The portfolio's members in its order, and the blocks Experiment 4 checks.
ALGORITHM_ORDER = (
    "BIPOP-CMA-ES",
    "Adaptive Nelder-Mead",
    "SHADE",
    "L-SHADE",
    "BOBYQA",
    "Dual Annealing",
)
FAMILY_BLOCKS = (
    "spline/warpheld",
    "knots-sparse/warpheld",
    "knots-dense/warpheld",
    "power/warpheld",
    "piecewise-linear/warpheld",
)
DECILE_COUNT = 10
ROLE_TUNING = "experiment3/role_tuning.csv"
SINGLE_BEST_SOLVER = "experiment3/single_best_solver.csv"
LEARNER_AGREEMENT = "learner_agreement/learner_agreement.csv"
SELECTION_GAIN = "experiment5/augmentation_gain.csv"


@dataclass(frozen=True)
class Column:
    """One value column of a document's table: a file column at the row."""

    header: str
    column: str
    digits: Digits = LEVEL
    # What the column adds to the row's selection, such as the block.
    where: Mapping[str, str] = field(default_factory=dict)
    # How the cell is written around its value.
    cell: str = "{}"


@dataclass(frozen=True)
class Table:
    """Rows of one report file a document shows, one quote per row.

    A row's first cells are its ``keys`` values (renamed by ``names``), the rest
    its ``columns``; ``fixed`` narrows every row.
    """

    document: str
    section: str
    path: str
    keys: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    columns: tuple[Column, ...]
    fixed: Mapping[str, str] = field(default_factory=dict)
    names: Mapping[str, str] = field(default_factory=dict)
    # The header as the document writes it, where the keys' names would not do.
    header_text: str = ""

    def text(self, row: tuple[str, ...]) -> str:
        """The row as the document writes it, a ``{}`` for each value."""
        cells = [self.names.get(value, value) for value in row]
        cells += [column.cell for column in self.columns]
        return "| " + " | ".join(cells) + " |"

    def header(self) -> str:
        """The table's two header lines."""
        if self.header_text:
            return self.header_text
        names = [*self.keys, *(column.header for column in self.columns)]
        return "| " + " | ".join(names) + " |\n|" + " --- |" * len(names)

    def quotes(self) -> Iterator[Quote]:
        """One quote per row."""
        for row in self.rows:
            where = {**self.fixed, **dict(zip(self.keys, row, strict=True))}
            yield Quote(
                self.document,
                self.section,
                self.text(row),
                tuple(
                    Source(self.path, {**where, **column.where}, column.column)
                    for column in self.columns
                ),
                digits_at={i: column.digits for i, column in enumerate(self.columns)},
                what=f"{self.path} at {', '.join(row)}",
            )


@dataclass(frozen=True)
class Statistics:
    """Single values a document shows in a two-column table, one quote each."""

    document: str
    section: str
    rows: tuple[tuple[str, Source, Digits], ...]
    header_text: str = "| Statistic | Value |\n| --- | --- |"

    def header(self) -> str:
        """The table's two header lines."""
        return self.header_text

    def quotes(self) -> Iterator[Quote]:
        """One quote per row."""
        for label, source, digits in self.rows:
            yield Quote(
                self.document,
                self.section,
                f"| {label} | {{}} |",
                (source,),
                digits=digits,
                what=f"{source.path}: {label}",
            )


TEST_COLUMNS = (
    Column("higher", "higher", WORD),
    Column("median paired difference", "median_difference"),
    Column("p (Holm)", "p_holm", P_VALUE),
    Column("significant", "significant", WORD),
)
TRAINING_TEST_COLUMNS = (
    Column("higher", "higher", WORD),
    Column("difference of medians", "median_difference"),
    Column("p", "p", P_VALUE),
    Column("p (Holm)", "p_holm", P_VALUE),
    Column("significant", "significant", WORD),
)
PAIR_NAMES = {pair: f"{group}: {label}" for group, (label, pair) in PAIRS.items()}


def shift_where(feature_set: str) -> dict[str, str]:
    """The preanalysis's reported rows of one feature set's features."""
    return {**REPORTED_SHIFT, "quantity_kind": "feature", "feature_set": feature_set}


def shift_statistics() -> Iterator[tuple[str, Source, Digits]]:
    """How many features moved and how far, per feature set."""
    for name, _, label in FEATURE_SETS:
        where = shift_where(name)

        def moved(reduce: str, keep: str = "positive", where=where) -> Source:
            return Source(SHIFT, where, "value", reduce=reduce, keep=keep)

        yield f"{label}: features", moved("count", "all"), COUNT
        yield f"{label}: features that moved (shift above 0)", moved("count"), COUNT
        yield f"{label}: features that did not move", moved("count", "zero"), COUNT
        for statistic_label, statistic in STATISTICS[1:]:
            yield (
                f"{label}: {statistic_label} shift of the moved",
                moved(statistic),
                LEVEL,
            )
        yield (
            f"{label}: feature with the largest shift",
            Source(SHIFT, where, "value", reduce="argmax", label="quantity"),
            WORD,
        )


def strength_statistics(
    path: str, problems: str
) -> tuple[tuple[str, Source, Digits], ...]:
    """The five-number summary of a block's map strengths."""
    return tuple(
        (f"{label} over the {problems} problems", strength(path, statistic), LEVEL)
        for label, statistic in STATISTICS
    )


# The closing appendices' report roots.
LEARNER_LEVELS = "learner_agreement/learner_levels.csv"
LEARNER_TESTS = "learner_agreement/learner_tests.csv"
BOOTSTRAP = "appendix_bootstrap/bootstrap_intervals{}.csv"
CHECKS = "appendix_checks/{}{}.csv"
BLOCKS = "appendix_blocks/block_{}{}.csv"
OTHER_LEARNERS = (("random-forest", "RandomForest"), ("xgboost", "XGBoost"))
FAMILIES = (
    "spline/warpheld",
    "knots-sparse/warpheld",
    "knots-dense/warpheld",
    "power/warpheld",
    "piecewise-linear/warpheld",
)
TRAININGS = ("plain", "augmented")
RESAMPLINGS = ("functions and instances", "instances within functions")
SCOPES = (UNSEEN_SCOPE, SPLINE_SCOPE)
# The control family's further blocks, the paper's draw first.
NEW_BLOCKS = (
    "spline-alpha0/warpheld",
    *(f"spline-draw{draw}/warpheld" for draw in range(1, 5)),
)
EVERY_CONTROL_BLOCK = (FAMILIES[0], *NEW_BLOCKS)
DOUBLED = "appendix_doubled/plain4800_{}{}.csv"
DOUBLED_TRAININGS = ("plain", "plain-4800", "augmented")
DOUBLED_SCOPES = ("untransformed test", UNSEEN_SCOPE, SPLINE_SCOPE)
# A blank cell of a file, as a document writes it.
BLANK = {"": "—"}


def bootstrap_rows(feature_set: str) -> tuple[tuple[str, ...], ...]:
    """The bootstrap table's rows, in the file's order.

    Each is question, part, group, view (the left side's), difference, the left
    side's representation and its predictor set ("" for a selection).
    """
    rows: list[tuple[str, ...]] = []
    if feature_set == "ela":
        rows += [
            (
                "RQ1",
                "regression",
                "G1",
                "post",
                "invariant - all",
                "value",
                "invariant",
            ),
            ("RQ1", "regression", "G1", "pre", "pre - post", "value", "all"),
            ("RQ1", "regression", "G1", "pre", "pre - post", "value", "invariant"),
            ("RQ1", "regression", "G1", "pre", "all - invariant", "value", "all"),
        ]
    for group in GROUPS:
        rows += [
            ("RQ2", "regression", group, "post", "rank - value", "rank", "all"),
            ("RQ2", "regression", group, "post", "rank - concat", "rank", "all"),
            ("RQ2", "regression", group, "pre", "rank - value", "rank", "all"),
        ]
    rows += [
        ("RQ2", "regression", group, "pre", "pre - post", variant, "all")
        for group in GROUPS[:2]
        for variant in VARIANTS
    ]
    rows += [
        ("RQ2", "selection", "Mixed", "post", "rank - value", "rank", ""),
        ("RQ2", "selection", "Mixed", "post", "rank - concat", "rank", ""),
    ]
    rows += [
        ("RQ3", "regression", group, view, "augmented - plain", variant, "all")
        for view in ("post", "pre")
        for group in GROUPS
        for variant in VARIANTS
    ]
    rows += [
        ("RQ3", "selection", "Mixed", "post", "augmented - plain", variant, "")
        for variant in VARIANTS
    ]
    after = (
        ("concat - rank", "concat"),
        ("concat - value", "concat"),
        ("rank - value", "rank"),
    )
    rows += [
        ("RQ3", "regression", group, "post", label, left, "all")
        for group in GROUPS
        for label, left in after
    ]
    rows += [
        ("RQ3", "selection", "Mixed", "post", label, left, "") for label, left in after
    ]
    return tuple(rows)


def learner_tables() -> list[Table | Statistics]:
    """RandomForest's and XGBoost's levels and tests, 2.10.2 on."""
    tables: list[Table | Statistics] = []
    index = 2
    for learner, _ in OTHER_LEARNERS:
        for name, _, _ in FEATURE_SETS:
            fixed = {"learner": learner, "feature_set": name}
            if name == "ela":
                tables.append(
                    Table(
                        SOURCES_DOCUMENT,
                        f"2.10.{index} ",
                        LEARNER_LEVELS,
                        ("predictors",),
                        (("all",), ("invariant",)),
                        (
                            Column("untransformed", "r2_pre_median"),
                            Column("transformed", "r2_post_median"),
                        ),
                        fixed={**fixed, "question": "RQ1"},
                    )
                )
            tables.append(
                Table(
                    SOURCES_DOCUMENT,
                    f"2.10.{index} ",
                    LEARNER_LEVELS,
                    ("question", "part", "group", "feature_variant"),
                    tuple(
                        (question, "regression", group, variant)
                        for question in ("RQ2", "RQ3")
                        for group in GROUPS
                        for variant in VARIANTS
                    )
                    + tuple(
                        (question, "selection", "", variant)
                        for question in ("RQ2", "RQ3")
                        for variant in VARIANTS
                    ),
                    tuple(
                        Column(training, "value", where={"training": training})
                        for training in TRAININGS
                    ),
                    fixed=fixed,
                    names=BLANK,
                )
            )
            tests = (
                Column("higher", "higher", WORD),
                Column("median difference", "median_difference"),
                Column("p", "p", P_VALUE),
                Column("p (Holm)", "p_holm", P_VALUE, cell="{}"),
            )
            if name == "ela":
                tables.append(
                    Table(
                        SOURCES_DOCUMENT,
                        f"2.10.{index + 1} ",
                        LEARNER_TESTS,
                        ("question", "left", "right"),
                        (("RQ1", "invariant", "all"),),
                        tests[:3],
                        fixed=fixed,
                    )
                )
            rows: list[tuple[str, ...]] = []
            rows += [
                ("RQ2", "regression", group, "", left, right)
                for group in GROUPS
                for left, right in (
                    ("concat", "rank"),
                    ("concat", "value"),
                    ("rank", "value"),
                )
            ]
            rows += [
                ("RQ2", "selection", "", "", left, right)
                for left, right in (
                    ("concat", "rank"),
                    ("concat", "value"),
                    ("rank", "value"),
                )
            ]
            rows += [
                ("RQ3", "regression", group, variant, "", "")
                for group in GROUPS
                for variant in VARIANTS
            ]
            rows += [("RQ3", "selection", "", variant, "", "") for variant in VARIANTS]
            tables.append(
                Table(
                    SOURCES_DOCUMENT,
                    f"2.10.{index + 1} ",
                    LEARNER_TESTS,
                    ("question", "part", "group", "feature_variant", "left", "right"),
                    tuple(rows),
                    tests,
                    fixed=fixed,
                    names=BLANK,
                )
            )
            index += 2
    return tables


def closing_tables() -> list[Table | Statistics]:
    """The bootstrap and checks appendices' numbers, 2.11 to 2.18."""
    tables: list[Table | Statistics] = []
    for offset, resampling in enumerate(RESAMPLINGS):
        for index, (name, suffix, _) in enumerate(FEATURE_SETS):
            tables.append(
                Table(
                    SOURCES_DOCUMENT,
                    f"2.11.{2 * offset + index + 1} ",
                    BOOTSTRAP.format(suffix),
                    (
                        "question",
                        "part",
                        "group",
                        "view",
                        "comparison",
                        "left_variant",
                        "left_predictors",
                    ),
                    bootstrap_rows(name),
                    (
                        Column("left level", "left_level"),
                        Column("right level", "right_level"),
                        Column("difference", "difference"),
                        Column("2.5%", "difference_lower"),
                        Column("97.5%", "difference_upper"),
                    ),
                    fixed={"resampling": resampling},
                    names=BLANK,
                )
            )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables += [
            Table(
                SOURCES_DOCUMENT,
                f"2.12.{2 * index + 1} ",
                CHECKS.format("untransformed_selection_levels", suffix),
                ("feature_variant",),
                tuple((variant,) for variant in VARIANTS),
                tuple(
                    Column(training, "value", where={"training": training})
                    for training in TRAININGS
                ),
            ),
            Table(
                SOURCES_DOCUMENT,
                f"2.12.{2 * index + 2} ",
                CHECKS.format("untransformed_selection_tests", suffix),
                ("feature_variant",),
                tuple((variant,) for variant in VARIANTS),
                TRAINING_TEST_COLUMNS,
            ),
            Table(
                SOURCES_DOCUMENT,
                f"2.13.{2 * index + 1} ",
                CHECKS.format("g1_prediction_change_levels", suffix),
                ("scope", "feature_variant"),
                tuple((scope, variant) for scope in SCOPES for variant in VARIANTS),
                tuple(
                    Column(training, "value", where={"training": training})
                    for training in TRAININGS
                ),
            ),
            Table(
                SOURCES_DOCUMENT,
                f"2.13.{2 * index + 2} ",
                CHECKS.format("g1_prediction_change_tests", suffix),
                ("scope", "feature_variant"),
                tuple(
                    (scope, variant)
                    for scope in SCOPES
                    for variant in ("value", "concat")
                ),
                TRAINING_TEST_COLUMNS,
            ),
        ]
    tables.append(
        Table(
            SOURCES_DOCUMENT,
            "2.14.1 ",
            CHECKS.format("target_spread", ""),
            ("audit", "group"),
            tuple((family, group) for family in FAMILIES for group in GROUPS),
            tuple(
                Column(f"{statistic}, {view}", statistic, where={"transform": view})
                for view in ("pre", "post")
                for statistic in ("mean", "variance")
            ),
        )
    )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.14.{index + 2} ",
                CHECKS.format("regression_error_levels", suffix),
                ("scope", "transform", "group", "feature_variant"),
                tuple(
                    (scope, view, group, variant)
                    for scope in SCOPES
                    for view in ("pre", "post")
                    for group in GROUPS
                    for variant in VARIANTS
                ),
                (
                    *(
                        Column(
                            f"{training} {statistic}",
                            column,
                            where={"training": training},
                        )
                        for training in TRAININGS
                        for statistic, column in (("R²", "r2"), ("MSE", "mse"))
                    ),
                    Column(
                        "target variance",
                        "target_variance",
                        where={"training": "plain"},
                    ),
                ),
            )
        )
    tables.append(
        Table(
            SOURCES_DOCUMENT,
            "2.15.1 ",
            CHECKS.format("g3_member_superiority", ""),
            ("measure", "audit", "algorithm"),
            tuple(
                (measure, family, algorithm)
                for measure in (
                    "superiority over the other G3 member (regression binding)",
                    "mean superiority over the portfolio (selection binding)",
                )
                for family in FAMILIES
                for algorithm in ("BOBYQA", "Dual Annealing")
            ),
            tuple(
                Column(view, "mean", where={"transform": view})
                for view in ("pre", "post")
            ),
        )
    )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.16.{index + 1} ",
                CHECKS.format("regression_family_levels", suffix),
                ("group", "feature_variant", "training"),
                tuple(
                    (group, variant, training)
                    for group in GROUPS
                    for variant in VARIANTS
                    for training in TRAININGS
                ),
                (
                    Column(
                        "untransformed",
                        "value",
                        where={"audit": FAMILIES[0], "metric": "r2_test_pre"},
                    ),
                    *(
                        Column(
                            family.split("/")[0],
                            "value",
                            where={"audit": family, "metric": "r2_test_post"},
                        )
                        for family in FAMILIES
                    ),
                ),
            )
        )
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.16.{index + 3} ",
                CHECKS.format("selection_family_levels", suffix),
                ("feature_variant", "training"),
                tuple(
                    (variant, training)
                    for variant in VARIANTS
                    for training in TRAININGS
                ),
                (
                    Column(
                        "untransformed",
                        "value",
                        where={"family": FAMILIES[0], "transform": "pre"},
                    ),
                    *(
                        Column(
                            family.split("/")[0],
                            "value",
                            where={"family": family, "transform": "post"},
                        )
                        for family in FAMILIES
                    ),
                ),
            )
        )
    tables += [
        Table(
            SOURCES_DOCUMENT,
            "2.17.1 ",
            CHECKS.format("feature_exclusion_rules", ""),
            ("rule",),
            (
                ("declared (every block the paper reads)",),
                ("recomputed over every block the paper reads",),
                ("training block alone (both views, instances 11-110)",),
            ),
            (Column("excluded", "excluded", WORD),),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.18.1 ",
            CHECKS.format("g1_trajectory_identity", ""),
            ("binding", "block", "algorithm"),
            tuple(
                (binding, block, algorithm)
                for binding, blocks in (
                    ("regression", ("training", *FAMILIES)),
                    ("selection", ("training", *FAMILIES)),
                    ("sbs", FAMILIES),
                )
                for block in blocks
                for algorithm in ("BIPOP-CMA-ES", "Adaptive Nelder-Mead")
            ),
            tuple(
                Column(header, column, COUNT)
                for header, column in (
                    ("runs", "runs"),
                    ("best_y differs", "best_y_differ"),
                    ("evaluations differ", "evaluations_used_differ"),
                    ("restarts differ", "restarts_differ"),
                    ("any differs", "runs_differ"),
                )
            ),
        ),
    ]
    return tables


def block_tables() -> list[Table | Statistics]:
    """The further blocks appendix's numbers, 2.19 to 2.21."""
    tables: list[Table | Statistics] = [
        Table(
            SOURCES_DOCUMENT,
            "2.19.1 ",
            BLOCKS.format("target_shift", ""),
            ("audit", "quantity_class"),
            tuple((block, group) for block in EVERY_CONTROL_BLOCK for group in GROUPS),
            (
                Column("moved", "p_moved"),
                Column("standardized shift", "value"),
                Column("raw shift", "value_raw"),
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.19.2 ",
            BLOCKS.format("g3_members", ""),
            ("audit", "algorithm"),
            tuple(
                (block, algorithm)
                for block in EVERY_CONTROL_BLOCK
                for algorithm in ("BOBYQA", "Dual Annealing")
            ),
            tuple(
                Column(view, "mean", where={"transform": view})
                for view in ("pre", "post")
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.19.3 ",
            BLOCKS.format("g1_identity", ""),
            ("binding", "block", "algorithm"),
            tuple(
                (binding, block, algorithm)
                for binding in ("regression", "selection", "sbs")
                for block in NEW_BLOCKS
                for algorithm in ("BIPOP-CMA-ES", "Adaptive Nelder-Mead")
            ),
            tuple(
                Column(header, column, COUNT)
                for header, column in (
                    ("runs", "runs"),
                    ("best_y differs", "best_y_differ"),
                    ("evaluations differ", "evaluations_used_differ"),
                    ("restarts differ", "restarts_differ"),
                    ("any differs", "runs_differ"),
                )
            ),
        ),
    ]
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables += [
            Table(
                SOURCES_DOCUMENT,
                f"2.20.{index + 1} ",
                BLOCKS.format("regression_levels", suffix),
                ("predictors", "group", "feature_variant", "training"),
                regression_rows_of(suffix),
                (
                    Column(
                        "untransformed",
                        "value",
                        where={"audit": FAMILIES[0], "metric": "r2_test_pre"},
                    ),
                    *(
                        Column(
                            block.split("/")[0],
                            "value",
                            where={"audit": block, "metric": "r2_test_post"},
                        )
                        for block in EVERY_CONTROL_BLOCK
                    ),
                ),
            ),
            Table(
                SOURCES_DOCUMENT,
                f"2.20.{index + 3} ",
                BLOCKS.format("selection_levels", suffix),
                ("feature_variant", "training"),
                tuple(
                    (variant, training)
                    for variant in VARIANTS
                    for training in TRAININGS
                ),
                (
                    Column(
                        "untransformed",
                        "value",
                        where={"family": FAMILIES[0], "transform": "pre"},
                    ),
                    *(
                        Column(
                            block.split("/")[0],
                            "value",
                            where={"family": block, "transform": "post"},
                        )
                        for block in EVERY_CONTROL_BLOCK
                    ),
                ),
            ),
        ]
    spread = tuple(Column(name, name) for name in ("min", "median", "max"))
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables += [
            Table(
                SOURCES_DOCUMENT,
                f"2.21.{index + 1} ",
                BLOCKS.format("regression_draw_spread", suffix),
                ("predictors", "group", "feature_variant", "training"),
                regression_rows_of(suffix),
                spread,
                fixed={"metric": "r2_test_post"},
            ),
            Table(
                SOURCES_DOCUMENT,
                f"2.21.{index + 3} ",
                BLOCKS.format("selection_draw_spread", suffix),
                ("feature_variant", "training"),
                tuple(
                    (variant, training)
                    for variant in VARIANTS
                    for training in TRAININGS
                ),
                spread,
                fixed={"transform": "post"},
            ),
        ]
    return tables


def doubled_tables() -> list[Table | Statistics]:
    """The plain-4800 appendix's numbers, 2.22 and 2.23."""
    untransformed = {"regression": {"metric": "r2_test_pre", "audit": FAMILIES[0]}}
    untransformed["selection"] = {"transform": "pre", "family": FAMILIES[0]}
    on_family = {
        "regression": lambda family: {"metric": "r2_test_post", "audit": family},
        "selection": lambda family: {"transform": "post", "family": family},
    }
    keys = {"regression": ("group",), "selection": ()}
    tables: list[Table | Statistics] = []
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        for offset, task in enumerate(("regression", "selection")):
            cells = [(group,) for group in GROUPS] if keys[task] else [()]
            tables.append(
                Table(
                    SOURCES_DOCUMENT,
                    f"2.22.{2 * index + offset + 1} ",
                    DOUBLED.format(f"{task}_levels", suffix),
                    (*keys[task], "feature_variant", "training"),
                    tuple(
                        (*cell, variant, training)
                        for cell in cells
                        for variant in VARIANTS
                        for training in DOUBLED_TRAININGS
                    ),
                    (
                        Column("untransformed", "value", where=untransformed[task]),
                        *(
                            Column(
                                family.split("/")[0],
                                "value",
                                where=on_family[task](family),
                            )
                            for family in FAMILIES
                        ),
                    ),
                )
            )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        for offset, task in enumerate(("regression", "selection")):
            cells = [(group,) for group in GROUPS] if keys[task] else [()]
            tables += [
                Table(
                    SOURCES_DOCUMENT,
                    f"2.23.{4 * index + 2 * offset + 1} ",
                    DOUBLED.format(f"{task}_scopes", suffix),
                    (*keys[task], "feature_variant", "training"),
                    tuple(
                        (*cell, variant, training)
                        for cell in cells
                        for variant in VARIANTS
                        for training in DOUBLED_TRAININGS
                    ),
                    tuple(
                        Column(scope, "value", where={"scope": scope})
                        for scope in DOUBLED_SCOPES
                    ),
                ),
                Table(
                    SOURCES_DOCUMENT,
                    f"2.23.{4 * index + 2 * offset + 2} ",
                    DOUBLED.format(f"{task}_tests", suffix),
                    ("scope", *keys[task], "feature_variant"),
                    tuple(
                        (scope, *cell, variant)
                        for scope in DOUBLED_SCOPES
                        for cell in cells
                        for variant in VARIANTS
                    ),
                    TRAINING_TEST_COLUMNS,
                ),
            ]
    return tables


LOFO = "appendix_lofo/lofo_regression_levels{}.csv"
REGRESSION_LEVELS = "experiment4/group_regression_levels.csv"


def lofo_tables() -> list[Table | Statistics]:
    """Appendix G's leave-one-function-out levels, 2.24."""
    tables: list[Table | Statistics] = [
        Table(
            SOURCES_DOCUMENT,
            f"2.24.{index + 1} ",
            LOFO.format(suffix),
            ("feature_variant", "training"),
            tuple(
                (variant, training) for variant in VARIANTS for training in TRAININGS
            ),
            tuple(
                Column(group, "lofo_median", where={"group": group}) for group in GROUPS
            ),
        )
        for index, (_, suffix, _) in enumerate(FEATURE_SETS)
    ]
    tables.append(
        Statistics(
            SOURCES_DOCUMENT,
            "2.24.3 ",
            tuple(
                (
                    f"{name}: {label}",
                    Source(LOFO.format(suffix), {}, column, reduce),
                    LEVEL,
                )
                for _, suffix, name in FEATURE_SETS
                for label, column, reduce in (
                    ("smallest median", "lofo_median", "min"),
                    ("largest median", "lofo_median", "max"),
                    ("largest upper quartile", "lofo_q3", "max"),
                )
            ),
        )
    )
    return tables


def untransformed_level(group: str, variant: str, training: str) -> Source:
    """Experiment 4's ELA level of a cell on the untransformed test block."""
    return Source(
        REGRESSION_LEVELS,
        {
            "scope": UNSEEN_SCOPE,
            "transform": "pre",
            "group": group,
            "feature_variant": variant,
            "training": training,
        },
        "level",
    )


def main_text_tables() -> list[Table | Statistics]:
    """Single values the main text quotes from the appendices, 2.25."""
    draws = BLOCKS.format("regression_levels", "")
    g1_value_plain = {
        "predictors": "all",
        "group": "G1",
        "feature_variant": "value",
        "training": "plain",
        "metric": "r2_test_post",
    }
    on_unseen = {"training": "plain-4800", "scope": UNSEEN_SCOPE}
    rows: list[tuple[str, Source, Digits]] = [
        (
            f"G1, value, plain, R² on {block} ({extreme} of draws 1–4)",
            Source(draws, {**g1_value_plain, "audit": f"{block}/warpheld"}, "value"),
            LEVEL,
        )
        for block, extreme in (
            ("spline-draw1", "smallest"),
            ("spline-draw2", "largest"),
        )
    ]
    rows.append(
        (
            "G3, standardized shift of J on spline-alpha0",
            Source(
                BLOCKS.format("target_shift", ""),
                {"audit": "spline-alpha0/warpheld", "quantity_class": "G3"},
                "value",
            ),
            LEVEL,
        )
    )
    rows += [
        (
            f"{group}, {variant}, plain-4800, R² over the unseen families",
            Source(
                DOUBLED.format("regression_scopes", ""),
                {**on_unseen, "group": group, "feature_variant": variant},
                "value",
            ),
            LEVEL,
        )
        for variant in ("rank", "value", "concat")
        for group in ("G1", "G2")
    ]
    rows += [
        (
            f"G3, rank, {training}, R² on the untransformed test block",
            untransformed_level("G3", "rank", training),
            LEVEL,
        )
        for training in TRAININGS
    ]
    rows += [
        (
            f"{group}, {variant}: plain − augmented, R² on the untransformed block",
            replace(
                untransformed_level(group, variant, "plain"),
                minus=untransformed_level(group, variant, "augmented"),
            ),
            LEVEL,
        )
        for group in GROUPS
        for variant in VARIANTS
    ]
    return [Statistics(SOURCES_DOCUMENT, "2.25.1 ", tuple(rows))]


def regression_rows_of(suffix: str) -> tuple[tuple[str, ...], ...]:
    """The regression rows of a block table: Experiment 1's cell only with ELA."""
    return tuple(
        (kind, group, variant, training)
        for kind in (("all", "invariant") if not suffix else ("all",))
        for group in GROUPS
        for variant in VARIANTS
        for training in TRAININGS
        if kind == "all" or (group, variant, training) == ("G1", "value", "plain")
    )


def document_tables() -> list[Table | Statistics]:
    """Every table of the two documents under docs/, in document order."""
    tables: list[Table | Statistics] = [
        # docs/SETTINGS.md
        Table(
            SETTINGS_DOCUMENT,
            "6.1 ",
            PAIR_TUNING,
            ("group", "configuration"),
            tuple(
                (group, side)
                for group in TUNED_MEMBERS
                for side in ("default", "selected")
            ),
            (
                Column("params", "params", WORD, cell="`{}`"),
                Column("D_KS", "score"),
                Column("saturated", "n_saturated", COUNT),
                Column("at the minimum", "n_at_minimum", COUNT),
            ),
            # The tuned member is a cell of its own, beside the group.
            names={
                group: f"{group} | {tuned}" for group, tuned in TUNED_MEMBERS.items()
            },
            header_text=(
                "| Group | Tuned member | Configuration | `params` | D_KS | "
                "Saturated (of 24) | At the minimum |\n"
                "| --- | --- | --- | --- | --- | --- | --- |"
            ),
        ),
        Table(
            SETTINGS_DOCUMENT,
            "6.2 ",
            ROLE_TUNING,
            ("algorithm",),
            tuple((name,) for name in ALGORITHM_ORDER),
            (
                Column(
                    "selection",
                    "params",
                    WORD,
                    where={"condition": "selection"},
                    cell="`{}`",
                ),
                Column("sbs", "params", WORD, where={"condition": "sbs"}, cell="`{}`"),
            ),
            fixed={"configuration": "selected"},
        ),
        Table(
            SETTINGS_DOCUMENT,
            "7.3 ",
            SINGLE_BEST_SOLVER,
            ("algorithm",),
            tuple((name,) for name in ALGORITHM_ORDER),
            (
                Column("mean superiority", "mean_superiority"),
                Column("SBS", "is_sbs", WORD),
            ),
        ),
        # docs/PAPER_SOURCES.md
        Table(
            SOURCES_DOCUMENT,
            "2.1.1 ",
            SHIFT,
            ("quantity",),
            (*((pair,) for _, pair in PAIRS.values()), ("deep_ela.large_50d_v1.X2",)),
            (
                Column("mean shift", "value"),
                Column("raw change", "value_raw", P_VALUE),
                Column("share of problems moved", "p_moved"),
            ),
            fixed=REPORTED_SHIFT,
            names=PAIR_NAMES,
        ),
        Statistics(SOURCES_DOCUMENT, "2.1.2 ", tuple(shift_statistics())),
        Statistics(
            SOURCES_DOCUMENT, "2.2.1 ", strength_statistics(TEST_STRENGTH, "240")
        ),
        Statistics(
            SOURCES_DOCUMENT, "2.2.2 ", strength_statistics(TRAINING_STRENGTH, "2,400")
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.3.1 ",
            HELD_OUT_PREMISES,
            ("group",),
            tuple((group,) for group in GROUPS),
            (
                Column("problems", "n_compared", COUNT),
                Column("superiority differs", "n_differing", COUNT),
                Column("largest difference", "max_difference"),
            ),
            fixed={"check": "target", "block": CONTROL_BLOCK},
        ),
        Statistics(
            SOURCES_DOCUMENT,
            "2.3.1 ",
            (
                (
                    "untransformed feature values equal to the training block's",
                    anchor(HELD_OUT_PREMISES, CONTROL_BLOCK),
                    COUNT,
                ),
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.3.2 ",
            EXPERIMENT4_PREMISES,
            ("block", "group"),
            tuple(
                (block, group)
                for block in (TRAINING_BLOCK, *FAMILY_BLOCKS)
                for group in GROUPS
            ),
            (
                Column("problems", "n_compared", COUNT),
                Column("superiority differs", "n_differing", COUNT),
                Column("largest difference", "max_difference"),
            ),
            fixed={"check": "target"},
        ),
        Statistics(
            SOURCES_DOCUMENT,
            "2.3.2 ",
            (
                (
                    "G1: most problems whose superiority differs, over the six blocks",
                    Source(
                        EXPERIMENT4_PREMISES,
                        {"check": "target", "group": "G1"},
                        "n_differing",
                        reduce="max",
                    ),
                    COUNT,
                ),
                (
                    "G1: fewest problems compared, over the six blocks",
                    Source(
                        EXPERIMENT4_PREMISES,
                        {"check": "target", "group": "G1"},
                        "n_compared",
                        reduce="min",
                    ),
                    COUNT,
                ),
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.4.1 ",
            PREDICTOR_LEVELS,
            ("predictors",),
            (("all",), ("invariant",)),
            (
                Column("columns", "n_features", COUNT),
                Column("R² untransformed", "r2_pre_median"),
                Column("R² transformed", "r2_post_median"),
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.4.2 ",
            PREDICTOR_TESTS,
            ("left", "right"),
            (("invariant", "all"),),
            (
                Column("seeds", "n_pairs", COUNT),
                Column("higher", "higher", WORD),
                Column("p", "p", P_VALUE),
                Column("median paired difference", "median_difference"),
                Column("significant", "significant", WORD),
            ),
        ),
    ]
    variant_rows = tuple((group, variant) for group in GROUPS for variant in VARIANTS)
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.5.{index + 1} ",
                f"experiment4/group_regression_levels{suffix}.csv",
                ("group", "feature_variant"),
                variant_rows,
                (
                    Column("untransformed", "level", where={"transform": "pre"}),
                    Column("transformed", "level", where={"transform": "post"}),
                ),
                fixed={"scope": SPLINE_SCOPE, "training": "plain"},
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.5.{index + 3} ",
                f"experiment4/representation_tests{suffix}.csv",
                ("group", "left", "right"),
                tuple((group, *pair) for group in GROUPS for pair in COMPARISONS),
                TEST_COLUMNS,
                fixed={"scope": SPLINE_SCOPE, "training": "plain"},
            )
        )
    representation_rows = tuple((variant,) for variant in VARIANTS)
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.6.{index + 1} ",
                f"experiment3/selection_levels{suffix}.csv",
                ("representation",),
                representation_rows,
                (
                    Column(
                        "untransformed",
                        "superiority_vs_sbs_mean",
                        where={"transform": "pre"},
                    ),
                    Column(
                        "transformed",
                        "superiority_vs_sbs_mean",
                        where={"transform": "post"},
                    ),
                    Column(
                        "oracle, untransformed", "oracle", where={"transform": "pre"}
                    ),
                    Column(
                        "oracle, transformed", "oracle", where={"transform": "post"}
                    ),
                ),
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.6.{index + 3} ",
                f"experiment3/representation_tests{suffix}.csv",
                ("left", "right"),
                COMPARISONS,
                TEST_COLUMNS,
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.7.{index + 1} ",
                f"experiment4/group_regression_levels{suffix}.csv",
                ("group", "feature_variant"),
                variant_rows,
                tuple(
                    Column(
                        f"{training}, {block}{' spread' if column == 'spread' else ''}",
                        column,
                        where={"training": training, "transform": view},
                    )
                    for column, block, view in (
                        ("level", "transformed", "post"),
                        ("spread", "transformed", "post"),
                        ("level", "untransformed", "pre"),
                        ("spread", "untransformed", "pre"),
                    )
                    for training in ("plain", "augmented")
                ),
                fixed={"scope": UNSEEN_SCOPE},
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.7.{index + 3} ",
                f"experiment4/representation_tests{suffix}.csv",
                ("training", "group", "left", "right"),
                tuple(
                    (training, group, *pair)
                    for training in ("plain", "augmented")
                    for group in GROUPS
                    for pair in COMPARISONS
                ),
                TEST_COLUMNS,
                fixed={"scope": UNSEEN_SCOPE},
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.7.{index + 5} ",
                f"experiment4/training_tests{suffix}.csv",
                ("group", "feature_variant"),
                variant_rows,
                TRAINING_TEST_COLUMNS,
                fixed={"scope": UNSEEN_SCOPE},
            )
        )
    tables.append(
        Statistics(
            SOURCES_DOCUMENT,
            "2.7.7 ",
            tuple(
                row
                for _, suffix, label in FEATURE_SETS
                for row in (
                    (
                        f"{label}: cells where augmented is significantly higher",
                        Source(
                            f"experiment4/training_tests{suffix}.csv",
                            {
                                "scope": UNSEEN_SCOPE,
                                "significant": "True",
                                "higher": "augmented",
                            },
                            "p_holm",
                            reduce="count",
                        ),
                        COUNT,
                    ),
                    (
                        f"{label}: cells with no significant difference",
                        Source(
                            f"experiment4/training_tests{suffix}.csv",
                            {"scope": UNSEEN_SCOPE, "significant": "False"},
                            "p_holm",
                            reduce="count",
                        ),
                        COUNT,
                    ),
                )
            ),
        )
    )
    tables.append(
        Table(
            SOURCES_DOCUMENT,
            "2.8.1 ",
            SELECTION_GAIN,
            ("scope", "feature_set", "representation"),
            tuple(
                (scope, name, variant)
                for scope in (UNSEEN_SCOPE, SPLINE_SCOPE)
                for name, _, _ in FEATURE_SETS
                for variant in VARIANTS
            ),
            (
                Column(
                    "plain", "superiority_vs_sbs_mean", where={"augmentation": "plain"}
                ),
                Column(
                    "augmented",
                    "superiority_vs_sbs_mean",
                    where={"augmentation": "augmented"},
                ),
                Column("oracle", "oracle", where={"augmentation": "plain"}),
            ),
        )
    )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.8.{index + 2} ",
                f"experiment5/representation_tests{suffix}.csv",
                ("training", "left", "right"),
                tuple(
                    (training, *pair)
                    for training in ("plain", "augmented")
                    for pair in COMPARISONS
                ),
                TEST_COLUMNS,
                fixed={"scope": UNSEEN_SCOPE},
            )
        )
    for index, (_, suffix, _) in enumerate(FEATURE_SETS):
        tables.append(
            Table(
                SOURCES_DOCUMENT,
                f"2.8.{index + 4} ",
                f"experiment5/training_tests{suffix}.csv",
                ("feature_variant",),
                representation_rows,
                TRAINING_TEST_COLUMNS,
                fixed={"scope": UNSEEN_SCOPE},
            )
        )
    tables += [
        Table(
            SOURCES_DOCUMENT,
            "2.9.1 ",
            CALIBRATION,
            ("block", "group"),
            tuple(
                (block, group) for block in ("tuning", "training") for group in GROUPS
            ),
            (
                Column("problems", "n_problems", COUNT),
                Column("D_KS", "ks_distance"),
                Column("floor", "ks_floor", ".2g"),
                Column("mean", "mean"),
                Column("variance", "variance"),
                Column("saturated", "saturated_share"),
                Column("informative", "informative_share"),
                Column("informative problems", "n_informative", COUNT),
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.9.2 ",
            CALIBRATION_DECILES,
            ("block", "group"),
            tuple(
                (block, group) for block in ("tuning", "training") for group in GROUPS
            ),
            tuple(
                Column(
                    # numpy's histogram closes the last bin.
                    f"[{decile / 10:.1f}, {(decile + 1) / 10:.1f}"
                    + ("]" if decile == DECILE_COUNT - 1 else ")"),
                    "n_problems",
                    COUNT,
                    where={"decile": str(decile)},
                )
                for decile in range(DECILE_COUNT)
            ),
        ),
        Table(
            SOURCES_DOCUMENT,
            "2.10.1 ",
            LEARNER_AGREEMENT,
            ("learner", "feature_set"),
            tuple(
                (learner, name)
                for learner in ("realmlp", "random-forest", "xgboost")
                for name, _, _ in FEATURE_SETS
            ),
            tuple(Column(rq, rq, WORD) for rq in ("RQ1", "RQ2", "RQ3")),
        ),
    ]
    return (
        tables
        + learner_tables()
        + closing_tables()
        + block_tables()
        + doubled_tables()
        + lofo_tables()
        + main_text_tables()
    )


def settings_text_quotes() -> Iterator[Quote]:
    """docs/SETTINGS.md's computed values written in its text and grid tables."""
    yield Quote(
        SETTINGS_DOCUMENT,
        "6.1 ",
        "1/(2·31²), is {} (`tune_algorithms.floor_for`)",
        (tuning("G1", "selected", "floor"),),
        what="the floor of the KS distance at 24 problems",
    )
    for group, text in (
        (
            "G1",
            "| Adaptive Nelder-Mead (G1) | `simplex_factor` ∈ {{0.01, 0.025, 0.05, "
            "0.1, 0.25}} | {} |",
        ),
        (
            "G2",
            "| SHADE (G2) | `population_factor` ∈ {{5, 10, 18, 30, 50}}, `f0` ∈ "
            "{{0.3, …, 0.7}}, `cr0` ∈ {{0.3, …, 0.7}} | {} |",
        ),
        (
            "G3",
            "| Dual Annealing (G3) | `initial_temp` ∈ {{500, 1000, 5230, 15000, "
            "40000}}, `visit` ∈ {{1.5, 2.0, 2.62, 2.85, 2.95}}, `accept` ∈ {{−5, "
            "−15, −50, −150, −500}} | {} |",
        ),
    ):
        yield Quote(
            SETTINGS_DOCUMENT,
            "6.1 ",
            text,
            (tuning(group, "selected", "n_candidates"),),
            digits=COUNT,
            what=f"{group}'s grid size",
        )
    for algorithm, text in (
        (
            "BIPOP-CMA-ES",
            "| BIPOP-CMA-ES | `sigma0_factor` ∈ {{0.1, 0.2, 0.3, 0.4, 0.5}}, "
            "`lambda0_factor` ∈ {{default, 2, 3, 5, 8}} (λ0 = max(4, "
            "round(factor·10))) | {} |",
        ),
        (
            "Adaptive Nelder-Mead",
            "| Adaptive Nelder-Mead | `simplex_factor` as in §6.1 | {} |",
        ),
        ("SHADE", "| SHADE, L-SHADE | as SHADE in §6.1 | {} each |"),
        (
            "BOBYQA",
            "| BOBYQA | `initial_step_factor` ∈ {{NLopt's default, 0.05, 0.1, 0.2, "
            "0.4}} × box width | {} |",
        ),
        ("Dual Annealing", "| Dual Annealing | as in §6.1 | {} |"),
    ):
        yield Quote(
            SETTINGS_DOCUMENT,
            "6.2 ",
            text,
            (
                Source(
                    ROLE_TUNING,
                    {
                        "condition": "selection",
                        "algorithm": algorithm,
                        "configuration": "selected",
                    },
                    "n_candidates",
                ),
            ),
            digits=COUNT,
            what=f"{algorithm}'s grid size in the role tuning",
        )


def document_quotes() -> Iterator[Quote]:
    """Every quote of the two documents under docs/."""
    yield from settings_text_quotes()
    for table in document_tables():
        yield from table.quotes()


def source_key(source: Source) -> str:
    """One string naming a source: file, rows, column and reduction."""
    where = ";".join(f"{key}={value}" for key, value in sorted(source.where.items()))
    return "|".join(
        (
            source.path,
            where,
            source.column,
            source.reduce or "",
            source.keep,
            source.label,
            "" if source.minus is None else f"minus {source_key(source.minus)}",
        )
    )


def build_quotes() -> list[Quote]:
    """Every registered quote, document by document."""
    return list(document_quotes())


QUOTES: list[Quote] = build_quotes()


# Every table row in this section must be registered, so a number cannot be
# added to the document without its source.
REGISTERED_SECTION = (SOURCES_DOCUMENT, "2. Numbers in the text")


def table_rows(body: str) -> Iterator[str]:
    """The data rows of every Markdown table in ``body``: not headers, not rules."""
    previous_in_table = False
    skip = 0
    for line in body.splitlines():
        in_table = line.startswith("|")
        if in_table and not previous_in_table:
            skip = 2  # the header and the rule below it
        if in_table:
            if skip:
                skip -= 1
            else:
                yield line.strip()
        previous_in_table = in_table


def unregistered_rows(
    quotes: Sequence[Quote] = QUOTES, repository: Path = REPOSITORY
) -> list[str]:
    """Rows of the registered section that no quote of its document registers."""
    document, section = REGISTERED_SECTION
    path = repository / document
    if not path.is_file():
        return [f"{document} does not exist"]
    body = section_text(path.read_text(encoding="utf-8"), section)
    patterns = [quote_pattern(q) for q in quotes if q.document == document]
    return [
        row
        for row in table_rows(body)
        if not any(pattern.fullmatch(row) for pattern in patterns)
    ]


def render(root: Path, document: str | None = None) -> str:
    """The documents' tables with values from ``root``, to paste in after a rerun."""
    blocks = []
    for table in document_tables():
        if document is not None and table.document != document:
            continue
        lines = [f"<!-- {table.document} § {table.section.strip()} -->", table.header()]
        for quote in table.quotes():
            values = [
                write_value(read_value(root, source), quote.digits_of(index))
                for index, source in enumerate(quote.sources)
            ]
            lines.append(quote.text.format(*values))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


# Two or more decimals, not part of a word or a longer number. LaTeX spells a
# minus as -, {-} or $-$; a dash after another dash is a range's, not a sign.
TEX_NUMBER = re.compile(
    r"(?<![\w.])(?:(?<!-)(?P<sign>-|\{-\}|\$-\$))?(?P<digits>\d+\.\d{2,})(?!\w|\.\d)"
)


def tex_numbers(path: Path) -> Iterator[tuple[Path, int, str]]:
    """Every number with two or more decimals in ``path``, skipping comments."""
    files = [path] if path.is_file() else sorted(path.rglob("*.tex"))
    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            code = re.split(r"(?<!\\)%", line, maxsplit=1)[0]
            for match in TEX_NUMBER.finditer(code):
                yield file, number, f"{'-' if match['sign'] else ''}{match['digits']}"


def registered_values(
    root: Path, quotes: Sequence[Quote] = QUOTES
) -> list[tuple[float, str]]:
    """Every numeric registered value in ``root``, with the quote it belongs to."""
    values = []
    seen = set()
    for quote in quotes:
        for source in quote.sources:
            key = (quote.document, source_key(source))
            if key in seen:
                continue
            seen.add(key)
            try:
                value = float(read_value(root, source))  # type: ignore[arg-type]
            except (LookupError, TypeError, ValueError):
                continue
            values.append((value, f"{quote.document}: {quote.what or quote.text}"))
    return values


def tex_matches(
    path: Path, root: Path, quotes: Sequence[Quote] = QUOTES
) -> Iterator[tuple[Path, int, str, list[str]]]:
    """Each number of a paper's source, with the registered values it rounds from."""
    values = registered_values(root, quotes)
    for file, line, text in tex_numbers(path):
        decimals = len(text.split(".")[1])
        names = [name for value, name in values if write_value(value, decimals) == text]
        yield file, line, text, names


def check_tex(path: Path, root: Path, quotes: Sequence[Quote] = QUOTES) -> list[str]:
    """The numbers of a paper's source that no registered source rounds to.

    Many values share a spelling at three decimals, so this only says a number has
    some source; ``--tex-matches`` lists the candidates.
    """
    return [
        f"{file}:{line}: {text}"
        for file, line, text, names in tex_matches(path, root, quotes)
        if not names
    ]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Check every number the documents quote from a result against the "
            "report files it comes from."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "the directory holding the report roots (preanalysis, experiment1, ...) "
            f"and their store (_stages); default {DEFAULT_ROOT}"
        ),
    )
    parser.add_argument(
        "--also",
        type=Path,
        action="append",
        default=[],
        help=(
            "a further directory of report roots, read for a file --root does not "
            "hold (an appendix's own output root); may be given more than once"
        ),
    )
    parser.add_argument(
        "--structure",
        action="store_true",
        help="read no report root: check only where each quote is and its shape",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="print the docs' tables with their values from --root, and check nothing",
    )
    parser.add_argument(
        "--tex",
        type=Path,
        help=(
            "a LaTeX file or a directory of them (the paper's source, which is not "
            "in the repository): list every number with two or more decimals that "
            "no registered source in --root rounds to"
        ),
    )
    parser.add_argument(
        "--tex-matches",
        action="store_true",
        help="with --tex, print every number with the registered values it matches",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and return a process exit status."""
    args = parse_args(argv)
    ALSO_ROOTS[:] = args.also
    if args.render:
        print(render(args.root), end="")
        return 0
    failures = []
    for quote in QUOTES:
        problem = (
            check_structure(quote) if args.structure else check_values(quote, args.root)
        )
        if problem is not None:
            failures.append((quote, problem))
    documents = sorted({quote.document for quote in QUOTES})
    checked = "where and in what shape" if args.structure else f"against {args.root}"
    print(f"{len(QUOTES)} quotes in {len(documents)} documents, checked {checked}")
    for quote, problem in failures:
        print(f"  FAIL {quote.what or quote.text}: {problem}")
    print("all agree" if not failures else f"{len(failures)} disagree")
    unregistered = unregistered_rows()
    for row in unregistered:
        print(f"  UNREGISTERED row in {SOURCES_DOCUMENT}: {row}")
    unmatched: list[str] = []
    if args.tex is not None and args.tex_matches:
        for file, line, text, names in tex_matches(args.tex, args.root):
            print(f"{file}:{line}: {text}")
            for name in names:
                print(f"    {name}")
    if args.tex is not None:
        unmatched = check_tex(args.tex, args.root)
        for problem in unmatched:
            print(f"  NOT REGISTERED in the paper: {problem}")
        print(f"{args.tex}: {len(unmatched)} number(s) no registered source rounds to")
    return 1 if failures or unregistered or unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
