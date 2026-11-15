"""The check of the docs' numbers against the report files.

The values themselves need a report root, which the suite does not have; what is
checked here is that every registered quote is where it is registered, and that
the check reads, writes and compares values the way the documents need.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_document_numbers as checker
from check_document_numbers import (
    QUOTES,
    SETTINGS_DOCUMENT,
    SOURCES_DOCUMENT,
    Column,
    Quote,
    Source,
    Table,
    check_structure,
    check_tex,
    check_values,
    quote_pattern,
    read_value,
    render,
    section_text,
    table_rows,
    tex_matches,
    tex_numbers,
    unregistered_rows,
    write_value,
)


class TestTheRegisteredQuotes:
    def test_no_quote_reads_a_table_written_as_rounded_text(self):
        """Experiment 4's main tables are read through their full-precision numbers."""
        rounded = re.compile(r"experiment4/group_regression_(pre|post)_")
        assert not [
            source.path
            for quote in QUOTES
            for source in quote.sources
            if rounded.match(source.path)
        ]

    @pytest.mark.parametrize(
        "quote", QUOTES, ids=[f"{q.document}:{q.section}:{q.what}" for q in QUOTES]
    )
    def test_each_is_where_it_is_registered(self, quote):
        assert check_structure(quote) is None

    def test_both_documents_are_covered(self):
        documents = {q.document for q in QUOTES}
        assert documents == {SETTINGS_DOCUMENT, SOURCES_DOCUMENT}


class TestTheDocuments:
    """The numbers section of docs/PAPER_SOURCES.md holds registered rows only."""

    def test_every_row_of_the_registered_section_is_registered(self):
        assert unregistered_rows() == []

    def test_a_row_without_a_source_is_named(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / SOURCES_DOCUMENT).write_text(
            "## 2. Numbers in the text\n\n| a | b |\n| --- | --- |\n| x | 1 |\n"
            "\n## 3. After\n",
            encoding="utf-8",
        )
        assert unregistered_rows([], tmp_path) == ["| x | 1 |"]


class TestWritingAValueToSignificantDigits:
    def test_a_format_specification(self):
        assert write_value("0.0009765625", ".3g") == "0.000977"
        assert write_value(2.835142154027603e-06, ".3g") == "2.84e-06"
        assert write_value(1.0, ".3g") == "1"

    def test_no_negative_zero(self):
        assert write_value(-0.0, ".3g") == "0"


class TestATextValue:
    def test_a_configuration_written_as_the_file_spells_it_has_a_shape(self):
        source = Source("experiment2/pair_tuning.csv", {"group": "G1"}, "params")
        quote = Quote("doc.md", "1", "| G1 | `{}` |", (source,), digits=None)
        assert quote_pattern(quote).fullmatch('| G1 | `{"eps_x": 0.0001}` |')


class TestTableRows:
    def test_the_header_and_the_rule_are_not_rows(self):
        body = (
            "text\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"
            "\n| c |\n| --- |\n| 5 |\n"
        )
        assert list(table_rows(body)) == ["| 1 | 2 |", "| 3 | 4 |", "| 5 |"]


class TestRendering:
    def test_a_rendered_row_is_the_registered_row(self, root, monkeypatch):
        table = Table(
            "doc.md",
            "2.1 ",
            "experiment1/levels.csv",
            ("predictors",),
            (("all",), ("invariant",)),
            (Column("pre", "r2_pre_median"), Column("post", "r2_post_median")),
        )
        monkeypatch.setattr(checker, "document_tables", lambda: [table])
        lines = render(root).splitlines()
        assert lines[1:3] == ["| predictors | pre | post |", "| --- | --- | --- |"]
        assert lines[3:] == ["| all | 0.872 | 0.117 |", "| invariant | 0.462 | 0.462 |"]


class TestThePapersSource:
    @pytest.fixture
    def paper(self, tmp_path) -> Path:
        directory = tmp_path / "paper"
        directory.mkdir()
        (directory / "main.tex").write_text(
            "R$^2$ falls from 0.872 to 0.12 % 0.999 is a comment\n"
            "a rate of 5\\% and section 3.10 and 0.4622 \\label{fig:0.55}\n"
            "It ends at 0.46. Version 1.2.3 is not a number.\n",
            encoding="utf-8",
        )
        return directory

    def test_numbers_with_two_or_more_decimals_outside_comments(self, paper):
        found = [text for _, _, text in tex_numbers(paper)]
        assert found == ["0.872", "0.12", "3.10", "0.4622", "0.55", "0.46"]

    def test_every_spelling_of_a_minus_is_read_as_a_sign(self, tmp_path):
        paper = tmp_path / "minus.tex"
        paper.write_text(
            "$-0.86$ and $[{-}0.326, {-}0.102]$ and \\textbf{$-$0.80} but 0.12--0.34\n",
            encoding="utf-8",
        )
        found = [text for _, _, text in tex_numbers(paper)]
        assert found == ["-0.86", "-0.326", "-0.102", "-0.80", "0.12", "0.34"]

    def test_a_number_no_source_rounds_to_is_listed(self, paper, root, monkeypatch):
        sources = (
            Source("experiment1/levels.csv", {"predictors": "all"}, "r2_pre_median"),
            Source("experiment1/levels.csv", {"predictors": "all"}, "r2_post_median"),
            Source(
                "experiment1/levels.csv", {"predictors": "invariant"}, "r2_pre_median"
            ),
        )
        quotes = [Quote("doc.md", "1", "{} {} {}", sources)]
        problems = check_tex(paper, root, quotes)
        assert [problem.rsplit(": ", 1)[1] for problem in problems] == ["3.10", "0.55"]

    def test_each_number_is_listed_with_the_values_it_rounds_from(self, paper, root):
        source = Source(
            "experiment1/levels.csv", {"predictors": "all"}, "r2_pre_median"
        )
        quotes = [Quote("doc.md", "1", "{}", (source,), what="all, untransformed")]
        found = {text: names for _, _, text, names in tex_matches(paper, root, quotes)}
        assert found["0.872"] == ["doc.md: all, untransformed"]
        assert found["0.12"] == []


DOCUMENT = """# Title

## 1. Design

| G1 | 0.500 |

## 2. Results

### 2.1 Levels

| G1 | 0.872 | 0.117 |

### 2.2 Tests

p = 0.001

## 3. After
"""


class TestTheSection:
    def test_the_body_runs_to_the_next_heading_of_its_level(self):
        body = section_text(DOCUMENT, "2. Results")
        assert "| G1 | 0.872 | 0.117 |" in body
        assert "p = 0.001" in body
        assert "After" not in body

    def test_a_subsection_stops_at_its_sibling(self):
        body = section_text(DOCUMENT, "2.1")
        assert "0.872" in body
        assert "p = 0.001" not in body

    def test_an_absent_heading_is_an_error(self):
        with pytest.raises(KeyError):
            section_text(DOCUMENT, "4. Nothing")

    def test_a_comment_in_fenced_code_is_not_a_heading(self):
        text = (
            "## 1. Run\n\n```bash\n# a comment\njust run\n```\n\nafter\n\n## 2. Next\n"
        )
        body = section_text(text, "1. Run")
        assert "just run" in body
        assert "after" in body
        assert "Next" not in body


class TestWritingAValue:
    def test_fixed_decimals(self):
        assert write_value("0.87210", 3) == "0.872"
        assert write_value(23, 0) == "23"

    def test_no_negative_zero(self):
        assert write_value(-0.0004, 3) == "0.000"

    def test_the_file_s_own_text(self):
        assert write_value("0.8721 (0.0000)", None) == "0.8721 (0.0000)"

    def test_thousands_grouped(self):
        assert write_value("14400", 0, grouped=True) == "14,400"
        assert write_value(2400, 0) == "2400"


@pytest.fixture
def root(tmp_path) -> Path:
    report = tmp_path / "root"
    (report / "experiment1").mkdir(parents=True)
    pd.DataFrame(
        {
            "predictors": ["all", "invariant"],
            "r2_pre_median": [0.8721090972, 0.4622238241],
            "r2_post_median": [0.1173077697, 0.4622238241],
        }
    ).to_csv(report / "experiment1" / "levels.csv", index=False)
    (report / "preanalysis").mkdir()
    pd.DataFrame(
        {
            "quantity": ["a", "b", "c", "d"],
            "kind": ["feature"] * 4,
            "value": [0.0, 0.3, 0.9, 0.5],
        }
    ).to_parquet(report / "preanalysis" / "shift.parquet")
    return report


class TestReadingAValue:
    def test_one_row_of_a_csv(self, root):
        source = Source(
            "experiment1/levels.csv", {"predictors": "all"}, "r2_pre_median"
        )
        assert write_value(read_value(root, source), 3) == "0.872"

    def test_a_value_is_rounded_from_its_full_precision(self, root):
        """A value is rounded from full precision, not from six-decimal text."""
        (root / "experiment5").mkdir()
        (root / "experiment5" / "gain.csv").write_text(
            "representation,level\nvalue,0.5864499902\n", encoding="utf-8"
        )
        source = Source("experiment5/gain.csv", {"representation": "value"}, "level")
        assert write_value(read_value(root, source), 4) == "0.5864"
        assert write_value(f"{0.5864499902:.6f}", 4) == "0.5865"

    def test_reductions_over_a_parquet_table(self, root):
        where = {"kind": "feature"}
        path = "preanalysis/shift.parquet"

        def value(reduce, keep="all", label=""):
            source = Source(path, where, "value", reduce=reduce, keep=keep, label=label)
            return read_value(root, source)

        assert value("count") == 4
        assert value("count", "positive") == 3
        assert value("count", "zero") == 1
        assert value("median", "positive") == pytest.approx(0.5)
        assert value("argmax", label="quantity") == "c"

    def test_a_difference_is_taken_at_full_precision(self, root):
        path = "experiment1/levels.csv"
        source = Source(
            path,
            {"predictors": "all"},
            "r2_pre_median",
            minus=Source(path, {"predictors": "invariant"}, "r2_pre_median"),
        )
        assert read_value(root, source) == pytest.approx(0.8721090972 - 0.4622238241)
        assert write_value(read_value(root, source), 3) == "0.410"

    def test_a_selection_of_several_rows_without_a_reduction_is_an_error(self, root):
        source = Source("preanalysis/shift.parquet", {"kind": "feature"}, "value")
        with pytest.raises(LookupError, match="selects 4 rows"):
            read_value(root, source)

    def test_only_the_one_root_is_read(self, tmp_path):
        """No second root stands in for a file the root does not hold."""
        empty = tmp_path / "empty"
        empty.mkdir()
        source = Source("experiment1/levels.csv", {"predictors": "all"}, "predictors")
        with pytest.raises(LookupError, match="does not hold"):
            read_value(empty, source)


class TestCheckingADocument:
    @pytest.fixture
    def repository(self, tmp_path) -> Path:
        (tmp_path / "doc.md").write_text(DOCUMENT, encoding="utf-8")
        return tmp_path

    @staticmethod
    def quote(text: str = "| G1 | {} | {} |") -> Quote:
        sources = tuple(
            Source("experiment1/levels.csv", {"predictors": "all"}, column)
            for column in ("r2_pre_median", "r2_post_median")
        )
        return Quote("doc.md", "2.1", text, sources)

    def test_a_document_that_writes_the_files_values_agrees(self, root, repository):
        assert check_values(self.quote(), root, repository) is None

    def test_a_changed_value_is_named(self, root, repository):
        changed = DOCUMENT.replace("0.117", "0.118")
        (repository / "doc.md").write_text(changed, encoding="utf-8")
        problem = check_values(self.quote(), root, repository)
        assert problem is not None
        assert "| G1 | 0.872 | 0.117 |" in problem

    def test_a_quote_outside_its_section_is_not_found(self, root, repository):
        quote = Quote("doc.md", "1. Design", "| G1 | {} | {} |", self.quote().sources)
        assert check_structure(quote, repository) is not None

    def test_a_grouped_number_is_read_and_written_as_the_document_has_it(
        self, root, tmp_path
    ):
        (tmp_path / "doc.md").write_text(
            "## 2.1 Levels\n\n**14,400 rows** agree\n", encoding="utf-8"
        )
        (root / "experiment1" / "premises.csv").write_text(
            "check,block,n_compared\nanchor,spline/warpheld,14400\n", encoding="utf-8"
        )
        source = Source("experiment1/premises.csv", {"check": "anchor"}, "n_compared")
        quote = Quote(
            "doc.md", "2.1", "**{} rows** agree", (source,), digits=0, grouped=True
        )
        assert check_structure(quote, tmp_path) is None
        assert check_values(quote, root, tmp_path) is None

    def test_a_doubled_brace_is_a_literal_one(self, root, tmp_path):
        (tmp_path / "doc.md").write_text(
            "## 2.1 Levels\n\n$R^2_\\text{pre} = 0.872$\n", encoding="utf-8"
        )
        source = Source(
            "experiment1/levels.csv", {"predictors": "all"}, "r2_pre_median"
        )
        quote = Quote("doc.md", "2.1", "$R^2_\\text{{pre}} = {}$", (source,))
        assert check_values(quote, root, tmp_path) is None

    def test_a_missing_file_is_named(self, repository, tmp_path):
        problem = check_values(self.quote(), tmp_path / "nowhere", repository)
        assert problem is not None
        assert "does not hold" in problem


def test_the_command_line_reports_agreement(capsys, monkeypatch):
    monkeypatch.setattr(checker, "QUOTES", [])
    assert checker.main(["--structure"]) == 0
    assert "all agree" in capsys.readouterr().out


class TestFurtherRoots:
    def test_a_file_the_root_lacks_is_read_from_a_further_root(
        self, tmp_path, monkeypatch
    ):
        root, appendix = tmp_path / "paper", tmp_path / "appendix"
        (appendix / "appendix_x").mkdir(parents=True)
        root.mkdir()
        pd.DataFrame({"k": ["a"], "v": ["0.5"]}).to_csv(
            appendix / "appendix_x" / "t.csv", index=False
        )
        source = Source("appendix_x/t.csv", {"k": "a"}, "v")
        with pytest.raises(LookupError):
            read_value(root, source)
        monkeypatch.setattr(checker, "ALSO_ROOTS", [appendix])
        assert read_value(root, source) == "0.5"

    def test_the_root_comes_first(self, tmp_path, monkeypatch):
        for name, value in (("paper", "1"), ("appendix", "2")):
            (tmp_path / name / "e").mkdir(parents=True)
            pd.DataFrame({"v": [value]}).to_csv(
                tmp_path / name / "e/t.csv", index=False
            )
        monkeypatch.setattr(checker, "ALSO_ROOTS", [tmp_path / "appendix"])
        assert read_value(tmp_path / "paper", Source("e/t.csv", {}, "v")) == "1"
