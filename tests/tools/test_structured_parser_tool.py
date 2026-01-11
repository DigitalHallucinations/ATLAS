"""Tests for the structured parser tool."""

from __future__ import annotations

import builtins
import sys
import types

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "modules" / "Tools" / "Base_Tools" / "structured_parser.py"

spec = importlib.util.spec_from_file_location("test_structured_parser_module", MODULE_PATH)
assert spec and spec.loader
structured_parser_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = structured_parser_module
spec.loader.exec_module(structured_parser_module)

DocumentTooLargeError = structured_parser_module.DocumentTooLargeError
InvalidInputError = structured_parser_module.InvalidInputError
StructuredParser = structured_parser_module.StructuredParser
StructuredParserDependencyError = structured_parser_module.StructuredParserDependencyError


@pytest.fixture()
def parser() -> StructuredParser:
    return StructuredParser(max_bytes=1024 * 1024)


def _install_fake_pdfminer(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("pdfminer")
    submodule = types.ModuleType("pdfminer.high_level")

    def fake_extract_text(stream):
        return "Hello from PDF"

    submodule.extract_text = fake_extract_text
    package.high_level = submodule

    monkeypatch.setitem(sys.modules, "pdfminer", package)
    monkeypatch.setitem(sys.modules, "pdfminer.high_level", submodule)


def _install_fake_docx(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("docx")

    class FakeCell:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeRow:
        def __init__(self, cells: list[str]) -> None:
            self.cells = [FakeCell(cell) for cell in cells]

    class FakeTable:
        def __init__(self, rows: list[list[str]]) -> None:
            self.rows = [FakeRow(row) for row in rows]

    class FakeDoc:
        def __init__(self) -> None:
            self.paragraphs = [types.SimpleNamespace(text="Intro"), types.SimpleNamespace(text="")]
            self.tables = [FakeTable([["H1", "H2"], ["R1", "R2"]])]

    def fake_document(_stream):
        return FakeDoc()

    module.Document = fake_document
    monkeypatch.setitem(sys.modules, "docx", module)


def test_parse_pdf_returns_text(monkeypatch: pytest.MonkeyPatch, parser: StructuredParser) -> None:
    _install_fake_pdfminer(monkeypatch)

    result = parser.parse(content=b"%PDF-1.4", source_format="pdf")

    assert result["text"] == "Hello from PDF"
    assert result["metadata"]["format"] == "pdf"


def test_parse_docx_returns_tables(monkeypatch: pytest.MonkeyPatch, parser: StructuredParser) -> None:
    _install_fake_docx(monkeypatch)

    result = parser.parse(content=b"FAKE-DOC", source_format="docx")

    assert "Intro" in result["text"]
    assert result["tables"][0]["rows"][0] == ["H1", "H2"]


def test_parse_html_without_beautifulsoup(monkeypatch: pytest.MonkeyPatch, parser: StructuredParser) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "bs4" or name.startswith("bs4."):
            raise ModuleNotFoundError("bs4 missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    html = b"<html><body><p>Hello <b>World</b></p></body></html>"
    result = parser.parse(content=html, source_format="html")

    assert result["text"] == "Hello World"
    assert result["metadata"]["extracted_with"] == "html.parser"


def test_parse_csv_yields_tables(parser: StructuredParser) -> None:
    csv_bytes = b"col1,col2\nval1,val2\n"
    result = parser.parse(content=csv_bytes, source_format="csv")

    assert result["tables"][0]["rows"] == [["col1", "col2"], ["val1", "val2"]]
    assert "val1, val2" in result["text"]


def test_parse_image_missing_dependencies(monkeypatch: pytest.MonkeyPatch, parser: StructuredParser) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PIL") or name.startswith("pytesseract"):
            raise ModuleNotFoundError("missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(StructuredParserDependencyError):
        parser.parse(content=b"fake", source_format="png")


def test_document_too_large(parser: StructuredParser) -> None:
    big_parser = StructuredParser(max_bytes=4)

    with pytest.raises(DocumentTooLargeError):
        big_parser.parse(content=b"12345", source_format="csv")


def test_invalid_arguments(parser: StructuredParser) -> None:
    with pytest.raises(InvalidInputError):
        parser.parse()

