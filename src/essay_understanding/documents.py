from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str
    source_kind: str
    provenance: dict[str, str]


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument: ...


class PlainTextParser(DocumentParser):
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(
            title=path.stem, text=path.read_text(encoding="utf-8"), source_kind="text",
            provenance={"source_path": str(path.resolve()), "parser": "plain_text"},
        )


class DoclingParser(DocumentParser):
    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise RuntimeError("Install the documents extra to parse this file type") from error
        self.converter = DocumentConverter()

    def parse(self, path: Path) -> ParsedDocument:
        result = self.converter.convert(path)
        return ParsedDocument(
            title=path.stem, text=result.document.export_to_markdown(),
            source_kind=path.suffix.lower().lstrip("."),
            provenance={"source_path": str(path.resolve()), "parser": "docling"},
        )


def parser_for(path: Path) -> DocumentParser:
    return PlainTextParser() if path.suffix.lower() in {".txt", ".md"} else DoclingParser()

