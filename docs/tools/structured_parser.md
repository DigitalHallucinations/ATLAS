## Structured Parser Tool

The structured parser is a shared tool that turns PDFs, Word documents, HTML,
CSV files, and common image formats into normalized text, tables, and metadata.
It surfaces a consistent schema so personas can build ingestion or summarization
workflows without juggling format-specific libraries.

### Supported Formats

* **PDF** via `pdfminer.six`
* **DOCX** via `python-docx`
* **HTML** using `beautifulsoup4` when available (falls back to a lightweight
  built-in parser for text-only extraction)
* **CSV** using Python's standard `csv` module
* **Images** (`png`, `jpg`, `jpeg`, `tiff`, `bmp`, `gif`) via `pytesseract` and
  Pillow for OCR

### Dependencies

All dependencies are optional and validated at runtime. If a required package is
missing, the tool raises a clear `StructuredParserDependencyError` so personas
can communicate the requirement. Install the libraries you need:

```bash
pip install pdfminer.six python-docx beautifulsoup4 pillow pytesseract
```

### Size Limits

Invocations enforce a configurable byte limit (5 MiB by default) to prevent
excessive resource usage. Override the ceiling by setting the
`ATLAS_STRUCTURED_PARSER_MAX_BYTES` environment variable when launching ATLAS.

### Response Schema

The tool returns a dictionary with up to three keys depending on the
`target_formats` request parameter:

* `text`: concatenated body text extracted from the document.
* `tables`: a list of tables, each represented by a `rows` matrix and optional
  metadata (such as column headers).
* `metadata`: format-specific context including byte size, extraction engine,
  detected titles, and basic OCR details.

Personas can request any subset of the outputs by passing
`target_formats: ["text", "metadata"]`, for example.
