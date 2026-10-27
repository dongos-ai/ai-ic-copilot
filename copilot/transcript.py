"""Read an uploaded earnings-call transcript (.txt or .pdf) and split prepared remarks from Q&A."""
from __future__ import annotations

import codecs
import io
import re
import unicodedata

from pypdf import PdfReader

MAX_UPLOAD_MB = 20
MIN_PART_CHARS = 200  # each side of a prepared-remarks / Q&A split must be at least this long

# A short heading line that opens the Q&A, e.g. "Question-and-Answer Session", "Q & A", "Questions and Answers:".
_QA_HEADING = re.compile(
    r"(q\s*&\s*a|q\s+and\s+a|questions?[\s\-\u2013\u2014]*(and|&)[\s\-\u2013\u2014]*answers?)"
    r"(\s+(session|section|portion))?",
    re.IGNORECASE,
)
_HEADING_DECORATION = " \t-\u2013\u2014=*#:.[]()"  # stripped before matching, e.g. "--- Q&A ---", "[Q&A]"
# The operator handing over to the first analyst, e.g. "Our first question comes from ...".
_QA_HANDOFF = re.compile(r"\bfirst question\b", re.IGNORECASE)


def read_transcript(filename: str, data: bytes) -> str:
    """Return the cleaned text of an uploaded .txt or .pdf transcript.

    Raises ValueError with a message that is safe to show in the UI.
    """
    name = (filename or "").lower()
    if not name.endswith((".txt", ".pdf")):
        raise ValueError(f"Unsupported file type for '{filename}'. Upload the transcript as a .txt or .pdf file.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"'{filename}' is larger than {MAX_UPLOAD_MB} MB. Upload a smaller transcript file.")
    if not data.strip():
        raise ValueError(f"'{filename}' is empty.")

    raw = _decode_text(data) if name.endswith(".txt") else _pdf_text(data)
    text = _clean_text(raw)
    if not text:
        raise ValueError(f"No readable text found in '{filename}'.")
    return text


def split_transcript(text: str) -> list[tuple[str, str]]:
    """Split a transcript into [("Prepared remarks", ...), ("Q&A", ...)].

    Falls back to [("Full transcript", text)] when no Q&A start is found or either side
    would be shorter than MIN_PART_CHARS.
    """
    lines = text.splitlines()
    for start in _qa_start_candidates(lines):
        prepared = "\n".join(lines[:start]).strip()
        qa = "\n".join(lines[start:]).strip()
        if len(prepared) >= MIN_PART_CHARS and len(qa) >= MIN_PART_CHARS:
            return [("Prepared remarks", prepared), ("Q&A", qa)]
    return [("Full transcript", text)]


def _qa_start_candidates(lines: list[str]) -> list[int]:
    """Line indexes where the Q&A may start: heading lines first, then operator hand-offs."""
    headings = [i for i, line in enumerate(lines) if _QA_HEADING.fullmatch(line.strip(_HEADING_DECORATION))]
    handoffs = []
    for i, line in enumerate(lines):
        if _QA_HANDOFF.search(line):
            # Keep a bare "Operator" speaker label with the line it introduces.
            if i > 0 and lines[i - 1].strip(" :").lower() == "operator":
                i -= 1
            handoffs.append(i)
    return headings + handoffs


def _decode_text(data: bytes) -> str:
    """Decode a text upload: UTF-16 (with or without BOM), UTF-8 (with or without BOM), Windows-1252, Latin-1."""
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    # Without a BOM, English UTF-16 text has a NUL byte next to every ASCII letter: "A\0" (LE) or "\0A" (BE).
    if data[1:2] == b"\x00":
        return data.decode("utf-16-le", errors="replace")
    if data[:1] == b"\x00":
        return data.decode("utf-16-be", errors="replace")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # never fails: every byte maps to a character


def _pdf_text(data: bytes) -> str:
    """Extract text from a PDF with pypdf, one page after another."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises several error types for damaged or non-PDF files
        raise ValueError("Could not open this PDF; the file looks damaged or is not a real PDF. "
                         "Try exporting it again, or upload a .txt version.") from exc

    if reader.is_encrypted:
        try:
            unlocked = bool(reader.decrypt(""))  # many "protected" PDFs open with an empty password
        except Exception:
            unlocked = False
        if not unlocked:
            raise ValueError("This PDF is password-protected. Remove the password or upload a .txt version.")

    try:
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError("Could not extract text from this PDF. Try uploading a .txt version.") from exc

    text = "\n".join(pages)
    if not text.strip():
        raise ValueError("This PDF has no extractable text (is it a scanned PDF?). "
                         "Export a text version of the transcript, or upload a .txt file.")
    return text


def _clean_text(text: str) -> str:
    """Normalize whitespace the same way as the 10-K text: one trimmed, non-empty line per line."""
    # NFKC turns PDF ligatures ("\ufb01" -> "fi") and odd spaces (\xa0, \u2009) into plain characters.
    text = unicodedata.normalize("NFKC", text).replace("\u200b", " ")
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)
