"""Tests for copilot.transcript: reading .txt/.pdf uploads and splitting prepared remarks from Q&A."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter

from copilot import transcript
from copilot.transcript import read_transcript, split_transcript

SAMPLE = Path(__file__).parent / "fixtures" / "sample_transcript.txt"


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a minimal valid PDF by hand: one Helvetica text line per string, one list per page."""
    font_id = 3 + 2 * len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()]
    for i, lines in enumerate(pages):
        ops = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1") if lines else b""
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, body))
    xref_at = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at))
    return out.getvalue()


def encrypted_pdf(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf([["Northwind Robotics call"]]))))
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# --- read_transcript: text files ---------------------------------------------------------------

def test_reads_utf8():
    text = "CEO \u2014 Revenue grew 18% at Caf\xe9 Robotics, a \u201crecord\u201d quarter."
    assert read_transcript("call.txt", text.encode("utf-8")) == text


def test_reads_utf8_with_bom():
    assert read_transcript("call.txt", b"\xef\xbb\xbfOperator: Good morning.") == "Operator: Good morning."


def test_reads_cp1252():
    text = "CFO: \u201cRecord\u201d margins \u2013 up 120 basis points."
    data = text.encode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")  # make sure this really exercises the fallback
    assert read_transcript("call.txt", data) == text


def test_bytes_undefined_in_cp1252_fall_back_to_latin1():
    assert read_transcript("call.txt", b"Margin \x81 expanded") == "Margin \x81 expanded"


def test_reads_utf16_with_bom():
    text = "Operator: Welcome to the call."
    assert read_transcript("call.txt", text.encode("utf-16")) == text


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_reads_utf16_without_bom(encoding):
    text = "Apple call text\nCFO: Revenue grew 6%."
    assert read_transcript("call.txt", text.encode(encoding)) == text


def test_normalizes_whitespace_and_keeps_line_breaks():
    data = "  Operator:\xa0 Good\tmorning.  \r\n\r\n\r\nCEO:  Thanks\u200b all.\n".encode("utf-8")
    assert read_transcript("CALL.TXT", data) == "Operator: Good morning.\nCEO: Thanks all."


# --- read_transcript: PDFs ---------------------------------------------------------------------

def test_reads_text_pdf():
    data = make_pdf([["Northwind Robotics Q3 call", "Operator: Good morning (everyone)."],
                     ["Page two: guidance raised."]])
    assert "Northwind Robotics Q3 call" in PdfReader(io.BytesIO(data)).pages[0].extract_text()  # valid PDF
    assert read_transcript("Call.PDF", data).splitlines() == [
        "Northwind Robotics Q3 call", "Operator: Good morning (everyone).", "Page two: guidance raised."]


def test_pdf_without_text_mentions_scanned_pdfs():
    with pytest.raises(ValueError, match="scanned PDF"):
        read_transcript("scan.pdf", make_pdf([[]]))


@pytest.mark.parametrize("data", [b"this is not a pdf at all " * 20, make_pdf([["Hello"]])[:80]],
                         ids=["garbage", "truncated"])
def test_unreadable_pdf(data):
    with pytest.raises(ValueError, match="PDF"):
        read_transcript("call.pdf", data)


def test_password_protected_pdf():
    with pytest.raises(ValueError, match="password-protected"):
        read_transcript("call.pdf", encrypted_pdf(user_password="secret"))


def test_pdf_with_only_an_owner_password_opens():
    assert read_transcript("call.pdf", encrypted_pdf(user_password="")) == "Northwind Robotics call"


# --- read_transcript: rejected uploads ---------------------------------------------------------

@pytest.mark.parametrize("filename", ["call.docx", "call", "notes.md", "call.txt.exe", ""])
def test_unsupported_extension(filename):
    with pytest.raises(ValueError, match=r"\.txt or \.pdf"):
        read_transcript(filename, b"Operator: Good morning.")


@pytest.mark.parametrize("filename,data", [("call.txt", b""), ("call.txt", b" \r\n\t "), ("call.pdf", b"")])
def test_empty_file(filename, data):
    with pytest.raises(ValueError, match="empty"):
        read_transcript(filename, data)


def test_file_with_only_a_byte_order_mark():
    with pytest.raises(ValueError, match="No readable text"):
        read_transcript("call.txt", b"\xef\xbb\xbf")


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(transcript, "MAX_UPLOAD_MB", 1)
    with pytest.raises(ValueError, match="larger than 1 MB"):
        read_transcript("call.txt", b"a" * (1024 * 1024 + 1))


# --- split_transcript --------------------------------------------------------------------------

PREPARED = "\n".join([
    "Operator: Welcome to the call. After the prepared remarks there will be a question-and-answer session.",
    "CEO: Revenue grew 18% this quarter on record shipments, and recurring software revenue grew 31%.",
    "CFO: For the fourth quarter we expect revenue between $430 million and $445 million.",
])
QA = "\n".join([
    "Analyst: Thanks. How should we think about pricing and tariffs next year?",
    "CEO: We expect hardware pricing to stay tight, but software attach keeps improving.",
    "Analyst: And why did receivables grow faster than revenue this quarter?",
    "CFO: Mostly timing; several large sites were accepted in the last two weeks of the quarter.",
])


def labels(parts: list[tuple[str, str]]) -> list[str]:
    return [label for label, _ in parts]


@pytest.mark.parametrize("heading", [
    "Question-and-Answer Session", "QUESTIONS AND ANSWERS", "Q&A", "Q & A Session:", "Questions & Answers:",
    "--- Q&A ---", "Question \u2013 and \u2013 Answer Section", "[Q&A]",
])
def test_split_on_qa_heading(heading):
    parts = split_transcript(f"{PREPARED}\n{heading}\n{QA}")
    assert labels(parts) == ["Prepared remarks", "Q&A"]
    assert parts[0][1] == PREPARED  # the intro's mention of a Q&A session is not a split point
    assert parts[1][1] == f"{heading}\n{QA}"


def test_split_on_operator_handoff_without_heading():
    handoff = "Operator: Thank you. Our first question comes from Jane Doe with Example Research."
    parts = split_transcript(f"{PREPARED}\n{handoff}\n{QA}")
    assert parts == [("Prepared remarks", PREPARED), ("Q&A", f"{handoff}\n{QA}")]


def test_bare_operator_label_stays_with_the_handoff():
    parts = split_transcript(f"{PREPARED}\nOperator\nThank you. Our first question comes from Jane Doe.\n{QA}")
    assert parts[1][1].startswith("Operator\nThank you. Our first question")


def test_heading_wins_over_an_earlier_first_question_mention():
    prepared = PREPARED + "\nCEO: The first question investors ask us is about demand, and demand is strong."
    parts = split_transcript(f"{prepared}\nQ&A Session\n{QA}")
    assert parts[0][1] == prepared
    assert parts[1][1].startswith("Q&A Session")


def test_table_of_contents_heading_is_skipped():
    contents = "Contents\nPrepared Remarks\nQuestions and Answers\nCall Participants"
    parts = split_transcript(f"{contents}\n{PREPARED}\nQuestions and Answers\n{QA}")
    assert parts[0][1] == f"{contents}\n{PREPARED}"
    assert parts[1][1] == f"Questions and Answers\n{QA}"


def test_no_qa_found_returns_full_transcript():
    assert split_transcript(PREPARED) == [("Full transcript", PREPARED)]


def test_short_qa_side_returns_full_transcript():
    text = f"{PREPARED}\nQ&A\nAnalyst: Just one question?"
    assert split_transcript(text) == [("Full transcript", text)]


def test_empty_text():
    assert split_transcript("") == [("Full transcript", "")]


def test_sample_transcript_fixture():
    text = read_transcript(SAMPLE.name, SAMPLE.read_bytes())
    assert "fictional company" in text.splitlines()[0]
    parts = split_transcript(text)
    assert labels(parts) == ["Prepared remarks", "Q&A"]
    prepared, qa = parts[0][1], parts[1][1]
    assert "there will be a question-and-answer session" in prepared
    assert "$430 million and $445 million" in prepared
    assert qa.startswith("Question-and-Answer Session\nOperator\nThank you. Our first question")
    assert "DSO went from 55 to 68 days" in qa
