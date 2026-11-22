"""
STEP 2 - TEXT EXTRACTION  (pure Python, no AI)

Reads a PDF, DOCX or TXT file and returns clean plain text.
This file is fully independent: it does not import any other project file.

Main function:
    extract_text(file_bytes, file_name) -> (text, warning)
    clean_text(text) -> text
"""

import io
import re

MIN_WORDS = 30          # below this, the text is probably unreadable
MAX_WORDS = 15000       # above this, the file is probably not a resume/JD
MAX_FILE_MB = 5


def clean_text(text):
    """Remove odd symbols and extra spaces, keep line breaks."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace common bullet symbols with a simple dash
    text = re.sub(r"[•●▪■◆►✓✔➢➤○◦·]", "-", text)
    # Remove other control characters
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than two blank lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _read_pdf(file_bytes):
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(file_bytes):
    import docx
    document = docx.Document(io.BytesIO(file_bytes))
    parts = [p.text for p in document.paragraphs]
    # Resumes often keep text inside tables, so read those too
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _read_txt(file_bytes):
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text(file_bytes, file_name):
    """
    Returns (text, warning).
    text    -> clean text ("" if it failed)
    warning -> "" if all fine, otherwise a message to show the user
    """
    if not file_bytes:
        return "", "The file is empty."

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return "", f"File is too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_MB} MB."

    name = (file_name or "").lower()
    try:
        if name.endswith(".pdf"):
            raw = _read_pdf(file_bytes)
        elif name.endswith(".docx"):
            raw = _read_docx(file_bytes)
        elif name.endswith(".txt"):
            raw = _read_txt(file_bytes)
        else:
            return "", "Unsupported file type. Please upload PDF, DOCX or TXT."
    except Exception as error:
        return "", f"Could not read the file: {error}"

    text = clean_text(raw)
    words = len(text.split())

    if words == 0:
        return "", ("No text found. The PDF may be a scanned image. "
                    "Please upload a text-based PDF or a DOCX file.")
    if words < MIN_WORDS:
        return text, f"Very little text found ({words} words). Results may be weak."
    if words > MAX_WORDS:
        return text, f"The document is very long ({words} words). Only the first part will be analysed."
    return text, ""


def check_pasted_text(text, label="Text"):
    """Validate pasted text. Returns (clean_text, warning)."""
    text = clean_text(text)
    words = len(text.split())
    if words == 0:
        return "", f"{label} is empty."
    if words < MIN_WORDS:
        return text, f"{label} is very short ({words} words). Results may be weak."
    return text, ""
