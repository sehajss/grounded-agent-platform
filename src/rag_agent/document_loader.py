from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def extract_document_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Use one of: {supported}")

    if not content:
        raise ValueError("The uploaded document is empty")

    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("The document exceeds the 10 MB upload limit")

    if extension == ".pdf":
        return _extract_pdf_text(content)

    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Text documents must use UTF-8 encoding") from exc

    if not text:
        raise ValueError("The document contains no text")

    return text


def _extract_pdf_text(content: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported")

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"## Page {page_number}\n\n{page_text}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read") from exc

    if not pages:
        raise ValueError("The PDF contains no extractable text")

    return "\n\n".join(pages)