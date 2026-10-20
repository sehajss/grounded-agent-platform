from __future__ import annotations

import re


_PROMPT_ATTACKS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|system|developer)\s+instructions\b", re.I),
    re.compile(r"\b(reveal|print|show|repeat)\s+(the\s+)?(system|developer)\s+prompt\b", re.I),
    re.compile(r"\b(show|print|reveal|return)\s+(me\s+)?(any\s+)?(secrets|credentials|api keys)\b", re.I),
)

_SECRET_PATTERNS = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I),
)

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


def screen_question(question: str) -> str | None:
    if "\x00" in question:
        return "The question contains invalid characters."

    if any(pattern.search(question) for pattern in _PROMPT_ATTACKS):
        return "I can help with documentation questions, but I can’t reveal prompts or secrets."

    return None


def validate_citation_markers(
    answer: str,
    citation_count: int,
) -> tuple[str, bool, list[int]]:
    unknown = sorted(
        {
            int(match.group(1))
            for match in _CITATION_MARKER.finditer(answer)
            if int(match.group(1)) < 1 or int(match.group(1)) > citation_count
        }
    )
    cleaned = _CITATION_MARKER.sub(
        lambda match: (
            match.group(0)
            if 1 <= int(match.group(1)) <= citation_count
            else ""
        ),
        answer,
    )
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
    return cleaned.strip(), not unknown, unknown


def redact_sensitive_output(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted