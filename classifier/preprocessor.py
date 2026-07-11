"""Text cleaning + field concatenation for the classifier input.

Mirrors src/preprocess.py used during training so dev and prod see the
same string format. Keep these two files in sync.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s\-]", re.UNICODE)


def clean_text(text: str) -> str:
    """Lowercase, strip special chars, preserve Cyrillic/Latin/digits."""
    if not text:
        return ""
    text = str(text).lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def build_input_text(
    title: str,
    attributes: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
    brand: Optional[str] = None,
    extra_fields: Optional[Iterable[str]] = None,
    description_chars: int = 200,
) -> str:
    """Concatenate available product fields into one classifier input string.

    Format: "title | brand | attr_value_1 | attr_value_2 | description[:200]"
    Empty/missing fields are skipped silently — the model was trained on
    augmented data with random field dropping, so this is safe.
    """
    parts = []
    title_clean = clean_text(title)
    if title_clean:
        parts.append(title_clean)

    if brand:
        b = clean_text(brand)
        if b:
            parts.append(b)

    if attributes:
        for value in attributes.values():
            if value is None:
                continue
            v = clean_text(str(value))
            if v:
                parts.append(v)

    if extra_fields:
        for value in extra_fields:
            if value is None:
                continue
            v = clean_text(str(value))
            if v:
                parts.append(v)

    if description:
        snippet = str(description)[:description_chars]
        d = clean_text(snippet)
        if d:
            parts.append(d)

    return " | ".join(parts)
