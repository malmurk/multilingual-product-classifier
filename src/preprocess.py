"""Training-time text assembly. MUST stay in sync with
classifier/preprocessor.py so the model sees the same string format in
dev and prod.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


_NON_WORD_RE = re.compile(r"[^\w\s\-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def clean_text(text: Any) -> str:
    """Lowercase, strip special chars, preserve Cyrillic, Latin, digits."""
    if text is None:
        return ""
    text = str(text).lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def price_band(price: Any) -> Optional[str]:
    """Coarse price bucket emitted as a plain-text token.
    Separates e.g. 30-lei cables from 30,000-lei TVs without leaking
    exact prices or overfitting.
    """
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    if p < 50:    return "цена_дешево"
    if p < 200:   return "цена_низкая"
    if p < 1000:  return "цена_средняя"
    if p < 5000:  return "цена_высокая"
    return "цена_премиум"


def build_input_text(
    title: str,
    attributes: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    brand: Optional[str] = None,
    extra_fields: Optional[Iterable[Any]] = None,
    description_chars: int = 200,
) -> str:
    """Concatenate available product fields into one classifier input string.

    Format:  title | brand | attr_value_1 | ... | extra_1 | ... | description[:200]
    Missing fields are skipped silently.
    """
    parts = []
    t = clean_text(title)
    if t:
        parts.append(t)

    if brand:
        b = clean_text(brand)
        if b:
            parts.append(b)

    if attributes:
        for value in attributes.values():
            v = clean_text(value)
            if v:
                parts.append(v)

    if extra_fields:
        for value in extra_fields:
            v = clean_text(value)
            if v:
                parts.append(v)

    if description:
        d = clean_text(str(description)[:description_chars])
        if d:
            parts.append(d)

    return " | ".join(parts)
