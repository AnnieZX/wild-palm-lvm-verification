"""Model-specific raw-response cleanup before shared parsing."""

from __future__ import annotations

# Ministral 3 (all sizes) emits otherwise valid JSON objects whose string values
# contain literal newlines/tabs (e.g. "visual_reasoning": "\nThe box ...\n"),
# which strict json.loads rejects as invalid control characters.
MINISTRAL3_MODEL_KEYS = frozenset({"ministral3_3b", "ministral3_8b", "ministral3_14b"})

_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def escape_control_chars_in_json_strings(text: str) -> str:
    """
    Escape raw control characters that occur inside JSON string literals.

    Equivalent to accepting what ``json.loads(..., strict=False)`` accepts:
    characters outside string literals, keys, values, quoting and structure are
    unchanged, and the decoded string content is identical (a literal newline
    decodes to the same newline). Nothing is inferred, added or removed.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif ord(char) < 0x20:
                out.append(_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
                continue
        elif char == '"':
            in_string = True
        out.append(char)
    return "".join(out)


def normalize_raw_response(raw_text: str, model_key: str = "") -> str:
    """
    Apply model-specific text cleanup before JSON extraction.

    Identity pass-through for every model except the Ministral 3 family, whose
    in-string control characters are escaped (see MINISTRAL3_MODEL_KEYS).
    """
    if model_key in MINISTRAL3_MODEL_KEYS:
        return escape_control_chars_in_json_strings(raw_text)
    return raw_text
