"""
Shared helpers for the Qwen2.5-VL label-robustness diagnostic.

DIAGNOSTIC ONLY — does not modify production parsers, A1–A5 prompts, or outputs.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.lvm.response_schema import parse_json_response

# Exact semantic definition bodies from canonical A1 (ablation_verification_prompts.py).
DEF_RELIABLE = (
    "The highlighted bounding box clearly contains a valid wild palm."
)
DEF_UNCERTAIN = (
    "Evidence is ambiguous due to occlusion, overlap, low image quality, "
    "partial visibility, or confusing vegetation."
)
DEF_UNRELIABLE = (
    "The highlighted object is not a palm or the detection is clearly incorrect."
)

CANONICAL_DECISION_BLOCK = """Decision definitions

Reliable
The highlighted bounding box clearly contains a valid wild palm.

Uncertain
Evidence is ambiguous due to occlusion, overlap, low image quality, partial visibility, or confusing vegetation.

Unreliable
The highlighted object is not a palm or the detection is clearly incorrect."""

CONSIDERED_RE = re.compile(
    r"A partially visible palm can still be considered Reliable if recognizable palm morphology is present\."
)
JSON_DECISION_RE = re.compile(
    r'"decision":\s*"Reliable \| Uncertain \| Unreliable"'
)
DECISION_BLOCK_RE = re.compile(
    r"Decision definitions\n\n"
    r"Reliable\n"
    r"The highlighted bounding box clearly contains a valid wild palm\.\n\n"
    r"Uncertain\n"
    r"Evidence is ambiguous due to occlusion, overlap, low image quality, "
    r"partial visibility, or confusing vegetation\.\n\n"
    r"Unreliable\n"
    r"The highlighted object is not a palm or the detection is clearly incorrect\.",
    re.MULTILINE,
)

CONDITION_DIRS = {
    "S1_semantic": "S1_semantic",
    "S2_neutral": "S2_neutral",
    "S3_token_permutation": "S3_token_permutation",
    "S4_order_permutation": "S4_order_permutation",
}

# Presentation order: list of (token, semantic_class, definition_body)
# semantic_class ∈ {Reliable, Uncertain, Unreliable}
S1_SPEC: list[tuple[str, str, str]] = [
    ("Reliable", "Reliable", DEF_RELIABLE),
    ("Uncertain", "Uncertain", DEF_UNCERTAIN),
    ("Unreliable", "Unreliable", DEF_UNRELIABLE),
]
S2_SPEC: list[tuple[str, str, str]] = [
    ("A", "Reliable", DEF_RELIABLE),
    ("B", "Uncertain", DEF_UNCERTAIN),
    ("C", "Unreliable", DEF_UNRELIABLE),
]
S3_SPEC: list[tuple[str, str, str]] = [
    ("B", "Reliable", DEF_RELIABLE),
    ("C", "Uncertain", DEF_UNCERTAIN),
    ("A", "Unreliable", DEF_UNRELIABLE),
]
# S4: same token→semantic map as S2, reversed presentation order
S4_SPEC: list[tuple[str, str, str]] = [
    ("C", "Unreliable", DEF_UNRELIABLE),
    ("B", "Uncertain", DEF_UNCERTAIN),
    ("A", "Reliable", DEF_RELIABLE),
]

CONDITION_SPECS = {
    "S1_semantic": S1_SPEC,
    "S2_neutral": S2_SPEC,
    "S3_token_permutation": S3_SPEC,
    "S4_order_permutation": S4_SPEC,
}

SEMANTIC_LABELS = ("Reliable", "Uncertain", "Unreliable")


@dataclass(frozen=True)
class ConditionConfig:
    key: str
    tokens: tuple[str, ...]
    token_to_semantic: dict[str, str]
    reliable_token: str  # for "considered X" line


def condition_config(key: str) -> ConditionConfig:
    spec = CONDITION_SPECS[key]
    token_to_semantic = {tok: sem for tok, sem, _ in spec}
    reliable_token = next(tok for tok, sem, _ in spec if sem == "Reliable")
    # For S4 presentation order differs, but mapping is still A/B/C as S2.
    if key == "S4_order_permutation":
        token_to_semantic = {"A": "Reliable", "B": "Uncertain", "C": "Unreliable"}
        reliable_token = "A"
    return ConditionConfig(
        key=key,
        tokens=tuple(tok for tok, _, _ in spec),
        token_to_semantic=token_to_semantic,
        reliable_token=reliable_token,
    )


def build_decision_block(spec: list[tuple[str, str, str]]) -> str:
    parts = ["Decision definitions", ""]
    for i, (token, _sem, body) in enumerate(spec):
        parts.append(token)
        parts.append(body)
        if i < len(spec) - 1:
            parts.append("")
    return "\n".join(parts)


def build_json_enum(spec: list[tuple[str, str, str]]) -> str:
    joined = " | ".join(tok for tok, _, _ in spec)
    return f'"decision": "{joined}"'


def transform_canonical_prompt(canonical: str, condition_key: str) -> str:
    """
    Build a condition prompt from an exact canonical A1 prompt.

    Only decision verbalizers / definition order / JSON enum are changed.
    """
    if not DECISION_BLOCK_RE.search(canonical):
        raise ValueError("Canonical prompt missing exact Decision definitions block.")
    if not CONSIDERED_RE.search(canonical):
        raise ValueError("Canonical prompt missing exact 'considered Reliable' sentence.")
    if not JSON_DECISION_RE.search(canonical):
        raise ValueError("Canonical prompt missing exact JSON decision enum.")

    if condition_key == "S1_semantic":
        return canonical

    spec = CONDITION_SPECS[condition_key]
    cfg = condition_config(condition_key)
    out = DECISION_BLOCK_RE.sub(build_decision_block(spec), canonical, count=1)
    out = CONSIDERED_RE.sub(
        f"A partially visible palm can still be considered {cfg.reliable_token} "
        f"if recognizable palm morphology is present.",
        out,
        count=1,
    )
    out = JSON_DECISION_RE.sub(build_json_enum(spec), out, count=1)
    return out


def normalize_for_parity(text: str, condition_key: str) -> str:
    """
    Map condition-specific verbalizers to semantic placeholders so that
    S1/S2/S3 should become identical after normalization (S4 differs only
    in definition-block order, which we re-sort for the order-insensitive check).
    """
    cfg = condition_config(condition_key)
    # Replace longest tokens first when semantic labels are used.
    tokens = sorted(cfg.token_to_semantic.keys(), key=len, reverse=True)
    out = text
    for tok in tokens:
        sem = cfg.token_to_semantic[tok]
        placeholder = f"__SEM_{sem.upper()}__"
        if condition_key == "S1_semantic":
            # Whole-word-ish replacements for multi-letter labels
            out = re.sub(rf"\b{re.escape(tok)}\b", placeholder, out)
        else:
            # A/B/C only appear as standalone decision tokens in controlled places.
            # Replace standalone line headers, "considered X", and JSON enum tokens.
            out = re.sub(
                rf"(?m)^({re.escape(tok)})$",
                placeholder,
                out,
            )
            out = out.replace(
                f"considered {tok} if",
                f"considered {placeholder} if",
            )
            out = out.replace(f'"{tok}"', f'"{placeholder}"')
            out = out.replace(f"{tok} |", f"{placeholder} |")
            out = out.replace(f"| {tok}", f"| {placeholder}")
    return out


def extract_decision_block(text: str) -> str:
    m = re.search(
        r"Decision definitions\n\n.*?(?=\n\n--------------------------------------------------)",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError("Could not extract Decision definitions block.")
    return m.group(0)


def sort_decision_block_by_semantic(block: str) -> str:
    """Reorder definition entries by semantic placeholder for order-insensitive compare."""
    body = block.replace("Decision definitions\n\n", "", 1)
    entries = re.split(r"\n\n+", body.strip())
    # Each entry: "TOKEN\nDEFINITION"
    keyed: list[tuple[str, str]] = []
    for entry in entries:
        lines = entry.split("\n", 1)
        if len(lines) != 2:
            raise ValueError(f"Malformed decision entry: {entry!r}")
        tok, defn = lines
        keyed.append((tok, defn))
    order = [
        "__SEM_RELIABLE__",
        "__SEM_UNCERTAIN__",
        "__SEM_UNRELIABLE__",
    ]
    by_tok = {t: d for t, d in keyed}
    missing = [t for t in order if t not in by_tok]
    if missing:
        raise ValueError(f"Missing semantic entries after normalize: {missing}; got {list(by_tok)}")
    rebuilt = ["Decision definitions", ""]
    for i, tok in enumerate(order):
        rebuilt.append(tok)
        rebuilt.append(by_tok[tok])
        if i < len(order) - 1:
            rebuilt.append("")
    return "\n".join(rebuilt)


def strip_decision_block(text: str) -> tuple[str, str]:
    block = extract_decision_block(text)
    stripped = text.replace(block, "__DECISION_BLOCK__", 1)
    return stripped, block


_JSON_ENUM_RE = re.compile(r'"decision":\s*"([^"]+)"')


def canonicalize_order_sensitive_fields(text: str) -> str:
    """
    For S2↔S4 parity: sort decision-block entries and JSON enum tokens by
    semantic placeholder so only unintended residual diffs remain.
    """
    rest, block = strip_decision_block(text)
    sorted_block = sort_decision_block_by_semantic(block)

    def _sort_enum(match: re.Match[str]) -> str:
        parts = [p.strip() for p in match.group(1).split("|")]
        order = [
            "__SEM_RELIABLE__",
            "__SEM_UNCERTAIN__",
            "__SEM_UNRELIABLE__",
        ]
        if set(parts) != set(order):
            raise ValueError(f"Unexpected JSON enum tokens: {parts}")
        return f'"decision": "{" | ".join(order)}"'

    rest_sorted = _JSON_ENUM_RE.sub(_sort_enum, rest, count=1)
    return rest_sorted.replace("__DECISION_BLOCK__", sorted_block, 1)


def unified_diff(a: str, b: str, fromfile: str, tofile: str) -> str:
    return "".join(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def audit_prompt_parity(
    prompts_by_condition: dict[str, str],
) -> dict[str, object]:
    """
    Verify controlled differences across S1–S4 for one sample's prompts.

    Returns a report dict with ok flag and diffs.
    """
    required = list(CONDITION_SPECS)
    for key in required:
        if key not in prompts_by_condition:
            raise KeyError(f"Missing prompt for {key}")

    s1 = prompts_by_condition["S1_semantic"]
    s2 = prompts_by_condition["S2_neutral"]
    s3 = prompts_by_condition["S3_token_permutation"]
    s4 = prompts_by_condition["S4_order_permutation"]

    n1 = normalize_for_parity(s1, "S1_semantic")
    n2 = normalize_for_parity(s2, "S2_neutral")
    n3 = normalize_for_parity(s3, "S3_token_permutation")
    n4 = normalize_for_parity(s4, "S4_order_permutation")

    # S1/S2/S3: fully identical after verbalizer normalization
    s1_s2_ok = n1 == n2
    s2_s3_ok = n2 == n3

    # S2 vs S4: only presentation order of definitions + JSON enum differs.
    s2_rest, s2_block = strip_decision_block(n2)
    s4_rest, s4_block = strip_decision_block(n4)
    s2_canon = canonicalize_order_sensitive_fields(n2)
    s4_canon = canonicalize_order_sensitive_fields(n4)
    s2_s4_order_canon_identical = s2_canon == s4_canon
    s2_s4_order_actually_differs = (s2_block != s4_block) or (s2_rest != s4_rest)
    s2_s4_ok = s2_s4_order_canon_identical and s2_s4_order_actually_differs

    # Raw diffs should be non-empty for intended changes
    raw_s1_s2 = unified_diff(s1, s2, "S1", "S2")
    raw_s2_s3 = unified_diff(s2, s3, "S2", "S3")
    raw_s2_s4 = unified_diff(s2, s4, "S2", "S4")

    ok = s1_s2_ok and s2_s3_ok and s2_s4_ok and bool(raw_s1_s2) and bool(raw_s2_s3) and bool(
        raw_s2_s4
    )

    return {
        "ok": ok,
        "s1_s2_normalized_identical": s1_s2_ok,
        "s2_s3_normalized_identical": s2_s3_ok,
        "s2_s4_order_canon_identical": s2_s4_order_canon_identical,
        "s2_s4_order_actually_differs": s2_s4_order_actually_differs,
        # Back-compat keys used by prepare report
        "s2_s4_rest_identical": s2_s4_order_canon_identical,
        "s2_s4_blocks_semantically_equal": sort_decision_block_by_semantic(s2_block)
        == sort_decision_block_by_semantic(s4_block),
        "diff_S1_S2": raw_s1_s2,
        "diff_S2_S3": raw_s2_s3,
        "diff_S2_S4": raw_s2_s4,
        "normalized_diff_S1_S2": unified_diff(n1, n2, "S1_norm", "S2_norm"),
        "normalized_diff_S2_S3": unified_diff(n2, n3, "S2_norm", "S3_norm"),
        "order_canon_diff_S2_S4": unified_diff(s2_canon, s4_canon, "S2_canon", "S4_canon"),
    }


def parse_condition_response(raw_text: str, condition_key: str) -> dict[str, str]:
    """
    Probe-local parser (does not use production decision normalizer for ABC tokens).

    Uses shared JSON extraction only (parse_json_response), then validates
    condition-specific decision tokens locally.
    """
    cfg = condition_config(condition_key)
    parsed = parse_json_response(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response must be a JSON object.")

    decision = str(parsed.get("decision", "")).strip()
    allowed = set(cfg.token_to_semantic.keys())
    if decision not in allowed:
        raise ValueError(
            f"decision must be one of [{', '.join(sorted(allowed))}], got {decision!r}"
        )

    mapped = cfg.token_to_semantic[decision]
    return {
        "raw_decision_token": decision,
        "mapped_semantic_decision": mapped,
        "confidence_reasoning": str(parsed.get("confidence_reasoning") or "").strip(),
        "visual_reasoning": str(parsed.get("visual_reasoning") or "").strip(),
    }


def expected_sample_ids() -> list[str]:
    """Exact InternVL label-semantics 20 sample_ids (manifest order)."""
    return [
        "sample_000071",
        "sample_000105",
        "sample_000163",
        "sample_000165",
        "sample_000174",
        "sample_000189",
        "sample_000190",
        "sample_000208",
        "sample_000210",
        "sample_000216",
        "sample_000019",
        "sample_000025",
        "sample_000036",
        "sample_000074",
        "sample_000077",
        "sample_000088",
        "sample_000093",
        "sample_000129",
        "sample_000130",
        "sample_000140",
    ]


def assert_no_unintended_label_residue(text: str, condition_key: str) -> None:
    """For ABC conditions, ensure semantic labels are fully replaced."""
    if condition_key == "S1_semantic":
        return
    for label in ("Unreliable", "Uncertain", "Reliable"):
        if re.search(rf"\b{label}\b", text):
            raise ValueError(
                f"{condition_key}: residual semantic label {label!r} remains in prompt."
            )
