"""
Evaluation gate for the Episodic Memory Constructor agent.

Validates LLM output against the expected schema before the pipeline proceeds.
Hard failures trigger a retry; soft failures log warnings but pass through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from omnimemory.core.utils import clean_and_parse_json

logger = logging.getLogger(__name__)

# ── Schema constants ────────────────────────────────────────────────────────

REQUIRED_TOP_LEVEL_KEYS = {"context", "behavioral_profile", "future_guidance"}

CONTEXT_REQUIRED_KEYS = {"available_data", "user_intent"}

BEHAVIORAL_PROFILE_REQUIRED_KEYS = {"communication", "learning"}

FUTURE_GUIDANCE_REQUIRED_KEYS = {"recommended_approaches"}

# Keys that are expected but whose absence is a soft failure only.
OPTIONAL_TOP_LEVEL_KEYS = {"what_worked", "what_failed", "interaction_insights"}

# Sub-field type expectations for hard checks.
_EXPECTED_DICT_FIELDS = {"context", "behavioral_profile", "future_guidance"}
_EXPECTED_DICT_OR_ABSENT_FIELDS = {"what_worked", "what_failed", "interaction_insights"}

# Soft-check length limits (word count) — mirrors the system prompt hints.
_MAX_SENTENCE_WORDS = 40  # "1 sentence" ≈ up to 40 words


# ── Result dataclass ────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """Result of an episodic agent output evaluation.

    Attributes:
        passed: True if no hard failures were found.
        hard_failures: Issues that should trigger a retry.
        soft_failures: Issues that are logged but don't block the pipeline.
    """

    passed: bool
    hard_failures: List[str] = field(default_factory=list)
    soft_failures: List[str] = field(default_factory=list)


# ── Public API ──────────────────────────────────────────────────────────────


def validate_episodic_output(raw_output: str) -> EvalResult:
    """Validate episodic agent LLM output against the expected schema.

    Args:
        raw_output: Raw string returned by the LLM.

    Returns:
        EvalResult with pass/fail status and categorised failure messages.
    """
    hard: List[str] = []
    soft: List[str] = []

    # ── Hard check 1: non-empty ──────────────────────────────────────────
    if not raw_output or not raw_output.strip():
        hard.append("Output is empty or whitespace-only")
        return EvalResult(passed=False, hard_failures=hard, soft_failures=soft)

    # ── Hard check 2: parseable JSON ─────────────────────────────────────
    try:
        data = clean_and_parse_json(raw_output)
    except (ValueError, TypeError) as exc:
        hard.append(f"JSON parsing failed: {exc}")
        return EvalResult(passed=False, hard_failures=hard, soft_failures=soft)

    if not isinstance(data, dict):
        hard.append(f"Top-level value is {type(data).__name__}, expected dict")
        return EvalResult(passed=False, hard_failures=hard, soft_failures=soft)

    # ── Hard check 3: required top-level keys ────────────────────────────
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
    if missing_keys:
        hard.append(f"Missing required top-level keys: {sorted(missing_keys)}")

    # ── Hard check 4: required fields are dicts ──────────────────────────
    for key in _EXPECTED_DICT_FIELDS:
        value = data.get(key)
        if value is not None and not isinstance(value, dict):
            hard.append(
                f"'{key}' should be a dict, got {type(value).__name__}"
            )

    # ── Hard check 5: required sub-keys ──────────────────────────────────
    _check_sub_keys(data, "context", CONTEXT_REQUIRED_KEYS, hard)
    _check_sub_keys(
        data, "behavioral_profile", BEHAVIORAL_PROFILE_REQUIRED_KEYS, hard
    )
    _check_sub_keys(data, "future_guidance", FUTURE_GUIDANCE_REQUIRED_KEYS, hard)

    # ── Hard check 6: recommended_approaches must be a list ──────────────
    guidance = data.get("future_guidance")
    if isinstance(guidance, dict):
        approaches = guidance.get("recommended_approaches")
        if approaches is not None and not isinstance(approaches, list):
            hard.append(
                f"'future_guidance.recommended_approaches' should be a list, "
                f"got {type(approaches).__name__}"
            )

    # ── Hard check 7: what_worked.strategies must be a list (if present) ─
    what_worked = data.get("what_worked")
    if isinstance(what_worked, dict):
        strategies = what_worked.get("strategies")
        if strategies is not None and not isinstance(strategies, list):
            hard.append(
                f"'what_worked.strategies' should be a list, "
                f"got {type(strategies).__name__}"
            )

    # ── Soft checks ──────────────────────────────────────────────────────
    _soft_check_optional_keys(data, soft)
    _soft_check_na_discipline(data, soft)
    _soft_check_strategy_limits(data, soft)

    passed = len(hard) == 0
    return EvalResult(passed=passed, hard_failures=hard, soft_failures=soft)


# ── Internal helpers ────────────────────────────────────────────────────────


def _check_sub_keys(
    data: Dict[str, Any],
    parent_key: str,
    required: set,
    hard: List[str],
) -> None:
    """Append hard failures for missing required sub-keys."""
    parent = data.get(parent_key)
    if not isinstance(parent, dict):
        return  # Already flagged as a type error above.
    missing = required - set(parent.keys())
    if missing:
        hard.append(
            f"'{parent_key}' missing required sub-keys: {sorted(missing)}"
        )


def _soft_check_optional_keys(data: Dict[str, Any], soft: List[str]) -> None:
    """Warn about missing optional top-level keys."""
    for key in OPTIONAL_TOP_LEVEL_KEYS:
        if key not in data:
            soft.append(f"Optional key '{key}' is absent")


def _soft_check_na_discipline(data: Dict[str, Any], soft: List[str]) -> None:
    """Warn if string fields are empty when they should be 'N/A'."""
    for section_key in ("context", "behavioral_profile", "interaction_insights"):
        section = data.get(section_key)
        if not isinstance(section, dict):
            continue
        for field_key, value in section.items():
            if isinstance(value, str) and value.strip() == "":
                soft.append(
                    f"'{section_key}.{field_key}' is empty string — should be 'N/A' "
                    f"if data is insufficient"
                )


def _soft_check_strategy_limits(data: Dict[str, Any], soft: List[str]) -> None:
    """Warn if strategy lists exceed the prompt-specified maximums."""
    what_worked = data.get("what_worked")
    if isinstance(what_worked, dict):
        strategies = what_worked.get("strategies", [])
        if isinstance(strategies, list) and len(strategies) > 3:
            soft.append(
                f"'what_worked.strategies' has {len(strategies)} items (max 3)"
            )

    what_failed = data.get("what_failed")
    if isinstance(what_failed, dict):
        strategies = what_failed.get("strategies", [])
        if isinstance(strategies, list) and len(strategies) > 2:
            soft.append(
                f"'what_failed.strategies' has {len(strategies)} items (max 2)"
            )

    guidance = data.get("future_guidance")
    if isinstance(guidance, dict):
        recommended = guidance.get("recommended_approaches", [])
        if isinstance(recommended, list) and len(recommended) > 3:
            soft.append(
                f"'future_guidance.recommended_approaches' has "
                f"{len(recommended)} items (max 3)"
            )
        avoid = guidance.get("avoid_approaches", [])
        if isinstance(avoid, list) and len(avoid) > 2:
            soft.append(
                f"'future_guidance.avoid_approaches' has "
                f"{len(avoid)} items (max 2)"
            )
