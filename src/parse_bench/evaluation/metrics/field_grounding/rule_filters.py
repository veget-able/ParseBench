"""Helpers for selecting extract_field rules for evaluation."""

from collections.abc import Iterable

from parse_bench.test_cases.schema import ExtractFieldTestRule


def filter_extract_field_rules(
    rules: Iterable[ExtractFieldTestRule],
    *,
    verified_only: bool,
    require_bboxes: bool = False,
) -> list[ExtractFieldTestRule]:
    """Return extract_field rules matching evaluator-level rule filters."""
    filtered: list[ExtractFieldTestRule] = []
    for rule in rules:
        if verified_only and not rule.verified:
            continue
        if require_bboxes and not rule.bboxes:
            continue
        filtered.append(rule)
    return filtered


def verified_only_metadata(
    *,
    enabled: bool,
    input_rule_count: int,
    scored_rule_count: int,
) -> dict[str, object]:
    """Metadata added to metrics when verified-only rule filtering is active."""
    if not enabled:
        return {}
    return {
        "rule_filter": "verified_only",
        "verified_only": True,
        "input_rule_count": input_rule_count,
        "scored_rule_count": scored_rule_count,
    }
