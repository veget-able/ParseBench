"""JSON subset matching metric for extract evaluation.

Ports json_subset_match_score from extract-tests with date normalization support.
"""

import re
from typing import Any

from autoevals.number import NumericDiff  # type: ignore[import-untyped]
from autoevals.string import EmbeddingSimilarity, Levenshtein  # type: ignore[import-untyped]
from dateutil import parser as date_parser  # type: ignore[import-untyped]


def normalize_date_string(date_str: str) -> str:
    """
    Normalize various date formats to a standard ISO format (YYYY-MM-DD).
    Returns the original string if it's not a recognizable date.

    :param date_str: Input date string
    :return: Normalized date string or original if not a date
    """
    if not isinstance(date_str, str):
        return date_str

    # Skip if it's clearly not a date (too short/long or contains non-date characters)
    if len(date_str) < 4 or len(date_str) > 50:
        return date_str

    # Skip strings that are just numbers (likely IDs, not dates)
    if date_str.isdigit():
        return date_str

    # Skip if it contains patterns that are unlikely to be dates
    # like very long numbers, special characters, etc.
    if re.search(r"\d{10,}", date_str):  # 10+ consecutive digits
        return date_str

    # Check for common date patterns first
    date_patterns = [
        r"\d{4}-\d{1,2}-\d{1,2}",  # YYYY-MM-DD
        r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
        r"\d{1,2}-\d{1,2}-\d{4}",  # MM-DD-YYYY
        r"[A-Za-z]+ \d{1,2},? \d{4}",  # Month DD, YYYY or Month DD YYYY
        r"[A-Za-z]+\.? [A-Za-z]+\.? \d{1,2},? \d{4}",  # Weekday Month DD YYYY
        r"\d{1,2} [A-Za-z]+ \d{4}",  # DD Month YYYY
    ]

    # Only try to parse if it matches common date patterns
    has_date_pattern = any(re.search(pattern, date_str) for pattern in date_patterns)
    if not has_date_pattern:
        return date_str

    try:
        # Try to parse the date
        parsed_date = date_parser.parse(date_str, fuzzy=False)
        # Return in ISO format (YYYY-MM-DD)
        return parsed_date.strftime("%Y-%m-%d")  # type: ignore[no-any-return]
    except (ValueError, TypeError):
        # If parsing fails, return original string
        return date_str


def _compute_score_with_weight(
    expected: Any,
    actual: Any,
    weighted: bool,
    case_sensitive: bool,
    cosine_similarity: bool,
    normalize_dates: bool,
    string_scorer: Any,
    number_scorer: Any,
) -> tuple[float, int]:
    """
    Recursively compute match score and weight.

    :param expected: Expected JSON structure
    :param actual: Actual JSON structure
    :param weighted: If True, aggregate by leaf node weights; if False, simple average
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings
    :param normalize_dates: Normalize date strings before comparison
    :param string_scorer: Scorer for string comparison
    :param number_scorer: Scorer for number comparison
    :return: (score, weight) where weight is the number of leaf nodes in expected
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)

        # Compute scores and weights for each key
        results: list[tuple[float, int]] = []
        for k in expected.keys():
            score, weight = _compute_score_with_weight(
                expected.get(k),
                actual.get(k),
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
            )
            results.append((score, weight))

        if not results:
            return (0.0, 1)

        total_weight = sum(w for _, w in results)
        # When weighted=False, treat each field as weight=1
        effective_weights = [w if weighted else 1 for _, w in results]
        total_eff_weight = sum(effective_weights)
        if total_eff_weight == 0:
            return (0.0, max(total_weight, 1))
        weighted_sum = sum(s * ew for (s, _), ew in zip(results, effective_weights, strict=True))
        agg_score = weighted_sum / total_eff_weight

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)
        if len(actual) == 0:
            # All expected items missing - compute total weight of expected
            total_weight = sum(
                _compute_score_with_weight(
                    e,
                    None,
                    weighted,
                    case_sensitive,
                    cosine_similarity,
                    normalize_dates,
                    string_scorer,
                    number_scorer,
                )[1]
                for e in expected
            )
            return (0.0, max(total_weight, 1))

        # Pair up elements by index
        min_len = min(len(expected), len(actual))
        list_results: list[tuple[float, int]] = []

        # Matched elements
        for i in range(min_len):
            score, weight = _compute_score_with_weight(
                expected[i],
                actual[i],
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
            )
            list_results.append((score, weight))

        # Missing expected elements (score = 0)
        for i in range(min_len, len(expected)):
            _, weight = _compute_score_with_weight(
                expected[i],
                None,
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
            )
            list_results.append((0.0, weight))

        if not list_results:
            return (0.0, 1)

        total_weight = sum(w for _, w in list_results)
        if weighted:
            # Weighted: each element contributes proportionally to its leaf count
            if total_weight == 0:
                return (0.0, 1)
            agg_score = sum(s * w for s, w in list_results) / total_weight
        else:
            # Unweighted: divide by max length to penalize extra items in actual
            agg_score = sum(s for s, _ in list_results) / max(len(expected), len(actual))

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, str):
        if not isinstance(actual, str):
            return (0.0, 1)

        expected_normalized = expected
        actual_normalized = actual

        if not case_sensitive:
            expected_normalized = expected_normalized.lower()
            actual_normalized = actual_normalized.lower()

        if normalize_dates:
            expected_normalized = normalize_date_string(expected_normalized)
            actual_normalized = normalize_date_string(actual_normalized)

        result = string_scorer.eval(expected_normalized, actual_normalized)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)):
            return (0.0, 1)
        result = number_scorer.eval(expected, actual)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif expected is None:
        if actual is None:
            return (1.0, 1)
        return (0.0, 1)

    else:
        # Type mismatch or unsupported type
        return (0.0, 1)


def json_subset_match_score(
    expected: Any,
    actual: Any,
    case_sensitive: bool = True,
    cosine_similarity: bool = False,
    normalize_dates: bool = True,
    weighted: bool = True,
) -> float:
    """
    Calculate similarity score between expected and actual JSON structures.

    Adapted from autoevals.JsonDiff to only test on the subset of keys within
    the expected json. This means extra keys in actual are ignored.

    :param expected: Expected JSON structure (dict, list, or primitive)
    :param actual: Actual JSON structure to compare
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings (slower but more semantic)
    :param normalize_dates: Normalize date strings before comparison
    :param weighted: If True (default), weight fields by their number of leaf nodes.
                     If False, use simple averaging (each field/element counts equally).
    :return: Similarity score between 0.0 and 1.0
    """
    string_scorer = Levenshtein() if not cosine_similarity else EmbeddingSimilarity()
    number_scorer = NumericDiff()

    score, _ = _compute_score_with_weight(
        expected=expected,
        actual=actual,
        weighted=weighted,
        case_sensitive=case_sensitive,
        cosine_similarity=cosine_similarity,
        normalize_dates=normalize_dates,
        string_scorer=string_scorer,
        number_scorer=number_scorer,
    )
    return score
