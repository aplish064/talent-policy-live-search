from __future__ import annotations

from datetime import date, datetime, timezone

from talent_activity_search.models import PolicyCard


def rank_policies(policies: list[PolicyCard]) -> list[PolicyCard]:
    scored = [policy.model_copy(update={"score": _score_policy(policy)}) for policy in policies]
    return sorted(
        scored,
        key=lambda policy: (-policy.score, policy.title.casefold(), policy.official_url),
    )


def _score_policy(policy: PolicyCard) -> float:
    official_url_score = 1.0 if policy.official_url else 0.0
    confidence_score = _clamp(policy.confidence)
    completeness_score = _clamp(policy.completeness)
    freshness_score = _freshness_score(policy.dates.published_date)
    evidence_score = min(len([text for text in policy.evidence_snippets if text.strip()]), 3) / 3

    score = (
        0.25 * official_url_score
        + 0.35 * confidence_score
        + 0.25 * completeness_score
        + 0.10 * freshness_score
        + 0.05 * evidence_score
    )
    return round(score, 4)


def _freshness_score(published_date: str | None) -> float:
    if not published_date:
        return 0.0

    parsed_date = _parse_date(published_date)
    if parsed_date is None:
        return 0.0

    today = datetime.now(timezone.utc).date()
    age_days = max((today - parsed_date).days, 0)
    return _clamp(1.0 - age_days / (5 * 365))


def _parse_date(value: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
        return parsed.date()
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))
