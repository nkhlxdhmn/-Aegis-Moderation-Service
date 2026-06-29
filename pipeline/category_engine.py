"""Category selection engine for the Aegis moderation pipeline.

This module maps model category scores to the single top heritage category that
is stored with a moderated post.
"""


def get_top_category(category_scores: dict[str, float]) -> tuple[str, float]:
    """Return the highest-scoring category name and confidence score."""

    if not category_scores:
        return "Uncategorized", 0.0

    return max(category_scores.items(), key=lambda item: item[1])

