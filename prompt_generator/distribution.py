"""
Category-based question distribution system.

This module provides the CategoryDistributor class which implements
a deterministic round-robin algorithm for distributing questions across
videos while ensuring even usage of all question types within each category.
"""

from collections import defaultdict
from typing import TYPE_CHECKING

from .templates import (
    QuestionCategory,
    QuestionType,
    QUESTION_CATEGORIES,
    QUESTIONS_PER_CATEGORY,
    QUESTION_TEMPLATES,
)

if TYPE_CHECKING:
    from .generator import QuestionGenerator


class CategoryDistributor:
    """
    Distributes questions across videos using category-based round-robin selection.

    Ensures:
    - Even distribution of question types within each category
    - No duplicate question types per video
    - Deterministic results (same input -> same output)
    """

    def __init__(self):
        """Initialize the distributor with category groupings and rotation indices."""
        # Group question types by their category
        self.types_by_category = self._group_types_by_category()

        # Track rotation index per category (global across all videos)
        # This ensures even distribution as we process videos sequentially
        self.rotation_indices: dict[QuestionCategory, int] = {
            cat: 0 for cat in QuestionCategory
        }

    def _group_types_by_category(self) -> dict[QuestionCategory, list[QuestionType]]:
        """
        Group question types by their category.

        Returns:
            Dictionary mapping each category to its list of question types,
            sorted for deterministic behavior.
        """
        groups: dict[QuestionCategory, list[QuestionType]] = defaultdict(list)

        for qtype, category in QUESTION_CATEGORIES.items():
            groups[category].append(qtype)

        # Sort for determinism
        return {
            cat: sorted(types, key=lambda t: t.value)
            for cat, types in groups.items()
        }

    def select_questions_for_video(
        self,
        entry: dict,
        generator: "QuestionGenerator",
    ) -> list[QuestionType]:
        """
        Select question types for a single video using round-robin distribution.

        Args:
            entry: Annotation entry for the video
            generator: QuestionGenerator instance to validate field requirements

        Returns:
            List of selected QuestionType enums for this video

        Algorithm:
        1. For each category (Simple, Compound, Complex, Counting):
           - Determine how many questions needed from QUESTIONS_PER_CATEGORY
           - Use round-robin index to select next type in rotation
           - Skip if type already used for this video
           - Validate video has required fields for selected type
           - Increment rotation index (persists across all videos)
        2. Return list of selected question types
        """
        selected: list[QuestionType] = []
        used_types: set[QuestionType] = set()

        # Process each category in order
        for category in [
            QuestionCategory.SIMPLE,
            QuestionCategory.COMPOUND,
            QuestionCategory.COMPLEX,
            # QuestionCategory.COUNTING,
            QuestionCategory.IDENTIFICATION,
        ]:
            num_needed = QUESTIONS_PER_CATEGORY[category]
            available_types = self.types_by_category[category]

            # Round-robin selection within this category
            for _ in range(num_needed):
                attempts = 0
                max_attempts = len(available_types) * 2

                while attempts < max_attempts:
                    # Get next type in rotation (round-robin)
                    idx = self.rotation_indices[category] % len(available_types)
                    qtype = available_types[idx]
                    self.rotation_indices[category] += 1
                    attempts += 1

                    # Skip if already used for this video (no duplicates)
                    if qtype in used_types:
                        continue

                    # Check if video has required fields for this question type
                    if self._can_generate(entry, qtype, generator):
                        selected.append(qtype)
                        used_types.add(qtype)
                        break
                else:
                    # Fallback: couldn't find valid type after max attempts
                    # Log warning and skip this question
                    video_name = entry.get("file_name", entry.get("video_name", "unknown"))
                    print(
                        f"Warning: Could not find valid {category.value} question "
                        f"for {video_name}"
                    )

        return selected

    def _can_generate(
        self,
        entry: dict,
        qtype: QuestionType,
        generator: "QuestionGenerator",
    ) -> bool:
        """
        Check if a question type can be generated for this entry.

        Args:
            entry: Annotation entry for the video
            qtype: Question type to check
            generator: QuestionGenerator instance

        Returns:
            True if the entry has all required fields for this question type
        """
        if qtype == QuestionType.ROLE_IDENTIFICATION:
            return generator._can_generate_role_identification(entry)
        if qtype == QuestionType.SEQUENCE_VERIFICATION:
            return generator._can_generate_sequence_verification(entry)
        template = QUESTION_TEMPLATES[qtype]
        return generator._has_required_fields(entry, template)
