import random

import re
from dataclasses import dataclass

from .answer_bank import AnswerBank, normalize_entry
from prompt_generator.hardness import (
    DEFAULT_RECIPES,
    HardnessRecipe,
    TRICK_RECIPE_FACTORY,
)
from prompt_generator.mutations import fulfill_recipe
from .templates import (
    QUESTION_TEMPLATES,
    COUNT_OPTIONS,
    ROLE_LABELS,
    ROLE_ID_NO_MATCH,
    SEQ_ORIGINAL_CORRECT,
    SEQ_NO_MATCH,
    QuestionType,
    QuestionTemplate,
    _format_people,
    _specific_bystander,
    _individual_bystanders,
)


@dataclass
class GeneratedQuestion:
    video_name: str
    question_type: str
    prompt: str
    answers: list[str]
    correct_answer: str
    correct_index: int
    is_trick: bool = False
    option_hardness: list[str] = None


NONE_DISTRACTOR_INJECTION_RATE = 0.20

class QuestionGenerator:
    def __init__(self, annotations: list[dict], num_distractors: int = 7, trick_probability: float = 0.0, recipes: dict = None):
        self.annotations = [normalize_entry(e) for e in annotations]
        self.bank = AnswerBank.from_annotations(annotations)
        self.num_distractors = num_distractors
        self.trick_probability = trick_probability
        self._recipes = recipes if recipes is not None else DEFAULT_RECIPES
        self._trick_counts: dict[str, int] = {}
        self._total_counts: dict[str, int] = {}
        # Longest-first so multi-word actions ("hit with an object") win the
        # substring match before shorter substrings ("hit") do.
        self._sorted_actions: list[str] = sorted(self.bank.actions, key=len, reverse=True)
        # Precompiled once for _rewrite_action.
        self._action_patterns: dict[str, "re.Pattern"] = {
            a: re.compile(re.escape(a), re.IGNORECASE) for a in self._sorted_actions
        }

    def _option_action(self, option: str) -> str | None:
        """Return the canonical action from `bank.actions` that appears inside
        `option`, or None if no known action is present.

        Every action string the generator ever embeds into an option comes from
        `bank.actions`, so this substring scan is an exact inverse. Abstention
        answers like "No meaningful interaction occurs" contain no action phrase
        and return None — they're never treated as duplicates.
        """
        if not self._sorted_actions:
            return None
        lower = option.lower()
        for act in self._sorted_actions:
            if act.lower() in lower:
                return act
        return None

    def _rewrite_action(self, option: str, old_action: str, new_action: str) -> str:
        """Replace the first case-insensitive occurrence of `old_action` in
        `option` with `new_action`, preserving the rest of the string.
        """
        pattern = self._action_patterns.get(old_action)
        if pattern is None:
            pattern = re.compile(re.escape(old_action), re.IGNORECASE)
        return pattern.sub(new_action, option, count=1)

    def _enforce_unique_actions(self, answers: list[str]) -> list[str]:
        """Return a copy of `answers` in which no two options reference the same
        action phrase from `bank.actions`.

        Scans left-to-right: the first option to use a given action claims it;
        subsequent options that parse to the same action have their action
        substring swapped for a random unused one from `bank.actions`.

        This preserves the option count (no drops) and preserves the assembled
        option structure (only the action token is rewritten), so a
        role-reversal distractor whose action would collide with the correct
        answer becomes a role-reversal-plus-different-action distractor.
        """
        if not self._sorted_actions:
            return list(answers)
        used: set[str] = set()
        result: list[str] = []
        for ans in answers:
            act = self._option_action(ans)
            if act is None:
                result.append(ans)
                continue
            if act not in used:
                used.add(act)
                result.append(ans)
                continue
            # Duplicate — swap in a random unused action from the canonical pool.
            candidates = [a for a in self._sorted_actions if a not in used]
            if not candidates:
                # No unused actions left (very rare; only when num_distractors+1
                # exceeds |bank.actions|). Keep the duplicate rather than fail.
                result.append(ans)
                continue
            new_action = random.choice(candidates)
            used.add(new_action)
            result.append(self._rewrite_action(ans, act, new_action))
        return result

    def _enforce_unique_actions_with_labels(
        self, answers: list[str], option_hardness: list[str], correct_answer: str,
        qtype: str, entry: dict,
    ) -> tuple[list[str], list[str]]:
        """Same as _enforce_unique_actions but re-classifies any rewritten
        distractor so its hardness label stays faithful to the emitted text.

        On the recipe-driven path fulfill_recipe already satisfies G1 by
        construction, so this method should be a no-op (answers unchanged).
        If the invariant ever slips (e.g. a future template embeds the
        correct action in an unexpected position), we re-run the classifier
        on the rewritten option so option_hardness doesn't drift.
        """
        rewritten = self._enforce_unique_actions(answers)
        if rewritten == answers:
            return rewritten, option_hardness
        # Defence in depth: re-classify anything the rewriter changed.
        from prompt_generator.hardness import classify_distractor
        new_labels = list(option_hardness)
        for i, (orig, new) in enumerate(zip(answers, rewritten)):
            if orig == new:
                continue
            if option_hardness[i] == "correct":
                # Correct answer's text should never be rewritten — defensive
                # guard only; if this fires, something upstream is wrong.
                continue
            new_labels[i] = classify_distractor(qtype, new, correct_answer, entry)
        return rewritten, new_labels

    @staticmethod
    def _length_balanced_sample(
        pool: list[str],
        target_length: int,
        count: int,
        tolerance: float = 0.4,
    ) -> list[str]:
        """Sample from pool preferring items within tolerance of target_length.

        Splits the pool into 'close' (within tolerance) and 'far' items.
        Draws from close first, then falls back to far if needed.
        """
        if not pool or count <= 0:
            return []

        close = []
        far = []
        lower = target_length * (1 - tolerance)
        upper = target_length * (1 + tolerance)

        for item in pool:
            if lower <= len(item) <= upper:
                close.append(item)
            else:
                far.append(item)

        random.shuffle(close)
        random.shuffle(far)

        result = close[:count]
        if len(result) < count:
            result.extend(far[: count - len(result)])
        return result

    @staticmethod
    def _is_semantically_similar(answer1: str, answer2: str) -> bool:
        """Check if two answers are semantically similar (e.g., both mean 'none')."""
        if answer1 is None or answer2 is None:
            return False
        
        # Normalize to lowercase for comparison
        a1 = answer1.lower().strip()
        a2 = answer2.lower().strip()
        
        # Exact match
        if a1 == a2:
            return True
        
        # Define "none" keywords
        none_keywords = {
            "no one", "no individual", "no person", "no people",
            "no bystander", "no victim", "no aggressor",
            "none", "not present", "no action",
            "not shown", "unclear", "not visible",
            "no meaningful", "no sequence"
        }
        
        # Check if both contain "none" keywords
        a1_is_none = any(keyword in a1 for keyword in none_keywords)
        a2_is_none = any(keyword in a2 for keyword in none_keywords)
        
        if a1_is_none and a2_is_none:
            return True
        
        # Check for substring matches (one contains the other)
        # Only if both are reasonably long to avoid false positives
        if len(a1) > 10 and len(a2) > 10:
            if a1 in a2 or a2 in a1:
                return True
        
        return False

    # Common separators used in compound answers: "action; Victim: X", "X in Y",
    # "Aggressor: X; Victim: Y", etc.
    _PREFIX_SEPARATORS = ("; ", " in ")

    @staticmethod
    def _extract_prefix(answer: str) -> str | None:
        """Extract the leading component of a compound answer.

        For "; " separators (e.g. "action; Victim: X"), splits at the first occurrence.
        For " in " separators (e.g. "person X in location"), splits at the *last*
        occurrence to avoid splitting person descriptions like "person in a red shirt".
        """
        if "; " in answer:
            return answer.split("; ", 1)[0].strip()
        if " in " in answer:
            # Split at the last " in " — the location is always the trailing component
            idx = answer.rfind(" in ")
            if idx > 0:
                return answer[:idx].strip()
        return None

    def _cap_prefix_repeats(
        self,
        correct_answer: str,
        distractors: list[str],
        pool_name: str | None = None,
        max_prefix_repeat: int = 2,
    ) -> list[str]:
        """Ensure the correct answer's prefix isn't the most frequent among all answers.

        Counts each leading component across [correct_answer] + distractors. If the
        correct prefix appears strictly more than any other prefix, replaces excess
        same-prefix distractors with different-prefix items from the bank pool.
        """
        correct_prefix = QuestionGenerator._extract_prefix(correct_answer)
        if correct_prefix is None:
            return distractors

        correct_prefix_lower = correct_prefix.lower()

        # Count all prefixes across answers (correct + distractors)
        from collections import Counter
        prefix_counts: Counter[str] = Counter()
        prefix_counts[correct_prefix_lower] += 1  # the correct answer itself

        matching_indices: list[int] = []
        for i, d in enumerate(distractors):
            d_prefix = QuestionGenerator._extract_prefix(d)
            if d_prefix is not None:
                p_lower = d_prefix.lower()
                prefix_counts[p_lower] += 1
                if p_lower == correct_prefix_lower:
                    matching_indices.append(i)

        # Find the max count among OTHER prefixes
        other_max = 0
        for prefix, count in prefix_counts.items():
            if prefix != correct_prefix_lower and count > other_max:
                other_max = count

        # If correct prefix doesn't dominate, nothing to do
        correct_total = prefix_counts[correct_prefix_lower]
        if correct_total <= max(other_max, 1):
            return distractors

        # How many same-prefix distractors to keep so that total
        # (correct answer + kept distractors) <= max(other_max, 1)
        target_distractors = max(other_max, 1) - 1  # subtract 1 for the correct answer
        target_distractors = max(target_distractors, 0)

        if len(matching_indices) <= target_distractors:
            return distractors

        # Build a replacement pool of different-prefix items from the bank
        existing = {d.lower() for d in distractors} | {correct_answer.lower()}
        replacements: list[str] = []
        if pool_name is not None:
            pool = self.bank.get_pool(pool_name)
            random.shuffle(pool)
            for item in pool:
                if item.lower() in existing:
                    continue
                item_prefix = QuestionGenerator._extract_prefix(item)
                if item_prefix is not None and item_prefix.lower() == correct_prefix_lower:
                    continue
                if self._is_semantically_similar(item, correct_answer):
                    continue
                replacements.append(item)

        random.shuffle(matching_indices)
        keep = set(matching_indices[:target_distractors])
        evict_indices = [i for i in matching_indices if i not in keep]

        result = list(distractors)
        replaced = 0
        # Replace evicted items with different-prefix alternatives in-place
        for idx in evict_indices:
            if replaced < len(replacements):
                result[idx] = replacements[replaced]
                replaced += 1
            else:
                result[idx] = None  # mark for removal

        # Remove any that couldn't be replaced
        result = [d for d in result if d is not None]

        return result

    def _should_be_trick(self, question_type: str) -> bool:
        """Decide if this question should be a trick, enforcing per-type rate targets."""
        if self.trick_probability <= 0:
            return False

        total = self._total_counts.get(question_type, 0)
        tricks = self._trick_counts.get(question_type, 0)

        if total < 4:
            return random.random() < self.trick_probability
        else:
            current_rate = tricks / total
            if current_rate > self.trick_probability:
                adjusted_prob = self.trick_probability * 0.5
            elif current_rate < self.trick_probability * 0.5:
                adjusted_prob = min(self.trick_probability * 1.5, 0.5)
            else:
                adjusted_prob = self.trick_probability
            return random.random() < adjusted_prob

    def _record_trick_outcome(self, question_type: str, was_trick: bool) -> None:
        """Record whether a successfully generated question was a trick."""
        self._total_counts[question_type] = self._total_counts.get(question_type, 0) + 1
        if was_trick:
            self._trick_counts[question_type] = self._trick_counts.get(question_type, 0) + 1

    def generate_question(
        self, entry: dict | None = None, question_type: QuestionType | None = None
    ) -> GeneratedQuestion | None:
        if entry is None:
            entry = random.choice(self.annotations)
        entry = normalize_entry(entry)

        if question_type is None:
            question_type = random.choice(list(QuestionType))

        if question_type == QuestionType.ROLE_IDENTIFICATION:
            return self._generate_role_identification(entry)

        if question_type == QuestionType.SEQUENCE_VERIFICATION:
            return self._generate_sequence_verification(entry)

        template = QUESTION_TEMPLATES[question_type]
        if not self._has_required_fields(entry, template):
            available_types = self._get_valid_question_types(entry)
            if not available_types:
                return None
            question_type = random.choice(available_types)
            template = QUESTION_TEMPLATES[question_type]

        return self._generate_from_template(entry, template)

    def generate_questions(
        self, count: int, allow_duplicates: bool = False
    ) -> list[GeneratedQuestion]:
        questions = []
        used_combinations = set()

        attempts = 0
        max_attempts = count * 10

        while len(questions) < count and attempts < max_attempts:
            attempts += 1
            question = self.generate_question()
            if question is None:
                continue

            combo_key = (question.video_name, question.question_type, question.prompt)
            if not allow_duplicates and combo_key in used_combinations:
                continue

            used_combinations.add(combo_key)
            questions.append(question)

        return questions

    def generate_all_questions(self) -> list[GeneratedQuestion]:
        """
        Generate all possible question types for all videos.
        """
        questions = []

        for entry in self.annotations:
            valid_types = self._get_valid_question_types(entry)

            for question_type in valid_types:
                question = self.generate_question(entry=entry, question_type=question_type)
                if question is not None:
                    questions.append(question)

        return questions

    def generate_distributed_questions_for_video(
        self,
        entry: dict,
        distributor: "CategoryDistributor",
    ) -> list[GeneratedQuestion]:
        """
        Generate questions for a video using category-based distribution.

        Args:
            entry: Annotation entry for the video
            distributor: CategoryDistributor instance to select question types

        Returns:
            List of GeneratedQuestion objects for this video
        """
        from .distribution import CategoryDistributor  # Import here to avoid circular dependency

        selected_types = distributor.select_questions_for_video(entry, self)

        questions = []
        for qtype in selected_types:
            question = self.generate_question(entry=entry, question_type=qtype)
            if question is not None:
                questions.append(question)

        return questions

    def _get_same_video_people_distractors(
        self, entry: dict, exclude_role: str
    ) -> list[str]:
        """Return individual people from all roles in the entry except exclude_role.

        Each person in a list field is yielded as a separate string.
        Bystanders described as "a group of people" are skipped.
        """
        distractors = []
        seen = set()
        for role in ("aggressor", "victim", "bystander"):
            if role == exclude_role:
                continue
            value = entry.get(role)
            if value is None:
                continue
            if isinstance(value, list):
                for person in value:
                    if person and isinstance(person, str) and person.strip():
                        if role == "bystander" and "group of" in person.lower():
                            continue
                        key = person.strip().lower()
                        if key not in seen:
                            distractors.append(person.strip())
                            seen.add(key)
            elif isinstance(value, str) and value.strip():
                if role == "bystander" and "group of" in value.lower():
                    continue
                key = value.strip().lower()
                if key not in seen:
                    distractors.append(value.strip())
                    seen.add(key)
        return distractors

    def _generate_from_template(
        self, entry: dict, template: QuestionTemplate
    ) -> GeneratedQuestion | None:
        correct_answer = template.correct_answer_builder(entry)

        if (
            template.static_distractor is not None
            and self._is_semantically_similar(correct_answer, template.static_distractor)
        ):
            if self._should_be_trick(template.question_type.value):
                result = self._generate_trick_from_template(entry, template)
                self._record_trick_outcome(template.question_type.value, True)
                return result
            else:
                return None

        is_trick = template.static_distractor is not None and self._should_be_trick(template.question_type.value)
        if is_trick:
            result = self._generate_trick_from_template(entry, template)
            self._record_trick_outcome(template.question_type.value, True)
            return result

        if template.distractors_override_builder is not None:
            distractors = template.distractors_override_builder(
                entry, self.bank, self.num_distractors, correct_answer,
            )
            answers = [correct_answer] + distractors
            option_hardness = ["correct"] + ["cross_video"] * len(distractors)
            paired = list(zip(answers, option_hardness))
            random.shuffle(paired)
            answers, option_hardness = map(list, zip(*paired))
            correct_index = option_hardness.index("correct")
            self._record_trick_outcome(template.question_type.value, False)
            return GeneratedQuestion(
                video_name=entry.get("video_name", "unknown"),
                question_type=template.question_type.value,
                prompt=template.prompt,
                answers=answers,
                correct_answer=correct_answer,
                correct_index=correct_index,
                is_trick=False,
                option_hardness=option_hardness,
            )

        recipe = self._recipes.get(template.question_type.value)
        if not recipe:
            # Fallback to a default cross-video recipe if missing
            recipe = HardnessRecipe({"cross_video": self.num_distractors})

        # Standalone frequency-inverted dispatch. Falls back to the balanced
        # recipe for this qtype if the builder returns None (G11 fail or
        # bank.actions too thin). Skips _enforce_unique_actions_with_labels
        # because this mode deliberately repeats actions across distractors.
        if getattr(recipe, "mode", "standard") == "frequency_inverted":
            from prompt_generator.frequency_inverted import build_frequency_inverted_question

            built = build_frequency_inverted_question(
                entry=entry, template=template, bank=self.bank,
                num_distractors=self.num_distractors,
            )
            if built is not None:
                answers, option_hardness, correct_index = built
                self._record_trick_outcome(template.question_type.value, False)
                return GeneratedQuestion(
                    video_name=entry.get("video_name", "unknown"),
                    question_type=template.question_type.value,
                    prompt=template.prompt,
                    answers=answers,
                    correct_answer=correct_answer,
                    correct_index=correct_index,
                    is_trick=False,
                    option_hardness=option_hardness,
                )
            # Fallback: use the balanced default recipe for this qtype.
            recipe = DEFAULT_RECIPES.get(
                template.question_type.value,
                HardnessRecipe({"cross_video": self.num_distractors}),
            )

        distractors, categories = fulfill_recipe(
            recipe=recipe,
            entry=entry,
            template=template,
            correct_answer=correct_answer,
            bank=self.bank,
            qtype=template.question_type.value,
            all_annotations=self.annotations,
        )

        answers = [correct_answer] + distractors
        option_hardness = ["correct"] + categories

        # Safety net. fulfill_recipe already satisfies G1 by construction, so
        # this should be a no-op; the labelled variant re-classifies any
        # distractor whose text is rewritten so option_hardness stays in sync.
        answers, option_hardness = self._enforce_unique_actions_with_labels(
            answers, option_hardness, correct_answer,
            template.question_type.value, entry,
        )

        paired = list(zip(answers, option_hardness))
        random.shuffle(paired)
        answers, option_hardness = map(list, zip(*paired))
        correct_index = option_hardness.index("correct")

        self._record_trick_outcome(template.question_type.value, False)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=False,
            option_hardness=option_hardness,
        )

    def _generate_trick_from_template(
        self, entry: dict, template: QuestionTemplate
    ) -> GeneratedQuestion:
        correct_answer = template.static_distractor
        actual_correct = template.correct_answer_builder(entry)

        recipe = TRICK_RECIPE_FACTORY(self.num_distractors)
        distractors, categories = fulfill_recipe(
            recipe=recipe,
            entry=entry,
            template=template,
            correct_answer=correct_answer,
            bank=self.bank,
            qtype=template.question_type.value,
            all_annotations=self.annotations,
        )

        # Fallback if fulfill_recipe failed to generate enough
        while len(distractors) < self.num_distractors:
            distractors.append(f"Option {len(distractors)+2}")
            categories.append("cross_video")

        answers = [correct_answer] + distractors
        option_hardness = ["correct"] + categories
        answers, option_hardness = self._enforce_unique_actions_with_labels(
            answers, option_hardness, correct_answer,
            template.question_type.value, entry,
        )

        paired = list(zip(answers, option_hardness))
        random.shuffle(paired)
        answers, option_hardness = map(list, zip(*paired))
        correct_index = option_hardness.index("correct")

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=True,
            option_hardness=option_hardness,
        )
    def _sample_distractors(
        self,
        pool_name: str,
        correct_answer: str,
        static_distractor: str | None = None,
        priority_distractors: list[str] | None = None,
        same_video_only: bool = False,
    ) -> list[str]:
        pool = self.bank.get_pool(pool_name)

        # Filter out exact matches and semantically similar answers
        pool = [
            p for p in pool
            if p != correct_answer and not self._is_semantically_similar(p, correct_answer)
        ]

        sampled: list[str] = []
        seen: set[str] = set()

        # 1. Add same-video people first (priority distractors), each as a separate option.
        if priority_distractors:
            for p in priority_distractors:
                if (
                    p != correct_answer
                    and not self._is_semantically_similar(p, correct_answer)
                    and p.lower() not in seen
                ):
                    sampled.append(p)
                    seen.add(p.lower())

        # 2. Check if static distractor is usable (reserve a slot for it).
        use_static = False
        if static_distractor and not self._is_semantically_similar(
            static_distractor, correct_answer
        ):
            if static_distractor.lower() not in seen:
                use_static = True

        # 3. Fill remaining slots (up to num_distractors) from the random pool.
        #    Priority distractors and the static slot both count toward num_distractors.
        slots_taken = len(sampled) + (1 if use_static else 0)
        remaining_needed = max(0, self.num_distractors - slots_taken)

        # Remove already-sampled entries from pool to avoid duplicates.
        pool = [p for p in pool if p.lower() not in seen]

        if not same_video_only and pool and remaining_needed > 0:
            sample_count = min(remaining_needed, len(pool))
            if pool_name == "actions" and self.bank.action_frequencies:
                # Weighted sampling: rare actions get higher probability
                weights = self.bank.get_action_weights()
                weighted_pool = [(p, weights.get(p, 1.0)) for p in pool]
                selected_from_pool = []
                for _ in range(sample_count):
                    if not weighted_pool:
                        break
                    total_w = sum(w for _, w in weighted_pool)
                    r = random.random() * total_w
                    cumulative = 0
                    for idx, (item, w) in enumerate(weighted_pool):
                        cumulative += w
                        if r <= cumulative:
                            selected_from_pool.append(item)
                            weighted_pool.pop(idx)
                            break
                sampled.extend(selected_from_pool)
            else:
                balanced = self._length_balanced_sample(
                    pool, len(correct_answer), sample_count
                )
                if balanced:
                    sampled.extend(balanced)
                else:
                    sampled.extend(random.sample(pool, sample_count))
            for p in sampled[len(sampled) - sample_count:]:
                seen.add(p.lower())

        if use_static:
            sampled.append(static_distractor)

        # 4. Fallback: fill any remaining slots with generic labels.
        #    Skipped when same_video_only=True — fewer options is preferable to
        #    obviously-wrong padding that models can trivially eliminate.
        #    Only triggered when total distractors < num_distractors
        #    (e.g., very small annotation sets).
        if not same_video_only:
            while len(sampled) < self.num_distractors:
                generic = f"Option {len(sampled) + 2}"
                if generic not in sampled and not self._is_semantically_similar(
                    generic, correct_answer
                ):
                    sampled.append(generic)
                else:
                    sampled.append(f"Alternative {len(sampled) + 2}")

        # Priority distractors are always kept; only cap random-pool overflow.
        # If same-video people + static already exceed num_distractors, return them all.
        return sampled

    def _has_required_fields(self, entry: dict, template: QuestionTemplate) -> bool:
        """Check if entry has required fields. None values are acceptable (for normal videos)."""
        for field in template.requires_fields:
            # Field must exist in entry (but can be None for normal videos)
            if field not in entry:
                return False
            
            value = entry.get(field)
            
            # None is acceptable - will use standardized "none" answers
            if value is None:
                continue
            
            # Empty strings/lists are not acceptable
            if isinstance(value, str) and not value.strip():
                return False
            if isinstance(value, list) and not any(
                v and str(v).strip() for v in value
            ):
                return False
        
        return True

    def _get_valid_question_types(self, entry: dict) -> list[QuestionType]:
        valid = []
        for qtype, template in QUESTION_TEMPLATES.items():
            if self._has_required_fields(entry, template):
                valid.append(qtype)
        if self._can_generate_role_identification(entry):
            valid.append(QuestionType.ROLE_IDENTIFICATION)
        if self._can_generate_sequence_verification(entry):
            valid.append(QuestionType.SEQUENCE_VERIFICATION)
        return valid

    def _can_generate_role_identification(self, entry: dict) -> bool:
        """Check if entry has at least one non-group person description."""
        for role in ("aggressor", "victim", "bystander"):
            value = entry.get(role)
            if value is None:
                continue
            if isinstance(value, list):
                for person in value:
                    if person and isinstance(person, str) and "group of" not in person.lower():
                        return True
            elif isinstance(value, str) and value.strip() and "group of" not in value.lower():
                return True
        return False

    def _generate_role_identification(
        self, entry: dict
    ) -> GeneratedQuestion | None:
        """Generate a role identification question with dynamic prompt."""
        candidates = []
        for role in ("aggressor", "victim", "bystander"):
            value = entry.get(role)
            if value is None:
                continue
            label = role.capitalize()
            if isinstance(value, list):
                for person in value:
                    if person and isinstance(person, str) and "group of" not in person.lower():
                        candidates.append((label, person.strip()))
            elif isinstance(value, str) and value.strip() and "group of" not in value.lower():
                candidates.append((label, value.strip()))

        if not candidates:
            return None

        current_descriptions = set()
        for role in ("aggressor", "victim", "bystander"):
            value = entry.get(role)
            if value is None:
                continue
            if isinstance(value, list):
                for person in value:
                    if person and isinstance(person, str):
                        current_descriptions.add(person.strip().lower())
            elif isinstance(value, str) and value.strip():
                current_descriptions.add(value.strip().lower())

        is_trick = self._should_be_trick(QuestionType.ROLE_IDENTIFICATION.value)

        if is_trick:
            foreign_pool = [
                desc for desc in self.bank.people
                if desc.strip().lower() not in current_descriptions
            ]
            if not foreign_pool:
                is_trick = False

        if is_trick:
            person_desc = random.choice(foreign_pool)
            correct_answer = ROLE_ID_NO_MATCH
            distractors = random.sample(ROLE_LABELS, min(self.num_distractors, len(ROLE_LABELS)))
            categories = ["cross_video"] * len(distractors)
        else:
            correct_role, person_desc = random.choice(candidates)
            correct_answer = correct_role
            other_labels = [l for l in ROLE_LABELS if l != correct_role]
            num_label_distractors = self.num_distractors - 1
            sampled_labels = random.sample(other_labels, min(num_label_distractors, len(other_labels)))
            distractors = sampled_labels + [ROLE_ID_NO_MATCH]
            
            categories = []
            for d in distractors:
                if d == ROLE_ID_NO_MATCH:
                    categories.append("none_claim")
                elif d == "Aggressor" or d == "Victim":
                    categories.append("role_reversal")
                elif d == "Bystander":
                    categories.append("bystander_substitution")
                else:
                    categories.append("cross_video")

        prompt = f"Concerning the {person_desc}, their role would best be described as:"

        answers = [correct_answer] + distractors
        option_hardness = ["correct"] + categories
        
        paired = list(zip(answers, option_hardness))
        
        random.shuffle(paired)
        answers, option_hardness = map(list, zip(*paired))
        correct_index = option_hardness.index("correct")

        self._record_trick_outcome(QuestionType.ROLE_IDENTIFICATION.value, is_trick)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.ROLE_IDENTIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=is_trick,
            option_hardness=option_hardness
        )
    def _can_generate_sequence_verification(self, entry: dict) -> bool:
        """Check if entry has non-None aggressor, action, and victim."""
        aggressor_raw = entry.get("aggressor")
        victim_raw = entry.get("victim")
        action = entry.get("action")
        # Guard against empty lists that would crash _format_people
        if isinstance(aggressor_raw, list) and not aggressor_raw:
            return False
        if isinstance(victim_raw, list) and not victim_raw:
            return False
        aggressor = _format_people(aggressor_raw)
        victim = _format_people(victim_raw)
        return aggressor is not None and action is not None and victim is not None

    @staticmethod
    def _build_sequence_str(aggressor: str, action: str, victim: str, style: int = 0) -> str:
        if style == 0:
            return (
                f"{aggressor}, who is the aggressor, performed the action of "
                f"{action} against {victim} who is the victim"
            )
        elif style == 1:
            return (
                f"The aggressor, {aggressor}, carried out {action} "
                f"against the victim, {victim}"
            )
        else:
            return (
                f"{aggressor} (aggressor) did {action} to {victim} (victim)"
            )

    def _generate_alternate_sequences(
        self, correct_seq: str, count: int, exclude: set[str] | None = None, style: int = 0
    ) -> list[str]:
        """Generate alternate incorrect sequences from other videos' data."""
        people_pool = list(self.bank.people)
        actions_pool = list(self.bank.actions)
        excluded = {correct_seq} | (exclude or set())
        alternates = set()
        attempts = 0
        max_attempts = count * 10

        while len(alternates) < count and attempts < max_attempts:
            attempts += 1
            alt_agg = random.choice(people_pool)
            alt_action = random.choice(actions_pool)
            alt_vic = random.choice(people_pool)
            seq = self._build_sequence_str(alt_agg, alt_action, alt_vic, style)
            if seq not in excluded:
                alternates.add(seq)

        return list(alternates)[:count]

    def _generate_sequence_verification(
        self, entry: dict
    ) -> GeneratedQuestion | None:
        """Recipe-path generation for sequence_verification.

        The sequence_verification template has no entry in QUESTION_TEMPLATES
        because the bespoke generator predates the template registry. We build
        a local QuestionTemplate-like shim whose `correct_answer_builder`
        renders a sequence sentence from the mutated entry — same mechanism
        used by every other Fake-Entry qtype.
        """
        aggressor = _format_people(entry.get("aggressor"))
        action = entry.get("action")
        victim = _format_people(entry.get("victim"))

        if aggressor is None or action is None or victim is None:
            return None

        seq_style = random.randint(0, 2)
        prompt = "Which of the following sequences best describes the interaction shown in the video?"

        def _seq_builder(e: dict) -> str:
            agg = _format_people(e.get("aggressor")) or "Unknown"
            act = e.get("action") or "unknown action"
            vic = _format_people(e.get("victim")) or "unknown target"
            return self._build_sequence_str(agg, act, vic, seq_style)

        is_trick = self._should_be_trick(QuestionType.SEQUENCE_VERIFICATION.value)

        # Shim template: exposes `correct_answer_builder` and optionally
        # `static_distractor` for fulfill_recipe's none_claim branch.
        shim_template = QuestionTemplate(
            question_type=QuestionType.SEQUENCE_VERIFICATION,
            prompt=prompt,
            correct_answer_builder=_seq_builder,
            distractor_pool="",  # unused — we go through fulfill_recipe
            static_distractor=SEQ_NO_MATCH if is_trick else None,
        )

        if is_trick:
            # Trick: correct answer is the null-claim; distractors are
            # cross_video sequences built from other annotations.
            correct_answer = SEQ_NO_MATCH
            recipe = TRICK_RECIPE_FACTORY(self.num_distractors)
        else:
            correct_answer = _seq_builder(entry)
            recipe = self._recipes.get(
                QuestionType.SEQUENCE_VERIFICATION.value,
                HardnessRecipe({"cross_video": self.num_distractors}),
            )

        # Frequency-inverted dispatch (non-trick only). Mirrors the dispatch
        # in _generate_from_template.
        if not is_trick and getattr(recipe, "mode", "standard") == "frequency_inverted":
            from prompt_generator.frequency_inverted import build_frequency_inverted_question

            built = build_frequency_inverted_question(
                entry=entry, template=shim_template, bank=self.bank,
                num_distractors=self.num_distractors,
            )
            if built is not None:
                answers, option_hardness, correct_index = built
                self._record_trick_outcome(QuestionType.SEQUENCE_VERIFICATION.value, False)
                return GeneratedQuestion(
                    video_name=entry.get("video_name", "unknown"),
                    question_type=QuestionType.SEQUENCE_VERIFICATION.value,
                    prompt=prompt,
                    answers=answers,
                    correct_answer=correct_answer,
                    correct_index=correct_index,
                    is_trick=False,
                    option_hardness=option_hardness,
                )
            # Fall back to the balanced recipe for sequence_verification.
            recipe = DEFAULT_RECIPES.get(
                QuestionType.SEQUENCE_VERIFICATION.value,
                HardnessRecipe({"cross_video": self.num_distractors}),
            )

        distractors, categories = fulfill_recipe(
            recipe=recipe,
            entry=entry,
            template=shim_template,
            correct_answer=correct_answer,
            bank=self.bank,
            qtype=QuestionType.SEQUENCE_VERIFICATION.value,
            all_annotations=self.annotations,
        )

        # Top up if the recipe couldn't reach the budget (rare when actions
        # or people pools are thin).
        while len(distractors) < self.num_distractors:
            alts = self._generate_alternate_sequences(
                correct_answer, self.num_distractors - len(distractors),
                exclude=set(distractors) | {correct_answer}, style=seq_style,
            )
            if not alts:
                break
            for alt in alts:
                distractors.append(alt)
                categories.append("cross_video")

        answers = [correct_answer] + distractors
        option_hardness = ["correct"] + categories
        answers, option_hardness = self._enforce_unique_actions_with_labels(
            answers, option_hardness, correct_answer,
            QuestionType.SEQUENCE_VERIFICATION.value, entry,
        )

        paired = list(zip(answers, option_hardness))
        random.shuffle(paired)
        answers, option_hardness = map(list, zip(*paired))
        correct_index = answers.index(correct_answer)

        self._record_trick_outcome(QuestionType.SEQUENCE_VERIFICATION.value, is_trick)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.SEQUENCE_VERIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=is_trick,
            option_hardness=option_hardness,
        )
