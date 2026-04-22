import random
import re
from dataclasses import dataclass

from .answer_bank import AnswerBank, normalize_entry
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


NONE_DISTRACTOR_INJECTION_RATE = 0.20

class QuestionGenerator:
    def __init__(self, annotations: list[dict], num_distractors: int = 7, trick_probability: float = 0.0):
        self.annotations = [normalize_entry(e) for e in annotations]
        self.bank = AnswerBank.from_annotations(annotations)
        self.num_distractors = num_distractors
        self.trick_probability = trick_probability
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
        # Trick question: correct answer is the static "none" distractor; all
        # other choices are plausible cross-video options so the model must
        # actually watch the video to know none of them apply.
        correct_answer = template.correct_answer_builder(entry)

        # When the natural correct answer equals the static distractor (e.g., no bystander
        # present), the question is inherently trick-like and must be rate-controlled the
        # same way as generated trick questions. Use _should_be_trick to decide; if the
        # rate is already saturated, skip (return None) rather than inflate the count.
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

        priority_distractors = None
        if template.source_role is not None:
            priority_distractors = self._get_same_video_people_distractors(
                entry, template.source_role
            )

        if template.same_entry_distractor_builder is not None:
            extra = template.same_entry_distractor_builder(entry)
            if extra is not None:
                priority_distractors = (priority_distractors or []) + [extra]

        if template.distractors_override_builder is not None:
            distractors = template.distractors_override_builder(
                entry, self.bank, self.num_distractors, correct_answer
            )
        else:
            distractors = self._sample_distractors(
                pool_name=template.distractor_pool,
                correct_answer=correct_answer,
                static_distractor=template.static_distractor,
                priority_distractors=priority_distractors,
                same_video_only=template.same_video_only,
            )

        # Inject "none"-type static distractor into ~20% of non-trick questions.
        # This breaks the signal that "none" options are always/only correct (trick Qs).
        if (
            template.static_distractor is not None
            and random.random() < NONE_DISTRACTOR_INJECTION_RATE
            and not self._is_semantically_similar(template.static_distractor, correct_answer)
        ):
            static = template.static_distractor
            # Only inject if not already present
            if static not in distractors:
                if len(distractors) >= self.num_distractors:
                    distractors[-1] = static  # Replace last distractor
                else:
                    distractors.append(static)

        # Limit how many distractors share the same leading component as the
        # correct answer. Without this, a model can pick the most-repeated prefix
        # (e.g. the action in compound_action_victims) without watching the video.
        # Override builders intentionally produce same-prefix hard negatives
        # (correct_aggressor + wrong_victim pairs test victim recognition), so
        # they opt out of the cap.
        if template.distractors_override_builder is None:
            distractors = self._cap_prefix_repeats(
                correct_answer, distractors,
                pool_name=template.distractor_pool,
                max_prefix_repeat=2,
            )

        answers = [correct_answer] + distractors
        # No two options may reference the same action from bank.actions.
        # The correct answer claims its action first; any distractor whose action
        # would collide has its action substring swapped for an unused one.
        # No-op for templates whose options carry no action (role_identification,
        # scene_location, count-based questions, etc.).
        answers = self._enforce_unique_actions(answers)
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        self._record_trick_outcome(template.question_type.value, False)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=False,
        )

    def _generate_trick_from_template(
        self, entry: dict, template: QuestionTemplate
    ) -> GeneratedQuestion:
        """Generate a trick question where the correct answer is the static 'none' distractor.

        All distractor choices are drawn from the global pool (cross-video), explicitly
        excluding the actual correct answer for this video. The model must recognise
        that none of the plausible-looking options match what is shown and select the
        'none' answer.
        """
        correct_answer = template.static_distractor
        actual_correct = template.correct_answer_builder(entry)

        pool = self.bank.get_pool(template.distractor_pool)
        pool = [
            p for p in pool
            if p != correct_answer
            and not self._is_semantically_similar(p, correct_answer)
            and p != actual_correct
            and not self._is_semantically_similar(p, actual_correct)
        ]

        sample_count = min(self.num_distractors, len(pool))
        distractors = random.sample(pool, sample_count) if pool else []

        # Fallback: draw from other pools rather than generating "Option N" labels
        if len(distractors) < self.num_distractors:
            fallback_pools = ["people", "actions", "environments", "action_statements"]
            used = set(distractors) | {correct_answer, actual_correct}
            for pool_name in fallback_pools:
                if len(distractors) >= self.num_distractors:
                    break
                extras = self.bank.get_pool(pool_name)
                random.shuffle(extras)
                for item in extras:
                    if item not in used and not self._is_semantically_similar(item, correct_answer):
                        distractors.append(item)
                        used.add(item)
                        if len(distractors) >= self.num_distractors:
                            break

        answers = [correct_answer] + distractors
        # Same uniqueness invariant as the non-trick path.
        answers = self._enforce_unique_actions(answers)
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=True,
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
        # Collect candidate (role_label, person_description) pairs
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

        # Collect all person descriptions from current entry for filtering
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
            # Pick a description from another video that doesn't match current entry
            foreign_pool = [
                desc for desc in self.bank.people
                if desc.strip().lower() not in current_descriptions
            ]
            if not foreign_pool:
                # Fall back to normal case if no foreign descriptions available
                is_trick = False

        if is_trick:
            person_desc = random.choice(foreign_pool)
            correct_answer = ROLE_ID_NO_MATCH
            # Fill distractor slots with role labels
            # Uncomment this and the additional roles in ROLE_LABELS to give up to 7 options
            distractors = random.sample(ROLE_LABELS, min(self.num_distractors, len(ROLE_LABELS)))

        else:
            # Normal case: pick a real person
            correct_role, person_desc = random.choice(candidates)
            correct_answer = correct_role
            # Build distractors: other role labels (excluding correct) + ROLE_ID_NO_MATCH
            other_labels = [l for l in ROLE_LABELS if l != correct_role]
            num_label_distractors = self.num_distractors - 1  # Reserve 1 slot for NO_MATCH
            sampled_labels = random.sample(other_labels, min(num_label_distractors, len(other_labels)))
            distractors = sampled_labels + [ROLE_ID_NO_MATCH]

        prompt = f"Concerning the {person_desc}, their role would best be described as:"

        # answers = [correct_answer] + distractors
        answers = [correct_answer] + distractors
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        self._record_trick_outcome(QuestionType.ROLE_IDENTIFICATION.value, is_trick)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.ROLE_IDENTIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
            is_trick=is_trick,
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
        """Generate a sequence selection question.

        The correct sequence and all distractors appear as answer choices — the
        prompt contains no sequence text, so the model cannot reason about
        correctness from the question alone and must watch the video.

        Distractor priority (all use only in-video people where possible):
          1. Role reversal — same action, aggressor and victim swapped
          2. Wrong action  — correct roles, different action
          3. Bystander as aggressor — bystander initiates correct action on victim
          4. Random cross-video sequences to fill remaining slots
        """
        aggressor = _format_people(entry.get("aggressor"))
        action = entry.get("action")
        victim = _format_people(entry.get("victim"))

        if aggressor is None or action is None or victim is None:
            return None

        seq_style = random.randint(0, 2)
        entry["_format_style_seq"] = seq_style
        correct_seq = self._build_sequence_str(aggressor, action, victim, seq_style)
        prompt = "Which of the following sequences best describes the interaction shown in the video?"

        # Trick question: the correct sequence is absent from the choices; all
        # options are plausible cross-video sequences so the model must watch
        # the video to know none of them apply.
        is_trick = self._should_be_trick(QuestionType.SEQUENCE_VERIFICATION.value)
        if is_trick:
            wrong_seqs = self._generate_alternate_sequences(correct_seq, self.num_distractors, style=seq_style)
            answers = [SEQ_NO_MATCH] + wrong_seqs
            answers = self._enforce_unique_actions(answers)
            random.shuffle(answers)
            self._record_trick_outcome(QuestionType.SEQUENCE_VERIFICATION.value, True)
            return GeneratedQuestion(
                video_name=entry.get("video_name", "unknown"),
                question_type=QuestionType.SEQUENCE_VERIFICATION.value,
                prompt=prompt,
                answers=answers,
                correct_answer=SEQ_NO_MATCH,
                correct_index=answers.index(SEQ_NO_MATCH),
                is_trick=True,
            )

        bystander = _specific_bystander(entry)

        distractors: list[str] = []
        seen: set[str] = {correct_seq}

        # 1. Role reversal
        reversal = self._build_sequence_str(victim, action, aggressor, seq_style)
        if reversal not in seen:
            distractors.append(reversal)
            seen.add(reversal)

        # 2. Wrong action (correct roles)
        wrong_actions = [a for a in self.bank.actions if a != action]
        if wrong_actions:
            wrong_action_seq = self._build_sequence_str(aggressor, random.choice(wrong_actions), victim, seq_style)
            if wrong_action_seq not in seen:
                distractors.append(wrong_action_seq)
                seen.add(wrong_action_seq)

        # 3. Bystander as aggressor
        if bystander:
            bys_seq = self._build_sequence_str(bystander, action, victim, seq_style)
            if bys_seq not in seen:
                distractors.append(bys_seq)
                seen.add(bys_seq)

        # 4. In-video bystanders as wrong aggressor / wrong victim — in-cast
        #    distractors the model cannot reject by absence.
        local_bystanders = [
            p for p in _individual_bystanders(entry)
            if p != aggressor and p != victim
        ]
        local_seqs: list[str] = []
        for p in local_bystanders:
            s = self._build_sequence_str(p, action, victim, seq_style)
            if s not in seen:
                local_seqs.append(s)
                seen.add(s)
        for p in local_bystanders:
            s = self._build_sequence_str(aggressor, action, p, seq_style)
            if s not in seen:
                local_seqs.append(s)
                seen.add(s)
        random.shuffle(local_seqs)
        for s in local_seqs:
            if len(distractors) >= self.num_distractors:
                break
            distractors.append(s)

        # 5. Fill remaining slots with cross-video random sequences
        needed = self.num_distractors - len(distractors)
        if needed > 0:
            alts = self._generate_alternate_sequences(correct_seq, needed, exclude=seen, style=seq_style)
            distractors.extend(alts)

        distractors = distractors[:self.num_distractors]

        answers = [correct_seq] + distractors
        # _enforce_unique_actions scans left-to-right; correct_seq sits at index 0
        # so it always claims its action first and is never rewritten.
        answers = self._enforce_unique_actions(answers)
        random.shuffle(answers)
        correct_index = answers.index(correct_seq)

        self._record_trick_outcome(QuestionType.SEQUENCE_VERIFICATION.value, False)
        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.SEQUENCE_VERIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_seq,
            correct_index=correct_index,
            is_trick=False,
        )