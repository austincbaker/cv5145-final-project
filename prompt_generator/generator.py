import random
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
)


@dataclass
class GeneratedQuestion:
    video_name: str
    question_type: str
    prompt: str
    answers: list[str]
    correct_answer: str
    correct_index: int


class QuestionGenerator:
    def __init__(self, annotations: list[dict], num_distractors: int = 7, trick_probability: float = 0.0):
        self.annotations = [normalize_entry(e) for e in annotations]
        self.bank = AnswerBank.from_annotations(annotations)
        self.num_distractors = num_distractors
        self.trick_probability = trick_probability

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
    ) -> GeneratedQuestion:
        # Trick question: correct answer is the static "none" distractor; all
        # other choices are plausible cross-video options so the model must
        # actually watch the video to know none of them apply.
        if template.static_distractor is not None and random.random() < self.trick_probability:
            return self._generate_trick_from_template(entry, template)

        correct_answer = template.correct_answer_builder(entry)

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

        answers = [correct_answer] + distractors
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
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

        while len(distractors) < self.num_distractors:
            generic = f"Option {len(distractors) + 2}"
            if generic not in distractors and not self._is_semantically_similar(generic, correct_answer):
                distractors.append(generic)
            else:
                distractors.append(f"Alternative {len(distractors) + 2}")

        answers = [correct_answer] + distractors
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=template.question_type.value,
            prompt=template.prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
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
        self, entry: dict, trick_probability: float = 0.25
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

        is_trick = random.random() < trick_probability

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

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.ROLE_IDENTIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
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
    def _build_sequence_str(aggressor: str, action: str, victim: str) -> str:
        return (
            f"{aggressor}, who is the aggressor, performed the action of "
            f"{action} against {victim} who is the victim"
        )

    def _generate_alternate_sequences(
        self, correct_seq: str, count: int, exclude: set[str] | None = None
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
            seq = self._build_sequence_str(alt_agg, alt_action, alt_vic)
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

        correct_seq = self._build_sequence_str(aggressor, action, victim)
        prompt = "Which of the following sequences best describes the interaction shown in the video?"

        # Trick question: the correct sequence is absent from the choices; all
        # options are plausible cross-video sequences so the model must watch
        # the video to know none of them apply.
        if random.random() < self.trick_probability:
            wrong_seqs = self._generate_alternate_sequences(correct_seq, self.num_distractors)
            answers = [SEQ_NO_MATCH] + wrong_seqs
            random.shuffle(answers)
            return GeneratedQuestion(
                video_name=entry.get("video_name", "unknown"),
                question_type=QuestionType.SEQUENCE_VERIFICATION.value,
                prompt=prompt,
                answers=answers,
                correct_answer=SEQ_NO_MATCH,
                correct_index=answers.index(SEQ_NO_MATCH),
            )

        bystander = _specific_bystander(entry)

        distractors: list[str] = []
        seen: set[str] = {correct_seq}

        # 1. Role reversal
        reversal = self._build_sequence_str(victim, action, aggressor)
        if reversal not in seen:
            distractors.append(reversal)
            seen.add(reversal)

        # 2. Wrong action (correct roles)
        wrong_actions = [a for a in self.bank.actions if a != action]
        if wrong_actions:
            wrong_action_seq = self._build_sequence_str(aggressor, random.choice(wrong_actions), victim)
            if wrong_action_seq not in seen:
                distractors.append(wrong_action_seq)
                seen.add(wrong_action_seq)

        # 3. Bystander as aggressor
        if bystander:
            bys_seq = self._build_sequence_str(bystander, action, victim)
            if bys_seq not in seen:
                distractors.append(bys_seq)
                seen.add(bys_seq)

        # 4. Fill remaining slots with cross-video random sequences
        needed = self.num_distractors - len(distractors)
        if needed > 0:
            alts = self._generate_alternate_sequences(correct_seq, needed, exclude=seen)
            distractors.extend(alts)

        distractors = distractors[:self.num_distractors]

        answers = [correct_seq] + distractors
        random.shuffle(answers)
        correct_index = answers.index(correct_seq)

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=QuestionType.SEQUENCE_VERIFICATION.value,
            prompt=prompt,
            answers=answers,
            correct_answer=correct_seq,
            correct_index=correct_index,
        )