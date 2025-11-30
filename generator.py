import random
from dataclasses import dataclass

from .answer_bank import AnswerBank, normalize_entry
from .templates import (
    QUESTION_TEMPLATES,
    ROLE_COUNT_PROMPTS,
    COUNT_OPTIONS,
    QuestionType,
    QuestionTemplate,
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
    def __init__(self, annotations: list[dict], num_distractors: int = 3):
        self.annotations = [normalize_entry(e) for e in annotations]
        self.bank = AnswerBank.from_annotations(annotations)
        self.num_distractors = num_distractors

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

        if question_type == QuestionType.ROLE_COUNT:
            return self._generate_role_count_question(entry)

        template = QUESTION_TEMPLATES[question_type]
        if not self._has_required_fields(entry, template):
            available_types = self._get_valid_question_types(entry)
            if not available_types:
                return None
            question_type = random.choice(available_types)
            if question_type == QuestionType.ROLE_COUNT:
                return self._generate_role_count_question(entry)
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
        """Generate all possible question types for all videos.
        """
        questions = []
        
        for entry in self.annotations:
            valid_types = self._get_valid_question_types(entry)
            
            for question_type in valid_types:
                question = self.generate_question(entry=entry, question_type=question_type)
                if question is not None:
                    questions.append(question)
        
        return questions

    def _generate_from_template(
        self, entry: dict, template: QuestionTemplate
    ) -> GeneratedQuestion:
        correct_answer = template.correct_answer_builder(entry)
        distractors = self._sample_distractors(
            pool_name=template.distractor_pool,
            correct_answer=correct_answer,
            static_distractor=template.static_distractor,
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

    def _generate_role_count_question(self, entry: dict) -> GeneratedQuestion:
        role = random.choice(["aggressor", "victim", "bystander"])
        prompt = ROLE_COUNT_PROMPTS[role]

        count = self._count_role(entry, role)
        if count == 0:
            correct_answer = "0"
        elif count == 1:
            correct_answer = "1"
        elif count == 2:
            correct_answer = "2"
        else:
            correct_answer = "More than 2"

        answers = COUNT_OPTIONS.copy()
        random.shuffle(answers)
        correct_index = answers.index(correct_answer)

        return GeneratedQuestion(
            video_name=entry.get("video_name", "unknown"),
            question_type=f"role_count_{role}",
            prompt=prompt,
            answers=answers,
            correct_answer=correct_answer,
            correct_index=correct_index,
        )

    def _count_role(self, entry: dict, role: str) -> int:
        value = entry.get(role)
        if value is None:
            return 0
        if isinstance(value, list):
            return len([v for v in value if v and str(v).strip()])
        if isinstance(value, str) and value.strip():
            return 1
        return 0

    def _sample_distractors(
        self,
        pool_name: str,
        correct_answer: str,
        static_distractor: str | None = None,
    ) -> list[str]:
        pool = self.bank.get_pool(pool_name)
        
        # Remove exact matches and semantically similar answers
        pool = [
            p for p in pool 
            if p != correct_answer and not self._is_semantically_similar(p, correct_answer)
        ]

        num_needed = self.num_distractors
        
        # Check if static distractor conflicts with the correct answer
        use_static = False
        if static_distractor and not self._is_semantically_similar(static_distractor, correct_answer):
            use_static = True
            num_needed -= 1

        sampled = []
        if pool:
            sample_count = min(num_needed, len(pool))
            sampled = random.sample(pool, sample_count)

        if use_static:
            sampled.append(static_distractor)

        # Fill remaining slots with generic stuff if needed
        while len(sampled) < self.num_distractors:
            generic = f"Option {len(sampled) + 2}"
            if generic not in sampled and not self._is_semantically_similar(generic, correct_answer):
                sampled.append(generic)
            else:
                # Fallback if somehow generic conflicts
                sampled.append(f"Alternative {len(sampled) + 2}")

        return sampled[: self.num_distractors]

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
            if qtype == QuestionType.ROLE_COUNT:
                valid.append(qtype)
            elif self._has_required_fields(entry, template):
                valid.append(qtype)
        return valid