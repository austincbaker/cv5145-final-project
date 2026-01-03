from dataclasses import dataclass
from enum import Enum
from typing import Callable


class QuestionType(Enum):
    PRIMARY_ACTION = "primary_action"
    AGGRESSOR_ID = "aggressor_identification"
    VICTIM_RECOGNITION = "victim_recognition"
    BYSTANDER_DETECTION = "bystander_detection"
    PERSPECTIVE_AGGRESSOR = "perspective_aggressor"
    SCENE_LOCATION = "scene_location"
    INTERACTION_SUMMARY = "interaction_summary"
    EVENT_SEQUENCE = "event_sequence"
    SOCIAL_APPROPRIATENESS = "social_appropriateness"
    ROLE_COUNT = "role_count"
    COMPOUND_AGGRESSOR_LOCATION = "compound_aggressor_location"
    COMPOUND_ACTION_VICTIMS = "compound_action_victims"
    COMPOUND_ACTION_LOCATION = "compound_action_location"
    COMPOUND_AGGRESSOR_VICTIM = "compound_aggressor_victim"
    COMPOUND_BYSTANDER_LOCATION = "compound_bystander_location"
    COMPOUND_AGGRESSOR_VICTIM_COUNT = "compound_aggressor_victim_count"
    COMPOUND_VICTIM_BYSTANDER_COUNT = "compound_victim_bystander_count"
    COMPOUND_AGGRESSOR_ACTION_VICTIM = "compound_aggressor_action_victim"


@dataclass
class QuestionTemplate:
    question_type: QuestionType
    prompt: str
    correct_answer_builder: Callable
    distractor_pool: str
    static_distractor: str | None = None
    requires_fields: tuple = ()


def build_action_answer(entry: dict, role_key: str, target_key: str) -> str:
    role = _format_people(entry.get(role_key))
    target = _format_people(entry.get(target_key))
    action = entry.get("action")
    
    # Handle None values
    if role is None and target is None:
        return "No action takes place"
    if role is None:
        role = "Unknown person"
    if target is None:
        target = "unknown target"
    if action is None:
        action = "unknown action"
    
    return f"{role} performs action of {action} on {target}"


def _format_people(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        elif len(value) == 2:
            return f"{value[0]} and {value[1]}"
        else:
            return ", ".join(value[:-1]) + f", and {value[-1]}"
    if isinstance(value, str):
        return value
    return None


def _get_person_value(entry: dict, key: str) -> str:
    formatted = _format_people(entry.get(key))
    if formatted is None:
        # Return standardized "none" answer based on role
        none_responses = {
            "aggressor": "No one displays aggressive behavior",
            "victim": "No one appears to be victimized",
            "bystander": "No bystanders are present",
        }
        return none_responses.get(key, "No one")
    return formatted


def _get_social_appropriateness_answer(entry: dict) -> str:
    """Get answer for social appropriateness question, handling None values."""
    formatted = _format_people(entry.get("aggressor"))
    if formatted is None:
        return "No actions performed are socially inappropriate"
    return f"The action performed by {formatted}"


def _build_compound_aggressor_location(entry: dict) -> str:
    """Build compound answer for aggressor description + location."""
    aggressor = _format_people(entry.get("aggressor"))
    environment = entry.get("environment")
    
    if aggressor is None and environment is None:
        return "No aggressor present and location unclear"
    elif aggressor is None:
        return f"No aggressor present; location is {environment}"
    elif environment is None:
        return f"{aggressor}; location unclear"
    else:
        return f"{aggressor} in {environment}"


def _build_compound_action_victims(entry: dict) -> str:
    """Build compound answer for action description + victim count."""
    action = entry.get("action")
    victim = entry.get("victim")
    
    # Count victims
    if victim is None:
        victim_count = 0
    elif isinstance(victim, list):
        victim_count = len([v for v in victim if v])
    else:
        victim_count = 1
    
    # Format action
    if action is None:
        action_text = "No action"
    else:
        action_text = action
    
    # Format count
    if victim_count == 0:
        count_text = "no victims"
    elif victim_count == 1:
        count_text = "1 victim"
    else:
        count_text = f"{victim_count} victims"
    
    return f"{action_text} with {count_text}"


def _build_compound_action_location(entry: dict) -> str:
    """Build compound answer for action + location."""
    action = entry.get("action")
    environment = entry.get("environment")
    
    if action is None and environment is None:
        return "No action and location unclear"
    elif action is None:
        return f"No action in {environment}"
    elif environment is None:
        return f"{action} in unclear location"
    else:
        return f"{action} in {environment}"


def _build_compound_aggressor_victim(entry: dict) -> str:
    """Build compound answer for aggressor + victim descriptions."""
    aggressor = _format_people(entry.get("aggressor"))
    victim = _format_people(entry.get("victim"))
    
    if aggressor is None and victim is None:
        return "No aggressor and no victim"
    elif aggressor is None:
        return f"No aggressor; Victim: {victim}"
    elif victim is None:
        return f"Aggressor: {aggressor}; No victim"
    else:
        return f"Aggressor: {aggressor}; Victim: {victim}"


def _build_compound_bystander_location(entry: dict) -> str:
    """Build compound answer for bystander + location."""
    bystander = _format_people(entry.get("bystander"))
    environment = entry.get("environment")
    
    if bystander is None and environment is None:
        return "No bystanders and location unclear"
    elif bystander is None:
        return f"No bystanders in {environment}"
    elif environment is None:
        return f"{bystander} in unclear location"
    else:
        return f"{bystander} in {environment}"


def _build_compound_aggressor_victim_count(entry: dict) -> str:
    """Build compound answer for aggressor count + victim count."""
    aggressor = entry.get("aggressor")
    victim = entry.get("victim")
    
    # Count aggressors
    if aggressor is None:
        aggressor_count = 0
    elif isinstance(aggressor, list):
        aggressor_count = len([a for a in aggressor if a])
    else:
        aggressor_count = 1
    
    # Count victims
    if victim is None:
        victim_count = 0
    elif isinstance(victim, list):
        victim_count = len([v for v in victim if v])
    else:
        victim_count = 1
    
    # Format counts
    agg_text = f"{aggressor_count} aggressor" + ("s" if aggressor_count != 1 else "")
    vic_text = f"{victim_count} victim" + ("s" if victim_count != 1 else "")
    
    return f"{agg_text} and {vic_text}"


def _build_compound_victim_bystander_count(entry: dict) -> str:
    """Build compound answer for victim count + bystander count."""
    victim = entry.get("victim")
    bystander = entry.get("bystander")
    
    # Count victims
    if victim is None:
        victim_count = 0
    elif isinstance(victim, list):
        victim_count = len([v for v in victim if v])
    else:
        victim_count = 1
    
    # Count bystanders
    if bystander is None:
        bystander_count = 0
    elif isinstance(bystander, list):
        bystander_count = len([b for b in bystander if b])
    else:
        bystander_count = 1
    
    # Format counts
    vic_text = f"{victim_count} victim" + ("s" if victim_count != 1 else "")
    bys_text = f"{bystander_count} bystander" + ("s" if bystander_count != 1 else "")
    
    return f"{vic_text} and {bys_text}"


def _build_compound_aggressor_action_victim(entry: dict) -> str:
    """Build compound answer for aggressor + action + victim (who did what to whom)."""
    aggressor = _format_people(entry.get("aggressor"))
    action = entry.get("action")
    victim = _format_people(entry.get("victim"))
    
    if aggressor is None and action is None and victim is None:
        return "No one did anything to anyone"
    elif aggressor is None:
        return f"Unknown person performed {action or 'unknown action'} on {victim or 'unknown target'}"
    elif action is None:
        return f"{aggressor} performed unknown action on {victim or 'unknown target'}"
    elif victim is None:
        return f"{aggressor} performed {action} on unknown target"
    else:
        return f"{aggressor} performed {action} on {victim}"


def _count_role(entry: dict, key: str) -> int:
    value = entry.get(key)
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        return 1
    return 0


QUESTION_TEMPLATES = {
    QuestionType.PRIMARY_ACTION: QuestionTemplate(
        question_type=QuestionType.PRIMARY_ACTION,
        prompt="Which of the following best describes the primary action taking place in the video?",
        correct_answer_builder=lambda e: build_action_answer(e, "aggressor", "victim"),
        distractor_pool="action_statements",
        static_distractor="No action takes place",
        requires_fields=("aggressor", "victim", "action"),
    ),
    QuestionType.AGGRESSOR_ID: QuestionTemplate(
        question_type=QuestionType.AGGRESSOR_ID,
        prompt="Who in the video is performing the aggressive behavior?",
        correct_answer_builder=lambda e: _get_person_value(e, "aggressor"),
        distractor_pool="people",
        static_distractor="No individual displays aggressive behavior",
        requires_fields=("aggressor",),
    ),
    QuestionType.VICTIM_RECOGNITION: QuestionTemplate(
        question_type=QuestionType.VICTIM_RECOGNITION,
        prompt="Which person appears to be the target or victim of the aggression shown?",
        correct_answer_builder=lambda e: _get_person_value(e, "victim"),
        distractor_pool="people",
        static_distractor="No one appears to be victimized",
        requires_fields=("victim",),
    ),
    QuestionType.BYSTANDER_DETECTION: QuestionTemplate(
        question_type=QuestionType.BYSTANDER_DETECTION,
        prompt="Who is present during the incident but does not directly participate?",
        correct_answer_builder=lambda e: _get_person_value(e, "bystander"),
        distractor_pool="people",
        static_distractor="No bystanders are present",
        requires_fields=("bystander",),
    ),
    QuestionType.PERSPECTIVE_AGGRESSOR: QuestionTemplate(
        question_type=QuestionType.PERSPECTIVE_AGGRESSOR,
        prompt="If you were observing this scene, which person would you describe as the aggressor?",
        correct_answer_builder=lambda e: _get_person_value(e, "aggressor"),
        distractor_pool="people",
        static_distractor="None of the above appear aggressive",
        requires_fields=("aggressor",),
    ),
    QuestionType.SCENE_LOCATION: QuestionTemplate(
        question_type=QuestionType.SCENE_LOCATION,
        prompt="What location or setting is the video taking place in?",
        correct_answer_builder=lambda e: e.get("environment") if e.get("environment") is not None else "Location unclear / not shown",
        distractor_pool="environments",
        static_distractor="Unclear / not shown",
        requires_fields=("environment",),
    ),
    QuestionType.INTERACTION_SUMMARY: QuestionTemplate(
        question_type=QuestionType.INTERACTION_SUMMARY,
        prompt="Which summary most accurately describes the interactions between the people in the video?",
        correct_answer_builder=lambda e: build_action_answer(e, "aggressor", "victim"),
        distractor_pool="action_statements",
        static_distractor="No meaningful interaction occurs",
        requires_fields=("aggressor", "victim", "action"),
    ),
    QuestionType.EVENT_SEQUENCE: QuestionTemplate(
        question_type=QuestionType.EVENT_SEQUENCE,
        prompt="Which sequence best matches the order of events that occur in the video?",
        correct_answer_builder=lambda e: build_action_answer(e, "aggressor", "victim"),
        distractor_pool="action_statements",
        static_distractor="No sequence observed",
        requires_fields=("aggressor", "victim", "action"),
    ),
    QuestionType.SOCIAL_APPROPRIATENESS: QuestionTemplate(
        question_type=QuestionType.SOCIAL_APPROPRIATENESS,
        prompt="Which action in the video would most likely be considered socially inappropriate or aggressive?",
        correct_answer_builder=lambda e: _get_social_appropriateness_answer(e),
        distractor_pool="action_by_role",
        static_distractor="No actions performed are socially inappropriate",
        requires_fields=("aggressor",),
    ),
    QuestionType.ROLE_COUNT: QuestionTemplate(
        question_type=QuestionType.ROLE_COUNT,
        prompt=None,  # Dynamically set based on role
        correct_answer_builder=None,  # Dynamically set based on role
        distractor_pool="counts",
        requires_fields=(),
    ),
    QuestionType.COMPOUND_AGGRESSOR_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_LOCATION,
        prompt="What is the aggressor wearing and what location is the event taking place?",
        correct_answer_builder=lambda e: _build_compound_aggressor_location(e),
        distractor_pool="compound_aggressor_location",
        static_distractor="Unable to determine aggressor or location",
        requires_fields=("aggressor", "environment"),
    ),
    QuestionType.COMPOUND_ACTION_VICTIMS: QuestionTemplate(
        question_type=QuestionType.COMPOUND_ACTION_VICTIMS,
        prompt="Describe the action taking place in the video and how many victims are present",
        correct_answer_builder=lambda e: _build_compound_action_victims(e),
        distractor_pool="compound_action_victims",
        static_distractor="No action with no victims",
        requires_fields=("action", "victim"),
    ),
    QuestionType.COMPOUND_ACTION_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_ACTION_LOCATION,
        prompt="What action is taking place and where is it happening?",
        correct_answer_builder=lambda e: _build_compound_action_location(e),
        distractor_pool="compound_action_location",
        static_distractor="No action and location unclear",
        requires_fields=("action", "environment"),
    ),
    QuestionType.COMPOUND_AGGRESSOR_VICTIM: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_VICTIM,
        prompt="Describe both the aggressor and the victim in this video",
        correct_answer_builder=lambda e: _build_compound_aggressor_victim(e),
        distractor_pool="compound_aggressor_victim",
        static_distractor="No aggressor and no victim",
        requires_fields=("aggressor", "victim"),
    ),
    QuestionType.COMPOUND_BYSTANDER_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_BYSTANDER_LOCATION,
        prompt="Who is present but not participating, and where is this taking place?",
        correct_answer_builder=lambda e: _build_compound_bystander_location(e),
        distractor_pool="compound_bystander_location",
        static_distractor="No bystanders and location unclear",
        requires_fields=("bystander", "environment"),
    ),
    QuestionType.COMPOUND_AGGRESSOR_VICTIM_COUNT: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_VICTIM_COUNT,
        prompt="How many aggressors and how many victims are in the video?",
        correct_answer_builder=lambda e: _build_compound_aggressor_victim_count(e),
        distractor_pool="compound_aggressor_victim_count",
        static_distractor="0 aggressors and 0 victims",
        requires_fields=("aggressor", "victim"),
    ),
    QuestionType.COMPOUND_VICTIM_BYSTANDER_COUNT: QuestionTemplate(
        question_type=QuestionType.COMPOUND_VICTIM_BYSTANDER_COUNT,
        prompt="How many people are victimized and how many are just observing?",
        correct_answer_builder=lambda e: _build_compound_victim_bystander_count(e),
        distractor_pool="compound_victim_bystander_count",
        static_distractor="0 victims and 0 bystanders",
        requires_fields=("victim", "bystander"),
    ),
    QuestionType.COMPOUND_AGGRESSOR_ACTION_VICTIM: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_ACTION_VICTIM,
        prompt="Who did what to whom?",
        correct_answer_builder=lambda e: _build_compound_aggressor_action_victim(e),
        distractor_pool="compound_aggressor_action_victim",
        static_distractor="No one did anything to anyone",
        requires_fields=("aggressor", "action", "victim"),
    ),
}

ROLE_COUNT_PROMPTS = {
    "aggressor": "How many aggressive people are in the video?",
    "victim": "How many victims are in the video?",
    "bystander": "How many bystanders are in the video?",
}

COUNT_OPTIONS = ["0", "1", "2", "More than 2"]