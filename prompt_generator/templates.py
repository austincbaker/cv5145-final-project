import random
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
    SOCIAL_APPROPRIATENESS = "social_appropriateness"
    ROLE_COUNT_AGGRESSOR = "role_count_aggressor"
    ROLE_COUNT_VICTIM = "role_count_victim"
    ROLE_COUNT_BYSTANDER = "role_count_bystander"
    COMPOUND_AGGRESSOR_LOCATION = "compound_aggressor_location"
    COMPOUND_ACTION_VICTIMS = "compound_action_victims"
    COMPOUND_ACTION_LOCATION = "compound_action_location"
    COMPOUND_AGGRESSOR_VICTIM = "compound_aggressor_victim"
    COMPOUND_BYSTANDER_LOCATION = "compound_bystander_location"
    COMPOUND_AGGRESSOR_VICTIM_COUNT = "compound_aggressor_victim_count"
    COMPOUND_VICTIM_BYSTANDER_COUNT = "compound_victim_bystander_count"
    COMPOUND_AGGRESSOR_ACTION_VICTIM = "compound_aggressor_action_victim"
    ROLE_IDENTIFICATION = "role_identification"
    SEQUENCE_VERIFICATION = "sequence_verification"


class QuestionCategory(Enum):
    SIMPLE = "simple"
    COMPOUND = "compound"
    COMPLEX = "complex"
    COUNTING = "counting"
    IDENTIFICATION = "identification"


@dataclass
class QuestionTemplate:
    question_type: QuestionType
    prompt: str
    correct_answer_builder: Callable
    distractor_pool: str
    static_distractor: str | None = None
    requires_fields: tuple = ()
    source_role: str | None = None  # Role providing the correct answer; used to pull other-role people as priority distractors
    same_entry_distractor_builder: Callable | None = None  # Optional: builds a same-annotation compound distractor
    distractors_override_builder: Callable | None = None  # Optional: (entry, bank, num, correct) -> list[str]; replaces _sample_distractors
    same_video_only: bool = False  # When True, _sample_distractors skips global pool and generic fallback


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
        filtered = [p for p in value if p and isinstance(p, str) and p.strip()]
        if not filtered:
            return None
        if len(filtered) == 1:
            return filtered[0]
        elif len(filtered) == 2:
            return f"{filtered[0]} and {filtered[1]}"
        else:
            return ", ".join(filtered[:-1]) + f", and {filtered[-1]}"
    if isinstance(value, str) and value.strip():
        return value.strip()
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
    """Build compound answer for action description + victim description."""
    action = entry.get("action")
    victim = _format_people(entry.get("victim"))
    
    # Get victim description 
    if victim is None:
        victim_text = "No one appears to be victimized"
    else:
        victim_text = victim
    
    # Format action
    if action is None:
        action_text = "No action"
    else:
        action_text = action
    
    return f"{action_text}; Victim: {victim_text}"


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
    style = random.randint(0, 2)
    entry["_format_style_av"] = style

    if aggressor is None and victim is None:
        return random.choice([
            "No aggressor described and no victim described",
            "Neither an aggressor nor a victim can be identified",
        ])
    elif aggressor is None:
        return random.choice([
            f"No aggressor; Victim: {victim}",
            f"No aggressor identified; {victim} is the victim",
        ])
    elif victim is None:
        return random.choice([
            f"Aggressor: {aggressor}; No victim",
            f"{aggressor} is the aggressor; no victim identified",
        ])
    else:
        return _format_aggressor_victim_pair(aggressor, victim, style)


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
    style = random.randint(0, 2)
    entry["_format_style_aav"] = style

    if aggressor is None and action is None and victim is None:
        return random.choice([
            "No one did anything to anyone",
            "No aggressive interaction occurred",
        ])
    elif aggressor is None:
        aggressor = "Unknown person"
    if action is None:
        action = "unknown action"
    if victim is None:
        victim = "unknown target"

    return _format_aggressor_action_victim(aggressor, action, victim, style)


def _count_role(entry: dict, key: str) -> int:
    value = entry.get(key)
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        return 1
    return 0


def _get_role_count_answer(entry: dict, role: str) -> str:
    """Get answer for role count questions, returning standardized count string."""
    count = _count_role(entry, role)
    if count == 0:
        return "0"
    elif count == 1:
        return "1"
    elif count == 2:
        return "2"
    else:
        return "More than 2"


def _format_aggressor_victim_pair(aggressor: str, victim: str, style: int = 0) -> str:
    """Format an aggressor/victim pair in the given style."""
    if style == 0:
        return f"Aggressor: {aggressor}; Victim: {victim}"
    elif style == 1:
        return f"{aggressor} is the aggressor and {victim} is the victim"
    else:
        return f"{aggressor} attacked {victim}"


def _format_aggressor_action_victim(aggressor: str, action: str, victim: str, style: int = 0) -> str:
    """Format an aggressor/action/victim triple in the given style."""
    if style == 0:
        return f"{aggressor} performed {action} on {victim}"
    elif style == 1:
        return f"{aggressor} committed {action} against {victim}"
    else:
        return f"The action of {action} was carried out by {aggressor} on {victim}"


def _specific_bystander(entry: dict) -> str | None:
    """Return the bystander description only if individually described (not a group)."""
    bystander = _format_people(entry.get("bystander"))
    if bystander is None or "group of" in bystander.lower():
        return None
    return bystander


def _pick_wrong_value(pool, exclude: set, fallback: str) -> str:
    """Pick one value from pool not in exclude; return fallback if none available."""
    candidates = [v for v in pool if v not in exclude]
    if not candidates:
        return fallback
    random.shuffle(candidates)
    return candidates[0]


def _build_compound_action_location_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str
) -> list[str]:
    """Build distractors for COMPOUND_ACTION_LOCATION as balanced mixed pairs.

    Always includes the static 'no action / no location' distractor.
    Remaining slots are split evenly between:
      - correct action + wrong location, and
      - wrong action + correct location
    """
    action = entry.get("action")
    environment = entry.get("environment")

    # Balanced grid for 4-option mode: each component appears exactly 2/4 times
    if num_distractors == 3 and action and environment:
        a_prime = _pick_wrong_value(bank.actions, {action}, "unknown action")
        b_prime = _pick_wrong_value(bank.environments, {environment}, "unclear location")
        d1 = f"{action} in {b_prime}"       # correct action, wrong env
        d2 = f"{a_prime} in {environment}"  # wrong action, correct env
        d3 = f"{a_prime} in {b_prime}"      # both wrong
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_actions = [a for a in bank.actions if a != action]
    wrong_envs = [e for e in bank.environments if e != environment]

    # Always include the static "none" option as a guaranteed distractor
    static = "No action and location unclear"
    guaranteed: list[str] = []
    if static != correct_answer:
        guaranteed.append(static)

    # Guarantee at least one wrong-action + correct-location distractor so the
    # correct location always appears paired with an incorrect action, forcing
    # the model to verify the action rather than confirming by location alone.
    if environment and wrong_actions:
        random.shuffle(wrong_actions)
        guaranteed_wrong_action = f"{wrong_actions[0]} in {environment}"
        if guaranteed_wrong_action != correct_answer and guaranteed_wrong_action not in guaranteed:
            guaranteed.append(guaranteed_wrong_action)

    remaining = num_distractors - len(guaranteed)
    half = remaining // 2

    # Build each type separately
    correct_action_wrong_env: list[str] = []
    if action:
        for env in wrong_envs:
            correct_action_wrong_env.append(f"{action} in {env}")

    wrong_action_correct_env: list[str] = []
    if environment:
        for act in wrong_actions:
            wrong_action_correct_env.append(f"{act} in {environment}")

    correct_action_wrong_env = [c for c in correct_action_wrong_env if c != correct_answer]
    wrong_action_correct_env = [c for c in wrong_action_correct_env if c != correct_answer]

    random.shuffle(correct_action_wrong_env)
    random.shuffle(wrong_action_correct_env)

    # Take half from each type, then backfill from whichever has remainder
    selected = correct_action_wrong_env[:half] + wrong_action_correct_env[:half]
    used = set(guaranteed + selected)
    leftover = [
        c for c in correct_action_wrong_env[half:] + wrong_action_correct_env[half:]
        if c not in used
    ]
    random.shuffle(leftover)

    pool = [c for c in selected + leftover if c not in set(guaranteed)]
    combined = guaranteed + pool

    # Fallback: randomly sample action/environment pairs from the bank
    if len(combined) < num_distractors:
        seen = set(combined)
        actions_list = list(bank.actions)
        envs_list = list(bank.environments)
        max_attempts = (num_distractors - len(combined)) * 30
        fallback_candidates: list[str] = []
        for _ in range(max_attempts):
            r = random.random()
            if actions_list and envs_list and r < 0.5:
                candidate = f"{random.choice(actions_list)} in {random.choice(envs_list)}"
            elif actions_list and r < 0.75:
                candidate = f"{random.choice(actions_list)} in unclear location"
            elif envs_list:
                candidate = f"No action in {random.choice(envs_list)}"
            else:
                break
            if candidate != correct_answer and candidate not in seen:
                fallback_candidates.append(candidate)
                seen.add(candidate)
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda c: abs(len(c) - len(correct_answer)))
        for candidate in fallback_candidates:
            if len(combined) >= num_distractors:
                break
            if candidate.lower() not in set(c.lower() for c in combined):
                combined.append(candidate)

    return combined[:num_distractors]


def _build_compound_aggressor_location_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str
) -> list[str]:
    """Build distractors for COMPOUND_AGGRESSOR_LOCATION as mixed correct/wrong pairs.

    Each distractor is either:
      - correct aggressor + wrong location, or
      - wrong person + correct location.
    The victim from the same annotation is always included as a guaranteed distractor.
    Falls back to bank.people × bank.environments when the main pool is insufficient.
    """
    aggressor = _format_people(entry.get("aggressor"))
    victim = _format_people(entry.get("victim"))
    environment = entry.get("environment")

    # Balanced grid for 4-option mode: each component appears exactly 2/4 times.
    # Wrong aggressor is always an in-video person (victim/bystander) so the model
    # cannot eliminate them by reasoning "that person isn't in this video."
    if num_distractors == 3 and aggressor and environment:
        bystander = _specific_bystander(entry)
        in_video_others = [p for p in [victim, bystander] if p and p != aggressor]
        if in_video_others:
            a_prime = random.choice(in_video_others)
        else:
            a_prime = _pick_wrong_value(bank.people, {aggressor}, "unknown person")
        b_prime = _pick_wrong_value(bank.environments, {environment}, "unclear location")
        d1 = f"{aggressor} in {b_prime}"
        d2 = f"{a_prime} in {environment}"
        d3 = f"{a_prime} in {b_prime}"
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_envs = [e for e in bank.environments if e != environment]
    wrong_people = [p for p in bank.people if p != aggressor]

    pool: list[str] = []

    # Correct aggressor + wrong location
    if aggressor:
        for env in wrong_envs:
            pool.append(f"{aggressor} in {env}")

    # Wrong person + correct location
    if environment:
        for p in wrong_people:
            pool.append(f"{p} in {environment}")

    pool = [c for c in pool if c != correct_answer]
    random.shuffle(pool)

    # Always include the victim and (if specific) the bystander from the same annotation
    guaranteed: list[str] = []
    if victim:
        vic_distractor = (
            f"{victim} in {environment}" if environment else f"{victim}; location unclear"
        )
        if vic_distractor != correct_answer:
            guaranteed.append(vic_distractor)

    bystander = _specific_bystander(entry)
    if bystander:
        bys_distractor = (
            f"{bystander} in {environment}" if environment else f"{bystander}; location unclear"
        )
        if bys_distractor != correct_answer and bys_distractor not in guaranteed:
            guaranteed.append(bys_distractor)

    pool = [c for c in pool if c not in guaranteed]
    combined = guaranteed + pool

    # Fallback: randomly sample person/environment pairs from the bank
    if len(combined) < num_distractors:
        seen = set(combined)
        people_list = list(bank.people)
        envs_list = list(bank.environments)
        max_attempts = (num_distractors - len(combined)) * 30
        fallback_candidates: list[str] = []
        for _ in range(max_attempts):
            r = random.random()
            if people_list and envs_list and r < 0.5:
                candidate = f"{random.choice(people_list)} in {random.choice(envs_list)}"
            elif envs_list and r < 0.75:
                candidate = f"No aggressor present; location is {random.choice(envs_list)}"
            elif people_list:
                candidate = f"{random.choice(people_list)}; location unclear"
            else:
                break
            if candidate != correct_answer and candidate not in seen:
                fallback_candidates.append(candidate)
                seen.add(candidate)
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda c: abs(len(c) - len(correct_answer)))
        for candidate in fallback_candidates:
            if len(combined) >= num_distractors:
                break
            if candidate.lower() not in set(c.lower() for c in combined):
                combined.append(candidate)

    return combined[:num_distractors]


def _build_compound_action_victims_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str,
    max_same_action: int = 3,
) -> list[str]:
    """Build distractors for COMPOUND_ACTION_VICTIMS as mixed correct/wrong pairs.

    Guaranteed distractor: role reversal (aggressor listed as victim).
    At most `max_same_action` distractors may share the correct action; remaining
    slots are filled with wrong-action distractors for variety.
    """
    action = entry.get("action")
    aggressor = _format_people(entry.get("aggressor"))
    victim = _format_people(entry.get("victim"))

    # Balanced grid for 4-option mode: each component appears exactly 2/4 times.
    # Wrong victim is always an in-video person (aggressor/bystander) so the model
    # must determine who actually received the action, not just who is in the video.
    if num_distractors == 3 and action and victim:
        a_prime = _pick_wrong_value(bank.actions, {action}, "unknown action")
        bystander = _specific_bystander(entry)
        in_video_others = [p for p in [aggressor, bystander] if p and p != victim]
        if in_video_others:
            b_prime = random.choice(in_video_others)
        else:
            b_prime = _pick_wrong_value(bank.people, {victim}, "unknown person")
        d1 = f"{action}; Victim: {b_prime}"
        d2 = f"{a_prime}; Victim: {victim}"
        d3 = f"{a_prime}; Victim: {b_prime}"
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_actions = [a for a in bank.actions if a != action]
    wrong_people = [p for p in bank.people if p != victim]

    # Guaranteed: role reversal (uses correct action)
    guaranteed: list[str] = []
    if action and aggressor:
        reversal = f"{action}; Victim: {aggressor}"
        if reversal != correct_answer:
            guaranteed.append(reversal)

    bystander = _specific_bystander(entry)
    if action and bystander:
        bys_distractor = f"{action}; Victim: {bystander}"
        if bys_distractor != correct_answer and bys_distractor not in guaranteed:
            guaranteed.append(bys_distractor)

    seen = set(g.lower() for g in guaranteed)
    same_action_count = len(guaranteed)  # guaranteed items use the correct action

    # Separate pool: distractors using correct action vs. a different action
    same_action_pool: list[str] = []
    diff_action_pool: list[str] = []

    if action:
        for p in wrong_people:
            item = f"{action}; Victim: {p}"
            if item != correct_answer and item.lower() not in seen:
                same_action_pool.append(item)

    if victim:
        for act in wrong_actions:
            item = f"{act}; Victim: {victim}"
            if item != correct_answer:
                diff_action_pool.append(item)

    random.shuffle(same_action_pool)
    random.shuffle(diff_action_pool)

    selected = list(guaranteed)

    # Fill with different-action distractors first for variety
    for item in diff_action_pool:
        if len(selected) >= num_distractors:
            break
        if item.lower() not in seen:
            selected.append(item)
            seen.add(item.lower())

    # Fill remaining slots with same-action distractors, up to the limit
    for item in same_action_pool:
        if len(selected) >= num_distractors:
            break
        if same_action_count >= max_same_action:
            break
        if item.lower() not in seen:
            selected.append(item)
            seen.add(item.lower())
            same_action_count += 1

    # Fallback: randomly sample action/person pairs from the bank
    if len(selected) < num_distractors:
        actions_list = list(bank.actions)
        people_list = list(bank.people)
        max_attempts = (num_distractors - len(selected)) * 30
        fallback_candidates: list[tuple[str, bool]] = []  # (candidate, is_same_action)
        for _ in range(max_attempts):
            if not actions_list:
                break
            # Prefer wrong actions in fallback to maintain variety
            wrong_acts = [a for a in actions_list if a != action]
            act = random.choice(wrong_acts if wrong_acts else actions_list)
            is_same_action = (act == action)
            if is_same_action and same_action_count >= max_same_action:
                continue
            if people_list and random.random() < 0.85:
                candidate = f"{act}; Victim: {random.choice(people_list)}"
            else:
                candidate = f"{act}; Victim: No one appears to be victimized"
            if candidate != correct_answer and candidate.lower() not in seen:
                fallback_candidates.append((candidate, is_same_action))
                seen.add(candidate.lower())
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda t: abs(len(t[0]) - len(correct_answer)))
        for candidate, is_same_action in fallback_candidates:
            if len(selected) >= num_distractors:
                break
            if is_same_action and same_action_count >= max_same_action:
                continue
            selected.append(candidate)
            if is_same_action:
                same_action_count += 1

    return selected[:num_distractors]


def _build_compound_aggressor_victim_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str
) -> list[str]:
    """Build distractors for COMPOUND_AGGRESSOR_VICTIM as mixed correct/wrong pairs.

    Guaranteed distractor: role reversal (victim labeled as aggressor, aggressor as victim).
    Remaining slots: correct aggressor + wrong victim, or wrong aggressor + correct victim.
    """
    style = entry.get("_format_style_av", 0)
    aggressor = _format_people(entry.get("aggressor"))
    victim = _format_people(entry.get("victim"))

    # Balanced grid for 4-option mode using only in-video people as distractors.
    # All three wrong options use only people who appear in the video so the model
    # cannot eliminate any option by reasoning "that person isn't visible."
    #   D1: role reversal — forces model to identify who is actually the aggressor
    #   D2: bystander as aggressor (or foreign fallback) + correct victim
    #   D3: correct aggressor + bystander as victim (or foreign fallback)
    if num_distractors == 3 and aggressor and victim:
        bystander = _specific_bystander(entry)
        d1 = _format_aggressor_victim_pair(victim, aggressor, style)  # role reversal
        if bystander:
            d2 = _format_aggressor_victim_pair(bystander, victim, style)
            d3 = _format_aggressor_victim_pair(aggressor, bystander, style)
        else:
            foreign = _pick_wrong_value(bank.people, {aggressor, victim}, "unknown person")
            d2 = _format_aggressor_victim_pair(foreign, victim, style)
            d3 = _format_aggressor_victim_pair(aggressor, foreign, style)
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_aggressors = [p for p in bank.people if p != aggressor and p != victim]
    wrong_victims = [p for p in bank.people if p != victim and p != aggressor]

    pool: list[str] = []

    # Correct aggressor + wrong victim
    if aggressor:
        for p in wrong_victims:
            pool.append(_format_aggressor_victim_pair(aggressor, p, style))

    # Wrong aggressor + correct victim
    if victim:
        for p in wrong_aggressors:
            pool.append(_format_aggressor_victim_pair(p, victim, style))

    pool = [c for c in pool if c != correct_answer]
    random.shuffle(pool)

    # Guaranteed: role reversal (swap aggressor and victim labels)
    guaranteed: list[str] = []
    if aggressor and victim:
        reversal = _format_aggressor_victim_pair(victim, aggressor, style)
        if reversal != correct_answer:
            guaranteed.append(reversal)

    bystander = _specific_bystander(entry)
    if bystander and victim:
        bys_distractor = _format_aggressor_victim_pair(bystander, victim, style)
        if bys_distractor != correct_answer and bys_distractor not in guaranteed:
            guaranteed.append(bys_distractor)
    if bystander and aggressor:
        bys_as_victim = _format_aggressor_victim_pair(aggressor, bystander, style)
        if bys_as_victim != correct_answer and bys_as_victim not in guaranteed:
            guaranteed.append(bys_as_victim)

    pool = [c for c in pool if c not in guaranteed]
    combined = guaranteed + pool

    # Fallback: randomly sample person pairs from the bank
    if len(combined) < num_distractors:
        seen = set(combined)
        people_list = list(bank.people)
        max_attempts = (num_distractors - len(combined)) * 30
        fallback_candidates: list[str] = []
        for _ in range(max_attempts):
            if not people_list:
                break
            r = random.random()
            if len(people_list) >= 2 and r < 0.6:
                p1, p2 = random.sample(people_list, 2)
                candidate = _format_aggressor_victim_pair(p1, p2, style)
            elif r < 0.8:
                candidate = f"No aggressor; Victim: {random.choice(people_list)}"
            else:
                candidate = f"Aggressor: {random.choice(people_list)}; No victim"
            if candidate != correct_answer and candidate not in seen:
                fallback_candidates.append(candidate)
                seen.add(candidate)
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda c: abs(len(c) - len(correct_answer)))
        for candidate in fallback_candidates:
            if len(combined) >= num_distractors:
                break
            if candidate.lower() not in set(c.lower() for c in combined):
                combined.append(candidate)

    return combined[:num_distractors]


def _build_compound_bystander_location_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str
) -> list[str]:
    """Build distractors for COMPOUND_BYSTANDER_LOCATION as mixed correct/wrong pairs.

    Each distractor is either:
      - correct bystander + wrong location, or
      - wrong person + correct location
    """
    bystander = _format_people(entry.get("bystander"))
    environment = entry.get("environment")

    # Balanced grid for 4-option mode: each component appears exactly 2/4 times
    if num_distractors == 3 and bystander and environment:
        a_prime = _pick_wrong_value(bank.people, {bystander}, "unknown person")
        b_prime = _pick_wrong_value(bank.environments, {environment}, "unclear location")
        d1 = f"{bystander} in {b_prime}"
        d2 = f"{a_prime} in {environment}"
        d3 = f"{a_prime} in {b_prime}"
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_envs = [e for e in bank.environments if e != environment]
    wrong_people = [p for p in bank.people if p != bystander]

    candidates: list[str] = []

    # Correct bystander + wrong location
    if bystander:
        for env in wrong_envs:
            candidates.append(f"{bystander} in {env}")

    # Wrong person + correct location
    if environment:
        for p in wrong_people:
            candidates.append(f"{p} in {environment}")

    candidates = [c for c in candidates if c != correct_answer]
    random.shuffle(candidates)
    combined = candidates

    # Fallback: randomly sample person/environment pairs from the bank
    if len(combined) < num_distractors:
        seen = set(combined)
        people_list = list(bank.people)
        envs_list = list(bank.environments)
        max_attempts = (num_distractors - len(combined)) * 30
        fallback_candidates: list[str] = []
        for _ in range(max_attempts):
            r = random.random()
            if people_list and envs_list and r < 0.5:
                candidate = f"{random.choice(people_list)} in {random.choice(envs_list)}"
            elif people_list and r < 0.75:
                candidate = f"{random.choice(people_list)} in unclear location"
            elif envs_list:
                candidate = f"No bystanders in {random.choice(envs_list)}"
            else:
                break
            if candidate != correct_answer and candidate not in seen:
                fallback_candidates.append(candidate)
                seen.add(candidate)
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda c: abs(len(c) - len(correct_answer)))
        for candidate in fallback_candidates:
            if len(combined) >= num_distractors:
                break
            if candidate.lower() not in set(c.lower() for c in combined):
                combined.append(candidate)

    return combined[:num_distractors]


def _build_compound_aggressor_action_victim_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str,
    max_same_action: int = 2,
    max_same_aggressor: int = 2,
    max_same_victim: int = 2,
) -> list[str]:
    """Build distractors for COMPOUND_AGGRESSOR_ACTION_VICTIM by swapping one component.

    Guaranteed distractor: role reversal (victim performed action on aggressor).
    Each swap type affects exactly two same-X counters (the one component that varies
    does not count). All three are capped independently for variety across dimensions.

      wrong aggressor → same_action + same_victim
      wrong action    → same_aggressor + same_victim
      wrong victim    → same_action + same_aggressor
    """
    style = entry.get("_format_style_aav", 0)
    aggressor = _format_people(entry.get("aggressor"))
    action = entry.get("action")
    victim = _format_people(entry.get("victim"))

    # Balanced grid for 4-option mode using only in-video people, each distractor
    # isolating exactly one dimension so the model must correctly identify all three:
    #   D1: role reversal        — tests who is aggressor vs victim
    #   D2: wrong action         — tests what action is occurring
    #   D3: bystander as aggressor (correct action+victim) — tests who initiates;
    #       falls back to role-reversal+wrong-action combo when no bystander exists
    if num_distractors == 3 and aggressor and action and victim:
        b_prime = _pick_wrong_value(bank.actions, {action}, "unknown action")
        bystander = _specific_bystander(entry)
        d1 = _format_aggressor_action_victim(victim, action, aggressor, style)       # role reversal
        d2 = _format_aggressor_action_victim(aggressor, b_prime, victim, style)      # wrong action
        if bystander:
            d3 = _format_aggressor_action_victim(bystander, action, victim, style)   # bystander as aggressor
        else:
            d3 = _format_aggressor_action_victim(victim, b_prime, aggressor, style)  # reversal + wrong action
        result = [d for d in [d1, d2, d3] if d != correct_answer]
        if len(result) == 3:
            return result

    wrong_aggressors = [p for p in bank.people if p != aggressor and p != victim]
    wrong_actions = [a for a in bank.actions if a != action]
    wrong_victims = [p for p in bank.people if p != victim and p != aggressor]

    # Guaranteed: role reversal — correct action, roles swapped (not correct aggressor/victim)
    # Bystander distractor — correct action + correct victim, not correct aggressor
    guaranteed: list[str] = []
    if aggressor and action and victim:
        reversal = _format_aggressor_action_victim(victim, action, aggressor, style)
        if reversal != correct_answer:
            guaranteed.append(reversal)

    bystander = _specific_bystander(entry)
    if bystander and action and victim:
        bys_distractor = _format_aggressor_action_victim(bystander, action, victim, style)
        if bys_distractor != correct_answer and bys_distractor not in guaranteed:
            guaranteed.append(bys_distractor)

    seen = set(g.lower() for g in guaranteed)
    same_action_count = len(guaranteed)      # reversal + bystander both use correct action
    same_aggressor_count = 0                  # neither uses correct aggressor
    same_victim_count = 1 if len(guaranteed) >= 2 else 0  # bystander uses correct victim

    # Tag each pool item with which same-X counters it increments.
    # Each tuple: (item, adds_to_action, adds_to_aggressor, adds_to_victim)
    tagged_pool: list[tuple[str, bool, bool, bool]] = []

    if aggressor and victim:
        for act in wrong_actions:
            item = _format_aggressor_action_victim(aggressor, act, victim, style)
            if item != correct_answer and item.lower() not in seen:
                tagged_pool.append((item, False, True, True))  # wrong action

    if action and victim:
        for p in wrong_aggressors:
            item = _format_aggressor_action_victim(p, action, victim, style)
            if item != correct_answer and item.lower() not in seen:
                tagged_pool.append((item, True, False, True))  # wrong aggressor

    if aggressor and action:
        for p in wrong_victims:
            item = _format_aggressor_action_victim(aggressor, action, p, style)
            if item != correct_answer and item.lower() not in seen:
                tagged_pool.append((item, True, True, False))  # wrong victim

    random.shuffle(tagged_pool)

    selected = list(guaranteed)

    for item, is_sa, is_sAgg, is_sv in tagged_pool:
        if len(selected) >= num_distractors:
            break
        if is_sa and same_action_count >= max_same_action:
            continue
        if is_sAgg and same_aggressor_count >= max_same_aggressor:
            continue
        if is_sv and same_victim_count >= max_same_victim:
            continue
        if item.lower() not in seen:
            selected.append(item)
            seen.add(item.lower())
            if is_sa:
                same_action_count += 1
            if is_sAgg:
                same_aggressor_count += 1
            if is_sv:
                same_victim_count += 1

    # Fallback: randomly sample person/action/person triples from the bank
    if len(selected) < num_distractors:
        people_list = list(bank.people)
        actions_list = list(bank.actions)
        max_attempts = (num_distractors - len(selected)) * 30
        fallback_candidates: list[tuple[str, bool, bool, bool]] = []
        for _ in range(max_attempts):
            if len(people_list) < 2 or not actions_list:
                break
            p1 = random.choice(people_list)
            act = random.choice(actions_list)
            p2_pool = [p for p in people_list if p != p1]
            if not p2_pool:
                break
            p2 = random.choice(p2_pool)
            is_sa = (act == action)
            is_sAgg = (p1 == aggressor)
            is_sv = (p2 == victim)
            if is_sa and same_action_count >= max_same_action:
                continue
            if is_sAgg and same_aggressor_count >= max_same_aggressor:
                continue
            if is_sv and same_victim_count >= max_same_victim:
                continue
            candidate = _format_aggressor_action_victim(p1, act, p2, style)
            if candidate != correct_answer and candidate.lower() not in seen:
                fallback_candidates.append((candidate, is_sa, is_sAgg, is_sv))
                seen.add(candidate.lower())
        # Sort by length proximity to correct answer, take closest matches
        fallback_candidates.sort(key=lambda t: abs(len(t[0]) - len(correct_answer)))
        for candidate, is_sa, is_sAgg, is_sv in fallback_candidates:
            if len(selected) >= num_distractors:
                break
            if is_sa and same_action_count >= max_same_action:
                continue
            if is_sAgg and same_aggressor_count >= max_same_aggressor:
                continue
            if is_sv and same_victim_count >= max_same_victim:
                continue
            selected.append(candidate)
            if is_sa:
                same_action_count += 1
            if is_sAgg:
                same_aggressor_count += 1
            if is_sv:
                same_victim_count += 1

    return selected[:num_distractors]


def _build_primary_action_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str,
    max_same_action: int = 3,
) -> list[str]:
    """Build distractors for PRIMARY_ACTION by swapping one component at a time.

    Guaranteed distractor: role reversal (victim performs action on aggressor).
    At most `max_same_action` distractors may share the correct action; remaining
    slots are filled with wrong-action distractors for variety.
    """
    aggressor = _format_people(entry.get("aggressor")) or "Unknown person"
    action = entry.get("action") or "unknown action"
    victim = _format_people(entry.get("victim")) or "unknown target"

    wrong_aggressors = [p for p in bank.people if p != aggressor and p != victim]
    wrong_actions = [a for a in bank.actions if a != action]
    wrong_victims = [p for p in bank.people if p != victim and p != aggressor]

    # Guaranteed: role reversal (uses correct action)
    guaranteed: list[str] = []
    reversal = f"{victim} performs action of {action} on {aggressor}"
    if reversal != correct_answer:
        guaranteed.append(reversal)

    bystander = _specific_bystander(entry)
    if bystander:
        bys_distractor = f"{bystander} performs action of {action} on {victim}"
        if bys_distractor != correct_answer and bys_distractor not in guaranteed:
            guaranteed.append(bys_distractor)

    seen = set(g.lower() for g in guaranteed)
    same_action_count = len(guaranteed)  # both guaranteed items use the correct action

    # Separate pool: distractors using correct action vs. a different action
    same_action_pool: list[str] = []
    diff_action_pool: list[str] = []

    for p in wrong_aggressors:
        item = f"{p} performs action of {action} on {victim}"
        if item != correct_answer and item.lower() not in seen:
            same_action_pool.append(item)

    for p in wrong_victims:
        item = f"{aggressor} performs action of {action} on {p}"
        if item != correct_answer and item.lower() not in seen:
            same_action_pool.append(item)

    for act in wrong_actions:
        item = f"{aggressor} performs action of {act} on {victim}"
        if item != correct_answer:
            diff_action_pool.append(item)

    random.shuffle(same_action_pool)
    random.shuffle(diff_action_pool)

    selected = list(guaranteed)

    # Fill with different-action distractors first for variety
    for item in diff_action_pool:
        if len(selected) >= num_distractors:
            break
        if item.lower() not in seen:
            selected.append(item)
            seen.add(item.lower())

    # Fill remaining slots with same-action distractors, up to the limit
    for item in same_action_pool:
        if len(selected) >= num_distractors:
            break
        if same_action_count >= max_same_action:
            break
        if item.lower() not in seen:
            selected.append(item)
            seen.add(item.lower())
            same_action_count += 1

    return selected[:num_distractors]


def _build_count_distractors(
    entry: dict, bank, num_distractors: int, correct_answer: str
) -> list[str]:
    """Return all COUNT_OPTIONS except the correct answer, ensuring no gaps in the number line."""
    distractors = [opt for opt in COUNT_OPTIONS if opt != correct_answer]
    # Always include the static distractor if it's not the correct answer
    static = "Unable to determine from the video"
    if static != correct_answer and static not in distractors:
        distractors.append(static)
    random.shuffle(distractors)
    return distractors[:num_distractors]


QUESTION_TEMPLATES = {
    QuestionType.PRIMARY_ACTION: QuestionTemplate(
        question_type=QuestionType.PRIMARY_ACTION,
        prompt="If there is an aggressive action taking place in this video, describe it as one of the following:",
        correct_answer_builder=lambda e: e.get("action") or "unknown action",
        distractor_pool="actions",
        static_distractor="no aggressive action is taking place",
        requires_fields=("action",),
    ),
    QuestionType.AGGRESSOR_ID: QuestionTemplate(
        question_type=QuestionType.AGGRESSOR_ID,
        prompt="Who in the video is performing the aggressive behavior?",
        correct_answer_builder=lambda e: _get_person_value(e, "aggressor"),
        distractor_pool="people",
        static_distractor="No individual displays aggressive behavior",
        requires_fields=("aggressor",),
        source_role="aggressor",
        same_video_only=False,
    ),
    QuestionType.VICTIM_RECOGNITION: QuestionTemplate(
        question_type=QuestionType.VICTIM_RECOGNITION,
        prompt="Which person appears to be the target or victim of the aggression shown?",
        correct_answer_builder=lambda e: _get_person_value(e, "victim"),
        distractor_pool="people",
        static_distractor="No one appears to be victimized",
        requires_fields=("victim",),
        source_role="victim",
        same_video_only=False,
    ),
    QuestionType.BYSTANDER_DETECTION: QuestionTemplate(
        question_type=QuestionType.BYSTANDER_DETECTION,
        prompt="Who is present during the incident but does not directly participate?",
        correct_answer_builder=lambda e: _get_person_value(e, "bystander"),
        distractor_pool="people",
        static_distractor="No bystanders are present",
        requires_fields=("bystander",),
        source_role="bystander",
    ),
    QuestionType.PERSPECTIVE_AGGRESSOR: QuestionTemplate(
        question_type=QuestionType.PERSPECTIVE_AGGRESSOR,
        prompt="If you were observing this scene, which person would you describe as the aggressor?",
        correct_answer_builder=lambda e: _get_person_value(e, "aggressor"),
        distractor_pool="people",
        static_distractor="None of the above appear aggressive",
        requires_fields=("aggressor",),
        source_role="aggressor",
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
        distractors_override_builder=_build_primary_action_distractors,
    ),
    QuestionType.SOCIAL_APPROPRIATENESS: QuestionTemplate(
        question_type=QuestionType.SOCIAL_APPROPRIATENESS,
        prompt="Which action in the video would most likely be considered socially inappropriate or aggressive?",
        correct_answer_builder=lambda e: _get_social_appropriateness_answer(e),
        distractor_pool="action_by_role",
        static_distractor="No actions performed are socially inappropriate",
        requires_fields=("aggressor",),
    ),
    QuestionType.ROLE_COUNT_AGGRESSOR: QuestionTemplate(
        question_type=QuestionType.ROLE_COUNT_AGGRESSOR,
        prompt="How many aggressive people are in the video?",
        correct_answer_builder=lambda e: _get_role_count_answer(e, "aggressor"),
        distractor_pool="counts",
        static_distractor="Unable to determine from the video",
        requires_fields=(),
        distractors_override_builder=_build_count_distractors,
    ),
    QuestionType.ROLE_COUNT_VICTIM: QuestionTemplate(
        question_type=QuestionType.ROLE_COUNT_VICTIM,
        prompt="How many victims are in the video?",
        correct_answer_builder=lambda e: _get_role_count_answer(e, "victim"),
        distractor_pool="counts",
        static_distractor="Unable to determine from the video",
        requires_fields=(),
        distractors_override_builder=_build_count_distractors,
    ),
    QuestionType.ROLE_COUNT_BYSTANDER: QuestionTemplate(
        question_type=QuestionType.ROLE_COUNT_BYSTANDER,
        prompt="How many bystanders are in the video?",
        correct_answer_builder=lambda e: _get_role_count_answer(e, "bystander"),
        distractor_pool="counts",
        static_distractor="Unable to determine from the video",
        requires_fields=(),
        distractors_override_builder=_build_count_distractors,
    ),
    QuestionType.COMPOUND_AGGRESSOR_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_LOCATION,
        prompt="What is the aggressor wearing and what location is the event taking place?",
        correct_answer_builder=lambda e: _build_compound_aggressor_location(e),
        distractor_pool="compound_aggressor_location",
        static_distractor="Unable to determine aggressor or location",
        requires_fields=("aggressor", "environment"),
        distractors_override_builder=_build_compound_aggressor_location_distractors,
    ),
    QuestionType.COMPOUND_ACTION_VICTIMS: QuestionTemplate(
        question_type=QuestionType.COMPOUND_ACTION_VICTIMS,
        prompt="Describe the action taking place in the video and describe any victims that are present",
        correct_answer_builder=lambda e: _build_compound_action_victims(e),
        distractor_pool="compound_action_victims",
        static_distractor="No action; Victim: No one appears to be victimized",
        requires_fields=("action", "victim"),
        distractors_override_builder=_build_compound_action_victims_distractors,
    ),
    QuestionType.COMPOUND_ACTION_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_ACTION_LOCATION,
        prompt="Which of the following most accurately describes both the specific type of aggressive action and the location where it occurs?",
        correct_answer_builder=lambda e: _build_compound_action_location(e),
        distractor_pool="compound_action_location",
        static_distractor="No action and location unclear",
        requires_fields=("action", "environment"),
        distractors_override_builder=_build_compound_action_location_distractors,
    ),
    QuestionType.COMPOUND_AGGRESSOR_VICTIM: QuestionTemplate(
        question_type=QuestionType.COMPOUND_AGGRESSOR_VICTIM,
        prompt="Which of the following correctly identifies the roles of the individuals shown?",
        correct_answer_builder=lambda e: _build_compound_aggressor_victim(e),
        distractor_pool="compound_aggressor_victim",
        static_distractor="No aggressor described and no victim described",
        requires_fields=("aggressor", "victim"),
        distractors_override_builder=_build_compound_aggressor_victim_distractors,
    ),
    QuestionType.COMPOUND_BYSTANDER_LOCATION: QuestionTemplate(
        question_type=QuestionType.COMPOUND_BYSTANDER_LOCATION,
        prompt="Who is present but not participating, and where is this taking place?",
        correct_answer_builder=lambda e: _build_compound_bystander_location(e),
        distractor_pool="compound_bystander_location",
        static_distractor="No bystanders and location unclear",
        requires_fields=("bystander", "environment"),
        distractors_override_builder=_build_compound_bystander_location_distractors,
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
        prompt="Which of the following best describes what happened in the video?",
        correct_answer_builder=lambda e: _build_compound_aggressor_action_victim(e),
        distractor_pool="compound_aggressor_action_victim",
        static_distractor="No one did anything to anyone",
        requires_fields=("aggressor", "action", "victim"),
        distractors_override_builder=_build_compound_aggressor_action_victim_distractors,
    ),
}

COUNT_OPTIONS = [
    "0", "1", "2", "3", "4", "5",
    "More than 2", "More than 3",
]

ROLE_LABELS = [
    "Aggressor", "Victim", "Bystander",
    # "Instigator", "Observer", "Mediator",
    # "Participant", "Witness", "Defender", "Facilitator",
]

ROLE_ID_NO_MATCH = "No one in the video fits that description"

SEQ_ORIGINAL_CORRECT = "The original sequence is correct"
SEQ_NO_MATCH = "No sequences describe the video"


# Category mappings for question distribution
QUESTION_CATEGORIES: dict[QuestionType, QuestionCategory] = {
    # Simple questions (single subject) - 6 types
    QuestionType.AGGRESSOR_ID: QuestionCategory.SIMPLE,
    QuestionType.VICTIM_RECOGNITION: QuestionCategory.SIMPLE,
    QuestionType.BYSTANDER_DETECTION: QuestionCategory.SIMPLE,
    # QuestionType.PERSPECTIVE_AGGRESSOR: QuestionCategory.SIMPLE,
    # QuestionType.SCENE_LOCATION: QuestionCategory.SIMPLE,
    QuestionType.PRIMARY_ACTION: QuestionCategory.SIMPLE,

    # Compound questions (multi-subject) - 7 types
    QuestionType.COMPOUND_AGGRESSOR_LOCATION: QuestionCategory.COMPOUND,
    QuestionType.COMPOUND_ACTION_VICTIMS: QuestionCategory.COMPOUND,
    QuestionType.COMPOUND_ACTION_LOCATION: QuestionCategory.COMPOUND,
    QuestionType.COMPOUND_AGGRESSOR_VICTIM: QuestionCategory.COMPOUND,
    QuestionType.COMPOUND_BYSTANDER_LOCATION: QuestionCategory.COMPOUND,
    # QuestionType.SOCIAL_APPROPRIATENESS: QuestionCategory.COMPOUND,
    # QuestionType.INTERACTION_SUMMARY: QuestionCategory.COMPOUND,

    # Complex questions (multi-subject + difficulty) - 2 types
    QuestionType.COMPOUND_AGGRESSOR_ACTION_VICTIM: QuestionCategory.COMPLEX,
    QuestionType.SEQUENCE_VERIFICATION: QuestionCategory.COMPLEX,

    # Counting questions - 5 types (3 role counts + 2 compound counts)
    QuestionType.ROLE_COUNT_AGGRESSOR: QuestionCategory.COUNTING,
    QuestionType.ROLE_COUNT_VICTIM: QuestionCategory.COUNTING,
    QuestionType.ROLE_COUNT_BYSTANDER: QuestionCategory.COUNTING,
    QuestionType.COMPOUND_AGGRESSOR_VICTIM_COUNT: QuestionCategory.COUNTING,
    QuestionType.COMPOUND_VICTIM_BYSTANDER_COUNT: QuestionCategory.COUNTING,

    # Identification questions - 1 type
    QuestionType.ROLE_IDENTIFICATION: QuestionCategory.IDENTIFICATION,
}

# Distribution configuration: number of questions per category per video
QUESTIONS_PER_CATEGORY = {
    QuestionCategory.SIMPLE: 2,
    QuestionCategory.COMPOUND: 3,
    QuestionCategory.COMPLEX: 1,
    QuestionCategory.COUNTING: 1,
    QuestionCategory.IDENTIFICATION: 1,
}

# Categories whose questions are tracked separately as secondary analysis
SECONDARY_CATEGORIES = {QuestionCategory.COUNTING}
SECONDARY_QUESTION_TYPES = (
    {
        qtype.value
        for qtype, cat in QUESTION_CATEGORIES.items()
        if cat in SECONDARY_CATEGORIES
    }
    | {QuestionType.COMPOUND_ACTION_LOCATION.value}
)