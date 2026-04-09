"""
Reward functions for GRPO training of Vietnamese Luc-Bat poem generation.
Uses check_poem.py rules: rhyme, tone, word count.
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_poem import check_luc_bat_rule

# Tags used in generation
POEM_START = "<POEM>"
POEM_END = "</POEM>"
THINK_START = "<think>"
THINK_END = "</think>"

poem_end_regex = re.compile(
    r"<POEM>(.*?)</POEM>[\s]*$",
    flags=re.MULTILINE | re.DOTALL,
)


def extract_poem(response: str) -> str | None:
    """Extract poem content from <POEM>...</POEM> tags."""
    match = poem_end_regex.search(response)
    if match:
        return match.group(1).strip()
    return None


def reward_format(completions, **kwargs) -> list[float]:
    """Reward model for using correct <POEM>...</POEM> format."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        count_open = response.count(POEM_START)
        count_close = response.count(POEM_END)
        if count_open == 1 and count_close == 1:
            score += 2.0
        else:
            score += -1.0 * abs(count_open - 1)
            score += -1.0 * abs(count_close - 1)
        scores.append(score)
    return scores


def reward_luc_bat_rules(completions, **kwargs) -> list[float]:
    """
    Main reward: score poem using check_luc_bat_rule.
    Score is in range (-100, 0], normalized to (-2, 2].
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        poem = extract_poem(response)
        if poem is None:
            scores.append(-2.0)
            continue
        try:
            raw_score = check_luc_bat_rule(poem)
            # raw_score is <=0, with 0 being perfect
            # Normalize: 0 -> +2.0, -100 -> -2.0
            normalized = max(-2.0, min(2.0, raw_score / 25.0 + 2.0))
            scores.append(normalized)
        except Exception:
            scores.append(-2.0)
    return scores


def reward_line_count(completions, **kwargs) -> list[float]:
    """Reward poems that have an even number of lines (luc-bat requirement)."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        poem = extract_poem(response)
        if poem is None:
            scores.append(-1.0)
            continue
        lines = [l.strip() for l in poem.splitlines() if l.strip()]
        if len(lines) == 0:
            scores.append(-1.0)
        elif len(lines) % 2 == 0:
            scores.append(1.0)
        else:
            scores.append(-0.5)
    return scores


def reward_word_count(completions, **kwargs) -> list[float]:
    """
    Reward poems where lines alternate 6-8 word counts (luc-bat pattern).
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        poem = extract_poem(response)
        if poem is None:
            scores.append(-1.0)
            continue
        lines = [l.strip() for l in poem.splitlines() if l.strip()]
        if len(lines) < 2:
            scores.append(-1.0)
            continue
        score = 0.0
        for i, line in enumerate(lines):
            words = line.split()
            expected = 6 if i % 2 == 0 else 8
            if len(words) == expected:
                score += 0.5
            else:
                score -= 0.3 * abs(len(words) - expected)
        scores.append(max(-2.0, min(2.0, score / max(len(lines), 1))))
    return scores


def reward_think_present(completions, **kwargs) -> list[float]:
    """Small bonus for using <think> reasoning before the poem."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        has_think = THINK_START in response and THINK_END in response
        scores.append(0.5 if has_think else 0.0)
    return scores
