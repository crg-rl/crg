import math
import re
from typing import Literal, Optional


def compute_content_correctness(answer: str, ground_truth: str) -> float:
    reward = 0.0
    if isinstance(answer, str) and len(answer) > 0:
        if answer == ground_truth:
            reward = 1.0
        elif ground_truth in answer:  # reward shaping used in ReasoningGYM
            reward = len(ground_truth) / len(answer)
    return reward


def compute_format_correctness(solution_str) -> float:
    solution_pattern = re.compile(
        r"\s*<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL
    )
    format_match = re.fullmatch(solution_pattern, solution_str)
    if not format_match:
        return 0.0
    think_matches = list(
        re.finditer(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    )
    answer_matches = list(
        re.finditer(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    )
    if len(think_matches) != 1 or len(answer_matches) != 1:
        return 0.0
    think_content = think_matches[0].group(1)
    if "<think>" in think_content or "<answer>" in think_content:
        return 0.0
    answer_content = answer_matches[0].group(1)
    if "<answer>" in answer_content or "<think>" in answer_content:
        return 0.0
    return 1.0


def compute_score(
    data_source,  # noqa: ARG001
    solution_str,
    ground_truth,
    extra_info=None,  # noqa: ARG001
) -> float:
    format_reward = compute_format_correctness(solution_str)
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    m = answer_pattern.search(solution_str)
    answer_str = None if not m else m.group(1).strip()
    content_reward = compute_content_correctness(answer_str, ground_truth)
    return format_reward * 0.2 + content_reward * 0.8
