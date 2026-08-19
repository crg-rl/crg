import re


def compute_format_correctness(solution_str) -> float:
    content = str(solution_str).strip()
    solution_pattern = re.compile(
        r'<think>.*?</think>.*<answer>.*?</answer>',
        re.DOTALL,
    )
    format_match = re.fullmatch(solution_pattern, content)
    if not format_match:
        return 0.0
    think_matches = list(
        re.finditer(r'<think>(.*?)</think>', content, re.DOTALL)
    )
    answer_matches = list(
        re.finditer(r'<answer>(.*?)</answer>', content, re.DOTALL)
    )
    if len(think_matches) != 1 or len(answer_matches) != 1:
        return 0.0
    think_content = think_matches[0].group(1)
    if '<think>' in think_content or '<answer>' in think_content:
        return 0.0
    answer_content = answer_matches[0].group(1)
    if '<answer>' in answer_content or '<think>' in answer_content:
        return 0.0
    return 1.0


def extract_answer(solution_str) -> str | None:
    content = str(solution_str)
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    match = answer_pattern.search(content)
    if match:
        content = match.group(1)
    for char in content:
        if char.isupper():
            return char
    return None


def compute_content_correctness(answer: str | None, ground_truth) -> float:
    if ground_truth is None:
        raise ValueError('VisuLogic reward requires non-empty ground_truth')
    truth = str(ground_truth).strip()
    if not truth:
        raise ValueError('VisuLogic reward requires non-empty ground_truth')
    return 1.0 if answer == truth[0] else 0.0


def compute_score(
    data_source,  # noqa: ARG001
    solution_str,
    ground_truth,
    extra_info=None,  # noqa: ARG001
) -> float:
    format_reward = compute_format_correctness(solution_str)
    answer_str = extract_answer(solution_str)
    content_reward = compute_content_correctness(answer_str, ground_truth)
    return format_reward * 0.2 + content_reward * 0.8
