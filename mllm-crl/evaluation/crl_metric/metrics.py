"""Core continual-learning metric formulas."""

from collections.abc import Sequence
from statistics import fmean

ScoreMatrix = Sequence[Sequence[float | None]]


def diagonal_success_average(matrix: ScoreMatrix) -> float | None:
    values = [
        row[index]
        for index, row in enumerate(matrix)
        if index < len(row) and row[index] is not None
    ]
    return fmean(values) if values else None


def final_average_success(matrix: ScoreMatrix) -> float | None:
    if not matrix:
        raise ValueError("matrix must not be empty")
    final_row = matrix[-1]
    values = [
        score
        for score_index, score in enumerate(final_row)
        if score is not None
        and score_index < len(matrix)
        and matrix[score_index][score_index] is not None
    ]
    return fmean(values) if values else None


def backward_transfer(matrix: ScoreMatrix) -> float:
    if not matrix:
        raise ValueError("matrix must not be empty")
    final_row = matrix[-1]
    values = [
        final_row[index] - matrix[index][index]
        for index in range(len(matrix) - 1)
        if (
            index < len(final_row)
            and index < len(matrix[index])
            and final_row[index] is not None
            and matrix[index][index] is not None
        )
    ]
    return fmean(values) if values else 0.0


def zero_shot_forward_transfer(
    base_scores: Sequence[float | None],
    pre_task_scores: Sequence[float | None],
    *,
    skip_first: bool = True,
) -> float | None:
    """Average zero-shot forward transfer over available task scores.

    ``base_scores[i]`` is the original base model's zero-shot score on task
    ``i``. ``pre_task_scores[i]`` is the score of the checkpoint before task
    ``i`` is trained. With ``skip_first=True``, task 0 is excluded because it
    has no preceding-task checkpoint.
    """

    start_index = 1 if skip_first else 0
    values = [
        pre_task_scores[index] - base_scores[index]
        for index in range(
            start_index,
            min(len(base_scores), len(pre_task_scores)),
        )
        if (
            base_scores[index] is not None
            and pre_task_scores[index] is not None
        )
    ]
    return fmean(values) if values else None


def summarize_continual_metrics(
    matrix: ScoreMatrix,
) -> dict[str, float | None]:
    if not matrix:
        raise ValueError("matrix must not be empty")
    final_avg = final_average_success(matrix)
    return {
        "R_ii": diagonal_success_average(matrix),
        "FinalAvg": final_avg,
        "BWT": backward_transfer(matrix),
    }
