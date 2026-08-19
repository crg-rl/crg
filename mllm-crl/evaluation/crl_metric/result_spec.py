"""Specification for CRL metric result documents."""

from __future__ import annotations

from typing import Any

INPUT_SCHEMA_VERSION = 'crl-metric-input-v1'
RESULT_SCHEMA_VERSION = 'crl-metric-result-v1'
SUMMARY_SCHEMA_VERSION = 'crl-metric-summary-v1'
MAX_RATIO_SCORE = 1.0
MAX_PERCENT_SCORE = 100.0

EXAMPLE_PAIR_SCORE_INPUT: dict[str, Any] = {
    'schema_version': INPUT_SCHEMA_VERSION,
    'suite_id': 'llm_algorithmic',
    'source_format': 'checkpoint_task_scores_v1',
    'task_order': ['task_a', 'task_b', 'task_c'],
    'checkpoints': [
        {
            'stage_id': 'task_a',
            'checkpoint_label': 'after_task_a',
            'checkpoint_path': '/path/to/global_step_50',
        },
        {
            'stage_id': 'task_b',
            'checkpoint_label': 'after_task_b',
            'checkpoint_path': '/path/to/global_step_100',
        },
        {
            'stage_id': 'task_c',
            'checkpoint_label': 'after_task_c',
            'checkpoint_path': '/path/to/global_step_150',
        },
    ],
    'score_records': [
        {
            'checkpoint_label': 'after_task_a',
            'task_name': 'task_a',
            'score': 0.70,
            'score_source': '/path/to/task_a_after_task_a.json',
        },
        {
            'checkpoint_label': 'after_task_a',
            'task_name': 'task_b',
            'score': 0.30,
            'score_source': '/path/to/task_b_pre_task.json',
        },
        {
            'checkpoint_label': 'after_task_b',
            'task_name': 'task_a',
            'score': 0.55,
        },
        {
            'checkpoint_label': 'after_task_b',
            'task_name': 'task_b',
            'score': 0.80,
        },
        {
            'checkpoint_label': 'after_task_b',
            'task_name': 'task_c',
            'score': 0.45,
        },
        {
            'checkpoint_label': 'after_task_c',
            'task_name': 'task_a',
            'score': 0.50,
        },
        {
            'checkpoint_label': 'after_task_c',
            'task_name': 'task_b',
            'score': 0.72,
        },
        {
            'checkpoint_label': 'after_task_c',
            'task_name': 'task_c',
            'score': 0.90,
        },
    ],
    'base_scores': {
        'task_a': 0.20,
        'task_b': 0.25,
        'task_c': 0.35,
    },
    'metadata': {
        'method': 'vanilla-crl',
        'note': 'scores already came from checkpoint-task evaluation jobs',
    },
}

EXAMPLE_MATRIX_INPUT: dict[str, Any] = {
    'schema_version': INPUT_SCHEMA_VERSION,
    'suite_id': 'llm_algorithmic',
    'source_format': 'performance_matrix_v1',
    'task_names': ['task_a', 'task_b', 'task_c'],
    'performance_matrix': [
        [0.70, 0.30, None],
        [0.55, 0.80, 0.45],
        [0.50, 0.72, 0.90],
    ],
    'base_scores': [0.20, 0.25, 0.35],
    'metadata': {
        'method': 'vanilla-crl',
    },
}


def result_spec_document() -> dict[str, Any]:
    """Return the human-readable CRL metric result specification."""

    return {
        'schema_version': INPUT_SCHEMA_VERSION,
        'result_schema_version': RESULT_SCHEMA_VERSION,
        'summary_schema_version': SUMMARY_SCHEMA_VERSION,
        'purpose': (
            'Specify and build a checkpoint-stage by task score matrix before '
            'computing CRL metrics such as R_ii, FinalAvg, BWT, and ZS-FWT.'
        ),
        'important_boundary': (
            'This layer consumes scores already produced by model evaluation. '
            'It does not launch training, checkpoint export, or model '
            'generation itself.'
        ),
        'accepted_inputs': {
            'pair_score_records': (
                'task_order + checkpoints + score_records, where each record '
                'identifies one checkpoint/stage, one task, and one score'
            ),
            'performance_matrix': (
                'task_names/tasks + performance_matrix/perf/perf_matrix'
            ),
        },
        'core_fields': {
            'task_order/task_names/tasks': 'ordered task list',
            'checkpoints': (
                'optional ordered stage metadata; defaults to one checkpoint '
                'per task'
            ),
            'score_records': (
                'pair-score records with checkpoint_label or stage_id, '
                'task_name, score, and optional source metadata'
            ),
            'performance_matrix': (
                'matrix[row][col] = score of checkpoint after task row on '
                'task col; future-task cells may be present for ZS-FWT'
            ),
            'base_scores': (
                'optional base-model zero-shot task scores used to compute '
                'ZS-FWT from row i-1, column i cells'
            ),
        },
        'output_fields': {
            'performance_matrix': 'full stage-by-task score matrix',
            'retention_matrix': 'lower-triangular seen-task subset',
            'stage_payloads': 'per-stage seen-task summaries',
            'summary': 'R_ii, FinalAvg/ACC, BWT, optional ZS-FWT',
            'complete': 'true when all lower-triangular retention cells exist',
            'missing_retention_cells': 'audit list for absent seen-task cells',
        },
        'score_scale': (
            'Scores must be ratios in [0, 1] or explicit percent strings such '
            'as "65%". Bare numeric 65 is rejected.'
        ),
        'example_pair_score_input': EXAMPLE_PAIR_SCORE_INPUT,
        'example_matrix_input': EXAMPLE_MATRIX_INPUT,
    }
