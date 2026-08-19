"""Build CRL metric result documents from evaluated scores."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from evaluation.crl_metric.metrics import (
    summarize_continual_metrics,
    zero_shot_forward_transfer,
)
from evaluation.crl_metric.result_spec import (
    EXAMPLE_MATRIX_INPUT,
    EXAMPLE_PAIR_SCORE_INPUT,
    INPUT_SCHEMA_VERSION,
    MAX_PERCENT_SCORE,
    MAX_RATIO_SCORE,
    RESULT_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    result_spec_document,
)


@dataclass(frozen=True)
class CheckpointStage:
    """One row in the CRL performance matrix."""

    stage_id: str
    task_name: str
    checkpoint_label: str
    checkpoint_path: str | None


@dataclass(frozen=True)
class ScoreCell:
    """One evaluated checkpoint-task score."""

    row_index: int
    column_index: int
    score: float
    score_source: str | None
    score_source_note: str | None


def example_pair_score_payload() -> dict[str, Any]:
    """Return an example pair-score input payload."""

    return dict(EXAMPLE_PAIR_SCORE_INPUT)


def example_matrix_payload() -> dict[str, Any]:
    """Return an example matrix input payload."""

    return dict(EXAMPLE_MATRIX_INPUT)


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, Mapping):
        raise TypeError(f'{path} must contain a JSON object')
    return dict(payload)


def normalize_score(value: Any) -> float:
    """Normalize an official ratio score or explicit percent string."""

    if isinstance(value, bool):
        raise TypeError('boolean is not a valid score')
    if isinstance(value, str):
        stripped = value.strip()
        has_percent_suffix = stripped.endswith('%')
        numeric_text = (
            stripped[:-1].strip() if has_percent_suffix else stripped
        )
        score = float(numeric_text)
        if has_percent_suffix:
            score /= MAX_PERCENT_SCORE
    else:
        score = float(value)
    if score < 0.0 or score > MAX_RATIO_SCORE:
        raise ValueError(
            'score must be a ratio in [0, 1] or an explicit percent string, '
            f'got {value!r}'
        )
    return score


def optional_score(value: Any) -> float | None:
    """Normalize a present score."""

    if value is None or value == '':
        return None
    return normalize_score(value)


def sequence_field(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> list[Any]:
    """Return the first non-empty sequence field from a payload."""

    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise TypeError(f'{key} must be a list')
        if value:
            return list(value)
    joined = '/'.join(keys)
    raise ValueError(f'missing non-empty {joined}')


def task_order_from_payload(payload: Mapping[str, Any]) -> list[str]:
    """Resolve the ordered task list from accepted field names."""

    values = sequence_field(payload, ('task_order', 'task_names', 'tasks'))
    return [str(value) for value in values]


def checkpoint_stages_from_payload(
    payload: Mapping[str, Any],
    task_order: Sequence[str],
) -> list[CheckpointStage]:
    """Resolve checkpoint/stage metadata for matrix rows."""

    raw_checkpoints = payload.get('checkpoints')
    if raw_checkpoints is None:
        return [
            CheckpointStage(
                stage_id=task_name,
                task_name=task_name,
                checkpoint_label=f'after_{task_name}',
                checkpoint_path=None,
            )
            for task_name in task_order
        ]
    if isinstance(raw_checkpoints, str) or not isinstance(
        raw_checkpoints,
        Sequence,
    ):
        raise TypeError('checkpoints must be a list')
    if len(raw_checkpoints) != len(task_order):
        raise ValueError(
            'checkpoints length must match task count: '
            f'checkpoints={len(raw_checkpoints)} tasks={len(task_order)}'
        )
    stages: list[CheckpointStage] = []
    for index, raw in enumerate(raw_checkpoints):
        if not isinstance(raw, Mapping):
            raise TypeError('every checkpoint must be a JSON object')
        task_name = str(raw.get('task_name') or task_order[index])
        stage_id = str(raw.get('stage_id') or task_name)
        checkpoint_label = str(
            raw.get('checkpoint_label')
            or raw.get('label')
            or raw.get('checkpoint')
            or f'after_{task_name}'
        )
        checkpoint_path = raw.get('checkpoint_path') or raw.get('path')
        stages.append(
            CheckpointStage(
                stage_id=stage_id,
                task_name=task_name,
                checkpoint_label=checkpoint_label,
                checkpoint_path=(
                    str(checkpoint_path)
                    if checkpoint_path is not None
                    else None
                ),
            )
        )
    return stages


def unique_index_map(values: Sequence[str], *, label: str) -> dict[str, int]:
    """Map unique string values to their index."""

    mapping: dict[str, int] = {}
    for index, value in enumerate(values):
        if value in mapping:
            raise ValueError(f'duplicate {label}: {value!r}')
        mapping[value] = index
    return mapping


def checkpoint_lookup(stages: Sequence[CheckpointStage]) -> dict[str, int]:
    """Build accepted checkpoint identifiers for score records."""

    lookup: dict[str, int] = {}
    for index, stage in enumerate(stages):
        for value in (
            stage.stage_id,
            stage.task_name,
            stage.checkpoint_label,
            stage.checkpoint_path,
        ):
            if value is None:
                continue
            if value in lookup and lookup[value] != index:
                raise ValueError(f'ambiguous checkpoint identifier: {value!r}')
            lookup[value] = index
    return lookup


def score_records_from_payload(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Resolve pair-score records from accepted field names."""

    for key in ('score_records', 'records', 'pair_scores'):
        raw_records = payload.get(key)
        if raw_records is None:
            continue
        if isinstance(raw_records, str) or not isinstance(
            raw_records,
            Sequence,
        ):
            raise TypeError(f'{key} must be a list')
        records: list[Mapping[str, Any]] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise TypeError(f'every item in {key} must be a JSON object')
            records.append(raw)
        if records:
            return records
    raise ValueError('payload is missing non-empty score_records')


def resolve_record_row(
    record: Mapping[str, Any],
    row_lookup: Mapping[str, int],
    row_count: int,
) -> int:
    """Resolve a score record's checkpoint/stage row."""

    index_value = record.get('checkpoint_index', record.get('stage_index'))
    if index_value is not None:
        index = int(index_value)
        if index < 0 or index >= row_count:
            raise ValueError(f'checkpoint index out of range: {index}')
        return index
    for key in ('checkpoint_label', 'checkpoint', 'stage_id'):
        value = record.get(key)
        if value is None:
            continue
        identifier = str(value)
        if identifier not in row_lookup:
            raise ValueError(
                'unknown checkpoint/stage identifier: '
                f'{identifier}'
            )
        return row_lookup[identifier]
    raise ValueError('score record is missing checkpoint identifier')


def resolve_record_column(
    record: Mapping[str, Any],
    task_lookup: Mapping[str, int],
    task_count: int,
) -> int:
    """Resolve a score record's task column."""

    if record.get('task_index') is not None:
        index = int(record['task_index'])
        if index < 0 or index >= task_count:
            raise ValueError(f'task index out of range: {index}')
        return index
    task_name = record.get('task_name') or record.get('task')
    if task_name is None:
        raise ValueError('score record is missing task_name')
    task_key = str(task_name)
    if task_key not in task_lookup:
        raise ValueError(f'unknown task_name: {task_key!r}')
    return task_lookup[task_key]


def empty_matrix(size: int) -> list[list[float | None]]:
    """Return an empty square score matrix."""

    return [[None for _ in range(size)] for _ in range(size)]


def matrix_from_score_records(
    payload: Mapping[str, Any],
    *,
    task_order: Sequence[str],
    stages: Sequence[CheckpointStage],
) -> tuple[list[list[float | None]], list[ScoreCell]]:
    """Build a performance matrix from checkpoint-task score records."""

    task_lookup = unique_index_map(task_order, label='task name')
    row_lookup = checkpoint_lookup(stages)
    matrix = empty_matrix(len(task_order))
    cells: list[ScoreCell] = []
    for record in score_records_from_payload(payload):
        row_index = resolve_record_row(record, row_lookup, len(stages))
        column_index = resolve_record_column(
            record,
            task_lookup,
            len(task_order),
        )
        if matrix[row_index][column_index] is not None:
            raise ValueError(
                'duplicate score for matrix cell '
                f'row={row_index} col={column_index}'
            )
        score = normalize_score(record.get('score'))
        matrix[row_index][column_index] = score
        cells.append(
            ScoreCell(
                row_index=row_index,
                column_index=column_index,
                score=score,
                score_source=(
                    str(record['score_source'])
                    if record.get('score_source') is not None
                    else None
                ),
                score_source_note=(
                    str(record['score_source_note'])
                    if record.get('score_source_note') is not None
                    else None
                ),
            )
        )
    return matrix, cells


def matrix_from_payload(
    payload: Mapping[str, Any],
) -> list[list[float | None]] | None:
    """Resolve an explicit performance matrix when present."""

    for key in ('performance_matrix', 'perf', 'perf_matrix'):
        raw_matrix = payload.get(key)
        if raw_matrix is None:
            continue
        if isinstance(raw_matrix, str) or not isinstance(raw_matrix, Sequence):
            raise TypeError(f'{key} must be a list of rows')
        matrix: list[list[float | None]] = []
        for row in raw_matrix:
            if isinstance(row, str) or not isinstance(row, Sequence):
                raise TypeError(f'every row in {key} must be a list')
            matrix.append([optional_score(value) for value in row])
        return matrix
    return None


def validate_square_matrix(
    *,
    matrix: Sequence[Sequence[float | None]],
    task_order: Sequence[str],
) -> None:
    """Fail fast on malformed CRL matrices."""

    if len(matrix) != len(task_order):
        raise ValueError(
            'matrix row count must match task count: '
            f'rows={len(matrix)} tasks={len(task_order)}'
        )
    for row_index, row in enumerate(matrix):
        if len(row) != len(task_order):
            raise ValueError(
                'matrix column count must match task count: '
                f'row={row_index} cols={len(row)} tasks={len(task_order)}'
            )
        if row[row_index] is None:
            raise ValueError(
                'matrix is missing diagonal/current-task score at '
                f'row={row_index} task={task_order[row_index]!r}'
            )


def retention_matrix(
    performance_matrix: Sequence[Sequence[float | None]],
) -> list[list[float | None]]:
    """Return the lower-triangular seen-task retention matrix."""

    matrix: list[list[float | None]] = []
    for row_index, row in enumerate(performance_matrix):
        matrix.append(
            [
                score if column_index <= row_index else None
                for column_index, score in enumerate(row)
            ]
        )
    return matrix


def missing_retention_cells(
    matrix: Sequence[Sequence[float | None]],
    task_order: Sequence[str],
) -> list[dict[str, Any]]:
    """List missing lower-triangular cells required for retention."""

    missing: list[dict[str, Any]] = []
    for row_index, row in enumerate(matrix):
        for column_index in range(row_index + 1):
            if row[column_index] is None:
                missing.append(
                    {
                        'stage_index': row_index,
                        'stage_task': task_order[row_index],
                        'task_index': column_index,
                        'task_name': task_order[column_index],
                    }
                )
    return missing


def stage_payloads(
    *,
    task_order: Sequence[str],
    stages: Sequence[CheckpointStage],
    performance_matrix: Sequence[Sequence[float | None]],
    suite_id: str,
    source_format: str,
) -> list[dict[str, Any]]:
    """Build per-stage retention summaries from a performance matrix."""

    payloads: list[dict[str, Any]] = []
    for row_index, row in enumerate(performance_matrix):
        seen_task_scores = {
            task_order[column_index]: float(score)
            for column_index, score in enumerate(row[: row_index + 1])
            if score is not None
        }
        seen_values = list(seen_task_scores.values())
        row_values = [float(score) for score in row if score is not None]
        stage = stages[row_index]
        payloads.append(
            {
                'suite_id': suite_id,
                'source_format': source_format,
                'stage_id': stage.stage_id,
                'task_name': stage.task_name,
                'checkpoint_label': stage.checkpoint_label,
                'checkpoint_path': stage.checkpoint_path,
                'current_task_success': float(row[row_index]),
                'seen_task_retention': (
                    fmean(seen_values) if seen_values else None
                ),
                'eval_overall_accuracy': (
                    fmean(row_values) if row_values else None
                ),
                'seen_task_scores': seen_task_scores,
                'matrix_row_index': row_index,
            }
        )
    return payloads


def score_cell_documents(
    *,
    cells: Sequence[ScoreCell],
    task_order: Sequence[str],
    stages: Sequence[CheckpointStage],
) -> list[dict[str, Any]]:
    """Return sorted score-cell audit records."""

    documents: list[dict[str, Any]] = []
    sorted_cells = sorted(
        cells,
        key=lambda item: (item.row_index, item.column_index),
    )
    for cell in sorted_cells:
        stage = stages[cell.row_index]
        documents.append(
            {
                'stage_index': cell.row_index,
                'stage_id': stage.stage_id,
                'checkpoint_label': stage.checkpoint_label,
                'checkpoint_path': stage.checkpoint_path,
                'task_index': cell.column_index,
                'task_name': task_order[cell.column_index],
                'score': cell.score,
                'score_source': cell.score_source,
                'score_source_note': cell.score_source_note,
            }
        )
    return documents


def base_scores_from_payload(
    payload: Mapping[str, Any],
    task_order: Sequence[str],
) -> list[float | None] | None:
    """Resolve optional base-model zero-shot scores."""

    raw = payload.get('base_scores') or payload.get('base_zero_shot_scores')
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return [optional_score(raw.get(task_name)) for task_name in task_order]
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError('base_scores must be a mapping or list')
    if len(raw) != len(task_order):
        raise ValueError(
            'base_scores length must match task count: '
            f'base_scores={len(raw)} tasks={len(task_order)}'
        )
    return [optional_score(value) for value in raw]


def pre_task_scores_from_matrix(
    matrix: Sequence[Sequence[float | None]],
) -> list[float | None]:
    """Return score(M_<i>, task_i) cells used for ZS-FWT."""

    scores: list[float | None] = []
    for task_index in range(len(matrix)):
        if task_index == 0:
            scores.append(None)
            continue
        scores.append(matrix[task_index - 1][task_index])
    return scores


def build_summary(
    *,
    performance_matrix: Sequence[Sequence[float | None]],
    stage_payload_documents: Sequence[Mapping[str, Any]],
    missing_cells: Sequence[Mapping[str, Any]],
    base_scores: Sequence[float | None] | None,
) -> dict[str, Any]:
    """Build CRL metric summary from a performance matrix."""

    summary: dict[str, Any] = dict(
        summarize_continual_metrics(performance_matrix)
    )
    summary['ACC'] = summary['FinalAvg']
    current_values = [
        float(payload['current_task_success'])
        for payload in stage_payload_documents
        if payload.get('current_task_success') is not None
    ]
    retention_values = [
        float(payload['seen_task_retention'])
        for payload in stage_payload_documents
        if payload.get('seen_task_retention') is not None
    ]
    eval_values = [
        float(payload['eval_overall_accuracy'])
        for payload in stage_payload_documents
        if payload.get('eval_overall_accuracy') is not None
    ]
    summary.update(
        {
            'schema_version': SUMMARY_SCHEMA_VERSION,
            'CurrentTaskSuccessAvg': fmean(current_values)
            if current_values
            else None,
            'SeenTaskRetentionAvg': fmean(retention_values)
            if retention_values
            else None,
            'EvalOverallAccuracyAvg': fmean(eval_values)
            if eval_values
            else None,
            'complete': not missing_cells,
            'missing_retention_cell_count': len(missing_cells),
        }
    )
    if base_scores is not None:
        pre_task_scores = pre_task_scores_from_matrix(performance_matrix)
        summary['ZS-FWT'] = zero_shot_forward_transfer(
            base_scores,
            pre_task_scores,
            skip_first=True,
        )
    return summary


def default_suite_id(payload: Mapping[str, Any], suite_id: str | None) -> str:
    """Resolve suite id from override or payload."""

    resolved = suite_id or payload.get('suite_id') or 'crl'
    return str(resolved)


def default_source_format(
    payload: Mapping[str, Any],
    source_format: str | None,
) -> str:
    """Resolve source format from override or payload."""

    resolved = source_format or payload.get('source_format')
    if resolved is not None:
        return str(resolved)
    if matrix_from_payload(payload) is not None:
        return 'performance_matrix_v1'
    return 'checkpoint_task_scores_v1'


def build_metric_result(
    payload: Mapping[str, Any],
    *,
    suite_id: str | None = None,
    source_format: str | None = None,
) -> dict[str, Any]:
    """Build a formal CRL metric result from a JSON payload."""

    task_order = task_order_from_payload(payload)
    stages = checkpoint_stages_from_payload(payload, task_order)
    matrix = matrix_from_payload(payload)
    cells: list[ScoreCell] = []
    if matrix is None:
        matrix, cells = matrix_from_score_records(
            payload,
            task_order=task_order,
            stages=stages,
        )
    validate_square_matrix(matrix=matrix, task_order=task_order)
    resolved_suite_id = default_suite_id(payload, suite_id)
    resolved_source_format = default_source_format(payload, source_format)
    retention = retention_matrix(matrix)
    missing_cells = missing_retention_cells(matrix, task_order)
    stage_payload_documents = stage_payloads(
        task_order=task_order,
        stages=stages,
        performance_matrix=matrix,
        suite_id=resolved_suite_id,
        source_format=resolved_source_format,
    )
    base_scores = base_scores_from_payload(payload, task_order)
    summary = build_summary(
        performance_matrix=matrix,
        stage_payload_documents=stage_payload_documents,
        missing_cells=missing_cells,
        base_scores=base_scores,
    )
    result = {
        'schema_version': RESULT_SCHEMA_VERSION,
        'input_schema_version': payload.get(
            'schema_version',
            INPUT_SCHEMA_VERSION,
        ),
        'suite_id': resolved_suite_id,
        'source_format': resolved_source_format,
        'task_names': list(task_order),
        'checkpoints': [stage.__dict__ for stage in stages],
        'performance_matrix': matrix,
        'retention_matrix': retention,
        'stage_payloads': stage_payload_documents,
        'summary': summary,
        'complete': summary['complete'],
        'missing_retention_cells': list(missing_cells),
        'metadata': dict(payload.get('metadata') or {}),
    }
    if base_scores is not None:
        result['base_scores'] = list(base_scores)
        result['pre_task_scores'] = pre_task_scores_from_matrix(matrix)
    if cells:
        result['score_records'] = score_cell_documents(
            cells=cells,
            task_order=task_order,
            stages=stages,
        )
    return result


def write_json_document(path: Path, document: Mapping[str, Any]) -> None:
    """Write one JSON document to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        + '\n',
        encoding='utf-8',
    )


def write_metric_result(
    *,
    input_path: Path,
    result_path: Path,
    summary_path: Path | None = None,
    suite_id: str | None = None,
    source_format: str | None = None,
) -> dict[str, Any]:
    """Load and write a CRL metric result."""

    result = build_metric_result(
        load_json_object(input_path),
        suite_id=suite_id,
        source_format=source_format,
    )
    write_json_document(result_path, result)
    if summary_path is not None:
        write_json_document(summary_path, result['summary'])
    return result
