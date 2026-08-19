"""Score normalization for VLM general-ability evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from evaluation.vlm_general_ability.suite_spec import (
    ALIASES,
    ALL_BENCHMARKS,
    BENCHMARK_SPECS,
    CORE_BENCHMARKS,
    MAX_PERCENT_SCORE,
    MAX_RATIO_SCORE,
    REFERENCE,
    RUN_RESULT_VERSION,
    SPEC_BY_NAME,
    SUITE_SPEC_VERSION,
    VLMEVAL_DATASET_ALIASES,
    ScoreRecord,
)


def canonical_benchmark_name(name: str) -> str:
    """Return the canonical benchmark name for an official dataset name."""

    normalized = name.strip()
    if normalized in ALIASES:
        return ALIASES[normalized]
    lowered = normalized.lower().replace('-', '_').replace(' ', '_')
    if lowered in VLMEVAL_DATASET_ALIASES:
        return VLMEVAL_DATASET_ALIASES[lowered]
    return ALIASES.get(lowered, lowered)


def canonical_metric_key(key: str) -> str:
    """Normalize metric keys such as ``acc,none`` or ``overall/acc``."""

    stripped = key.strip().split(',', maxsplit=1)[0]
    return stripped.rsplit('/', maxsplit=1)[-1]


def normalize_score(value: Any) -> float:
    """Normalize an official ratio score or explicit percent string."""

    if isinstance(value, bool):
        raise TypeError('boolean is not a valid benchmark score')
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
            f'score must be a ratio in [0, 1] or an explicit percent string, '
            f'got {value!r}'
        )
    return score


def find_score(
    payload: Mapping[str, Any],
    benchmark: str,
) -> tuple[float, str] | None:
    """Find the benchmark's explicitly configured metric key."""

    for key in SPEC_BY_NAME[benchmark].preferred_score_keys:
        if key in payload and payload[key] is not None:
            return normalize_score(payload[key]), key
    for raw_key, value in payload.items():
        if not isinstance(raw_key, str) or value is None:
            continue
        canonical_key = canonical_metric_key(raw_key)
        if canonical_key in SPEC_BY_NAME[benchmark].preferred_score_keys:
            return normalize_score(value), raw_key
    return None


def records_from_named_results(
    results: Mapping[str, Any],
) -> dict[str, ScoreRecord]:
    """Extract one official score record per benchmark from one section."""

    records: dict[str, ScoreRecord] = {}
    for raw_name, raw_value in results.items():
        if not isinstance(raw_name, str):
            continue
        benchmark = canonical_benchmark_name(raw_name)
        if benchmark not in SPEC_BY_NAME:
            continue
        if benchmark in records:
            previous = records[benchmark]
            raise ValueError(
                f'duplicate VLM score for benchmark {benchmark!r}: '
                f'{previous.input_name!r} and {raw_name!r}'
            )
        if isinstance(raw_value, Mapping):
            found = find_score(raw_value, benchmark)
            if found is None:
                continue
            score, metric_key = found
        else:
            score = normalize_score(raw_value)
            metric_key = 'value'
        records[benchmark] = ScoreRecord(
            benchmark=benchmark,
            score=score,
            input_name=raw_name,
            metric_key=metric_key,
        )
    return records


def collect_score_records(payload: Mapping[str, Any]) -> list[ScoreRecord]:
    """Collect normalized VLM scores from structured evaluator JSON."""

    structured_sections = [
        payload[key]
        for key in ('results', 'groups', 'benchmarks')
        if isinstance(payload.get(key), Mapping)
    ]
    if not structured_sections:
        raise ValueError(
            'score artifact must contain a structured results, groups, or '
            'benchmarks object'
        )

    records: list[ScoreRecord] = []
    for section in structured_sections:
        records.extend(records_from_named_results(section).values())
    return records


def choose_records(records: Sequence[ScoreRecord]) -> dict[str, ScoreRecord]:
    """Return one score record per benchmark, failing on duplicates."""

    chosen: dict[str, ScoreRecord] = {}
    for record in records:
        if record.benchmark in chosen:
            previous = chosen[record.benchmark]
            raise ValueError(
                f'duplicate score for benchmark {record.benchmark!r}: '
                f'{previous.input_name}/{previous.metric_key} and '
                f'{record.input_name}/{record.metric_key}'
            )
        chosen[record.benchmark] = record
    return chosen


def load_json_file(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, Mapping):
        raise TypeError(f'{path} must contain a JSON object')
    return payload


def benchmark_from_score_path(path: Path) -> str:
    """Infer the benchmark name from an EvalScope score artifact path."""

    path_text = str(path).lower().replace('-', '_').replace(' ', '_')
    for benchmark in ALL_BENCHMARKS:
        names = (benchmark, *SPEC_BY_NAME[benchmark].aliases)
        for name in names:
            normalized = name.lower().replace('-', '_').replace(' ', '_')
            if normalized in path_text:
                return benchmark
    raise ValueError(f'cannot infer VLM benchmark from {path}')


def load_tabular_score_file_records(path: Path) -> dict[str, ScoreRecord]:
    """Load one EvalScope CSV/TSV score artifact."""

    benchmark = benchmark_from_score_path(path)
    delimiter = '\t' if path.suffix.lower() == '.tsv' else ','
    with path.open(newline='', encoding='utf-8') as file_obj:
        rows = list(csv.DictReader(file_obj, delimiter=delimiter))
    if not rows:
        raise ValueError(f'{path} contains no score rows')

    value: Any = None
    metric_key = ''
    for row in rows:
        for key in SPEC_BY_NAME[benchmark].preferred_score_keys:
            if key in row and row[key] not in {None, ''}:
                value = row[key]
                metric_key = key
                break
        if value is not None:
            break
        for key in ('Overall', 'ALL', 'overall', 'score', 'accuracy', 'acc'):
            if key in row and row[key] not in {None, ''}:
                value = row[key]
                metric_key = key
                break
        if value is not None:
            break
    if value is None:
        raise ValueError(f'{path} contains no configured VLM score')
    return {
        benchmark: ScoreRecord(
            benchmark=benchmark,
            score=normalize_score(value),
            input_name=path.stem,
            metric_key=metric_key,
        )
    }


def has_structured_score_sections(payload: Mapping[str, Any]) -> bool:
    """Return true if JSON has explicit benchmark result sections."""

    return any(
        isinstance(payload.get(key), Mapping)
        for key in ('results', 'groups', 'benchmarks')
    )


def load_score_file_records(path: Path) -> dict[str, ScoreRecord]:
    """Load one VLM evaluator score artifact."""

    suffix = path.suffix.lower()
    if suffix in {'.csv', '.tsv'}:
        return load_tabular_score_file_records(path)
    payload = load_json_file(path)
    if has_structured_score_sections(payload):
        records = choose_records(collect_score_records(payload))
        if not records:
            raise ValueError(f'{path} contains no configured VLM scores')
        return records

    benchmark = benchmark_from_score_path(path)
    if benchmark == 'ocrbench' and 'Final Score Norm' in payload:
        return {
            benchmark: ScoreRecord(
                benchmark=benchmark,
                score=normalize_score(f"{payload['Final Score Norm']}%"),
                input_name=path.stem,
                metric_key='Final Score Norm',
            )
        }
    raise ValueError(f'{path} contains no configured VLM scores')


def summarize_score_files(paths: Sequence[Path]) -> dict[str, Any]:
    """Summarize structured evaluator JSON artifacts; fail on duplicates."""

    records_by_benchmark: dict[str, tuple[Path, ScoreRecord]] = {}
    for path in paths:
        for record in load_score_file_records(path).values():
            if record.benchmark in records_by_benchmark:
                previous_path = records_by_benchmark[record.benchmark][0]
                raise ValueError(
                    f'duplicate score for benchmark {record.benchmark!r}: '
                    f'{previous_path} and {path}'
                )
            records_by_benchmark[record.benchmark] = (path, record)

    ordered_scores = {
        benchmark: records_by_benchmark[benchmark][1].score
        for benchmark in ALL_BENCHMARKS
        if benchmark in records_by_benchmark
    }
    required_scores = {
        benchmark: ordered_scores[benchmark]
        for benchmark in CORE_BENCHMARKS
        if benchmark in ordered_scores
    }
    missing = [
        benchmark
        for benchmark in CORE_BENCHMARKS
        if benchmark not in required_scores
    ]
    return {
        'schema_version': RUN_RESULT_VERSION,
        'suite_spec_version': SUITE_SPEC_VERSION,
        'reference': REFERENCE,
        'required_benchmarks': list(CORE_BENCHMARKS),
        'optional_benchmarks': [
            spec.name for spec in BENCHMARK_SPECS if not spec.required
        ],
        'benchmarks': ordered_scores,
        'suite_average': (
            fmean(required_scores.values()) if required_scores else None
        ),
        'diagnostic_average': (
            fmean(ordered_scores.values()) if ordered_scores else None
        ),
        'missing_benchmarks': missing,
        'complete': not missing,
        'sources': {
            benchmark: str(records_by_benchmark[benchmark][0])
            for benchmark in ordered_scores
        },
        'benchmark_details': {
            benchmark: {
                'display_name': SPEC_BY_NAME[benchmark].display_name,
                'input_name': records_by_benchmark[benchmark][1].input_name,
                'metric_key': records_by_benchmark[benchmark][1].metric_key,
                'score': records_by_benchmark[benchmark][1].score,
                'required': SPEC_BY_NAME[benchmark].required,
            }
            for benchmark in ordered_scores
        },
    }


def parse_benchmark_selection(values: Sequence[str] | None) -> tuple[str, ...]:
    """Return canonical VLM benchmark names from Hydra config values."""

    if not values:
        return CORE_BENCHMARKS

    benchmarks: list[str] = []
    for value in values:
        for item in value.split(','):
            benchmark = canonical_benchmark_name(item)
            if not benchmark:
                continue
            if benchmark not in SPEC_BY_NAME:
                supported = ', '.join(ALL_BENCHMARKS)
                raise ValueError(
                    f'unsupported VLM benchmark {item!r}; '
                    f'supported: {supported}'
                )
            if benchmark not in benchmarks:
                benchmarks.append(benchmark)
    if not benchmarks:
        raise ValueError('at least one VLM benchmark must be selected')
    return tuple(benchmarks)
