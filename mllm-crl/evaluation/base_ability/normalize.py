"""Score normalization for text base-ability evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from evaluation.base_ability.suite_spec import (
    ALIASES,
    CORE_BENCHMARKS,
    MAX_PERCENT_SCORE,
    MAX_RATIO_SCORE,
    PREFERRED_SCORE_KEYS,
    RUN_RESULT_VERSION,
    SPEC_BY_NAME,
    SUITE_SPEC_VERSION,
    ScoreRecord,
)


def canonical_benchmark_name(name: str) -> str:
    """Return the canonical benchmark name for an official task/group name."""

    normalized = name.strip()
    if normalized in ALIASES:
        return ALIASES[normalized]

    lowered = normalized.lower().replace('-', '_').replace(' ', '_')
    return ALIASES.get(lowered, lowered)


def canonical_metric_key(key: str) -> str:
    """Normalize lm-eval metric keys such as ``acc_norm,none``."""

    return key.strip().split(',', maxsplit=1)[0]


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
    for preferred_key in SPEC_BY_NAME[benchmark].preferred_score_keys:
        for raw_key, value in payload.items():
            if not isinstance(raw_key, str) or value is None:
                continue
            if canonical_metric_key(raw_key) == preferred_key:
                return normalize_score(value), raw_key
    return None


def records_from_named_results(
    results: Mapping[str, Any],
) -> dict[str, ScoreRecord]:
    """Extract one official score record per benchmark from one section."""

    records: dict[str, ScoreRecord] = {}
    for raw_name, raw_result in results.items():
        if not isinstance(raw_name, str):
            continue
        input_name = raw_name.split(',', maxsplit=1)[0]
        benchmark = canonical_benchmark_name(input_name)
        if benchmark not in CORE_BENCHMARKS:
            continue
        if benchmark in records:
            previous = records[benchmark]
            raise ValueError(
                f'duplicate {benchmark!r} records in one lm-eval section: '
                f'{previous.input_name!r} and {raw_name!r}'
            )
        if not isinstance(raw_result, Mapping):
            raise TypeError(
                f'lm-eval result for {raw_name!r} must be a metric mapping'
            )
        score = find_score(raw_result, benchmark)
        if score is None:
            continue
        score_value, metric_key = score
        records[benchmark] = ScoreRecord(
            benchmark=benchmark,
            score=score_value,
            input_name=input_name,
            metric_key=metric_key,
        )
    return records


def extract_lm_eval_records(
    payload: Mapping[str, Any],
) -> dict[str, ScoreRecord]:
    """Extract records from official lm-eval ``results``/``groups`` JSON.

    ``groups.mmlu`` is preferred for MMLU because lm-eval reports the official
    aggregate there. Other benchmarks prefer ``results`` and use ``groups``
    only when the benchmark is absent from ``results``.
    """

    results_section = payload.get('results')
    groups_section = payload.get('groups')
    if not isinstance(results_section, Mapping) and not isinstance(
        groups_section,
        Mapping,
    ):
        raise ValueError('score file must contain lm-eval results or groups')

    results = (
        records_from_named_results(results_section)
        if isinstance(results_section, Mapping)
        else {}
    )
    groups = (
        records_from_named_results(groups_section)
        if isinstance(groups_section, Mapping)
        else {}
    )

    records: dict[str, ScoreRecord] = {}
    for benchmark in CORE_BENCHMARKS:
        if benchmark == 'mmlu' and benchmark in groups:
            records[benchmark] = groups[benchmark]
        elif benchmark in results:
            records[benchmark] = results[benchmark]
        elif benchmark in groups:
            records[benchmark] = groups[benchmark]
    return records


def extract_lm_eval_scores(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract benchmark scores from lm-eval-style JSON."""

    return {
        benchmark: record.score
        for benchmark, record in extract_lm_eval_records(payload).items()
    }


def load_score_file_records(path: Path) -> dict[str, ScoreRecord]:
    """Load one official lm-eval JSON file and return score records."""

    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, Mapping):
        raise TypeError(f'{path} must contain a JSON object')

    records = extract_lm_eval_records(payload)
    if not records:
        raise ValueError(f'{path} contains no configured lm-eval scores')
    return records


def load_score_file(path: Path) -> dict[str, float]:
    """Load one official lm-eval JSON file and return normalized scores."""

    return {
        benchmark: record.score
        for benchmark, record in load_score_file_records(path).items()
    }


def summarize_scores(scores: Mapping[str, float]) -> dict[str, Any]:
    """Build the suite summary from already-normalized scores."""

    normalized = {
        benchmark: normalize_score(scores[benchmark])
        for benchmark in CORE_BENCHMARKS
        if benchmark in scores
    }
    missing = [
        benchmark
        for benchmark in CORE_BENCHMARKS
        if benchmark not in normalized
    ]
    suite_average = fmean(normalized.values()) if normalized else None
    return {
        'suite_spec_version': SUITE_SPEC_VERSION,
        'benchmarks': dict(normalized),
        'suite_average': suite_average,
        'missing_benchmarks': missing,
        'complete': not missing,
    }


def benchmark_detail(record: ScoreRecord, source: str) -> dict[str, Any]:
    """Return a JSON-friendly detail object for a selected score."""

    spec = SPEC_BY_NAME[record.benchmark]
    return {
        'display_name': spec.display_name,
        'score': record.score,
        'source': source,
        'input_name': record.input_name,
        'metric_key': record.metric_key,
    }


def summarize_score_files(paths: Sequence[Path]) -> dict[str, Any]:
    """Load official lm-eval files and fail on duplicate benchmarks."""

    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    records: dict[str, ScoreRecord] = {}
    for path in paths:
        for benchmark, record in load_score_file_records(path).items():
            if benchmark in records:
                raise ValueError(
                    f'duplicate score for benchmark {benchmark!r}: '
                    f'{sources[benchmark]} and {path}'
                )
            scores[benchmark] = record.score
            sources[benchmark] = str(path)
            records[benchmark] = record

    summary = summarize_scores(scores)
    summary['schema_version'] = RUN_RESULT_VERSION
    summary['suite_spec_version'] = SUITE_SPEC_VERSION
    summary['sources'] = sources
    summary['benchmark_details'] = {
        benchmark: benchmark_detail(records[benchmark], sources[benchmark])
        for benchmark in CORE_BENCHMARKS
        if benchmark in records
    }
    return summary


def parse_benchmark_selection(values: Sequence[str] | None) -> tuple[str, ...]:
    """Return canonical benchmark names from Hydra config values."""

    if not values:
        return CORE_BENCHMARKS

    benchmarks: list[str] = []
    for value in values:
        for item in value.split(','):
            benchmark = canonical_benchmark_name(item)
            if not benchmark:
                continue
            if benchmark not in CORE_BENCHMARKS:
                supported = ', '.join(CORE_BENCHMARKS)
                raise ValueError(
                    f'unsupported benchmark {item!r}; supported: {supported}'
                )
            if benchmark not in benchmarks:
                benchmarks.append(benchmark)
    if not benchmarks:
        raise ValueError('at least one benchmark must be selected')
    return tuple(benchmarks)
