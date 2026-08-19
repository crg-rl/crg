"""EvalScope/VLMEvalKit runner for VLM general-ability evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from evaluation.vlm_general_ability.normalize import (
    load_score_file_records,
    summarize_score_files,
)
from evaluation.vlm_general_ability.suite_spec import (
    RUN_RESULT_VERSION,
    SUITE_SPEC_VERSION,
    VLMEVAL_DATASETS,
    BenchmarkRunPlan,
)


def validate_eval_parallelism(
    eval_batch_size: int,
    judge_worker_num: int,
    *,
    allow_eval_batch_size_one: bool,
) -> None:
    """Reject evaluator settings that silently underuse accelerator lanes."""

    if eval_batch_size < 1:
        raise ValueError('--eval-batch-size must be positive')
    if eval_batch_size == 1 and not allow_eval_batch_size_one:
        raise ValueError(
            '--eval-batch-size=1 is blocked for formal eval. Use a larger '
            'batch size, or pass --allow-eval-batch-size-one only for an '
            'explicit diagnostic run.'
        )
    if judge_worker_num < 1:
        raise ValueError('--judge-worker-num must be positive')


def build_evalscope_command(
    *,
    checkpoint: str,
    dataset_name: str,
    output_dir: Path,
    evalscope_binary: str,
    eval_batch_size: int,
    judge_worker_num: int,
    model_type: str,
    model_id: str,
    model_kwargs: Mapping[str, Any] | None,
    limit: int | None,
    extra_args: Sequence[str] | None,
) -> tuple[str, ...]:
    """Build one EvalScope command using the VLMEvalKit backend."""

    model_config: dict[str, Any] = {
        'name': model_type,
        'model_path': checkpoint,
    }
    if model_kwargs:
        model_config.update(dict(model_kwargs))
    eval_config = {
        'data': [dataset_name],
        'model': [model_config],
        'nproc': eval_batch_size,
        'work_dir': str(output_dir),
    }
    if limit is not None:
        eval_config['limit'] = limit
    command = [
        evalscope_binary,
        'eval',
        '--eval-backend',
        'VLMEvalKit',
        '--eval-config',
        json.dumps(eval_config),
        '--model-id',
        model_id,
        '--eval-batch-size',
        str(eval_batch_size),
        '--judge-worker-num',
        str(judge_worker_num),
        '--work-dir',
        str(output_dir),
        '--no-timestamp',
    ]
    if limit is not None:
        command.extend(['--limit', str(limit)])
    if extra_args:
        command.extend(extra_args)
    return tuple(command)


def build_run_plans(
    *,
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    evalscope_binary: str,
    eval_batch_size: int,
    judge_worker_num: int,
    model_type: str,
    model_id: str,
    model_kwargs: Mapping[str, Any] | None,
    limit: int | None,
    extra_args: Sequence[str] | None,
) -> list[BenchmarkRunPlan]:
    """Build all selected VLM benchmark run plans."""

    plans: list[BenchmarkRunPlan] = []
    for benchmark in benchmarks:
        if benchmark not in VLMEVAL_DATASETS:
            raise ValueError(
                f'{benchmark} has no default VLMEvalKit dataset mapping'
            )
        dataset_name = VLMEVAL_DATASETS[benchmark]
        benchmark_output_dir = output_dir / benchmark
        plans.append(
            BenchmarkRunPlan(
                benchmark=benchmark,
                dataset_name=dataset_name,
                output_dir=benchmark_output_dir,
                command=build_evalscope_command(
                    checkpoint=checkpoint,
                    dataset_name=dataset_name,
                    output_dir=benchmark_output_dir,
                    evalscope_binary=evalscope_binary,
                    eval_batch_size=eval_batch_size,
                    judge_worker_num=judge_worker_num,
                    model_type=model_type,
                    model_id=model_id,
                    model_kwargs=model_kwargs,
                    limit=limit,
                    extra_args=extra_args,
                ),
            )
        )
    return plans


def score_artifact_has_records(path: Path) -> bool:
    """Return true if one artifact yields a known score."""

    try:
        return bool(load_score_file_records(path))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def find_score_artifacts(output_dir: Path) -> list[Path]:
    """Find score artifacts under one benchmark output directory."""

    if not output_dir.exists():
        return []
    return [
        path
        for path in sorted(output_dir.rglob('*'))
        if path.suffix.lower() in {'.json', '.csv', '.tsv'}
        and score_artifact_has_records(path)
    ]


def collect_score_artifacts(
    output_dir: Path,
    benchmarks: Sequence[str],
) -> list[Path]:
    """Collect score artifacts for selected VLM benchmarks."""

    paths: list[Path] = []
    for benchmark in benchmarks:
        paths.extend(find_score_artifacts(output_dir / benchmark))
    return paths


def run_plan(
    plan: BenchmarkRunPlan,
    *,
    skip_existing: bool,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Run one VLM benchmark plan."""

    existing = find_score_artifacts(plan.output_dir)
    if skip_existing and existing:
        return {
            'benchmark': plan.benchmark,
            'dataset_name': plan.dataset_name,
            'status': 'skipped_existing',
            'output_dir': str(plan.output_dir),
            'score_artifacts': [str(path) for path in existing],
            'command': list(plan.command),
        }

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = plan.output_dir / 'evalscope.stdout.log'
    stderr_path = plan.output_dir / 'evalscope.stderr.log'
    start_time = time.time()
    with (
        stdout_path.open('w', encoding='utf-8') as stdout_file,
        stderr_path.open('w', encoding='utf-8') as stderr_file,
    ):
        completed = subprocess.run(
            plan.command,
            env=dict(env),
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    elapsed_seconds = time.time() - start_time
    artifacts = find_score_artifacts(plan.output_dir)
    if completed.returncode != 0:
        status = 'failed'
        summary = summarize_output_dir(
            plan.output_dir.parent,
            [plan.benchmark],
        )
    elif artifacts:
        status = 'succeeded'
        summary = summarize_output_dir(
            plan.output_dir.parent,
            [plan.benchmark],
        )
    else:
        status = 'missing_results'
        summary = summarize_output_dir(
            plan.output_dir.parent,
            [plan.benchmark],
        )
    return {
        'benchmark': plan.benchmark,
        'dataset_name': plan.dataset_name,
        'status': status,
        'returncode': completed.returncode,
        'elapsed_seconds': elapsed_seconds,
        'output_dir': str(plan.output_dir),
        'stdout': str(stdout_path),
        'stderr': str(stderr_path),
        'score_artifacts': [str(path) for path in artifacts],
        'summary': summary,
        'command': list(plan.command),
    }


def summarize_output_dir(
    output_dir: Path,
    benchmarks: Sequence[str],
) -> dict[str, Any]:
    """Summarize existing VLM evaluator outputs under one directory."""

    artifacts = collect_score_artifacts(output_dir, benchmarks)
    summary = summarize_score_files(artifacts)
    summary['schema_version'] = RUN_RESULT_VERSION
    summary['suite_spec_version'] = SUITE_SPEC_VERSION
    summary['mode'] = 'collect_output_dir'
    summary['output_dir'] = str(output_dir)
    summary['selected_benchmarks'] = list(benchmarks)
    summary['score_artifacts'] = [str(path) for path in artifacts]
    return summary


def dry_run_document(
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    plans: Sequence[BenchmarkRunPlan],
) -> dict[str, Any]:
    """Build a dry-run document without launching heavy VLM eval."""

    return {
        'schema_version': RUN_RESULT_VERSION,
        'suite_spec_version': SUITE_SPEC_VERSION,
        'mode': 'run_checkpoint',
        'dry_run': True,
        'checkpoint': checkpoint,
        'output_dir': str(output_dir),
        'selected_benchmarks': list(benchmarks),
        'complete': False,
        'run': {
            'plans': [
                {
                    'benchmark': plan.benchmark,
                    'dataset_name': plan.dataset_name,
                    'output_dir': str(plan.output_dir),
                    'command': list(plan.command),
                }
                for plan in plans
            ],
        },
    }


def evaluate_checkpoint(
    *,
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    evalscope_binary: str,
    eval_batch_size: int,
    judge_worker_num: int,
    model_type: str,
    model_id: str,
    model_kwargs: Mapping[str, Any] | None,
    limit: int | None,
    extra_args: Sequence[str] | None,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run selected VLM general ability benchmarks for a checkpoint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plans = build_run_plans(
        checkpoint=checkpoint,
        output_dir=output_dir,
        benchmarks=benchmarks,
        evalscope_binary=evalscope_binary,
        eval_batch_size=eval_batch_size,
        judge_worker_num=judge_worker_num,
        model_type=model_type,
        model_id=model_id,
        model_kwargs=model_kwargs,
        limit=limit,
        extra_args=extra_args,
    )
    if dry_run:
        return dry_run_document(
            checkpoint=checkpoint,
            output_dir=output_dir,
            benchmarks=benchmarks,
            plans=plans,
        )

    statuses: list[dict[str, Any]] = []
    env = os.environ.copy()
    for plan in plans:
        status = run_plan(
            plan=plan,
            skip_existing=skip_existing,
            env=env,
        )
        statuses.append(status)
        if status['status'] == 'failed':
            break

    summary = summarize_output_dir(output_dir, benchmarks)
    summary['mode'] = 'run_checkpoint'
    summary['checkpoint'] = checkpoint
    summary['run'] = {
        'dry_run': False,
        'skip_existing': skip_existing,
        'evalscope_binary': evalscope_binary,
        'eval_batch_size': eval_batch_size,
        'judge_worker_num': judge_worker_num,
        'model_type': model_type,
        'model_id': model_id,
        'model_kwargs': dict(model_kwargs or {}),
        'statuses': statuses,
    }
    summary['run_complete'] = all(
        status['status'] in {'succeeded', 'skipped_existing'}
        for status in statuses
    ) and len(statuses) == len(plans)
    return summary
