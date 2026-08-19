"""lm-eval runner for text base-ability evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from evaluation.base_ability.normalize import (
    summarize_score_files,
    summarize_scores,
)
from evaluation.base_ability.suite_spec import (
    CORE_BENCHMARKS,
    LM_EVAL_TASKS,
    RUN_RESULT_VERSION,
    SUITE_SPEC_VERSION,
    UNSAFE_CODE_BENCHMARKS,
    BenchmarkRunPlan,
)


def normalize_lm_eval_device(device: str) -> str:
    """Normalize common accelerator aliases for lm-eval."""

    normalized = device.strip()
    if normalized.lower() == 'npu':
        return 'npu:0'
    return normalized


def validate_lm_eval_batch_size(
    batch_size: str,
    *,
    allow_eval_batch_size_one: bool,
) -> None:
    """Reject accidental single-sample formal lm-eval runs."""

    normalized = batch_size.strip().lower()
    if normalized in {'1', '1.0'} and not allow_eval_batch_size_one:
        raise ValueError(
            '--batch-size=1 is blocked for formal eval. Use auto or a larger '
            'batch size, or pass --allow-eval-batch-size-one only for an '
            'explicit diagnostic run.'
        )


def lm_eval_entrypoint_command(
    entrypoint: str,
    lm_eval_binary: str,
) -> list[str]:
    """Return the executable prefix for lm-eval."""

    if entrypoint == 'module':
        return [sys.executable, '-m', 'lm_eval']
    if entrypoint == 'binary':
        return [lm_eval_binary]
    raise ValueError(f'unsupported lm-eval entrypoint: {entrypoint!r}')


def build_model_args(
    checkpoint: str,
    *,
    trust_remote_code: bool,
    extra_model_args: Sequence[str] | None,
) -> str:
    """Build the lm-eval HF model_args string."""

    args = [f'pretrained={checkpoint}']
    if trust_remote_code:
        args.append('trust_remote_code=True')
    if extra_model_args:
        args.extend(extra_model_args)
    return ','.join(args)


def build_lm_eval_command(
    checkpoint: str,
    benchmark: str,
    output_dir: Path,
    device: str,
    batch_size: str,
    entrypoint: str,
    lm_eval_binary: str,
    *,
    trust_remote_code: bool,
    extra_model_args: Sequence[str] | None,
) -> tuple[str, ...]:
    """Build one lm-eval command for a canonical benchmark."""

    command = lm_eval_entrypoint_command(entrypoint, lm_eval_binary)
    command.extend(
        [
            '--model',
            'hf',
            '--model_args',
            build_model_args(
                checkpoint,
                trust_remote_code=trust_remote_code,
                extra_model_args=extra_model_args,
            ),
            '--tasks',
            LM_EVAL_TASKS[benchmark],
            '--device',
            normalize_lm_eval_device(device),
            '--batch_size',
            batch_size,
            '--output_path',
            str(output_dir),
            '--log_samples',
        ]
    )
    if benchmark in UNSAFE_CODE_BENCHMARKS:
        command.append('--confirm_run_unsafe_code')
    return tuple(command)


def build_run_plans(
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    device: str,
    batch_size: str,
    entrypoint: str,
    lm_eval_binary: str,
    *,
    trust_remote_code: bool,
    extra_model_args: Sequence[str] | None,
) -> list[BenchmarkRunPlan]:
    """Build all benchmark run plans."""

    plans: list[BenchmarkRunPlan] = []
    for benchmark in benchmarks:
        benchmark_output_dir = output_dir / benchmark
        plans.append(
            BenchmarkRunPlan(
                benchmark=benchmark,
                task_name=LM_EVAL_TASKS[benchmark],
                output_dir=benchmark_output_dir,
                command=build_lm_eval_command(
                    checkpoint=checkpoint,
                    benchmark=benchmark,
                    output_dir=benchmark_output_dir,
                    device=device,
                    batch_size=batch_size,
                    entrypoint=entrypoint,
                    lm_eval_binary=lm_eval_binary,
                    trust_remote_code=trust_remote_code,
                    extra_model_args=extra_model_args,
                ),
            )
        )
    return plans


def has_lm_eval_results(path: Path) -> bool:
    """Return true if a JSON file looks like an lm-eval result file."""

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and isinstance(
        payload.get('results'),
        Mapping,
    )


def find_lm_eval_result_files(output_dir: Path) -> list[Path]:
    """Find lm-eval result JSON files under one benchmark output directory."""

    exact = output_dir / 'results.json'
    if exact.is_file() and has_lm_eval_results(exact):
        return [exact]
    if not output_dir.exists():
        return []
    return [
        path
        for path in sorted(output_dir.rglob('*.json'))
        if has_lm_eval_results(path)
    ]


def collect_lm_eval_result_files(
    output_dir: Path,
    benchmarks: Sequence[str],
) -> list[Path]:
    """Collect lm-eval result files for selected benchmarks."""

    result_files: list[Path] = []
    for benchmark in benchmarks:
        result_files.extend(find_lm_eval_result_files(output_dir / benchmark))
    return result_files


def run_lm_eval_plan(
    plan: BenchmarkRunPlan,
    *,
    skip_existing: bool,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Run one benchmark plan and return JSON-friendly status."""

    existing_result_files = find_lm_eval_result_files(plan.output_dir)
    if skip_existing and existing_result_files:
        return {
            'benchmark': plan.benchmark,
            'task_name': plan.task_name,
            'status': 'skipped_existing',
            'output_dir': str(plan.output_dir),
            'result_files': [str(path) for path in existing_result_files],
            'command': list(plan.command),
        }

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = plan.output_dir / 'lm_eval.stdout.log'
    stderr_path = plan.output_dir / 'lm_eval.stderr.log'
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
    result_files = find_lm_eval_result_files(plan.output_dir)
    if completed.returncode != 0:
        status = 'failed'
    elif result_files:
        status = 'succeeded'
    else:
        status = 'missing_results'
    return {
        'benchmark': plan.benchmark,
        'task_name': plan.task_name,
        'status': status,
        'returncode': completed.returncode,
        'elapsed_seconds': elapsed_seconds,
        'output_dir': str(plan.output_dir),
        'stdout': str(stdout_path),
        'stderr': str(stderr_path),
        'result_files': [str(path) for path in result_files],
        'command': list(plan.command),
    }


def summarize_lm_eval_output_dir(
    output_dir: Path,
    benchmarks: Sequence[str],
) -> dict[str, Any]:
    """Summarize lm-eval outputs already present under an output directory."""

    result_files = collect_lm_eval_result_files(output_dir, benchmarks)
    summary = summarize_score_files(result_files) if result_files else (
        summarize_scores({})
    )
    summary['schema_version'] = RUN_RESULT_VERSION
    summary['suite_spec_version'] = SUITE_SPEC_VERSION
    summary['mode'] = 'collect_lm_eval_output_dir'
    summary['output_dir'] = str(output_dir)
    summary['selected_benchmarks'] = list(benchmarks)
    summary['result_files'] = [str(path) for path in result_files]
    return summary


def dry_run_document(
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    plans: Sequence[BenchmarkRunPlan],
) -> dict[str, Any]:
    """Build a dry-run run document without launching lm-eval."""

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
                    'task_name': plan.task_name,
                    'output_dir': str(plan.output_dir),
                    'command': list(plan.command),
                }
                for plan in plans
            ],
        },
    }


def evaluate_checkpoint_base_ability(
    checkpoint: str,
    output_dir: Path,
    benchmarks: Sequence[str],
    device: str,
    batch_size: str,
    entrypoint: str,
    lm_eval_binary: str,
    *,
    trust_remote_code: bool,
    extra_model_args: Sequence[str] | None,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run selected base-ability benchmarks for a checkpoint and summarize."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plans = build_run_plans(
        checkpoint=checkpoint,
        output_dir=output_dir,
        benchmarks=benchmarks,
        device=device,
        batch_size=batch_size,
        entrypoint=entrypoint,
        lm_eval_binary=lm_eval_binary,
        trust_remote_code=trust_remote_code,
        extra_model_args=extra_model_args,
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
        status = run_lm_eval_plan(
            plan=plan,
            skip_existing=skip_existing,
            env=env,
        )
        statuses.append(status)
        if status['status'] == 'failed':
            break

    summary = summarize_lm_eval_output_dir(output_dir, benchmarks)
    summary['mode'] = 'run_checkpoint'
    summary['checkpoint'] = checkpoint
    summary['run'] = {
        'dry_run': False,
        'skip_existing': skip_existing,
        'device': normalize_lm_eval_device(device),
        'batch_size': batch_size,
        'statuses': statuses,
    }
    summary['run_complete'] = all(
        status['status'] in {'succeeded', 'skipped_existing'}
        for status in statuses
    ) and len(statuses) == len(plans)
    return summary
