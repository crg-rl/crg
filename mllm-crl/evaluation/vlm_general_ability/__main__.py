"""Command entrypoint for VLM general-ability evaluation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

from evaluation.cli_utils import (
    optional_path,
    optional_string_list,
    path_list,
    required_path,
    required_str,
    validate_output_outside_repo,
    write_json_document,
)
from evaluation.vlm_general_ability.evalscope_runner import (
    evaluate_checkpoint,
    summarize_output_dir,
    validate_eval_parallelism,
)
from evaluation.vlm_general_ability.normalize import (
    parse_benchmark_selection,
    summarize_score_files,
)
from evaluation.vlm_general_ability.suite_spec import (
    EXAMPLE_INPUT,
    suite_spec_document,
)


def validate_mode(mode: str) -> None:
    """Fail fast on unsupported evaluation modes."""

    supported_modes = {
        'checkpoint',
        'collect',
        'build',
        'spec',
        'example',
    }
    if mode not in supported_modes:
        raise ValueError(
            f'unsupported mode={mode!r}; expected one of '
            f'{sorted(supported_modes)}'
        )


def selected_benchmarks(config: DictConfig) -> tuple[str, ...]:
    """Return canonical VLM benchmark names selected by Hydra config."""

    return parse_benchmark_selection(optional_string_list(config.benchmarks))


def optional_mapping(config_value: Any) -> dict[str, Any]:
    """Return a plain dict for optional Hydra mapping values."""

    if config_value is None:
        return {}
    value = OmegaConf.to_container(config_value, resolve=True)
    if not isinstance(value, dict):
        raise TypeError('model_kwargs must be a mapping')
    return value


def run_checkpoint(config: DictConfig, output: Any) -> int:
    """Run EvalScope/VLMEvalKit for one checkpoint and write a summary."""

    output_dir = required_path(config.output_dir, 'output_dir')
    checkpoint = required_str(config.checkpoint, 'checkpoint')
    validate_output_outside_repo(
        output_dir,
        'output_dir',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    validate_output_outside_repo(
        output,
        'output',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    validate_eval_parallelism(
        int(config.eval_batch_size),
        int(config.judge_worker_num),
        allow_eval_batch_size_one=bool(config.allow_eval_batch_size_one),
    )
    summary = evaluate_checkpoint(
        checkpoint=checkpoint,
        output_dir=output_dir,
        benchmarks=selected_benchmarks(config),
        evalscope_binary=str(config.evalscope_binary),
        eval_batch_size=int(config.eval_batch_size),
        judge_worker_num=int(config.judge_worker_num),
        model_type=str(config.model_type),
        model_id=str(config.model_id),
        model_kwargs=optional_mapping(config.model_kwargs),
        limit=config.limit,
        extra_args=optional_string_list(config.eval_args),
        skip_existing=bool(config.skip_existing),
        dry_run=bool(config.dry_run),
    )
    write_json_document(summary, output)
    if config.dry_run:
        return 0
    if not summary.get('run_complete', False):
        return 1
    if config.strict and not summary['complete']:
        missing = ', '.join(summary['missing_benchmarks'])
        sys.stderr.write(f'missing required VLM benchmarks: {missing}\n')
        return 2
    return 0


def run_collect(config: DictConfig, output: Any) -> int:
    """Summarize existing EvalScope/VLMEvalKit output directories."""

    collect_output_dir = required_path(
        config.collect_output_dir,
        'collect_output_dir',
    )
    validate_output_outside_repo(
        output,
        'output',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    summary = summarize_output_dir(
        output_dir=collect_output_dir,
        benchmarks=selected_benchmarks(config),
    )
    write_json_document(summary, output)
    if config.strict and not summary['complete']:
        missing = ', '.join(summary['missing_benchmarks'])
        sys.stderr.write(f'missing required VLM benchmarks: {missing}\n')
        return 2
    return 0


def run_build(config: DictConfig, output: Any) -> int:
    """Summarize explicit VLM score artifacts."""

    score_files = path_list(config.score_files)
    if not score_files:
        raise ValueError('score_files is required in build mode')
    validate_output_outside_repo(
        output,
        'output',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    summary = summarize_score_files(score_files)
    write_json_document(summary, output)
    if config.strict and not summary['complete']:
        missing = ', '.join(summary['missing_benchmarks'])
        sys.stderr.write(f'missing required VLM benchmarks: {missing}\n')
        return 2
    return 0


def run(config: DictConfig) -> int:
    """Dispatch the configured VLM general-ability evaluation."""

    mode = str(config.mode)
    validate_mode(mode)
    output = optional_path(config.output)
    if mode == 'spec':
        write_json_document(suite_spec_document(), output)
        return 0
    if mode == 'example':
        write_json_document(EXAMPLE_INPUT, output)
        return 0
    if mode == 'checkpoint':
        return run_checkpoint(config, output)
    if mode == 'collect':
        return run_collect(config, output)
    return run_build(config, output)


@hydra.main(
    config_path='../conf',
    config_name='vlm_general_ability',
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Command main function."""

    raise SystemExit(run(config))


if __name__ == '__main__':
    main()
