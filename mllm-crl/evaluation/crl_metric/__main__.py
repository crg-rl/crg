"""Command entrypoint for CRL metric result generation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

from evaluation.cli_utils import (
    optional_path,
    optional_str,
    required_path,
    validate_output_outside_repo,
    write_json_document,
)
from evaluation.crl_metric.result import (
    build_metric_result,
    example_matrix_payload,
    example_pair_score_payload,
    load_json_object,
    result_spec_document,
)


def validate_mode(mode: str) -> None:
    """Fail fast on unsupported modes."""

    supported_modes = {
        'build',
        'spec',
        'example_pairs',
        'example_matrix',
    }
    if mode not in supported_modes:
        raise ValueError(
            f'unsupported mode={mode!r}; expected one of '
            f'{sorted(supported_modes)}'
        )


def config_mapping(value: Any) -> dict[str, Any]:
    """Convert an optional Hydra mapping to a plain dict."""

    if value is None:
        return {}
    return dict(value)


def write_optional_output(document: dict[str, Any], output: Any) -> None:
    """Write JSON through the shared stdout/file helper."""

    write_json_document(document, optional_path(output))


def run_build(config: DictConfig) -> int:
    """Build a CRL metric result from an input payload."""

    input_path = required_path(config.input, 'input')
    output = required_path(config.output, 'output')
    summary_output = optional_path(config.summary_output)
    validate_output_outside_repo(
        output,
        'output',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    validate_output_outside_repo(
        summary_output,
        'summary_output',
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    result = build_metric_result(
        load_json_object(input_path),
        suite_id=optional_str(config.suite_id),
        source_format=optional_str(config.source_format),
    )
    write_json_document(result, output)
    if summary_output is not None:
        write_json_document(result['summary'], summary_output)
    if config.strict and not result['complete']:
        missing = result['missing_retention_cells']
        sys.stderr.write(
            'CRL retention matrix is incomplete; '
            f'missing_retention_cell_count={len(missing)}\n'
        )
        return 2
    return 0


def run(config: DictConfig) -> int:
    """Dispatch the configured CRL metric command."""

    mode = str(config.mode)
    validate_mode(mode)
    if mode == 'spec':
        write_optional_output(result_spec_document(), config.output)
        return 0
    if mode == 'example_pairs':
        write_optional_output(example_pair_score_payload(), config.output)
        return 0
    if mode == 'example_matrix':
        write_optional_output(example_matrix_payload(), config.output)
        return 0
    if mode == 'build':
        return run_build(config)
    raise AssertionError(f'unhandled mode={mode!r}')


@hydra.main(
    config_path='../conf',
    config_name='crl_metric',
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Command main function."""

    raise SystemExit(run(config))


if __name__ == '__main__':
    main()
