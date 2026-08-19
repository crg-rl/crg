"""Command entrypoint for entropy metric evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra

if TYPE_CHECKING:
    from omegaconf import DictConfig

from evaluation.cli_utils import (
    optional_path,
    optional_str,
    required_path,
    required_str,
    validate_output_outside_repo,
    write_json_document,
)
from evaluation.entropy_metric.evaluator import evaluate_entropy
from evaluation.entropy_metric.result_spec import (
    EXAMPLE_INPUT,
    result_spec_document,
)


def validate_mode(mode: str) -> None:
    """Fail fast on unsupported modes."""

    supported_modes = {"evaluate", "spec", "example"}
    if mode not in supported_modes:
        raise ValueError(
            f"unsupported mode={mode!r}; expected one of "
            f"{sorted(supported_modes)}"
        )


def optional_int(value: Any) -> int | None:
    """Convert an optional integer config value."""

    if value is None or value == "":
        return None
    return int(value)


def write_optional_output(document: dict[str, Any], output: Any) -> None:
    """Write JSON to stdout or a file."""

    write_json_document(document, optional_path(output))


def run_evaluate(config: DictConfig) -> int:
    """Run fixed-prompt entropy for one final checkpoint."""

    output_jsonl = required_path(config.output_jsonl, "output_jsonl")
    summary_json = required_path(config.summary_json, "summary_json")
    validate_output_outside_repo(
        output_jsonl,
        "output_jsonl",
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    validate_output_outside_repo(
        summary_json,
        "summary_json",
        allow_output_in_repo=bool(config.allow_output_in_repo),
    )
    evaluate_entropy(
        prompt_set=required_path(config.prompt_set, "prompt_set"),
        model_path=required_path(config.model_path, "model_path"),
        modality=required_str(config.modality, "modality"),
        output_jsonl=output_jsonl,
        summary_json=summary_json,
        max_new_tokens=int(config.max_new_tokens),
        batch_size=int(config.batch_size),
        score_batch_size=int(config.score_batch_size),
        dtype=str(config.dtype),
        device_map=optional_str(config.device_map),
        limit=optional_int(config.limit),
    )
    return 0


def run(config: DictConfig) -> int:
    """Dispatch the configured entropy metric command."""

    mode = str(config.mode)
    validate_mode(mode)
    if mode == "spec":
        write_optional_output(result_spec_document(), config.output)
        return 0
    if mode == "example":
        write_optional_output(EXAMPLE_INPUT, config.output)
        return 0
    if mode == "evaluate":
        return run_evaluate(config)
    raise AssertionError(f"unreachable mode={mode!r}")


@hydra.main(
    config_path="../conf",
    config_name="entropy_metric",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Command main function."""

    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
