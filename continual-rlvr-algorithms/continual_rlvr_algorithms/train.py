from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import ray
from mllm_crl.train import load_and_merge_task_config
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

if TYPE_CHECKING:
    from omegaconf import DictConfig

ppo_config_path = (
    Path(run_ppo.__code__.co_filename).resolve().parent / "config"
)

DEFAULT_TASK_RUNNER_CLS = (
    "continual_rlvr_algorithms.method.task_runner:MethodHookReasoningGymRunner"
)


def config_get(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_import_path(path: str):
    module_name, separator, attr_name = path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(
            f"import path must use 'module:attribute' format; got {path!r}"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_task_runner_cls(config: DictConfig) -> type:
    path = config_get(config, "task_runner_cls", DEFAULT_TASK_RUNNER_CLS)
    return resolve_import_path(str(path))


def run_reasoning_gym_ppo(
    config: DictConfig,
    task_runner_cls: type,
) -> None:
    config = load_and_merge_task_config(config)
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(task_runner_cls)
    run_ppo(config, task_runner_class)


def build_dynamic_reasoning_gym_ppo_main():
    @hydra.main(config_path=str(ppo_config_path), version_base=None)
    def main(config):
        run_reasoning_gym_ppo(
            config,
            resolve_task_runner_cls(config),
        )

    return main


main = build_dynamic_reasoning_gym_ppo_main()


if __name__ == "__main__":
    main()
