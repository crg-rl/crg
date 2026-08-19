from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from mllm_crl.task import ReasoningGymRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator


def upstream_runner_globals() -> dict[str, Any]:
    globals_map = ReasoningGymRunner.run.__globals__
    task_runner_module = globals_map.get("task_runner")
    if task_runner_module is not None:
        globals_map.setdefault(
            "prepare_datasets",
            task_runner_module.prepare_datasets,
        )
        globals_map.setdefault(
            "RayPPOTrainer",
            task_runner_module.RayPPOTrainer,
        )
    return globals_map


class MethodHookReasoningGymRunner(ReasoningGymRunner):
    """mllm-crl ReasoningGymRunner plus algorithm-owned hook points."""

    trainer_cls = RayPPOTrainer

    def before_setup(self, config) -> None:
        del config

    def prepare_runner_datasets(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        del config, tokenizer
        return train_dataset, val_dataset

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del config, trainer, train_dataset, val_dataset

    def train_reward_num_examine(self, config, default: int) -> int:
        method_debug = config.get("method_debug", {}) or {}
        return int(method_debug.get("train_reward_num_examine", default))

    @contextmanager
    def patch_upstream_runner(self, config) -> Iterator[None]:
        upstream_globals = upstream_runner_globals()
        original_prepare_datasets = upstream_globals["prepare_datasets"]
        original_trainer_cls = upstream_globals["RayPPOTrainer"]
        original_load_reward_manager = upstream_globals.get(
            "load_reward_manager"
        )
        runner = self
        captured: dict[str, Any] = {}

        def prepare_datasets_with_method_hooks(config_arg, tokenizer):
            train_dataset, val_dataset = original_prepare_datasets(
                config_arg,
                tokenizer,
            )
            train_dataset, val_dataset = runner.prepare_runner_datasets(
                config_arg,
                tokenizer,
                train_dataset,
                val_dataset,
            )
            captured["train_dataset"] = train_dataset
            captured["val_dataset"] = val_dataset
            return train_dataset, val_dataset

        class MethodHookTrainer(runner.trainer_cls):
            def __init__(self, *args, **kwargs) -> None:
                captured["train_dataset"] = kwargs.get(
                    "train_dataset",
                    captured.get("train_dataset"),
                )
                captured["val_dataset"] = kwargs.get(
                    "val_dataset",
                    captured.get("val_dataset"),
                )
                super().__init__(*args, **kwargs)

            def init_workers(self, *args, **kwargs):
                result = super().init_workers(*args, **kwargs)
                runner.after_init_workers(
                    config,
                    self,
                    captured.get("train_dataset"),
                    captured.get("val_dataset"),
                )
                return result

        def load_reward_manager_with_method_hooks(
            config_arg,
            tokenizer,
            num_examine,
            **reward_kwargs,
        ):
            if num_examine == 0:
                num_examine = runner.train_reward_num_examine(
                    config_arg,
                    num_examine,
                )
            return original_load_reward_manager(
                config_arg,
                tokenizer,
                num_examine,
                **reward_kwargs,
            )

        upstream_globals["prepare_datasets"] = (
            prepare_datasets_with_method_hooks
        )
        upstream_globals["RayPPOTrainer"] = MethodHookTrainer
        if original_load_reward_manager is not None:
            upstream_globals["load_reward_manager"] = (
                load_reward_manager_with_method_hooks
            )
        try:
            yield
        finally:
            upstream_globals["prepare_datasets"] = original_prepare_datasets
            upstream_globals["RayPPOTrainer"] = original_trainer_cls
            if original_load_reward_manager is not None:
                upstream_globals["load_reward_manager"] = (
                    original_load_reward_manager
                )

    def run(self, config):
        self.before_setup(config)
        with self.patch_upstream_runner(config):
            return super().run(config)
