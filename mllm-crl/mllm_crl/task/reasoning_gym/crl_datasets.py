from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import reasoning_gym
import reasoning_gym.utils
from omegaconf import OmegaConf

from mllm_crl.task.reasoning_gym.datasets import ReasoningGymDataset

if TYPE_CHECKING:
    from reasoning_gym.dataset import ProceduralDataset
    from transformers import PreTrainedTokenizer


@dataclass
class TaskSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class CrlTaskSwitchingDataset(ReasoningGymDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task_specs: list[TaskSpec],
        seed_base: int,
        seed_stride: int,
        dataset_size: int,
        steps_per_task: int,
        developer_prompt: str,
        max_prompt_length: int = 2048,
        truncation: str = "error",
        val_dataset: CrlTaskSwitchingDataset | None = None,
    ) -> None:
        self.task_specs = task_specs
        self.seed_base = int(seed_base)
        self.seed_stride = int(seed_stride)
        self.dataset_size = int(dataset_size)
        steps_per_task = int(steps_per_task)
        if steps_per_task <= 0:
            raise ValueError("steps_per_task must be a positive int.")
        self.steps_per_task = steps_per_task
        self.task_idx = 0
        self.train_steps = 0
        self.val_dataset = val_dataset

        data_source = self.make_task_dataset(self.task_idx)
        super().__init__(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation=truncation,
        )

    @property
    def current_task(self) -> str:
        return self.task_specs[self.task_idx].name

    def task_seed(self, task_idx: int) -> int:
        return self.seed_base + task_idx * self.seed_stride

    def make_task_dataset(self, task_idx: int) -> ProceduralDataset:
        task_spec = self.task_specs[task_idx]
        return reasoning_gym.create_dataset(
            task_spec.name,
            seed=self.task_seed(task_idx),
            size=self.dataset_size,
            **task_spec.params,
        )

    def set_task(self, task_idx: int) -> None:
        if not (0 <= task_idx < len(self.task_specs)):
            raise ValueError(f"Invalid task index: {task_idx}")
        self.task_idx = task_idx
        self.data = self.make_task_dataset(task_idx)

    def on_batch_end(self, batch: Any) -> None:
        if batch is None:
            return

        # Track steps locally because VERL omits `global_steps` from the batch
        # passed to `train_dataset.on_batch_end(batch=...)`.
        self.train_steps += 1
        step = self.train_steps
        if step % self.steps_per_task != 0:
            return

        next_task_idx = self.task_idx + 1
        if next_task_idx >= len(self.task_specs):
            next_task_idx = 0

        prev_task = self.current_task
        self.set_task(next_task_idx)
        if self.val_dataset is not None:
            self.val_dataset.set_task(next_task_idx)

        message = (
            f"[CRL] step {step}: switch task {prev_task} -> "
            f"{self.current_task} ({next_task_idx + 1}/{len(self.task_specs)})"
        )
        print(message)


def prepare_crl_datasets(
    config, tokenizer: PreTrainedTokenizer, valid_ratio: float = 0.1
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    if config.curriculum.enabled:
        raise ValueError("CRL is not compatible with curriculum training.")

    if config.data.dataloader_num_workers != 0:
        raise ValueError(
            "CRL task switching requires data.dataloader_num_workers=0 "
            "(dataloader workers get a copy of the dataset)."
        )

    steps_per_task = config.crl.steps_per_task

    dataset_size = int(config.reasoning_gym.dataset_size)
    task_specs: list[TaskSpec] = []
    for name, ds in config.reasoning_gym.datasets.items():
        task_specs.append(
            TaskSpec(
                name=name,
                params=(
                    OmegaConf.to_container(ds.config, resolve=True)
                    if "config" in ds
                    else {}
                ),
            )
        )

    if not task_specs:
        raise ValueError("CRL requires at least one reasoning_gym dataset.")

    developer_prompt_setting = config.reasoning_gym.developer_prompt
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS[
        developer_prompt_setting
    ]

    val_dataset_size = config.crl.get("val_dataset_size", None)
    if val_dataset_size is None:
        crl_valid_ratio = float(config.crl.get("valid_ratio", valid_ratio))
        val_size = int(dataset_size * crl_valid_ratio)
    else:
        val_size = int(val_dataset_size)

    val_dataset = CrlTaskSwitchingDataset(
        tokenizer=tokenizer,
        task_specs=task_specs,
        seed_base=int(config.crl.get("val_seed", 2)),
        seed_stride=int(config.crl.get("task_seed_stride", 1000)),
        dataset_size=val_size,
        steps_per_task=steps_per_task,
        developer_prompt=developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
    )
    train_dataset = CrlTaskSwitchingDataset(
        tokenizer=tokenizer,
        task_specs=task_specs,
        seed_base=int(config.crl.get("train_seed", 1)),
        seed_stride=int(config.crl.get("task_seed_stride", 1000)),
        dataset_size=dataset_size,
        steps_per_task=steps_per_task,
        developer_prompt=developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
        val_dataset=val_dataset,
    )
    return train_dataset, val_dataset
