from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from verl.utils.dataset.rl_dataset import RLHFDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class VisuLogicCrlTaskSpec:
    name: str
    sample_indices: np.ndarray


class VisuLogicCrlTaskSwitchingDataset(RLHFDataset):
    """VisuLogic domain-internal CRL dataset for labeled parquet data.

    The caller prepares one parquet per first-level domain and stores the
    second-level task label, such as ``subcategory``, in ``extra_info``.
    """

    shared_state: dict[str, Any] | None = None

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: DictConfig,
        processor: Any | None = None,
        max_samples: int = -1,
    ) -> None:
        if isinstance(data_files, str):
            parquet_files = [data_files]
        else:
            parquet_files = list(data_files)
        for path in parquet_files:
            if not str(path).endswith(".parquet"):
                raise ValueError(
                    "VisuLogicCrlTaskSwitchingDataset only supports parquet "
                    f"files. Got: {path}"
                )

        if config.dataloader_num_workers != 0:
            raise ValueError(
                "CRL task switching requires data.dataloader_num_workers=0 "
                "(dataloader workers get a copy of the dataset)."
            )

        visulogic_crl_config = config.visulogic_crl

        steps_per_task = int(visulogic_crl_config["steps_per_task"])
        if steps_per_task <= 0:
            raise ValueError(
                "data.visulogic_crl.steps_per_task must be a positive int."
            )
        self.steps_per_task = steps_per_task

        start_step = int(visulogic_crl_config.get("start_step", 0))
        if start_step < 0:
            raise ValueError("data.visulogic_crl.start_step must be >= 0.")
        self.start_step = start_step

        task_field = str(visulogic_crl_config["task_field"]).strip()
        if not task_field:
            raise ValueError("data.visulogic_crl.task_field must be set.")
        self.task_field = task_field

        split = self.infer_split(data_files)
        self.split = split

        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=-1,
        )

        if len(self.dataframe) == 0:
            raise ValueError("Empty dataset after loading parquet files.")
        self.task_labels = self.resolve_task_labels()
        if not any(label is not None for label in self.task_labels):
            raise ValueError(
                f"Missing required extra_info[{self.task_field!r}] labels "
                "in parquet dataset."
            )

        dataset_size_key = (
            "train_dataset_size" if split == "train" else "val_dataset_size"
        )
        dataset_size = int(visulogic_crl_config[dataset_size_key])
        if dataset_size <= 0:
            raise ValueError(
                f"data.visulogic_crl.{dataset_size_key} must be a "
                "positive int."
            )
        if max_samples > 0:
            dataset_size = min(dataset_size, max_samples)
        self.dataset_size = int(dataset_size)

        seed_key = "train_seed" if split == "train" else "val_seed"
        default_seed = 1 if split == "train" else 2
        seed_base = int(visulogic_crl_config.get(seed_key, default_seed))
        task_seed_stride = int(
            visulogic_crl_config.get("task_seed_stride", 1000)
        )
        task_names = self.resolve_task_names(visulogic_crl_config)
        self.task_specs = self.build_task_specs(
            task_names=task_names,
            seed_base=seed_base,
            task_seed_stride=task_seed_stride,
        )
        self.initialize_shared_state(
            task_names=task_names,
            start_step=start_step,
        )

    def infer_split(self, data_files: str | list[str]) -> str:
        if isinstance(data_files, str):
            text = data_files
        else:
            text = " ".join(map(str, data_files))
        return "train" if "train" in text else "test"

    def resolve_task_labels(self) -> list[str | None]:
        labels: list[str | None] = []
        if "extra_info" not in self.dataframe.column_names:
            raise ValueError(
                "Missing required extra_info column in parquet dataset."
            )

        has_top_level_field = self.task_field in self.dataframe.column_names
        extra_infos = self.dataframe["extra_info"]
        top_level_labels = (
            self.dataframe[self.task_field]
            if has_top_level_field
            else [None] * len(self.dataframe)
        )

        for idx, (extra_info, top_level_label) in enumerate(
            zip(extra_infos, top_level_labels, strict=True)
        ):
            if not isinstance(extra_info, dict):
                raise ValueError(
                    f"Row {idx} has invalid extra_info: {extra_info!r}"
                )
            raw_label = extra_info.get(self.task_field)
            if raw_label is None:
                labels.append(None)
                continue

            label = str(raw_label)
            if (
                top_level_label is not None
                and str(top_level_label) != label
            ):
                raise ValueError(
                    f"Row {idx} has inconsistent task labels: "
                    f"extra_info[{self.task_field!r}]={label!r}, "
                    f"{self.task_field}={top_level_label!r}"
                )
            labels.append(label)
        return labels

    def resolve_task_names(
        self, visulogic_crl_config: DictConfig
    ) -> list[str]:
        tasks_config = visulogic_crl_config.get("tasks", None)
        if tasks_config is None or len(tasks_config) == 0:
            task_names = sorted(
                {label for label in self.task_labels if label is not None}
            )
        else:
            task_names = [str(name) for name in tasks_config]
        if not task_names:
            raise ValueError(
                "CRL requires at least one task in data.visulogic_crl.tasks "
                "or populated extra_info task labels."
            )
        return task_names

    def build_task_specs(
        self,
        task_names: list[str],
        seed_base: int,
        task_seed_stride: int,
    ) -> list[VisuLogicCrlTaskSpec]:
        indices_by_task: dict[str, list[int]] = {
            name: [] for name in task_names
        }
        for idx, raw_task_name in enumerate(self.task_labels):
            if raw_task_name is None:
                continue
            task_name = str(raw_task_name)
            if task_name not in indices_by_task:
                continue
            indices_by_task[task_name].append(idx)

        task_specs: list[VisuLogicCrlTaskSpec] = []
        for task_idx, task_name in enumerate(task_names):
            indices = np.asarray(indices_by_task[task_name], dtype=np.int64)
            if indices.size == 0:
                raise ValueError(
                    f"No samples found for task {task_name!r} in "
                    f"extra_info[{self.task_field!r}] and split "
                    f"{self.split!r}."
                )
            rng_seed = seed_base + task_idx * task_seed_stride
            rng = np.random.default_rng(rng_seed)
            replace = self.dataset_size > indices.size
            sample_indices = rng.choice(
                indices,
                size=self.dataset_size,
                replace=replace,
            )
            task_specs.append(
                VisuLogicCrlTaskSpec(
                    name=task_name,
                    sample_indices=sample_indices,
                )
            )
        return task_specs

    def initialize_shared_state(
        self,
        task_names: list[str],
        start_step: int,
    ) -> None:
        if VisuLogicCrlTaskSwitchingDataset.shared_state is None:
            task_idx = (start_step // self.steps_per_task) % len(task_names)
            VisuLogicCrlTaskSwitchingDataset.shared_state = {
                "task_idx": task_idx,
                "train_steps": start_step,
                "task_names": task_names,
            }
            return

        if (
            VisuLogicCrlTaskSwitchingDataset.shared_state["task_names"]
            != task_names
        ):
            raise ValueError(
                "data.visulogic_crl.tasks must be identical for train "
                "and val datasets."
            )

    def get_shared_state(self) -> dict[str, Any]:
        state = VisuLogicCrlTaskSwitchingDataset.shared_state
        if state is None:
            raise ValueError(
                "VisuLogicCrlTaskSwitchingDataset.shared_state is not "
                "initialized."
            )
        return state

    def __len__(self) -> int:
        return self.dataset_size

    @property
    def current_task_idx(self) -> int:
        return int(self.get_shared_state()["task_idx"])

    @property
    def current_task(self) -> str:
        return self.task_specs[self.current_task_idx].name

    def set_task(self, task_idx: int) -> None:
        if not (0 <= task_idx < len(self.task_specs)):
            raise ValueError(f"Invalid task index: {task_idx}")
        self.get_shared_state()["task_idx"] = task_idx

    def task_switch_signature(self) -> tuple[int, str]:
        return self.current_task_idx, self.current_task

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_spec = self.task_specs[self.current_task_idx]
        sample_index = int(task_spec.sample_indices[index % self.dataset_size])
        item = super().__getitem__(sample_index)

        if "extra_info" not in item or item["extra_info"] is None:
            item["extra_info"] = {}
        item["extra_info"]["task"] = task_spec.name
        item["extra_info"]["task_field"] = self.task_field
        item["extra_info"].setdefault("split", self.split)
        return item

    def on_batch_end(self, batch: Any) -> None:
        if batch is None:
            return

        state = self.get_shared_state()
        state["train_steps"] += 1
        step = state["train_steps"]
        if step % self.steps_per_task != 0:
            return

        prev_task = self.current_task
        next_task_idx = self.current_task_idx + 1
        if next_task_idx >= len(self.task_specs):
            next_task_idx = 0
        self.set_task(next_task_idx)

        print(
            f"[CRL] step {step}: switch task {prev_task} -> "
            f"{self.current_task} ({next_task_idx + 1}/"
            f"{len(self.task_specs)})"
        )
