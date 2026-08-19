import importlib
from dataclasses import replace
from typing import Literal, Optional

import numpy as np
import reasoning_gym
import reasoning_gym.utils
import torch
import verl.utils.torch_functional as verl_f
from omegaconf import OmegaConf
from reasoning_gym.coaching.curriculum_config import (
    CurriculumAttributeConfig,
    CurriculumExperimentConfig,
)
from reasoning_gym.coaching.experiment import CurriculumExperiment, Experiment
from reasoning_gym.composite import CompositeDataset, DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: ProceduralDataset | None = None,
        experiment: Experiment | None = None,
        developer_prompt: str | None = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, (
            "One of `procedural_dataset` or `experiment` must be provided"
        )
        assert procedural_dataset is None or experiment is None, (
            "Only one of `procedural_dataset` or `experiment` may be provided"
        )

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        row_dict["data_source"] = "reasoning_gym"

        q = row_dict["question"]
        chat = []
        if self.developer_prompt is not None:
            chat.append(
                {"role": self.developer_role, "content": self.developer_prompt}
            )
        chat.append({"role": "user", "content": q})
        row_dict["raw_prompt"] = chat

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        row_dict["index"] = index
        row_dict["reward_model"] = {
            "ground_truth": row_dict["answer"],
            "style": "rule",
        }
        return row_dict

    def update_experiment_difficulty(
        self, dataset_name: str, method: Literal["increment", "decrement"]
    ):
        """Update the difficulty of the underlying dataset."""
        if self.experiment is None:
            raise ValueError(
                "Cannot update difficulty: dataset is not a "
                "CurriculumExperiment"
            )
        if method not in ["increment", "decrement"]:
            raise ValueError(
                "Invalid method: must be 'increment' or 'decrement'"
            )
        self.experiment.score_board.clear(dataset_name)
        self.experiment.update_difficulty(dataset_name, method)
        self.data = self.experiment.composite
        return True

    def aggregate(self, last_n: int | None = None):
        """Aggregate scores from the underlying experiment"""
        if self.experiment is None:
            raise ValueError(
                "Cannot aggregate scores: dataset is not a "
                "CurriculumExperiment"
            )

        results = self.experiment.score_board.aggregate(last_n=last_n)
        output_results = {}

        for key, value in results.items():
            output_results[key] = {}
            scores = value.scores
            first_key = next(iter(scores.keys()))
            output_results[key]["results"] = np.mean(scores[first_key])
            output_results[key]["total_samples"] = value.total_scores
        return output_results


def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    max_prompt_length: int = 2048,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset
    or Experiment as the underlying data source.
    """
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    else:
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )


def prepare_datasets(
    config, tokenizer, valid_ratio=0.1
) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and validation datasets."""
    if bool(config.get("crl", {}).get("enabled", False)):
        prepare_crl_datasets = importlib.import_module(
            "mllm_crl.task.reasoning_gym.crl_datasets"
        ).prepare_crl_datasets
        return prepare_crl_datasets(
            config=config, tokenizer=tokenizer, valid_ratio=float(valid_ratio)
        )

    dataset_size = config.reasoning_gym.dataset_size
    developer_prompt_setting = config.reasoning_gym.developer_prompt
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS[
        developer_prompt_setting
    ]

    if config.curriculum.enabled:
        curricula = config.curriculum.curricula
        curriculum_config = CurriculumExperimentConfig(
            curricula={
                curriculum_name: CurriculumAttributeConfig(**curriculum_config)
                for curriculum_name, curriculum_config in curricula.items()
            }
        )

        train_data_source = CurriculumExperiment(
            name=config.trainer.experiment_name,
            config=curriculum_config,
            size=dataset_size,
            seed=1,
        )
        val_data_source = CompositeDataset(
            config=replace(train_data_source.composite.config, seed=2)
        )
    else:
        dataset_specs = [
            DatasetSpec(
                name=name,
                weight=ds.weight,
                config=OmegaConf.to_container(ds.config, resolve=True)
                if "config" in ds
                else {},
            )
            for name, ds in config.reasoning_gym.datasets.items()
        ]
        train_data_source = reasoning_gym.create_dataset(
            "composite", seed=1, size=dataset_size, datasets=dataset_specs
        )
        val_data_source = reasoning_gym.create_dataset(
            "composite",
            seed=2,
            size=int(dataset_size * valid_ratio),
            datasets=dataset_specs,
        )
    train_dataset = make_dataset(
        tokenizer,
        train_data_source,
        developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
    )
    val_dataset = make_dataset(
        tokenizer,
        val_data_source,
        developer_prompt,
        max_prompt_length=config.data.max_prompt_length,
    )
    return train_dataset, val_dataset
