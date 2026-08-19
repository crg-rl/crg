from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sized
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

CURRENT_PROMPT_ROLE = "current_prompt_reward"
OLD_PROMPT_ROLE = "old_prompt_kl_only"
KL_MODE_TO_OLD_POLICY_OLD_PROMPTS = "kl_to_old_policy_old_prompts"
KL_MODES = frozenset({KL_MODE_TO_OLD_POLICY_OLD_PROMPTS})


def mapping_get(payload: Any, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def mapping_get_bool(payload: Any, key: str, *, default: bool) -> bool:
    return bool(mapping_get(payload, key, default))


def normalize_mode(value: Any) -> str:
    mode = (
        str(value or KL_MODE_TO_OLD_POLICY_OLD_PROMPTS)
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "old": KL_MODE_TO_OLD_POLICY_OLD_PROMPTS,
        "old_policy": KL_MODE_TO_OLD_POLICY_OLD_PROMPTS,
        "kl_to_old_policy": KL_MODE_TO_OLD_POLICY_OLD_PROMPTS,
        "kl_to_old_policy_on_old_prompts": KL_MODE_TO_OLD_POLICY_OLD_PROMPTS,
    }
    mode = aliases.get(mode, mode)
    if mode not in KL_MODES:
        raise ValueError(
            "kl_regularization.mode must be "
            f"{KL_MODE_TO_OLD_POLICY_OLD_PROMPTS!r}, got {value!r}"
        )
    return mode


@dataclass(frozen=True)
class KLRegularizationConfig:
    enabled: bool = True
    mode: str = KL_MODE_TO_OLD_POLICY_OLD_PROMPTS
    old_prompt_fraction: float = 0.5
    prompts_per_step: int | None = None
    seed: int = 1
    old_policy_checkpoint_path: str = ""
    update_old_policy_on_task_switch: bool = True
    require_task_boundary_checkpoint: bool = True

    @classmethod
    def from_config(cls, config: Any) -> KLRegularizationConfig:
        raw_config = mapping_get(config, "kl_regularization", {}) or {}
        data_config = mapping_get(config, "data", {}) or {}
        sampler_config = mapping_get(data_config, "sampler", {}) or {}
        raw_prompts_per_step = mapping_get(
            sampler_config,
            "prompts_per_step",
            None,
        )
        if raw_prompts_per_step in (None, "", -1):
            prompts_per_step = None
        else:
            prompts_per_step = max(1, int(raw_prompts_per_step))
        return cls(
            enabled=mapping_get_bool(raw_config, "enabled", default=True),
            mode=normalize_mode(
                mapping_get(
                    raw_config, "mode", KL_MODE_TO_OLD_POLICY_OLD_PROMPTS
                )
            ),
            old_prompt_fraction=normalize_fraction(
                mapping_get(
                    raw_config,
                    "old_prompt_fraction",
                    mapping_get(sampler_config, "old_prompt_fraction", 0.5),
                )
            ),
            prompts_per_step=prompts_per_step,
            seed=int(
                mapping_get(
                    raw_config,
                    "seed",
                    mapping_get(
                        sampler_config,
                        "seed",
                        mapping_get(data_config, "seed", 1),
                    ),
                )
                or 1
            ),
            old_policy_checkpoint_path=str(
                mapping_get(raw_config, "old_policy_checkpoint_path", "") or ""
            ),
            update_old_policy_on_task_switch=mapping_get_bool(
                raw_config,
                "update_old_policy_on_task_switch",
                default=True,
            ),
            require_task_boundary_checkpoint=mapping_get_bool(
                raw_config,
                "require_task_boundary_checkpoint",
                default=True,
            ),
        )

    def is_old_policy_mode(self) -> bool:
        return self.enabled and self.mode == KL_MODE_TO_OLD_POLICY_OLD_PROMPTS


def normalize_fraction(value: Any) -> float:
    fraction = float(value)
    if fraction < 0.0 or fraction >= 1.0:
        raise ValueError(
            "old_prompt_fraction must be in [0, 1) so each batch keeps at "
            f"least one current-task prompt, got {fraction}"
        )
    return fraction


def dummy_tensor() -> torch.Tensor:
    return torch.tensor([0], dtype=torch.uint8)


def build_prompt_key(task_name: str, dataset_index: int) -> str:
    return f"{task_name}::{int(dataset_index)}"


def resolve_task_name(
    extra_info: Mapping[str, Any] | None,
    fallback: str,
) -> str:
    if extra_info is None:
        return str(fallback)
    for key in (
        "kl_regularization_source_task",
        "algorithm_task",
        "task",
        "current_task",
    ):
        value = extra_info.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback)


def dataset_current_task(data_source: Any) -> str | None:
    for source in (data_source, getattr(data_source, "base_dataset", None)):
        if source is None:
            continue
        current_task = getattr(source, "current_task", None)
        if current_task not in (None, ""):
            return str(current_task)
    return None


def is_old_prompt_extra(extra_info: Any) -> bool:
    if not isinstance(extra_info, Mapping):
        return False
    return extra_info.get("kl_regularization_role") == OLD_PROMPT_ROLE


def old_prompt_row_mask(
    batch: Any, *, device: Any | None = None
) -> torch.Tensor:
    extras = batch.non_tensor_batch.get("extra_info")
    batch_size = len(batch)
    if extras is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)
    values = [is_old_prompt_extra(extra_info) for extra_info in extras]
    return torch.tensor(values, dtype=torch.bool, device=device)


def add_kl_regularization_mask(batch: Any, *, mode: str) -> Any:
    if "response_mask" not in batch.batch:
        return batch
    response_mask = batch.batch["response_mask"]
    response_mask = response_mask.to(response_mask.dtype)
    if mode == KL_MODE_TO_OLD_POLICY_OLD_PROMPTS:
        row_mask = old_prompt_row_mask(batch, device=response_mask.device)
        old_prompt_mask = row_mask.unsqueeze(-1).to(response_mask.dtype)
        kl_mask = old_prompt_mask * response_mask
        policy_mask = (1.0 - old_prompt_mask) * response_mask
    else:
        kl_mask = response_mask
        policy_mask = response_mask
    batch.batch["kl_regularization_mask"] = kl_mask
    batch.batch["policy_training_mask"] = policy_mask
    return batch


def zero_old_prompt_advantages(batch: Any) -> Any:
    if "response_mask" not in batch.batch:
        return batch
    row_mask = old_prompt_row_mask(
        batch, device=batch.batch["response_mask"].device
    )
    if not bool(row_mask.any().item()):
        return batch
    for key in ("advantages", "returns"):
        if key in batch.batch:
            value = batch.batch[key].clone()
            value[row_mask] = 0
            batch.batch[key] = value
    return batch


def zero_old_prompt_rewards(
    batch: Any, reward_tensor: torch.Tensor
) -> torch.Tensor:
    row_mask = old_prompt_row_mask(batch, device=reward_tensor.device)
    if not bool(row_mask.any().item()):
        return reward_tensor
    reward_tensor = reward_tensor.clone()
    reward_tensor[row_mask] = 0
    return reward_tensor


@dataclass(frozen=True)
class KLPromptPayload:
    prompt_key: str
    task_name: str
    dataset_index: int
    row_payload: dict[str, Any]


class KLOldPromptDataset:
    def __init__(self, base_dataset: Any):
        self.base_dataset = base_dataset
        self.old_prompt_overrides: dict[int, KLPromptPayload] = {}
        self.prompt_row_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def install_old_prompt_override(
        self,
        slot_index: int,
        payload: KLPromptPayload,
    ) -> None:
        self.old_prompt_overrides[int(slot_index)] = deepcopy(payload)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slot_index = int(index)
        payload = self.old_prompt_overrides.pop(slot_index, None)
        if payload is not None:
            return self.payload_to_old_prompt_row(payload)

        row = deepcopy(self.base_dataset[slot_index])
        row = self.normalize_current_prompt_row(row, slot_index=slot_index)
        prompt_key = str(row["extra_info"]["kl_regularization_prompt_key"])
        self.prompt_row_cache[prompt_key] = deepcopy(row)
        return row

    def on_batch_end(self, batch: Any) -> None:
        if hasattr(self.base_dataset, "on_batch_end"):
            self.base_dataset.on_batch_end(batch)

    def normalize_current_prompt_row(
        self,
        row: dict[str, Any],
        *,
        slot_index: int,
    ) -> dict[str, Any]:
        extra_info = deepcopy(row.get("extra_info") or {})
        task_name = resolve_task_name(
            extra_info,
            fallback=getattr(self.base_dataset, "current_task", "task_0"),
        )
        dataset_index = int(row.get("index", slot_index))
        prompt_key = build_prompt_key(task_name, dataset_index)
        extra_info.setdefault("algorithm_task", task_name)
        extra_info["kl_regularization_role"] = CURRENT_PROMPT_ROLE
        extra_info["kl_regularization_prompt_key"] = prompt_key
        extra_info["kl_regularization_source_task"] = task_name
        row["extra_info"] = extra_info
        row["index"] = dataset_index
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        if "dummy_tensor" not in row:
            row["dummy_tensor"] = dummy_tensor()
        return row

    def payload_to_old_prompt_row(
        self, payload: KLPromptPayload
    ) -> dict[str, Any]:
        row = deepcopy(payload.row_payload)
        extra_info = deepcopy(row.get("extra_info") or {})
        extra_info["algorithm_task"] = payload.task_name
        extra_info["kl_regularization_role"] = OLD_PROMPT_ROLE
        extra_info["kl_regularization_prompt_key"] = payload.prompt_key
        extra_info["kl_regularization_source_task"] = payload.task_name
        row["extra_info"] = extra_info
        row["index"] = int(payload.dataset_index)
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        row["dummy_tensor"] = dummy_tensor()
        return row


@dataclass(frozen=True)
class KLOldPromptSamplerConfig:
    enabled: bool = True
    old_prompt_fraction: float = 0.5
    prompts_per_step: int | None = None
    seed: int = 1

    @classmethod
    def from_data_config(cls, data_config: Any) -> KLOldPromptSamplerConfig:
        sampler_config = mapping_get(data_config, "sampler", {}) or {}
        raw_prompts_per_step = mapping_get(
            sampler_config, "prompts_per_step", None
        )
        if raw_prompts_per_step in (None, "", -1):
            prompts_per_step = int(
                mapping_get(
                    data_config,
                    "gen_batch_size",
                    mapping_get(data_config, "train_batch_size", 1),
                )
            )
        else:
            prompts_per_step = int(raw_prompts_per_step)
        return cls(
            enabled=mapping_get_bool(sampler_config, "enabled", default=True),
            old_prompt_fraction=normalize_fraction(
                mapping_get(sampler_config, "old_prompt_fraction", 0.5)
            ),
            prompts_per_step=max(1, prompts_per_step),
            seed=int(
                mapping_get(
                    sampler_config,
                    "seed",
                    mapping_get(data_config, "seed", 1),
                )
                or 1
            ),
        )


class KLOldPromptSampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Sized, data_config: Any):
        if not isinstance(data_source, KLOldPromptDataset):
            raise TypeError(
                "KLOldPromptSampler requires KLOldPromptDataset. Use "
                "KLRegularizationReasoningGymRunner so old-prompt rows can "
                "be injected without mutating the mllm_crl dataset."
            )
        self.data_source = data_source
        self.data_config = data_config
        self.config = KLOldPromptSamplerConfig.from_data_config(data_config)
        self.dataset_size = len(data_source)
        self.shuffle = mapping_get_bool(data_config, "shuffle", default=True)
        self.rng = random.Random(self.config.seed)
        self.order: list[int] = []
        self.cursor = 0
        self.emitted_steps = 0
        self.old_prompt_payloads: dict[str, KLPromptPayload] = {}
        self.active_old_prompt_keys: set[str] = set()

    def __iter__(self) -> Iterator[int]:
        remaining = len(self.data_source)
        while remaining > 0:
            batch_size = min(
                self.config.prompts_per_step or remaining, remaining
            )
            yield from self.next_batch_indices(batch_size=batch_size)
            remaining -= batch_size

    def __len__(self) -> int:
        return self.dataset_size

    def next_batch_indices(self, batch_size: int) -> list[int]:
        batch_size = max(1, int(batch_size))
        old_prompt_budget = self.max_old_prompt_budget(batch_size)
        reserved_slots: set[int] = set()
        indices: list[int] = []
        issued_old_prompts = 0
        for _ in range(batch_size):
            slot_index = self.next_new_slot(exclude=reserved_slots)
            reserved_slots.add(slot_index)
            if issued_old_prompts < old_prompt_budget:
                payload = self.next_old_prompt_payload()
                if payload is not None:
                    self.data_source.install_old_prompt_override(
                        slot_index, payload
                    )
                    self.active_old_prompt_keys.add(payload.prompt_key)
                    issued_old_prompts += 1
            indices.append(slot_index)
        self.emitted_steps += 1
        return indices

    def max_old_prompt_budget(self, batch_size: int) -> int:
        if not self.config.enabled or batch_size <= 1:
            return 0
        raw_budget = math.floor(batch_size * self.config.old_prompt_fraction)
        return max(0, min(raw_budget, batch_size - 1))

    def next_new_slot(self, exclude: set[int]) -> int:
        while True:
            if self.cursor >= len(self.order):
                self.order = list(range(self.dataset_size))
                if self.shuffle:
                    self.rng.shuffle(self.order)
                self.cursor = 0
            slot_index = int(self.order[self.cursor])
            self.cursor += 1
            if slot_index not in exclude:
                return slot_index

    def next_old_prompt_payload(self) -> KLPromptPayload | None:
        current_task = dataset_current_task(self.data_source)
        candidates = [
            payload
            for payload in self.old_prompt_payloads.values()
            if payload.task_name != current_task
            and payload.prompt_key not in self.active_old_prompt_keys
        ]
        if not candidates:
            return None
        return deepcopy(self.rng.choice(candidates))

    def update(self, batch: Any) -> None:
        self.refresh_payloads_from_dataset_cache()
        extras = batch.non_tensor_batch.get("extra_info", [])
        for extra_info in extras:
            if not isinstance(extra_info, Mapping):
                continue
            self.active_old_prompt_keys.discard(
                str(extra_info.get("kl_regularization_prompt_key", ""))
            )

    def refresh_payloads_from_dataset_cache(self) -> None:
        for prompt_key, row in self.data_source.prompt_row_cache.items():
            extra_info = row.get("extra_info") or {}
            if is_old_prompt_extra(extra_info):
                continue
            task_name = resolve_task_name(extra_info, fallback="task_0")
            dataset_index = int(row.get("index", 0))
            self.old_prompt_payloads[str(prompt_key)] = KLPromptPayload(
                prompt_key=str(prompt_key),
                task_name=task_name,
                dataset_index=dataset_index,
                row_payload=deepcopy(row),
            )

    def get_state(self) -> dict[str, Any]:
        return {
            "order": self.order.copy(),
            "cursor": self.cursor,
            "emitted_steps": self.emitted_steps,
            "rng_state": self.rng.getstate(),
            "old_prompt_payloads": deepcopy(self.old_prompt_payloads),
            "active_old_prompt_keys": sorted(self.active_old_prompt_keys),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.order = [int(index) for index in state.get("order", [])]
        self.cursor = int(state.get("cursor", 0))
        self.emitted_steps = int(state.get("emitted_steps", 0))
        if "rng_state" in state:
            self.rng.setstate(state["rng_state"])
        self.old_prompt_payloads = deepcopy(
            state.get("old_prompt_payloads", {})
        )
        self.active_old_prompt_keys = {
            str(key) for key in state.get("active_old_prompt_keys", [])
        }
