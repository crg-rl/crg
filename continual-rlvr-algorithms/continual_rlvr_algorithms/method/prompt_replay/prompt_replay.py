from __future__ import annotations

import heapq
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sized

import torch
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


def mapping_get(payload: Any, key: str, default: Any) -> Any:
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def mapping_get_bool(payload: Any, key: str, *, default: bool) -> bool:
    return bool(mapping_get(payload, key, default))


def build_prompt_key(task_name: str, dataset_index: int) -> str:
    return f"{task_name}::{int(dataset_index)}"


def dummy_tensor():
    return torch.tensor([0], dtype=torch.uint8)


SAFE_ROW_PAYLOAD_KEYS = frozenset(
    {
        "raw_prompt",
        "reward_model",
        "data_source",
        "extra_info",
        "index",
        "tools_kwargs",
        "interaction_kwargs",
        "question",
        "answer",
        "multi_modal_inputs",
    }
)

# A stored behavior-policy sample contains the exact sequence tensors,
# response, and behavior log-probabilities used for scoring. The response mask
# is recomputed after injection and may be retained for diagnostics.
OFF_POLICY_REQUIRED_TRAJECTORY_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "rollout_log_probs",
)
OFF_POLICY_OPTIONAL_TRAJECTORY_KEYS = ("response_mask",)
OFF_POLICY_TRAJECTORY_KEYS = (
    *OFF_POLICY_REQUIRED_TRAJECTORY_KEYS,
    *OFF_POLICY_OPTIONAL_TRAJECTORY_KEYS,
)


class StaleTrajectoryContractError(RuntimeError):
    """Raised when an off-policy replay row cannot be replayed faithfully."""


def stale_trajectory_static_errors(trajectory: dict[str, Any]) -> list[str]:
    """Validate a stored trajectory before it is admitted to the replay buffer."""
    errors: list[str] = []
    for key in OFF_POLICY_REQUIRED_TRAJECTORY_KEYS:
        if key not in trajectory:
            errors.append(f"missing {key}")
        elif not isinstance(trajectory[key], torch.Tensor):
            errors.append(f"non-tensor {key}")
    return errors



@dataclass(frozen=True)
class PromptReplayConfig:
    enabled: bool = True
    replay_scope: str = "previous_tasks"
    replay_fraction: float = 0.5
    cooldown_steps: int = 5
    min_pass_rate: float | None = 0.24
    max_pass_rate: float | None = 0.7
    prompts_per_step: int | None = None
    seed: int = 1
    # Off-policy ablation: when True the replay buffer keeps each prompt's
    # response and rollout log-probs from the step it was collected, and the
    # trainer reuses that stale trajectory instead of regenerating under the
    # current policy. Importance-sampling correction for the resulting
    # distribution shift is delegated to verl's rollout_correction. Default
    # False keeps the on-policy CPR behaviour untouched.
    off_policy: bool = False

    @classmethod
    def from_data_config(cls, data_config: Any) -> PromptReplayConfig:
        sampler_cfg = mapping_get(data_config, "sampler", {}) or {}
        raw_prompts_per_step = mapping_get(
            sampler_cfg,
            "prompts_per_step",
            None,
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
        min_pass_rate = normalize_rate_bound(
            mapping_get(sampler_cfg, "min_pass_rate", 0.24)
        )
        max_pass_rate = normalize_rate_bound(
            mapping_get(sampler_cfg, "max_pass_rate", 0.7)
        )
        if (
            min_pass_rate is not None
            and max_pass_rate is not None
            and min_pass_rate > max_pass_rate
        ):
            min_pass_rate, max_pass_rate = max_pass_rate, min_pass_rate
        seed = mapping_get(
            sampler_cfg,
            "seed",
            mapping_get(data_config, "seed", 1),
        )
        return cls(
            enabled=mapping_get_bool(
                sampler_cfg,
                "enabled",
                default=True,
            ),
            replay_scope=normalize_replay_scope(
                mapping_get(sampler_cfg, "replay_scope", "previous_tasks")
            ),
            replay_fraction=min(
                max(
                    float(mapping_get(sampler_cfg, "replay_fraction", 0.5)),
                    0.0,
                ),
                0.5,
            ),
            cooldown_steps=max(
                0,
                int(mapping_get(sampler_cfg, "cooldown_steps", 5)),
            ),
            min_pass_rate=min_pass_rate,
            max_pass_rate=max_pass_rate,
            prompts_per_step=max(1, prompts_per_step),
            seed=int(seed or 1),
            off_policy=mapping_get_bool(
                sampler_cfg,
                "off_policy",
                default=False,
            ),
        )

    def hydra_overrides(self) -> list[str]:
        overrides = [
            "++data.sampler.class_path=pkg://continual_rlvr_algorithms.method.prompt_replay.prompt_replay",
            "++data.sampler.class_name=PromptReplaySampler",
            f"++data.sampler.enabled={str(self.enabled).lower()}",
            f"++data.sampler.replay_scope={self.replay_scope}",
            f"++data.sampler.replay_fraction={self.replay_fraction}",
            f"++data.sampler.cooldown_steps={self.cooldown_steps}",
            f"++data.sampler.seed={self.seed}",
            f"++data.sampler.off_policy={str(self.off_policy).lower()}",
        ]
        min_pass_rate = (
            -1 if self.min_pass_rate is None else self.min_pass_rate
        )
        max_pass_rate = (
            -1 if self.max_pass_rate is None else self.max_pass_rate
        )
        if self.prompts_per_step is not None:
            overrides.append(
                f"++data.sampler.prompts_per_step={self.prompts_per_step}"
            )
        overrides.append(f"++data.sampler.min_pass_rate={min_pass_rate}")
        overrides.append(f"++data.sampler.max_pass_rate={max_pass_rate}")
        return overrides


@dataclass(frozen=True)
class PromptReplayPayload:
    prompt_key: str
    task_name: str
    dataset_index: int
    raw_prompt: Any
    data_source: Any
    reward_model: Any
    extra_info: dict[str, Any] = field(default_factory=dict)
    tools_kwargs: dict[str, Any] = field(default_factory=dict)
    interaction_kwargs: dict[str, Any] = field(default_factory=dict)
    row_payload: dict[str, Any] = field(default_factory=dict)
    # Store one frozen trajectory per rollout.n sample for the off-policy ablation.
    # It stores the historical response and behavior log-probs; reward and
    # advantage are recomputed when that trajectory is replayed.
    trajectories: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class PromptReplayEntry:
    prompt_key: str
    task_name: str
    dataset_index: int
    training_step: int
    pass_rate: float
    was_reused: bool = False
    cooldown_ready_step: int | None = None


@dataclass
class StepReplayState:
    replay_budget: int
    issued_total: int = 0
    issued_replay: int = 0
    issued_new: int = 0
    reserved_slots: set[int] = field(default_factory=set)


class PromptReplayDataset:
    def __init__(self, base_dataset: Any):
        self.base_dataset = base_dataset
        self.replay_overrides: dict[int, PromptReplayPayload] = {}
        self.prompt_row_cache: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def install_replay_override(
        self,
        slot_index: int,
        payload: PromptReplayPayload,
    ) -> None:
        if not payload.row_payload:
            cached_row = self.prompt_row_cache.get(payload.prompt_key)
            if cached_row is not None:
                payload = replace(
                    payload,
                    row_payload=deepcopy(cached_row),
                )
        self.replay_overrides[int(slot_index)] = deepcopy(payload)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slot_index = int(index)
        payload = self.replay_overrides.pop(slot_index, None)
        if payload is not None:
            return self.payload_to_row(payload)
        row = deepcopy(self.base_dataset[slot_index])
        row = self.normalize_base_row(row, slot_index=slot_index)
        prompt_key = str(
            row.get("extra_info", {}).get("prompt_replay_key", "")
        )
        if prompt_key:
            self.prompt_row_cache[prompt_key] = deepcopy(row)
        return row

    def on_batch_end(self, batch: Any) -> None:
        if hasattr(self.base_dataset, "on_batch_end"):
            self.base_dataset.on_batch_end(batch)

    def normalize_base_row(
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
        extra_info["prompt_replay_key"] = prompt_key
        extra_info["prompt_replay_source"] = "base"
        row["extra_info"] = extra_info
        row["index"] = dataset_index
        row.setdefault("tools_kwargs", {})
        row.setdefault("interaction_kwargs", {})
        if "dummy_tensor" not in row:
            row["dummy_tensor"] = dummy_tensor()
        return row

    def payload_to_row(self, payload: PromptReplayPayload) -> dict[str, Any]:
        has_full_row_payload = bool(payload.row_payload)
        if has_full_row_payload:
            row = deepcopy(payload.row_payload)
        else:
            row = {
                "raw_prompt": deepcopy(payload.raw_prompt),
                "data_source": payload.data_source,
                "reward_model": deepcopy(payload.reward_model),
                "extra_info": deepcopy(payload.extra_info),
                "index": int(payload.dataset_index),
                "tools_kwargs": deepcopy(payload.tools_kwargs),
                "interaction_kwargs": deepcopy(payload.interaction_kwargs),
            }
        extra_info = deepcopy(row.get("extra_info") or payload.extra_info)
        extra_info["algorithm_task"] = payload.task_name
        extra_info["prompt_replay_key"] = payload.prompt_key
        extra_info["prompt_replay_source"] = "replay"
        if not has_full_row_payload:
            row["raw_prompt"] = deepcopy(payload.raw_prompt)
            row["data_source"] = payload.data_source
            row["reward_model"] = deepcopy(payload.reward_model)
            row["tools_kwargs"] = deepcopy(payload.tools_kwargs)
            row["interaction_kwargs"] = deepcopy(payload.interaction_kwargs)
        row.setdefault("data_source", payload.data_source)
        row.setdefault("reward_model", deepcopy(payload.reward_model))
        row.setdefault("tools_kwargs", deepcopy(payload.tools_kwargs))
        row.setdefault("interaction_kwargs", deepcopy(payload.interaction_kwargs))
        row["extra_info"] = extra_info
        row["index"] = int(payload.dataset_index)
        row["dummy_tensor"] = dummy_tensor()
        return row


def resolve_task_name(
    extra_info: Mapping[str, Any] | None,
    fallback: str,
) -> str:
    if extra_info is None:
        return str(fallback)
    for key in ("algorithm_task", "task", "current_task"):
        value = extra_info.get(key)
        if value not in (None, ""):
            return str(value)
    return str(fallback)


def normalize_rate_bound(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value < 0.0:
        return None
    return float(min(max(value, 0.0), 1.0))


def normalize_replay_scope(value: Any) -> str:
    scope = str(value or "previous_tasks").strip().lower().replace("-", "_")
    aliases = {
        "previous": "previous_tasks",
        "prior": "previous_tasks",
        "past": "previous_tasks",
        "all": "all_seen",
        "all_tasks": "all_seen",
        "all_seen_tasks": "all_seen",
    }
    scope = aliases.get(scope, scope)
    if scope not in {"previous_tasks", "all_seen"}:
        raise ValueError(
            "prompt replay scope must be 'previous_tasks' or 'all_seen'"
        )
    return scope


def to_python_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def extract_score_vector(batch: Any) -> list[float]:
    if "acc" in batch.batch:
        tensor = batch.batch["acc"]
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu().tolist()
        return [float(item) for item in tensor]
    scores = batch.batch["token_level_scores"]
    if hasattr(scores, "sum"):
        scores = scores.sum(-1)
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().tolist()
    return [float(item) for item in scores]


def dataset_current_task(data_source: Any) -> str | None:
    for source in (data_source, getattr(data_source, "base_dataset", None)):
        if source is None:
            continue
        current_task = getattr(source, "current_task", None)
        if current_task not in (None, ""):
            return str(current_task)
    return None


class PromptReplaySampler(AbstractCurriculumSampler):
    def __init__(self, data_source: Sized, data_config: Any):
        self.data_source = data_source
        self.data_config = data_config
        self.config = PromptReplayConfig.from_data_config(data_config)
        self.dataset_size = len(data_source)
        self.shuffle = mapping_get_bool(
            data_config,
            "shuffle",
            default=True,
        )
        self.rng = random.Random(self.config.seed)
        self.order: list[int] = []
        self.cursor = 0
        self.emitted_steps = 0
        self.completed_steps = 0
        self.prompt_pass_rates: dict[str, float] = {}
        self.prompt_last_step: dict[str, int] = {}
        self.prompt_payloads: dict[str, PromptReplayPayload] = {}
        self.prompt_meta: dict[str, tuple[str, int]] = {}
        self.active_replay_keys: set[str] = set()
        self.replay_heap: list[tuple[float, float, str]] = []

        if not isinstance(self.data_source, PromptReplayDataset):
            raise TypeError(
                "PromptReplaySampler requires PromptReplayDataset so "
                "replayed prompts "
                "can be injected without mutating mllm_crl substrate."
            )

    def __iter__(self) -> Iterator[int]:
        remaining = len(self.data_source)
        while remaining > 0:
            batch_size = min(self.config.prompts_per_step, remaining)
            yield from self.next_batch_indices(batch_size=batch_size)
            remaining -= batch_size

    def __len__(self) -> int:
        return self.dataset_size

    def next_batch_indices(self, batch_size: int | None = None) -> list[int]:
        batch_size = (
            self.config.prompts_per_step
            if batch_size is None
            else int(batch_size)
        )
        batch_size = max(1, batch_size)
        state = StepReplayState(
            replay_budget=self.max_replay_budget(batch_size)
        )
        indices: list[int] = []
        for _ in range(batch_size):
            slot_index = self.next_new_slot(exclude=state.reserved_slots)
            state.reserved_slots.add(slot_index)
            prompt_key = None
            if (
                state.replay_budget > 0
                and state.issued_replay < state.replay_budget
            ):
                prompt_key = self.pop_next_replay_candidate(
                    target_step=self.emitted_steps
                )
            if prompt_key is not None:
                payload = deepcopy(self.prompt_payloads[prompt_key])
                self.data_source.install_replay_override(slot_index, payload)
                self.active_replay_keys.add(prompt_key)
                state.issued_replay += 1
            else:
                state.issued_new += 1
            state.issued_total += 1
            indices.append(slot_index)
        if state.issued_replay and os.environ.get("CRG_REPLAY_TRACE"):
            print(
                "[CPR] batch "
                f"{self.emitted_steps + 1}: replay={state.issued_replay} "
                f"current={state.issued_new}",
                flush=True,
            )
        self.emitted_steps += 1
        return indices

    def get_state(self) -> dict[str, Any]:
        return {
            "order": self.order.copy(),
            "cursor": self.cursor,
            "emitted_steps": self.emitted_steps,
            "completed_steps": self.completed_steps,
            "rng_state": self.rng.getstate(),
            "prompt_pass_rates": self.prompt_pass_rates.copy(),
            "prompt_last_step": self.prompt_last_step.copy(),
            "prompt_payloads": deepcopy(self.prompt_payloads),
            "prompt_meta": deepcopy(self.prompt_meta),
            "active_replay_keys": sorted(self.active_replay_keys),
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.order = [int(index) for index in state.get("order", [])]
        self.cursor = int(state.get("cursor", 0))
        self.emitted_steps = int(state.get("emitted_steps", 0))
        self.completed_steps = int(state.get("completed_steps", 0))
        self.rng.setstate(state["rng_state"])
        self.prompt_pass_rates = {
            str(key): float(value)
            for key, value in state.get("prompt_pass_rates", {}).items()
        }
        self.prompt_last_step = {
            str(key): int(value)
            for key, value in state.get("prompt_last_step", {}).items()
        }
        self.prompt_payloads = deepcopy(state.get("prompt_payloads", {}))
        self.prompt_meta = {
            str(key): (str(value[0]), int(value[1]))
            for key, value in state.get("prompt_meta", {}).items()
        }
        self.active_replay_keys = {
            str(key) for key in state.get("active_replay_keys", [])
        }
        self.rebuild_replay_heap()

    def max_replay_budget(self, batch_size: int) -> int:
        if not self.config.enabled or batch_size <= 1:
            return 0
        raw_budget = math.floor(batch_size * self.config.replay_fraction)
        return max(0, min(raw_budget, batch_size // 2))

    def rebuild_replay_heap(self) -> None:
        self.replay_heap = []
        for prompt_key in self.prompt_pass_rates:
            entry = self.make_replay_heap_entry(prompt_key)
            if entry is not None:
                heapq.heappush(self.replay_heap, entry)

    def next_new_slot(self, exclude: set[int]) -> int:
        while True:
            if self.cursor >= len(self.order):
                self.order = list(range(self.dataset_size))
                if self.shuffle:
                    self.rng.shuffle(self.order)
                self.cursor = 0
            slot_index = self.order[self.cursor]
            self.cursor += 1
            if slot_index not in exclude:
                return int(slot_index)

    def pop_next_replay_candidate(self, *, target_step: int) -> str | None:
        buffered: list[tuple[float, float, str]] = []
        candidate: str | None = None
        while self.replay_heap:
            distance, rate_snapshot, prompt_key = heapq.heappop(
                self.replay_heap
            )
            rate = self.prompt_pass_rates.get(prompt_key)
            if rate is None or not self.is_rate_within_window(rate):
                continue
            if prompt_key in self.active_replay_keys:
                buffered.append((distance, rate_snapshot, prompt_key))
                continue
            if not self.is_prompt_allowed_by_scope(prompt_key):
                buffered.append((distance, rate_snapshot, prompt_key))
                continue
            if not self.is_prompt_cooled_down(prompt_key, target_step):
                buffered.append((distance, rate_snapshot, prompt_key))
                continue
            candidate = prompt_key
            break
        for item in buffered:
            heapq.heappush(self.replay_heap, item)
        return candidate

    def update(self, batch: Any) -> None:
        entries = self.build_entries_from_batch(
            batch,
            training_step=self.completed_steps,
        )
        for entry, payload in entries:
            self.prompt_pass_rates[entry.prompt_key] = entry.pass_rate
            self.prompt_last_step[entry.prompt_key] = entry.training_step
            self.prompt_payloads[entry.prompt_key] = payload
            self.prompt_meta[entry.prompt_key] = (
                entry.task_name,
                entry.dataset_index,
            )
            self.active_replay_keys.discard(entry.prompt_key)
            if self.is_rate_within_window(entry.pass_rate):
                replay_entry = self.make_replay_heap_entry(
                    entry.prompt_key,
                    entry.pass_rate,
                )
                if replay_entry is not None:
                    heapq.heappush(self.replay_heap, replay_entry)
        if self.config.off_policy:
            captured = sum(
                len(payload.trajectories or []) for _, payload in entries
            )
            eligible = sum(
                1
                for prompt_key, payload in self.prompt_payloads.items()
                if payload.trajectories
                and self.is_rate_within_window(
                    self.prompt_pass_rates[prompt_key]
                )
            )
            print(
                "[CPR-offpolicy] capture: "
                f"captured={captured} eligible={eligible} "
                f"buffered_prompts={len(self.prompt_payloads)}",
                flush=True,
            )
        self.completed_steps += 1

    def build_entries_from_batch(
        self,
        batch: Any,
        *,
        training_step: int,
    ) -> list[tuple[PromptReplayEntry, PromptReplayPayload]]:
        scores = extract_score_vector(batch)
        non_tensor = batch.non_tensor_batch
        raw_prompts = non_tensor.get("raw_prompt", [None] * len(scores))
        reward_models = non_tensor.get("reward_model", [None] * len(scores))
        data_sources = non_tensor.get(
            "data_source",
            ["reasoning_gym"] * len(scores),
        )
        extras = non_tensor.get("extra_info", [None] * len(scores))
        indices = non_tensor.get("index", list(range(len(scores))))
        uids = non_tensor.get("uid", list(range(len(scores))))
        tools_kwargs = non_tensor.get(
            "tools_kwargs",
            [{} for _ in range(len(scores))],
        )
        interaction_kwargs = non_tensor.get(
            "interaction_kwargs",
            [{} for _ in range(len(scores))],
        )

        grouped: dict[str, dict[str, Any]] = {}
        for position, uid in enumerate(uids):
            extra_info = deepcopy(extras[position] or {})
            task_name = resolve_task_name(extra_info, fallback="task_0")
            dataset_index = int(indices[position])
            prompt_key = str(
                extra_info.get(
                    "prompt_replay_key",
                    build_prompt_key(task_name, dataset_index),
                )
            )
            cached_row_payload: dict[str, Any] = {}
            cached_row = self.data_source.prompt_row_cache.get(prompt_key)
            if cached_row is not None:
                cached_row_payload = deepcopy(cached_row)
            group = grouped.setdefault(
                str(uid),
                {
                    "scores": [],
                    "prompt_key": prompt_key,
                    "task_name": task_name,
                    "dataset_index": dataset_index,
                    "raw_prompt": raw_prompts[position],
                    "reward_model": reward_models[position],
                    "data_source": data_sources[position],
                    "extra_info": extra_info,
                    "tools_kwargs": deepcopy(tools_kwargs[position] or {}),
                    "interaction_kwargs": deepcopy(
                        interaction_kwargs[position] or {}
                    ),
                    "row_payload": (
                        cached_row_payload
                        or self.build_row_payload_from_batch(
                            non_tensor,
                            position=position,
                            batch_size=len(scores),
                        )
                    ),
                    "trajectories": [],
                },
            )
            group["scores"].append(to_python_float(scores[position]))
            if self.config.off_policy:
                group["trajectories"].append(
                    self.collect_trajectory_from_batch(batch, position=position)
                )

        results: list[tuple[PromptReplayEntry, PromptReplayPayload]] = []
        for group in grouped.values():
            prompt_key = str(group["prompt_key"])
            pass_rate = float(fmean(group["scores"]))
            was_reused = prompt_key in self.active_replay_keys or (
                group["extra_info"].get("prompt_replay_source") == "replay"
            )
            cooldown_ready_step = training_step + self.config.cooldown_steps
            entry = PromptReplayEntry(
                prompt_key=prompt_key,
                task_name=str(group["task_name"]),
                dataset_index=int(group["dataset_index"]),
                training_step=int(training_step),
                pass_rate=pass_rate,
                was_reused=was_reused,
                cooldown_ready_step=cooldown_ready_step,
            )
            payload = PromptReplayPayload(
                prompt_key=prompt_key,
                task_name=str(group["task_name"]),
                dataset_index=int(group["dataset_index"]),
                raw_prompt=deepcopy(group["raw_prompt"]),
                data_source=group["data_source"],
                reward_model=deepcopy(group["reward_model"]),
                extra_info=deepcopy(group["extra_info"]),
                tools_kwargs=deepcopy(group["tools_kwargs"]),
                interaction_kwargs=deepcopy(group["interaction_kwargs"]),
                row_payload=deepcopy(group["row_payload"]),
                trajectories=(
                    group["trajectories"] if self.config.off_policy else None
                ),
            )
            results.append((entry, payload))
        return results

    def build_row_payload_from_batch(
        self,
        non_tensor: Mapping[str, Any],
        *,
        position: int,
        batch_size: int,
    ) -> dict[str, Any]:
        row_payload: dict[str, Any] = {}
        for key, values in non_tensor.items():
            normalized_key = str(key)
            if normalized_key == "uid" or normalized_key.startswith("__"):
                continue
            if normalized_key not in SAFE_ROW_PAYLOAD_KEYS:
                continue
            if not hasattr(values, "__len__") or len(values) != batch_size:
                continue
            row_payload[normalized_key] = deepcopy(values[position])
        return row_payload

    def collect_trajectory_from_batch(
        self,
        batch: Any,
        *,
        position: int,
    ) -> dict[str, Any]:
        """Snapshot one sample's response tensors for off-policy replay.

        Tensors are detached and moved to CPU, avoiding GPU-memory and
        autograd-graph retention across stages.
        """
        trajectory: dict[str, Any] = {}
        tensor_batch = batch.batch
        if tensor_batch is None:
            return trajectory
        for key in OFF_POLICY_TRAJECTORY_KEYS:
            if key not in tensor_batch:
                continue
            value = tensor_batch[key]
            if not isinstance(value, torch.Tensor):
                continue
            sample = value[position].detach().to("cpu").clone()
            trajectory[key] = sample
        errors = stale_trajectory_static_errors(trajectory)
        if errors:
            raise StaleTrajectoryContractError(
                "cannot capture stale trajectory at "
                f"batch position {position}: {', '.join(errors)}"
            )
        return trajectory

    def is_prompt_cooled_down(self, prompt_key: str, target_step: int) -> bool:
        if self.config.cooldown_steps <= 0:
            return True
        last_step = self.prompt_last_step.get(prompt_key)
        if last_step is None:
            return True
        return (target_step - last_step) >= self.config.cooldown_steps

    def is_prompt_allowed_by_scope(self, prompt_key: str) -> bool:
        if self.config.replay_scope == "all_seen":
            return True
        current_task = dataset_current_task(self.data_source)
        if current_task is None:
            return False
        prompt_meta = self.prompt_meta.get(prompt_key)
        if prompt_meta is not None:
            prompt_task = str(prompt_meta[0])
        else:
            payload = self.prompt_payloads.get(prompt_key)
            prompt_task = None if payload is None else str(payload.task_name)
        return prompt_task is not None and prompt_task != current_task

    def is_rate_within_window(self, rate: float) -> bool:
        if rate <= 0.0:
            return False
        if (
            self.config.min_pass_rate is not None
            and rate < self.config.min_pass_rate
        ):
            return False
        return not (
            self.config.max_pass_rate is not None
            and rate > self.config.max_pass_rate
        )

    def make_replay_heap_entry(
        self,
        prompt_key: str,
        rate: float | None = None,
    ) -> tuple[float, float, str] | None:
        if rate is None:
            rate = self.prompt_pass_rates.get(prompt_key)
        if rate is None or not self.is_rate_within_window(rate):
            return None
        distance = abs(rate - 0.5)
        return (distance, float(rate), prompt_key)
