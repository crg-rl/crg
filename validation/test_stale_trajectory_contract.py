#!/usr/bin/env python3
"""Static contract for CPR stale-trajectory/sample replay.

It proves the healthy ``calculate_log_probs=false`` path: stale sequence
injection happens before reward, then an ordinary pre-update actor logprob pass
creates fresh behavior anchors and stale rows overwrite those anchors exactly.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask

from continual_rlvr_algorithms.method.prompt_replay.offpolicy_replay import (
    PENDING_BEHAVIOR_ANCHORS_KEY,
    hydrate_offpolicy_behavior_log_probs,
    inject_offpolicy_trajectories,
    validate_offpolicy_replay_config,
)
from continual_rlvr_algorithms.method.prompt_replay.prompt_replay import (
    PromptReplayDataset,
    PromptReplaySampler,
    StaleTrajectoryContractError,
)


class BaseDataset:
    current_task = "task_current"

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        return {
            "raw_prompt": [{"role": "user", "content": f"p{index}"}],
            "data_source": "reasoning_gym",
            "reward_model": {"ground_truth": "x"},
            "extra_info": {"algorithm_task": "task_old"},
            "index": index,
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }


def make_batch(
    *,
    sources: list[str],
    prompt_keys: list[str],
    responses: list[list[int]],
    log_probs: list[list[float]],
    include_rollout_log_probs: bool = True,
) -> DataProto:
    batch_size = len(sources)
    response_tensor = torch.tensor(responses, dtype=torch.long)
    tensors = {
        "input_ids": torch.tensor(
            [[10 + i, 11 + i, *response] for i, response in enumerate(responses)],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3] for _ in range(batch_size)], dtype=torch.long
        ),
        "responses": response_tensor,
        "response_mask": torch.ones_like(response_tensor),
        "token_level_scores": torch.tensor(
            [[0.0, 0.5] for _ in range(batch_size)], dtype=torch.float32
        ),
    }
    if include_rollout_log_probs:
        tensors["rollout_log_probs"] = torch.tensor(log_probs, dtype=torch.float32)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "p"}]] * batch_size, dtype=object
            ),
            "reward_model": np.array([{"ground_truth": "x"}] * batch_size, dtype=object),
            "data_source": np.array(["reasoning_gym"] * batch_size, dtype=object),
            "extra_info": np.array(
                [
                    {
                        "algorithm_task": "task_old",
                        "prompt_replay_source": source,
                        "prompt_replay_key": key,
                    }
                    for source, key in zip(sources, prompt_keys, strict=True)
                ],
                dtype=object,
            ),
            "index": np.arange(batch_size),
            "uid": np.array(["old-group"] * batch_size, dtype=object),
            "tools_kwargs": np.array([{}] * batch_size, dtype=object),
            "interaction_kwargs": np.array([{}] * batch_size, dtype=object),
        },
    )


def make_sampler() -> PromptReplaySampler:
    dataset = PromptReplayDataset(BaseDataset())
    return PromptReplaySampler(
        dataset,
        {
            "train_batch_size": 2,
            "sampler": {
                "enabled": True,
                "off_policy": True,
                "replay_fraction": 0.5,
                "replay_scope": "previous_tasks",
                "cooldown_steps": 0,
                "min_pass_rate": 0.0,
                "max_pass_rate": 1.0,
                "seed": 1,
            },
        },
    )


def assert_equal(actual: torch.Tensor, expected: torch.Tensor, message: str) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(message)


def assert_config_error(config: dict, expected: str) -> None:
    try:
        validate_offpolicy_replay_config(config)
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(f"expected {expected!r}, got {exc!r}") from exc
    else:
        raise AssertionError(f"invalid config did not fail: {expected}")


def check_offpolicy_config_contract() -> list[str]:
    valid = {
        "actor_rollout_ref": {"rollout": {"calculate_log_probs": False}},
        "algorithm": {
            "rollout_correction": {
                "bypass_mode": False,
                "rollout_is": "token",
                "rollout_is_threshold": 2.0,
            }
        },
    }
    validate_offpolicy_replay_config(valid)
    failures = [
        (
            {
                "actor_rollout_ref": {"rollout": {"calculate_log_probs": True}},
                "algorithm": valid["algorithm"],
            },
            "calculate_log_probs=false",
        ),
        (
            {
                "actor_rollout_ref": valid["actor_rollout_ref"],
                "algorithm": {
                    "rollout_correction": {
                        "bypass_mode": True,
                        "rollout_is": "token",
                        "rollout_is_threshold": 2.0,
                    }
                },
            },
            "bypass_mode",
        ),
        (
            {
                "actor_rollout_ref": valid["actor_rollout_ref"],
                "algorithm": {
                    "rollout_correction": {
                        "bypass_mode": False,
                        "rollout_is": "none",
                        "rollout_is_threshold": 2.0,
                    }
                },
            },
            "rollout_is",
        ),
        (
            {
                "actor_rollout_ref": valid["actor_rollout_ref"],
                "algorithm": {
                    "rollout_correction": {
                        "bypass_mode": False,
                        "rollout_is": "token",
                        "rollout_is_threshold": 0.0,
                    }
                },
            },
            "rollout_is_threshold",
        ),
    ]
    for config, expected in failures:
        assert_config_error(config, expected)
    return [expected for _, expected in failures]


def main() -> None:
    config_failures_checked = check_offpolicy_config_contract()
    sampler = make_sampler()
    captured_batch = make_batch(
        sources=["base", "base"],
        prompt_keys=["task_old::0", "task_old::0"],
        responses=[[71, 0], [72, 0]],
        log_probs=[[-7.1, -7.2], [-7.3, -7.4]],
    )
    captured_batch.batch["attention_mask"] = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long
    )
    captured_batch.batch["position_ids"] = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24]], dtype=torch.long
    )
    sampler.update(captured_batch)
    payload = sampler.prompt_payloads["task_old::0"]
    assert payload.trajectories is not None and len(payload.trajectories) == 2

    # Stage captured behavior anchors until the pre-update actor pass returns.
    live_batch = make_batch(
        sources=["base", "replay", "replay"],
        prompt_keys=["task_current::3", "task_old::0", "task_old::0"],
        responses=[[11, 0], [21, 0], [22, 0]],
        log_probs=[[-1.1, -1.2], [-2.1, -2.2], [-2.3, -2.4]],
        include_rollout_log_probs=False,
    )
    live_batch.batch["attention_mask"] = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long
    )
    live_batch.batch["position_ids"] = torch.tensor(
        [[31, 32, 33, 34], [41, 42, 43, 44], [51, 52, 53, 54]],
        dtype=torch.long,
    )
    sequence_keys = ("input_ids", "attention_mask", "position_ids", "responses")
    base_before = {key: live_batch.batch[key][0].clone() for key in sequence_keys}
    live_batch.meta_info["global_token_num"] = [-1] * len(live_batch.batch["responses"])
    stats = inject_offpolicy_trajectories(live_batch, sampler)
    assert stats["replay_rows"] == 2 and stats["replay_hit"] == 2
    for key in (
        "exact_input_ids_match",
        "exact_attention_mask_match",
        "exact_position_ids_match",
        "exact_response_match",
        "exact_behavior_logprob_match",
    ):
        assert stats[key] == 2, key
    assert "rollout_log_probs" not in live_batch.batch
    expected_global_token_num = torch.sum(live_batch.batch["attention_mask"], dim=-1).tolist()
    assert live_batch.meta_info["global_token_num"] == expected_global_token_num
    assert_equal(
        live_batch.batch["response_mask"],
        compute_response_mask(live_batch),
        "response mask was not refreshed after stale injection",
    )
    for key in sequence_keys:
        assert_equal(live_batch.batch[key][0], base_before[key], f"base {key} was modified")
    for offset, position in enumerate([1, 2]):
        for key in sequence_keys:
            assert_equal(
                live_batch.batch[key][position],
                payload.trajectories[offset][key],
                f"replay {key} did not equal stored trajectory",
            )

    pre_update_log_probs = torch.tensor(
        [[-1.1, -1.2], [-9.1, -9.2], [-9.3, -9.4]], dtype=torch.float32
    )
    anchors = live_batch.meta_info.pop(PENDING_BEHAVIOR_ANCHORS_KEY)
    hydration = hydrate_offpolicy_behavior_log_probs(
        live_batch, pre_update_log_probs, anchors
    )
    assert hydration == {
        "hydrated_behavior_logprob_match": 2,
        "fresh_behavior_logprob_rows": 1,
    }
    assert_equal(
        live_batch.batch["rollout_log_probs"][0],
        pre_update_log_probs[0],
        "fresh row did not receive its pre-update actor anchor",
    )
    for offset, position in enumerate([1, 2]):
        assert_equal(
            live_batch.batch["rollout_log_probs"][position],
            payload.trajectories[offset]["rollout_log_probs"],
            "stale row did not receive its historical behavior anchor",
        )

    # Verify rejection when behavior log-probabilities are missing.
    missing_behavior_batch = make_batch(
        sources=["base"], prompt_keys=["task_old::1"], responses=[[81, 0]],
        log_probs=[[-8.1, -8.2]],
    )
    del missing_behavior_batch.batch["rollout_log_probs"]
    try:
        make_sampler().build_entries_from_batch(missing_behavior_batch, training_step=0)
    except StaleTrajectoryContractError as exc:
        assert "rollout_log_probs" in str(exc)
    else:
        raise AssertionError("capture without behavior logprob did not fail closed")

    # Verify atomic rejection of missing samples and malformed anchors.
    missing_batch = make_batch(
        sources=["replay"], prompt_keys=["missing::0"], responses=[[31, 0]],
        log_probs=[[-3.1, -3.2]], include_rollout_log_probs=False,
    )
    missing_before = missing_batch.batch["responses"].clone()
    try:
        inject_offpolicy_trajectories(missing_batch, sampler)
    except StaleTrajectoryContractError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing replay trajectory did not fail closed")
    assert_equal(missing_batch.batch["responses"], missing_before, "missing row was modified")

    bad_sequence = dict(payload.trajectories[0])
    bad_sequence["responses"] = bad_sequence["responses"][:1]
    sampler.prompt_payloads["bad-sequence::0"] = replace(
        payload, prompt_key="bad-sequence::0", trajectories=[bad_sequence]
    )
    shape_batch = make_batch(
        sources=["replay"], prompt_keys=["bad-sequence::0"], responses=[[41, 0]],
        log_probs=[[-4.1, -4.2]], include_rollout_log_probs=False,
    )
    shape_before = shape_batch.batch["responses"].clone()
    try:
        inject_offpolicy_trajectories(shape_batch, sampler)
    except StaleTrajectoryContractError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("shape-mismatched replay trajectory did not fail closed")
    assert_equal(shape_batch.batch["responses"], shape_before, "shape-mismatched row was modified")

    bad_behavior = dict(payload.trajectories[0])
    bad_behavior["rollout_log_probs"] = bad_behavior["rollout_log_probs"][:1]
    sampler.prompt_payloads["bad-behavior::0"] = replace(
        payload, prompt_key="bad-behavior::0", trajectories=[bad_behavior]
    )
    behavior_batch = make_batch(
        sources=["replay"], prompt_keys=["bad-behavior::0"], responses=[[51, 0]],
        log_probs=[[-5.1, -5.2]], include_rollout_log_probs=False,
    )
    try:
        inject_offpolicy_trajectories(behavior_batch, sampler)
    except StaleTrajectoryContractError as exc:
        assert "rollout_log_probs" in str(exc) and "shape mismatch" in str(exc)
    else:
        raise AssertionError("malformed behavior anchor did not fail closed")
    assert "rollout_log_probs" not in behavior_batch.batch

    print(json.dumps({
        "status": "PASS",
        "captured_trajectories": len(payload.trajectories),
        "replay_hit": stats["replay_hit"],
        "exact_input_ids_match": stats["exact_input_ids_match"],
        "exact_attention_mask_match": stats["exact_attention_mask_match"],
        "exact_position_ids_match": stats["exact_position_ids_match"],
        "exact_response_match": stats["exact_response_match"],
        "exact_behavior_logprob_match": stats["exact_behavior_logprob_match"],
        "hydrated_behavior_logprob_match": hydration["hydrated_behavior_logprob_match"],
        "fresh_behavior_logprob_rows": hydration["fresh_behavior_logprob_rows"],
        "failure_modes_checked": [
            "calculate_log_probs_true", "missing", "shape_mismatch",
            "missing_behavior_logprob_capture", "malformed_behavior_anchor",
        ],
        "derived_fields_checked": ["response_mask", "global_token_num"],
        "config_failures_checked": config_failures_checked,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
