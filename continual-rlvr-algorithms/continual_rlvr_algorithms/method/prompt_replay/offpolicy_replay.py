from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from continual_rlvr_algorithms.method.prompt_replay.prompt_replay import (
    OFF_POLICY_REQUIRED_TRAJECTORY_KEYS,
    StaleTrajectoryContractError,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask

if TYPE_CHECKING:
    from verl import DataProto

    from continual_rlvr_algorithms.method.prompt_replay.prompt_replay import (
        PromptReplaySampler,
    )


REPLAY_SOURCE_KEY = "prompt_replay_source"
REPLAY_PROMPT_KEY = "prompt_replay_key"

# ``calculate_log_probs=false`` keeps the normal vLLM generation path healthy
# on the Qwen3 / Reasoning Gym deployment.  The ordinary pre-update actor
# log-probability pass below then provides the behavior anchor for fresh rows;
# stale rows replace that anchor with their stored historical values.
OFF_POLICY_BEHAVIOR_LOGPROB_KEY = "rollout_log_probs"
OFF_POLICY_SEQUENCE_KEYS = tuple(
    key
    for key in OFF_POLICY_REQUIRED_TRAJECTORY_KEYS
    if key != OFF_POLICY_BEHAVIOR_LOGPROB_KEY
)
PENDING_BEHAVIOR_ANCHORS_KEY = "_cpr_stale_behavior_logprob_anchors"


def _config_get(payload: Any, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def validate_offpolicy_replay_config(config: Any) -> None:
    """Fail before workers start if stale replay lacks its IS data contract."""
    rollout = _config_get(_config_get(config, "actor_rollout_ref", {}), "rollout", {})
    if bool(_config_get(rollout, "calculate_log_probs", False)):
        raise ValueError(
            "stale-trajectory replay requires "
            "actor_rollout_ref.rollout.calculate_log_probs=false: behavior "
            "rollout_log_probs are hydrated from the ordinary pre-update actor "
            "log-probability pass, avoiding the broken live-generation path"
        )

    algorithm = _config_get(config, "algorithm", {})
    correction = _config_get(algorithm, "rollout_correction", None)
    if correction is None:
        raise ValueError(
            "stale-trajectory replay requires algorithm.rollout_correction "
            "with explicit importance sampling"
        )
    if bool(_config_get(correction, "bypass_mode", False)):
        raise ValueError(
            "stale-trajectory replay forbids rollout_correction.bypass_mode=true; "
            "use the decoupled old-logprob path for explicit IS correction"
        )
    rollout_is = _config_get(correction, "rollout_is", None)
    if rollout_is not in {"token", "sequence"}:
        raise ValueError(
            "stale-trajectory replay requires "
            "algorithm.rollout_correction.rollout_is to be 'token' or 'sequence'"
        )
    threshold = _config_get(correction, "rollout_is_threshold", None)
    try:
        valid_threshold = float(threshold) > 0.0
    except (TypeError, ValueError):
        valid_threshold = False
    if not valid_threshold:
        raise ValueError(
            "stale-trajectory replay requires a positive "
            "algorithm.rollout_correction.rollout_is_threshold"
        )


def _sample_extra_info(batch: DataProto, position: int) -> dict[str, Any]:
    extras = batch.non_tensor_batch.get("extra_info")
    if extras is None:
        return {}
    value = extras[position]
    return value if isinstance(value, dict) else {}


def replay_positions_by_prompt(batch: DataProto) -> dict[str, list[int]]:
    """Group replay rows by buffer key.

    Each replay-labelled row requires a prompt key; a missing key raises
    ``StaleTrajectoryContractError``.
    """
    grouped: dict[str, list[int]] = {}
    batch_size = len(batch.batch["responses"])
    for position in range(batch_size):
        extra_info = _sample_extra_info(batch, position)
        if extra_info.get(REPLAY_SOURCE_KEY) != "replay":
            continue
        prompt_key = extra_info.get(REPLAY_PROMPT_KEY)
        if not prompt_key:
            raise StaleTrajectoryContractError(
                f"replay row {position} has no {REPLAY_PROMPT_KEY!r}"
            )
        grouped.setdefault(str(prompt_key), []).append(position)
    return grouped


def _trajectory_row_contract_errors(
    batch: DataProto,
    position: int,
    trajectory: dict[str, Any],
) -> list[str]:
    """Validate a stored stale trajectory before mutating a live row.

    Live generation omits ``rollout_log_probs``. The stored behavior anchor is
    checked against the response shape and installed after the normal
    pre-update actor log-probability pass.
    """
    errors: list[str] = []
    for key in OFF_POLICY_SEQUENCE_KEYS:
        if key not in batch.batch:
            errors.append(f"live batch missing {key}")
            continue
        if key not in trajectory:
            errors.append(f"stored trajectory missing {key}")
            continue
        live = batch.batch[key]
        stored = trajectory[key]
        if not isinstance(live, torch.Tensor):
            errors.append(f"live batch {key} is not a tensor")
            continue
        if not isinstance(stored, torch.Tensor):
            errors.append(f"stored trajectory {key} is not a tensor")
            continue
        target = live[position]
        if stored.shape != target.shape:
            errors.append(
                f"shape mismatch for {key}: stored={tuple(stored.shape)} "
                f"live={tuple(target.shape)}"
            )

    behavior = trajectory.get(OFF_POLICY_BEHAVIOR_LOGPROB_KEY)
    if behavior is None:
        errors.append(f"stored trajectory missing {OFF_POLICY_BEHAVIOR_LOGPROB_KEY}")
    elif not isinstance(behavior, torch.Tensor):
        errors.append(f"stored trajectory {OFF_POLICY_BEHAVIOR_LOGPROB_KEY} is not a tensor")
    elif not torch.is_floating_point(behavior):
        errors.append(f"stored trajectory {OFF_POLICY_BEHAVIOR_LOGPROB_KEY} is not floating point")
    else:
        responses = batch.batch.get("responses")
        if not isinstance(responses, torch.Tensor):
            errors.append("live batch responses is not a tensor")
        elif behavior.shape != responses[position].shape:
            errors.append(
                "shape mismatch for rollout_log_probs: "
                f"stored={tuple(behavior.shape)} "
                f"responses={tuple(responses[position].shape)}"
            )
    return errors

def _overwrite_sample(
    batch: DataProto,
    position: int,
    trajectory: dict[str, Any],
) -> tuple[dict[str, bool], torch.Tensor]:
    """Install sequence tensors and stage the historical behavior anchor.

    ``rollout_log_probs`` is staged outside the live batch because ordinary
    generation runs with ``calculate_log_probs=false``. The anchor moves to
    ``_compute_old_log_prob``, where it is merged with freshly recomputed
    pre-update actor log-probabilities for the base rows.
    """
    errors = _trajectory_row_contract_errors(batch, position, trajectory)
    if errors:
        raise StaleTrajectoryContractError("; ".join(errors))

    staged: dict[str, torch.Tensor] = {}
    for key in OFF_POLICY_SEQUENCE_KEYS:
        target = batch.batch[key][position]
        staged[key] = trajectory[key].to(device=target.device, dtype=target.dtype)
    behavior_anchor = (
        trajectory[OFF_POLICY_BEHAVIOR_LOGPROB_KEY].detach().to("cpu").clone()
    )

    # Batch validation completes before mutation, making row injection atomic.
    for key, source in staged.items():
        batch.batch[key][position].copy_(source)

    exact_matches = {
        "input_ids": torch.equal(batch.batch["input_ids"][position], staged["input_ids"]),
        "attention_mask": torch.equal(
            batch.batch["attention_mask"][position], staged["attention_mask"]
        ),
        "position_ids": torch.equal(
            batch.batch["position_ids"][position], staged["position_ids"]
        ),
        "responses": torch.equal(batch.batch["responses"][position], staged["responses"]),
        "rollout_log_probs": torch.equal(
            behavior_anchor,
            trajectory[OFF_POLICY_BEHAVIOR_LOGPROB_KEY].detach().to("cpu"),
        ),
    }
    if not all(exact_matches.values()):
        raise StaleTrajectoryContractError(
            "stale trajectory staging did not preserve exact sequence, response, "
            "and behavior-logprob values"
        )
    return exact_matches, behavior_anchor

def _stats_summary(stats: dict[str, int]) -> str:
    ordered = (
        "replay_prompts",
        "replay_rows",
        "replay_hit",
        "missing",
        "short",
        "invalid",
        "shape_mismatch",
        "exact_input_ids_match",
        "exact_attention_mask_match",
        "exact_position_ids_match",
        "exact_response_match",
        "exact_behavior_logprob_match",
        "hydrated_behavior_logprob_match",
        "fresh_behavior_logprob_rows",
    )
    return " ".join(f"{key}={stats[key]}" for key in ordered if key in stats)

def inject_offpolicy_trajectories(
    batch: DataProto,
    sampler: PromptReplaySampler,
) -> dict[str, int]:
    """Install stale sequences and stage their historical behavior anchors.

    Missing, short, or malformed trajectories raise
    ``StaleTrajectoryContractError``. Fresh rows retain sequence tensors from
    generation and receive anchors from the pre-update actor pass.
    """
    if OFF_POLICY_BEHAVIOR_LOGPROB_KEY in batch.batch:
        raise StaleTrajectoryContractError(
            "stale-trajectory injection expected calculate_log_probs=false "
            "with no live rollout_log_probs"
        )
    grouped = replay_positions_by_prompt(batch)
    stats = {
        "replay_prompts": len(grouped),
        "replay_rows": sum(len(positions) for positions in grouped.values()),
        "replay_hit": 0,
        "missing": 0,
        "short": 0,
        "invalid": 0,
        "shape_mismatch": 0,
        "exact_input_ids_match": 0,
        "exact_attention_mask_match": 0,
        "exact_position_ids_match": 0,
        "exact_response_match": 0,
        "exact_behavior_logprob_match": 0,
    }
    behavior_anchors: dict[int, torch.Tensor] = {}
    for prompt_key, positions in grouped.items():
        payload = sampler.prompt_payloads.get(prompt_key)
        trajectories = None if payload is None else payload.trajectories
        if not trajectories:
            stats["missing"] += len(positions)
            raise StaleTrajectoryContractError(
                "stale trajectory missing for replay prompt "
                f"{prompt_key!r}; {_stats_summary(stats)}"
            )
        if len(trajectories) < len(positions):
            stats["short"] += len(positions) - len(trajectories)
            raise StaleTrajectoryContractError(
                "stale trajectory count is shorter than rollout.n for replay "
                f"prompt {prompt_key!r}; {_stats_summary(stats)}"
            )
        for offset, position in enumerate(positions):
            try:
                exact_matches, behavior_anchor = _overwrite_sample(
                    batch,
                    position,
                    trajectories[offset],
                )
            except StaleTrajectoryContractError as exc:
                stats["invalid"] += 1
                if "shape mismatch" in str(exc):
                    stats["shape_mismatch"] += 1
                raise StaleTrajectoryContractError(
                    f"invalid stale trajectory for replay prompt {prompt_key!r}, "
                    f"row={position}: {exc}; {_stats_summary(stats)}"
                ) from exc
            behavior_anchors[position] = behavior_anchor
            stats["replay_hit"] += 1
            stats["exact_input_ids_match"] += int(exact_matches["input_ids"])
            stats["exact_attention_mask_match"] += int(
                exact_matches["attention_mask"]
            )
            stats["exact_position_ids_match"] += int(
                exact_matches["position_ids"]
            )
            stats["exact_response_match"] += int(exact_matches["responses"])
            stats["exact_behavior_logprob_match"] += int(
                exact_matches["rollout_log_probs"]
            )

    if stats["replay_hit"] != stats["replay_rows"]:
        raise StaleTrajectoryContractError(
            "not every replay row received a stale trajectory; "
            f"{_stats_summary(stats)}"
        )
    if stats["replay_rows"]:
        if PENDING_BEHAVIOR_ANCHORS_KEY in batch.meta_info:
            raise StaleTrajectoryContractError(
                "stale behavior anchors already pending for this batch"
            )
        # Remove this rollout-only metadata before the actor RPC.
        batch.meta_info[PENDING_BEHAVIOR_ANCHORS_KEY] = behavior_anchors
        # Recompute masks and token counts from the injected sequence tensors.
        batch.batch["response_mask"] = compute_response_mask(batch)
        batch.meta_info["global_token_num"] = torch.sum(
            batch.batch["attention_mask"], dim=-1
        ).tolist()
    return stats


def hydrate_offpolicy_behavior_log_probs(
    batch: DataProto,
    pre_update_log_probs: torch.Tensor,
    stale_behavior_anchors: Mapping[int, torch.Tensor] | None = None,
) -> dict[str, int]:
    """Create the complete behavior-policy denominator for rollout IS.

    Fresh rows use the exact log-probabilities recomputed by the frozen
    pre-update actor.  Replayed rows replace those values atomically with their
    historical behavior anchors. Hydration runs after sequence injection and
    raises ``StaleTrajectoryContractError`` for a missing or malformed anchor.
    """
    if not isinstance(pre_update_log_probs, torch.Tensor):
        raise StaleTrajectoryContractError("pre-update old_log_probs is not a tensor")
    responses = batch.batch.get("responses")
    if not isinstance(responses, torch.Tensor):
        raise StaleTrajectoryContractError("live batch responses is not a tensor")
    if pre_update_log_probs.shape != responses.shape:
        raise StaleTrajectoryContractError(
            "pre-update old_log_probs shape does not match responses: "
            f"old={tuple(pre_update_log_probs.shape)} "
            f"responses={tuple(responses.shape)}"
        )
    if not torch.is_floating_point(pre_update_log_probs):
        raise StaleTrajectoryContractError("pre-update old_log_probs is not floating point")

    anchors = {} if stale_behavior_anchors is None else dict(stale_behavior_anchors)
    behavior = pre_update_log_probs.detach().clone()
    exact = 0
    for position, stored in anchors.items():
        if not isinstance(position, int) or position < 0 or position >= len(responses):
            raise StaleTrajectoryContractError(f"invalid stale behavior anchor position {position!r}")
        if not isinstance(stored, torch.Tensor):
            raise StaleTrajectoryContractError(
                f"stale behavior anchor at row {position} is not a tensor"
            )
        if not torch.is_floating_point(stored):
            raise StaleTrajectoryContractError(
                f"stale behavior anchor at row {position} is not floating point"
            )
        if stored.shape != behavior[position].shape:
            raise StaleTrajectoryContractError(
                "shape mismatch for stale behavior anchor at row "
                f"{position}: stored={tuple(stored.shape)} "
                f"old={tuple(behavior[position].shape)}"
            )
        staged = stored.detach().to(device=behavior.device, dtype=behavior.dtype)
        behavior[position].copy_(staged)
        if not torch.equal(behavior[position], staged):
            raise StaleTrajectoryContractError(
                f"stale behavior anchor copy failed at row {position}"
            )
        exact += 1
    batch.batch[OFF_POLICY_BEHAVIOR_LOGPROB_KEY] = behavior
    return {
        "hydrated_behavior_logprob_match": exact,
        "fresh_behavior_logprob_rows": len(responses) - len(anchors),
    }


class OffPolicyReplayTrainer(RayPPOTrainer):
    """RayPPOTrainer for the stale-trajectory CPR ablation.

    The common rollout pipeline first generates replay-labelled rows. Before
    scoring, each row is replaced atomically with its stored historical
    trajectory. The verifier reward and
    GRPO advantage are then recomputed under the current run, while
    ``rollout_log_probs`` remain the old behavior policy's values for explicit
    importance-sampling correction. Current-task/base rows remain untouched.
    """

    def _offpolicy_sampler(self) -> PromptReplaySampler | None:
        dataloader = getattr(self, "train_dataloader", None)
        sampler = getattr(dataloader, "sampler", None)
        config = getattr(sampler, "config", None)
        if config is not None and getattr(config, "off_policy", False):
            return sampler
        return None

    def _compute_old_log_prob(self, batch: DataProto):
        # Pop staged behavior anchors before the actor RPC.
        anchors = batch.meta_info.pop(PENDING_BEHAVIOR_ANCHORS_KEY, {})
        if not isinstance(anchors, Mapping):
            raise StaleTrajectoryContractError("pending stale behavior anchors are malformed")
        old_log_prob, old_log_prob_mfu = super()._compute_old_log_prob(batch)
        if "old_log_probs" not in old_log_prob.batch:
            raise StaleTrajectoryContractError(
                "pre-update actor log-probability output lacks old_log_probs"
            )
        hydration = hydrate_offpolicy_behavior_log_probs(
            batch,
            old_log_prob.batch["old_log_probs"],
            anchors,
        )
        replay_stats = batch.meta_info.get("offpolicy_replay_stats")
        if isinstance(replay_stats, dict) and replay_stats.get("replay_rows", 0):
            expected = int(replay_stats["replay_rows"])
            if hydration["hydrated_behavior_logprob_match"] != expected:
                raise StaleTrajectoryContractError(
                    "not every replay row received its historical behavior "
                    "log-probability anchor"
                )
            replay_stats.update(hydration)
            print(
                "[CPR-offpolicy] behavior-logprob hydration: "
                f"replay_rows={expected} "
                f"hydrated_behavior_logprob_match="
                f"{hydration['hydrated_behavior_logprob_match']} "
                f"fresh_behavior_logprob_rows="
                f"{hydration['fresh_behavior_logprob_rows']}",
                flush=True,
            )
        return old_log_prob, old_log_prob_mfu

    def _compute_or_extract_reward(self, batch, *args, **kwargs):
        sampler = self._offpolicy_sampler()
        if sampler is not None and "responses" in batch.batch:
            try:
                stats = inject_offpolicy_trajectories(batch, sampler)
            except StaleTrajectoryContractError as exc:
                print(f"[CPR-offpolicy] contract failure: {exc}", flush=True)
                raise
            if stats["replay_rows"]:
                # The injector has already refreshed response_mask and
                # global_token_num from the stale sequence before reward/IS.
                batch.meta_info["offpolicy_replay_stats"] = stats
                print(
                    "[CPR-offpolicy] stale-trajectory replay hit: "
                    f"{_stats_summary(stats)}",
                    flush=True,
                )
        return super()._compute_or_extract_reward(batch, *args, **kwargs)
