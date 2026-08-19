from __future__ import annotations

import os

import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import (
    agg_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_corr_metrics_from_logprobs,
)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


def config_get(payload, key, default=None):
    if payload is None:
        return default
    if hasattr(payload, "get"):
        return payload.get(key, default)
    return getattr(payload, key, default)


def masked_stats(tensor, mask):
    values = tensor.detach()[mask.detach().bool()].float().cpu()
    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "max": values.max().item(),
    }


def prefix_metrics(prefix, values):
    return {f"{prefix}/{key}": value for key, value in values.items()}


def scalar_metric(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return value.item()
    return value


def sanitize_actor_metrics(metrics):
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            sanitized[key] = [scalar_metric(item) for item in value]
        else:
            sanitized[key] = scalar_metric(value)
    return sanitized


class KLRegularizedDataParallelPPOActor(DataParallelPPOActor):
    def debug_config(self):
        return config_get(self.config, "kl_regularization_debug", {}) or {}

    def should_log_debug(self, global_step):
        debug_config = self.debug_config()
        if not bool(config_get(debug_config, "enabled", False)):
            return False
        max_steps = int(config_get(debug_config, "max_steps", 1))
        return int(global_step or 0) <= max_steps

    def select_policy_data(self, data):
        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch:
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
            if "kl_regularization_mask" in data.batch:
                select_keys.append("kl_regularization_mask")
            if "policy_training_mask" in data.batch:
                select_keys.append("policy_training_mask")
        if "rollout_is_weights" in data.batch:
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch:
            select_keys.append("rollout_log_probs")

        non_tensor_select_keys = []
        if "multi_modal_inputs" in data.non_tensor_batch:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch:
            non_tensor_select_keys.append("uid")

        return data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

    def build_micro_batches(self, mini_batch):
        if self.config.use_dynamic_bsz:
            max_token_len = (
                self.config.ppo_max_token_len_per_gpu
                * self.ulysses_sequence_parallel_size
            )
            micro_batches, _ = prepare_dynamic_batch(
                mini_batch,
                max_token_len=max_token_len,
            )
            return micro_batches

        self.gradient_accumulation = (
            self.config.ppo_mini_batch_size
            // self.config.ppo_micro_batch_size_per_gpu
        )
        return mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

    def compute_kl_loss(
        self, log_prob, ref_log_prob, response_mask, model_inputs
    ):
        kl_loss_mask = model_inputs.get(
            "kl_regularization_mask", response_mask
        )
        if not bool(torch.sum(kl_loss_mask).detach().item() > 0):
            return log_prob.sum() * 0.0, 0.0, 0.0

        kld = kl_penalty(
            logprob=log_prob,
            ref_logprob=ref_log_prob,
            kl_penalty=self.config.kl_loss_type,
        )
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=kl_loss_mask,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        active_fraction = (
            torch.sum(kl_loss_mask).detach().float()
            / torch.sum(response_mask).detach().float().clamp(min=1.0)
        ).item()
        return kl_loss, kl_loss.detach().item(), active_fraction

    def update_policy(self, data):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        data = self.select_policy_data(data)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        global_step = data.meta_info.get("global_steps")
        debug_enabled = self.should_log_debug(global_step)
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                micro_batches = self.build_micro_batches(mini_batch)
                self.actor_optimizer.zero_grad()
                for micro_batch_index, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {
                        **micro_batch.batch,
                        **micro_batch.non_tensor_batch,
                        "pad_token_id": pad_token_id,
                    }
                    micro_batch_metrics = {}
                    response_mask = model_inputs["response_mask"]
                    policy_mask = model_inputs.get(
                        "policy_training_mask",
                        response_mask,
                    )
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = (
                        self.config.calculate_entropy or entropy_coeff != 0
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = (
                            response_mask.shape[0]
                            / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    policy_loss_scale_factor = (
                        loss_scale_factor
                        if bool(torch.sum(policy_mask).detach().item() > 0)
                        else 0.0
                    )

                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = (
                        outputs["entropys"] if calculate_entropy else None
                    )

                    use_rollout_log_probs = (
                        hasattr(self.config, "use_rollout_log_probs")
                        and self.config.use_rollout_log_probs
                    )
                    if use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get(
                        "loss_mode", "vanilla"
                    )
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    has_policy_tokens = bool(
                        torch.sum(policy_mask).detach().item() > 0
                    )
                    if has_policy_tokens:
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=policy_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=model_inputs.get(
                                "rollout_is_weights"
                            ),
                        )
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        pg_loss = log_prob.sum() * 0.0

                    if debug_enabled and micro_batch_index == 0:
                        old_log_prob_stats = masked_stats(
                            old_log_prob,
                            response_mask,
                        )
                        actor_log_prob_stats = masked_stats(
                            log_prob,
                            response_mask,
                        )
                        ref_log_prob = model_inputs.get("ref_log_prob")
                        if ref_log_prob is not None:
                            ref_log_prob_stats = masked_stats(
                                ref_log_prob,
                                response_mask,
                            )
                            log_prob_delta_stats = masked_stats(
                                ref_log_prob - log_prob,
                                response_mask,
                            )
                            kld = kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_token_stats = masked_stats(kld, response_mask)
                        else:
                            ref_log_prob_stats = {}
                            log_prob_delta_stats = {}
                            kl_token_stats = {}
                        debug_metrics = {
                            "debug/advantages_absmax": (
                                advantages.detach().abs().max().item()
                            ),
                            "debug/advantages_nonzero_rate": (
                                advantages.detach().ne(0).float().mean().item()
                            ),
                            "debug/response_mask_tokens": (
                                response_mask.detach().float().sum().item()
                            ),
                            "debug/policy_mask_tokens": (
                                policy_mask.detach().float().sum().item()
                            ),
                            "debug/raw_pg_loss": pg_loss.detach().item(),
                        }
                        debug_metrics.update(
                            prefix_metrics(
                                "debug/old_log_prob",
                                old_log_prob_stats,
                            )
                        )
                        debug_metrics.update(
                            prefix_metrics(
                                "debug/actor_log_prob",
                                actor_log_prob_stats,
                            )
                        )
                        debug_metrics.update(
                            prefix_metrics(
                                "debug/ref_log_prob",
                                ref_log_prob_stats,
                            )
                        )
                        debug_metrics.update(
                            prefix_metrics(
                                "debug/ref_minus_actor_log_prob",
                                log_prob_delta_stats,
                            )
                        )
                        debug_metrics.update(
                            prefix_metrics("debug/low_var_kl", kl_token_stats)
                        )
                        micro_batch_metrics.update(debug_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs")
                    if (
                        has_policy_tokens
                        and loss_mode != "bypass_mode"
                        and rollout_log_prob is not None
                    ):
                        micro_batch_metrics.update(
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=policy_mask,
                            )
                        )

                    policy_loss = pg_loss
                    if (
                        calculate_entropy
                        and entropy is not None
                        and has_policy_tokens
                    ):
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=policy_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        micro_batch_metrics["actor/entropy"] = (
                            entropy_agg.detach().item()
                        )
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff
                        if debug_enabled and micro_batch_index == 0:
                            micro_batch_metrics["debug/entropy_agg"] = (
                                entropy_agg.detach().item()
                            )
                            micro_batch_metrics["debug/entropy_term"] = (
                                (-entropy_agg * entropy_coeff).detach().item()
                            )

                    if self.config.use_kl_loss:
                        kl_loss, kl_loss_value, active_fraction = (
                            self.compute_kl_loss(
                                log_prob,
                                model_inputs["ref_log_prob"],
                                response_mask,
                                model_inputs,
                            )
                        )
                        policy_loss = (
                            policy_loss + kl_loss * self.config.kl_loss_coef
                        )
                        metrics["actor/kl_loss"] += (
                            kl_loss_value * loss_scale_factor
                        )
                        micro_batch_metrics["actor/kl_coef"] = (
                            self.config.kl_loss_coef
                        )
                        micro_batch_metrics[
                            "actor/kl_regularization_active_fraction"
                        ] = active_fraction
                        if debug_enabled and micro_batch_index == 0:
                            kl_term = kl_loss * self.config.kl_loss_coef
                            kl_mask = model_inputs.get(
                                "kl_regularization_mask",
                                response_mask,
                            )
                            micro_batch_metrics["debug/kl_mask_tokens"] = (
                                kl_mask.detach().float().sum().item()
                            )
                            micro_batch_metrics["debug/kl_loss"] = (
                                kl_loss.detach().item()
                            )
                            micro_batch_metrics["debug/kl_term"] = (
                                kl_term.detach().item()
                            )
                            micro_batch_metrics["debug/policy_loss"] = (
                                policy_loss.detach().item()
                            )

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += (
                        pg_loss.detach().item() * policy_loss_scale_factor
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {"actor/grad_norm": grad_norm.detach().item()},
                )
        self.actor_optimizer.zero_grad()
        return sanitize_actor_metrics(metrics)


class KLRegularizationActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = KLRegularizedDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_ref_policy_from_actor_checkpoint(
        self,
        local_path: str,
    ) -> dict[str, object] | None:
        if not self._is_ref:
            return None
        if local_path is None or not str(local_path):
            raise ValueError(
                "local_path is required to refresh old-policy reference"
            )
        local_path = str(local_path)
        if not os.path.isdir(local_path):
            raise FileNotFoundError(
                f"old-policy actor checkpoint directory does not exist: {local_path}"
            )
        checkpoint_config = OmegaConf.create(
            {"load_contents": ["model"], "save_contents": []}
        )
        checkpoint_manager = FSDPCheckpointManager(
            model=self.ref_module_fsdp,
            optimizer=None,
            lr_scheduler=None,
            processing_class=(
                self.processor
                if self.processor is not None
                else self.tokenizer
            ),
            checkpoint_config=checkpoint_config,
        )
        checkpoint_manager.load_checkpoint(local_path)
        summary = {
            "loaded": True,
            "path": local_path,
            "rank": int(getattr(self, "rank", 0)),
        }
        if getattr(self, "rank", 0) == 0:
            print(f"[KL] refreshed old-policy reference from {local_path}")
        return summary
