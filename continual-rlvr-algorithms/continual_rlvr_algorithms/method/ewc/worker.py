from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.core_algos import (
    agg_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_corr_metrics_from_logprobs,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from continual_rlvr_algorithms.method.ewc.ewc import (
    EWCConfig,
    OnlineEWCState,
)

EWC_STATE_FILENAME = "ewc_state.pt"


@dataclass(frozen=True)
class EWCRuntimeSummary:
    consolidated: bool
    task_name: str | None
    fisher_samples: int
    num_params: int


def save_runtime_state(path: str, actor) -> None:
    torch.save(
        {
            "current_task_name": actor.current_task_name,
            "ewc_state": actor.ewc_state.state_dict(),
        },
        path,
    )


def load_runtime_state(path: str, actor) -> None:
    payload = torch.load(path, map_location="cpu")
    actor.current_task_name = payload.get("current_task_name")
    actor.ewc_state.load_state_dict(
        payload.get("ewc_state", {}),
        module=actor.actor_module,
    )


class EWCDataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        *,
        config,
        actor_module,
        actor_optimizer,
        ewc_config: EWCConfig,
    ):
        super().__init__(
            config=config,
            actor_module=actor_module,
            actor_optimizer=actor_optimizer,
        )
        self.ewc_config = ewc_config
        self.ewc_state = OnlineEWCState(ewc_config)
        self.current_task_name: str | None = None

    def set_current_task(self, task_name: str) -> None:
        self.current_task_name = str(task_name)

    def consolidate_current_task(
        self,
        task_name: str | None,
    ) -> dict[str, object]:
        summary = self.ewc_state.consolidate_current_task(
            self.actor_module,
            task_name=task_name,
        )
        if task_name is not None:
            self.current_task_name = str(task_name)
        return summary

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

    def build_model_inputs(self, micro_batch, *, pad_token_id: int):
        micro_batch_data = micro_batch.to(get_device_id())
        return {
            **micro_batch_data.batch,
            **micro_batch_data.non_tensor_batch,
            "pad_token_id": pad_token_id,
        }

    def compute_policy_step(
        self,
        model_inputs,
        *,
        temperature,
        on_policy: bool,
    ) -> dict[str, object]:
        micro_batch_metrics = {}
        response_mask = model_inputs["response_mask"]
        advantages = model_inputs["advantages"]
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        calculate_entropy = self.config.calculate_entropy or entropy_coeff != 0
        if self.config.use_dynamic_bsz:
            loss_scale_factor = (
                response_mask.shape[0] / self.config.ppo_mini_batch_size
            )
        else:
            loss_scale_factor = 1 / self.gradient_accumulation

        outputs = self._forward_micro_batch(
            model_inputs,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
        )
        log_prob = outputs["log_probs"]
        entropy = outputs["entropys"] if calculate_entropy else None

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
            "loss_mode",
            "vanilla",
        )
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.config,
            rollout_is_weights=model_inputs.get("rollout_is_weights"),
        )
        micro_batch_metrics.update(pg_metrics)

        rollout_log_prob = model_inputs.get("rollout_log_probs")
        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
            micro_batch_metrics.update(
                compute_rollout_corr_metrics_from_logprobs(
                    log_prob=log_prob,
                    rollout_log_prob=rollout_log_prob,
                    response_mask=response_mask,
                )
            )

        policy_loss = pg_loss
        if calculate_entropy and entropy is not None:
            entropy_agg = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
            )
            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
            if entropy_coeff != 0:
                policy_loss -= entropy_agg * entropy_coeff

        kl_loss_value = 0.0
        if self.config.use_kl_loss:
            ref_log_prob = model_inputs["ref_log_prob"]
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=ref_log_prob,
                kl_penalty=self.config.kl_loss_type,
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
            )
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            kl_loss_value = kl_loss.detach().item() * loss_scale_factor
            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

        return {
            "policy_loss": policy_loss,
            "pg_loss": pg_loss,
            "kl_loss_value": kl_loss_value,
            "loss_scale_factor": loss_scale_factor,
            "micro_batch_metrics": micro_batch_metrics,
        }

    def estimate_ewc_penalty(self, policy_loss):
        named_params = self.ewc_state.iter_trainable_named_params(
            self.actor_module
        )
        policy_grads = torch.autograd.grad(
            policy_loss,
            [param for _, param in named_params],
            retain_graph=True,
            allow_unused=True,
        )
        self.ewc_state.accumulate_fisher_from_grads(
            named_params=named_params,
            grads=policy_grads,
            task_name=self.current_task_name,
        )
        return self.ewc_state.estimate_penalty(self.actor_module)

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
            "actor/ewc_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                micro_batches = self.build_micro_batches(mini_batch)
                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    model_inputs = self.build_model_inputs(
                        micro_batch,
                        pad_token_id=pad_token_id,
                    )
                    policy_step = self.compute_policy_step(
                        model_inputs,
                        temperature=temperature,
                        on_policy=on_policy,
                    )
                    micro_batch_metrics = policy_step["micro_batch_metrics"]
                    loss_scale_factor = policy_step["loss_scale_factor"]
                    metrics["actor/kl_loss"] += policy_step["kl_loss_value"]

                    ewc_penalty = self.estimate_ewc_penalty(
                        policy_step["policy_loss"]
                    )
                    micro_batch_metrics["actor/ewc_penalty"] = (
                        ewc_penalty.detach().item()
                    )
                    loss = (
                        policy_step["policy_loss"]
                        + self.ewc_config.penalty_coef * ewc_penalty
                    ) * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += (
                        policy_step["pg_loss"].detach().item()
                        * loss_scale_factor
                    )
                    metrics["actor/ewc_loss"] += (
                        ewc_penalty.detach().item() * loss_scale_factor
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {"actor/grad_norm": grad_norm.detach().item()},
                )
        self.actor_optimizer.zero_grad()
        return metrics


class EWCActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return
        ewc_config = EWCConfig.from_mapping(self.config.get("ewc", {}))
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = EWCDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
            ewc_config=ewc_config,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_ewc_task(self, task_name: str) -> None:
        if self._is_actor:
            self.actor.set_current_task(task_name)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def consolidate_ewc_task(
        self,
        previous_task: str,
        current_task: str,
    ) -> dict[str, object] | None:
        if not self._is_actor:
            return None
        summary = self.actor.consolidate_current_task(previous_task)
        self.actor.set_current_task(current_task)
        if getattr(self, "rank", 0) == 0:
            print(
                "[EWC] consolidated task "
                f"{previous_task} -> {current_task}: {summary}"
            )
        return summary

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        super().save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        if not self._is_actor or getattr(self, "rank", 0) != 0:
            return
        save_runtime_state(
            os.path.join(local_path, EWC_STATE_FILENAME),
            self.actor,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self,
        local_path,
        hdfs_path=None,
        del_local_after_load=False,  # noqa: FBT002
    ):
        super().load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if not self._is_actor:
            return
        if local_path is None:
            return
        state_path = os.path.join(local_path, EWC_STATE_FILENAME)
        if not os.path.exists(state_path):
            return
        load_runtime_state(state_path, self.actor)
