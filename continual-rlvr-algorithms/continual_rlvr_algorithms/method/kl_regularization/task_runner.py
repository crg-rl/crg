from __future__ import annotations

import os

import ray
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role

from continual_rlvr_algorithms.method.kl_regularization.kl_regularization import (
    KL_MODE_TO_OLD_POLICY_OLD_PROMPTS,
    KLOldPromptDataset,
    KLRegularizationConfig,
    add_kl_regularization_mask,
    zero_old_prompt_advantages,
    zero_old_prompt_rewards,
)
from continual_rlvr_algorithms.method.kl_regularization.worker import (
    KLRegularizationActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.task_runner import (
    MethodHookReasoningGymRunner,
)
from continual_rlvr_algorithms.method.task_switch import (
    install_task_switch_hook,
)


class KLRegularizationRayPPOTrainer(RayPPOTrainer):
    def kl_regularization_config(self) -> KLRegularizationConfig:
        return KLRegularizationConfig.from_config(self.config)

    def _compute_or_extract_reward(self, batch, *args, **kwargs):
        result = super()._compute_or_extract_reward(batch, *args, **kwargs)
        config = self.kl_regularization_config()
        if not config.is_old_policy_mode() or bool(
            kwargs.get("reward_for_val", False)
        ):
            return result

        if isinstance(result, tuple):
            reward_tensor, reward_extra_infos_dict = result
            reward_tensor = zero_old_prompt_rewards(batch, reward_tensor)
            return reward_tensor, reward_extra_infos_dict
        return zero_old_prompt_rewards(batch, result)

    def _update_actor(self, batch):
        config = self.kl_regularization_config()
        if config.enabled:
            add_kl_regularization_mask(batch, mode=config.mode)
        if config.is_old_policy_mode():
            zero_old_prompt_advantages(batch)
        return super()._update_actor(batch)


class KLRegularizationReasoningGymRunner(MethodHookReasoningGymRunner):
    trainer_cls = KLRegularizationRayPPOTrainer

    def before_setup(self, config) -> None:
        kl_config = KLRegularizationConfig.from_config(config)
        if not kl_config.enabled:
            return
        if not bool(config.actor_rollout_ref.actor.use_kl_loss):
            raise ValueError(
                "KL regularization baselines require "
                "actor_rollout_ref.actor.use_kl_loss=true."
            )

    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get("worker_impl_mode", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        actor_rollout_cls = KLRegularizationActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def prepare_runner_datasets(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        del tokenizer
        kl_config = KLRegularizationConfig.from_config(config)
        if kl_config.is_old_policy_mode():
            train_dataset = KLOldPromptDataset(train_dataset)
        return train_dataset, val_dataset

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del val_dataset
        kl_config = KLRegularizationConfig.from_config(config)
        if not kl_config.is_old_policy_mode():
            return

        if kl_config.old_policy_checkpoint_path:
            self.load_old_policy_reference(
                trainer,
                kl_config.old_policy_checkpoint_path,
            )

        if not kl_config.update_old_policy_on_task_switch:
            return

        def on_task_switch(previous_task, current_task) -> None:
            del previous_task, current_task
            actor_path = self.resolve_task_boundary_actor_checkpoint(
                config,
                train_dataset,
            )
            if not os.path.isdir(actor_path):
                message = (
                    "KL-to-old-policy requires a task-boundary actor "
                    f"checkpoint, but it was not found: {actor_path}. "
                    "Set trainer.save_freq=crl.steps_per_task or provide "
                    "++kl_regularization.old_policy_checkpoint_path."
                )
                if kl_config.require_task_boundary_checkpoint:
                    raise FileNotFoundError(message)
                print(f"[KL] skip old-policy refresh: {message}")
                return
            self.load_old_policy_reference(trainer, actor_path)

        installed = install_task_switch_hook(
            train_dataset,
            on_task_switch=on_task_switch,
            hook_attr="_kl_regularization_hook_installed",
        )
        if not installed:
            raise RuntimeError(
                "KL-to-old-policy task-switch hook was not installed. The "
                "old-policy baseline requires a CRL dataset with "
                "current_task/on_batch_end lifecycle."
            )

    @staticmethod
    def resolve_task_boundary_actor_checkpoint(config, train_dataset) -> str:
        train_steps = KLRegularizationReasoningGymRunner.resolve_train_steps(
            train_dataset
        )
        if train_steps is None:
            raise RuntimeError(
                "Cannot resolve CRL train_steps for old-policy checkpoint."
            )

        base_dir = str(config.trainer.default_local_dir)
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(os.getcwd(), base_dir)
        return os.path.join(
            base_dir, f"global_step_{int(train_steps)}", "actor"
        )

    @staticmethod
    def resolve_train_steps(train_dataset) -> int | None:
        for dataset in (
            train_dataset,
            getattr(train_dataset, "base_dataset", None),
        ):
            if dataset is None:
                continue
            train_steps = getattr(dataset, "train_steps", None)
            if train_steps is not None:
                return int(train_steps)
            shared_state_getter = getattr(dataset, "get_shared_state", None)
            if callable(shared_state_getter):
                shared_state = shared_state_getter()
                if shared_state is not None and "train_steps" in shared_state:
                    return int(shared_state["train_steps"])
        return None

    @staticmethod
    def load_old_policy_reference(trainer, actor_path: str) -> None:
        ref_policy_wg = getattr(trainer, "ref_policy_wg", None)
        if ref_policy_wg is None:
            raise RuntimeError(
                "KL-to-old-policy requires a separate reference-policy worker. "
                "Ensure actor_rollout_ref.actor.use_kl_loss=true and avoid "
                "LoRA ref-in-actor mode for this baseline."
            )
        ref_policy_wg.load_ref_policy_from_actor_checkpoint(str(actor_path))
        print(f"[KL] old-policy reference path: {actor_path}")
