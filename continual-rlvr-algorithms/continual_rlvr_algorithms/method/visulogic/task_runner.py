from __future__ import annotations

import os

import ray
from omegaconf import open_dict
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role

from continual_rlvr_algorithms.method.ewc.worker import (
    EWCActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.fire.worker import (
    FIREActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.kl_regularization.kl_regularization import (
    KLOldPromptDataset,
    KLRegularizationConfig,
)
from continual_rlvr_algorithms.method.kl_regularization.task_runner import (
    KLRegularizationRayPPOTrainer,
)
from continual_rlvr_algorithms.method.kl_regularization.worker import (
    KLRegularizationActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.prompt_replay.prompt_replay import (
    PromptReplayDataset,
)
from continual_rlvr_algorithms.method.redo.worker import (
    REDOActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.task_switch import (
    install_task_switch_hook,
    resolve_initial_task_name,
)
from continual_rlvr_algorithms.method.verl_task_runner import (
    MethodHookTaskRunner,
)


class VisuLogicEWCRunner(MethodHookTaskRunner):
    """VisuLogic CRL runner with online EWC task consolidation."""

    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get("worker_impl_mode", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        actor_rollout_cls = EWCActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del config, val_dataset
        initial_task_name = resolve_initial_task_name(train_dataset, "task_0")
        trainer.actor_rollout_wg.set_ewc_task(initial_task_name)

        def on_task_switch(previous_task, current_task) -> None:
            trainer.actor_rollout_wg.consolidate_ewc_task(
                previous_task,
                current_task,
            )

        install_task_switch_hook(
            train_dataset,
            on_task_switch=on_task_switch,
            hook_attr="_ewc_hook_installed",
        )


class VisuLogicREDORunner(MethodHookTaskRunner):
    """VisuLogic CRL runner with ReDo task-switch resets."""

    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get("worker_impl_mode", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        actor_rollout_cls = REDOActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del config, val_dataset

        def apply_redo_task_switch(previous_task, current_task, batch) -> None:
            trainer.actor_rollout_wg.apply_redo_task_switch(
                previous_task,
                current_task,
                batch,
            )

        install_task_switch_hook(
            train_dataset,
            on_task_switch=apply_redo_task_switch,
            hook_attr="_redo_hook_installed",
            pass_batch=True,
        )


class VisuLogicFIRERunner(MethodHookTaskRunner):
    """VisuLogic CRL runner with FIRE task-switch resets."""

    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get("worker_impl_mode", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        self.attach_fire_config_to_actor_rollout(config)
        actor_rollout_cls = FIREActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del val_dataset

        def handle_task_switch(previous_task, current_task) -> None:
            rollout_was_slept = self.sleep_fire_rollout_before_reset(
                trainer,
                config,
            )
            summaries = trainer.actor_rollout_wg.apply_fire_task_switch(
                previous_task,
                current_task,
            )
            if not self.fire_should_sync_rollout(trainer, summaries):
                if rollout_was_slept:
                    self.sync_fire_weights_to_rollout(
                        trainer,
                        sleep_first=False,
                    )
                return
            self.sync_fire_weights_to_rollout(trainer, sleep_first=False)

        install_task_switch_hook(
            train_dataset,
            on_task_switch=handle_task_switch,
            hook_attr="_fire_hook_installed",
        )

    @staticmethod
    def attach_fire_config_to_actor_rollout(config) -> None:
        fire_config = config.get("fire", None)
        if fire_config is None:
            return

        actor_rollout_config = config.actor_rollout_ref
        if actor_rollout_config.get("fire", None) is not None:
            return

        with open_dict(actor_rollout_config):
            actor_rollout_config.fire = fire_config

    @staticmethod
    def fire_should_sync_rollout(trainer, summaries) -> bool:
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        update_weights = getattr(checkpoint_manager, "update_weights", None)
        if not callable(update_weights):
            return False

        def summary_applied(summary) -> bool:
            return (
                isinstance(summary, dict)
                and bool(summary.get("applied"))
                and bool(summary.get("sync_rollout_after_reset", True))
            )

        if isinstance(summaries, list | tuple):
            return any(summary_applied(summary) for summary in summaries)
        return summary_applied(summaries)

    @staticmethod
    def fire_config_sync_enabled(config) -> bool:
        fire_config = config.get("fire", None)
        if fire_config is None:
            return False

        value = fire_config.get("sync_rollout_after_reset", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def sleep_fire_rollout_before_reset(trainer, config) -> bool:
        if not VisuLogicFIRERunner.fire_config_sync_enabled(config):
            return False

        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if not callable(getattr(checkpoint_manager, "update_weights", None)):
            return False

        sleep_replicas = getattr(checkpoint_manager, "sleep_replicas", None)
        if not callable(sleep_replicas):
            return False
        sleep_replicas()
        return True

    @staticmethod
    def sync_fire_weights_to_rollout(
        trainer,
        *,
        sleep_first: bool = True,
    ) -> None:
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if not callable(getattr(checkpoint_manager, "update_weights", None)):
            return

        if sleep_first:
            sleep_replicas = getattr(
                checkpoint_manager, "sleep_replicas", None
            )
            if callable(sleep_replicas):
                sleep_replicas()
        checkpoint_manager.update_weights()


class VisuLogicKLRegularizationRunner(MethodHookTaskRunner):
    """VisuLogic CRL runner for KL-to-old-policy old-prompts baseline."""

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
                    "Set trainer.save_freq to the VLM CRL steps_per_task or "
                    "provide ++kl_regularization.old_policy_checkpoint_path."
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
                "old-policy baseline requires a VLM CRL dataset with "
                "current_task/on_batch_end lifecycle."
            )

    @staticmethod
    def resolve_task_boundary_actor_checkpoint(config, train_dataset) -> str:
        train_steps = VisuLogicKLRegularizationRunner.resolve_train_steps(
            train_dataset
        )
        if train_steps is None:
            raise RuntimeError(
                "Cannot resolve VLM CRL train_steps for old-policy checkpoint."
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
                "Ensure actor_rollout_ref.actor.use_kl_loss=true."
            )
        ref_policy_wg.load_ref_policy_from_actor_checkpoint(str(actor_path))
        print(f"[KL] old-policy reference path: {actor_path}")


class VisuLogicMuonRunner(MethodHookTaskRunner):
    """VisuLogic CRL runner using Muon through config-native optimizer hooks."""


class VisuLogicOSFTRunner(MethodHookTaskRunner):
    """VisuLogic CRL runner that keeps OSFT on the multimodal-native VLM path."""

    def add_actor_rollout_worker(self, config):
        load_osft_config, actor_rollout_cls = self._load_osft_dependencies()
        osft_config = load_osft_config(config.actor_rollout_ref)
        worker_impl_mode = config.trainer.get("osft_worker_impl", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        if osft_config.enabled:
            print("[OSFT] VLM actor-side wrapper is enabled for actor worker")
        else:
            print("[OSFT] VLM wrapper worker is in pass-through mode")

        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            actor_rollout_cls
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del val_dataset
        load_osft_config, _ = self._load_osft_dependencies()
        osft_config = load_osft_config(config.actor_rollout_ref)
        if not osft_config.enabled or not osft_config.reinit_on_task_switch:
            return

        def handle_task_switch(previous_task, current_task) -> None:
            summaries = (
                trainer.actor_rollout_wg.apply_osft_task_switch_reprotect(
                    previous_task,
                    current_task,
                )
            )
            if self._osft_should_sync_rollout(trainer, summaries, osft_config):
                self._sync_osft_weights_to_rollout(trainer)

        installed = install_task_switch_hook(
            train_dataset,
            on_task_switch=handle_task_switch,
            hook_attr="_osft_hook_installed",
        )
        if not installed:
            raise RuntimeError(
                "OSFT task-switch re-SVD/re-protect hook was not installed. "
                "VLM OSFT requires a CRL dataset with current_task/on_batch_end "
                "lifecycle."
            )

    @staticmethod
    def _osft_should_sync_rollout(trainer, summaries, osft_config) -> bool:
        if not osft_config.sync_rollout_on_reinit:
            return False
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        update_weights = getattr(checkpoint_manager, "update_weights", None)
        if not callable(update_weights):
            return False

        def summary_applied(summary) -> bool:
            return isinstance(summary, dict) and bool(summary.get("applied"))

        if isinstance(summaries, list | tuple):
            return any(summary_applied(summary) for summary in summaries)
        return summary_applied(summaries)

    @staticmethod
    def _sync_osft_weights_to_rollout(trainer) -> None:
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if not callable(getattr(checkpoint_manager, "update_weights", None)):
            return
        sleep_replicas = getattr(checkpoint_manager, "sleep_replicas", None)
        if callable(sleep_replicas):
            sleep_replicas()
        checkpoint_manager.update_weights()

    @staticmethod
    def _load_osft_dependencies():
        from continual_rlvr_algorithms.method.osft.osft_config import (
            load_osft_config,
        )
        from continual_rlvr_algorithms.method.osft.workers import (
            OSFTAsyncActorRolloutRefWorker,
        )

        return load_osft_config, OSFTAsyncActorRolloutRefWorker


class VisuLogicPromptReplayRunner(MethodHookTaskRunner):
    """VisuLogic CRL runner with prompt replay sampler/dataset state."""

    def prepare_runner_datasets(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        del config, tokenizer
        return PromptReplayDataset(train_dataset), val_dataset
