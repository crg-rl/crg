from __future__ import annotations

import ray
from omegaconf import open_dict
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role

from continual_rlvr_algorithms.method.fire.worker import (
    FIREActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.task_runner import (
    MethodHookReasoningGymRunner,
)
from continual_rlvr_algorithms.method.task_switch import (
    install_task_switch_hook,
)


class FIREResettingReasoningGymRunner(MethodHookReasoningGymRunner):
    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get(
            "worker_impl_mode",
            "auto",
        )
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        self._attach_fire_config_to_actor_rollout(config)
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
            rollout_was_slept = self._sleep_fire_rollout_before_reset(
                trainer,
                config,
            )
            summaries = trainer.actor_rollout_wg.apply_fire_task_switch(
                previous_task,
                current_task,
            )
            if not self._fire_should_sync_rollout(trainer, summaries):
                if rollout_was_slept:
                    self._sync_fire_weights_to_rollout(
                        trainer,
                        sleep_first=False,
                    )
                return

            # VERL syncs rollout/vLLM weights immediately after normal actor
            # optimizer updates.  FIRE mutates actor weights later in the
            # dataset task-switch hook, so the next task would otherwise sample
            # from stale rollout weights.
            #
            # Task switches are commonly aligned with save checkpoints.  At
            # that point VERL may have just woken the vLLM weights after
            # checkpoint saving, so calling update_weights() again can try to
            # wake already-resident vLLM weight buffers.  Sleep replicas first
            # when the checkpoint manager supports it, then reuse the standard
            # weight-sync path.
            self._sync_fire_weights_to_rollout(trainer, sleep_first=False)

        install_task_switch_hook(
            train_dataset,
            on_task_switch=handle_task_switch,
            hook_attr="_fire_hook_installed",
        )

    @staticmethod
    def _attach_fire_config_to_actor_rollout(config) -> None:
        fire_config = config.get("fire", None)
        if fire_config is None:
            return

        actor_rollout_config = config.actor_rollout_ref
        if actor_rollout_config.get("fire", None) is not None:
            return

        with open_dict(actor_rollout_config):
            actor_rollout_config.fire = fire_config

    @staticmethod
    def _fire_should_sync_rollout(trainer, summaries) -> bool:
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
    def _fire_config_sync_enabled(config) -> bool:
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
    def _sleep_fire_rollout_before_reset(trainer, config) -> bool:
        if not FIREResettingReasoningGymRunner._fire_config_sync_enabled(
            config,
        ):
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
    def _sync_fire_weights_to_rollout(
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
