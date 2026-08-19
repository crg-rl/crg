from __future__ import annotations

import ray
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import Role

from continual_rlvr_algorithms.method.redo.worker import (
    REDOActorRolloutRefWorker,
)
from continual_rlvr_algorithms.method.task_runner import (
    MethodHookReasoningGymRunner,
)
from continual_rlvr_algorithms.method.task_switch import (
    install_task_switch_hook,
)


class REDOReasoningGymRunner(MethodHookReasoningGymRunner):
    def add_actor_rollout_worker(self, config):
        worker_impl_mode = config.trainer.get(
            "worker_impl_mode",
            "auto",
        )
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
