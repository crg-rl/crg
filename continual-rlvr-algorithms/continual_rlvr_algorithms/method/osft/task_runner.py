from __future__ import annotations

import os

import ray

from continual_rlvr_algorithms.method.task_runner import (
    MethodHookReasoningGymRunner,
)
from continual_rlvr_algorithms.method.task_switch import (
    install_task_switch_hook,
)

from .osft_config import load_osft_config
from .resume_state import repair_crl_dataloader_state_dict
from .upstream import ensure_upstream_paths, ensure_verl_config_compatibility
from .workers import OSFTAsyncActorRolloutRefWorker

ensure_upstream_paths()
ensure_verl_config_compatibility()

import mllm_crl.task.reasoning_gym.task_runner as upstream_reasoning_gym_task_runner
from mllm_crl.task.reasoning_gym.crl_datasets import CrlTaskSwitchingDataset
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path


def sync_crl_dataset_to_global_step(
    dataset: CrlTaskSwitchingDataset, global_step: int
) -> None:
    if global_step < 0:
        raise ValueError(
            f"global_step must be non-negative, got {global_step}"
        )

    task_idx = (global_step // dataset.steps_per_task) % len(
        dataset.task_specs
    )
    dataset.train_steps = int(global_step)
    dataset.set_task(task_idx)
    if dataset.val_dataset is not None:
        dataset.val_dataset.set_task(task_idx)


def install_crl_dataset_state_patch() -> None:
    if hasattr(CrlTaskSwitchingDataset, "sync_to_global_step"):
        return

    def state_dict(self) -> dict[str, int]:
        return {
            "task_idx": int(self.task_idx),
            "train_steps": int(self.train_steps),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        task_idx = int(state_dict["task_idx"])
        train_steps = int(state_dict["train_steps"])
        self.train_steps = train_steps
        self.set_task(task_idx)
        if self.val_dataset is not None:
            self.val_dataset.set_task(task_idx)

    def sync_to_global_step(self, global_step: int) -> None:
        sync_crl_dataset_to_global_step(self, global_step)

    CrlTaskSwitchingDataset.state_dict = state_dict
    CrlTaskSwitchingDataset.load_state_dict = load_state_dict
    CrlTaskSwitchingDataset.sync_to_global_step = sync_to_global_step


def resolve_resume_global_step_folder(config) -> str | None:
    if config.trainer.resume_mode == "disable":
        return None

    if config.trainer.resume_mode == "resume_path":
        global_step_folder = config.trainer.resume_from_path
        if not isinstance(global_step_folder, str):
            raise ValueError(
                "trainer.resume_from_path must be a string when resume_mode=resume_path"
            )
        if not os.path.isabs(global_step_folder):
            global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        return global_step_folder

    checkpoint_folder = config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
    return find_latest_ckpt_path(checkpoint_folder)


class OSFTRayPPOTrainer(RayPPOTrainer):
    def _load_checkpoint(self):
        super()._load_checkpoint()

        if not bool(self.config.get("crl", {}).get("enabled", False)):
            return
        if self.global_steps <= 0:
            return

        global_step_folder = resolve_resume_global_step_folder(self.config)
        if global_step_folder is None:
            return

        repaired_state_dict, repaired = repair_crl_dataloader_state_dict(
            self.train_dataloader.state_dict(),
            global_step=self.global_steps,
            dataloader_len=len(self.train_dataloader),
            train_batch_size=int(self.config.data.train_batch_size),
            steps_per_task=int(self.config.crl.steps_per_task),
            task_count=len(self.train_dataset.task_specs),
        )
        if repaired:
            self.train_dataloader.load_state_dict(repaired_state_dict)
            dataset_state = repaired_state_dict["dataset_state"]
            self.train_dataset.load_state_dict(dataset_state)
            print(
                "[OSFT] repaired CRL dataloader state for global_step "
                f"{self.global_steps}: yielded={repaired_state_dict['_num_yielded']}, "
                f"train_steps={dataset_state['train_steps']}, task_idx={dataset_state['task_idx']}"
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            return

        steps_per_task = int(self.config.crl.steps_per_task)
        if self.global_steps % steps_per_task != 0:
            raise ValueError(
                "Missing CRL dataloader state at a non-boundary checkpoint. "
                f"global_step={self.global_steps}, steps_per_task={steps_per_task}, "
                f"checkpoint={global_step_folder}"
            )

        for dataset in (self.train_dataset, self.val_dataset):
            if not hasattr(dataset, "sync_to_global_step"):
                raise ValueError(
                    f"Dataset does not support CRL resume inference: {type(dataset)!r}"
                )
            dataset.sync_to_global_step(self.global_steps)

        print(
            "[OSFT] inferred CRL dataset state from global_step "
            f"{self.global_steps} because {dataloader_local_path} is missing"
        )


install_crl_dataset_state_patch()
upstream_reasoning_gym_task_runner.RayPPOTrainer = OSFTRayPPOTrainer


class OSFTReasoningGymRunner(MethodHookReasoningGymRunner):
    trainer_cls = OSFTRayPPOTrainer

    """ReasoningGym task runner that routes actor/ref/rollout through the wrapper worker.

    The wrapper is a no-op for baseline runs and only activates OSFT logic when
    ``actor_rollout_ref.osft.enabled=true``.
    """

    def add_actor_rollout_worker(self, config):
        osft_config = load_osft_config(config.actor_rollout_ref)
        worker_impl_mode = config.trainer.get("osft_worker_impl", "auto")
        if worker_impl_mode == "disable":
            return super().add_actor_rollout_worker(config)

        strategy = config.actor_rollout_ref.actor.strategy
        if strategy not in {"fsdp", "fsdp2"}:
            return super().add_actor_rollout_worker(config)

        if osft_config.enabled:
            print("[OSFT] actor-side wrapper is enabled for the actor worker")
        else:
            print(
                "[OSFT] using wrapper worker in pass-through mode for baseline diagnostics"
            )

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(
            OSFTAsyncActorRolloutRefWorker
        )
        self.mapping[Role.ActorRollout] = "global_pool"
        return OSFTAsyncActorRolloutRefWorker, RayWorkerGroup

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del val_dataset
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
                "CRL OSFT requires a dataset current_task/on_batch_end lifecycle."
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
