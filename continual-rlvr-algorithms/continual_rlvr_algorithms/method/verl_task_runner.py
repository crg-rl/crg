from __future__ import annotations

import os
import socket
from pprint import pprint

from omegaconf import OmegaConf
from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local


class MethodHookTaskRunner(TaskRunner):
    """VERL TaskRunner with algorithm-owned dataset/worker hooks.

    This runner mirrors verl.trainer.main_ppo.TaskRunner.run and exposes
    algorithm-owned hook points without replacing VERL multimodal dataset
    creation.  VLM/VisuLogic method runners use it so their path stays on the
    upstream processor-aware dataset route.
    """

    trainer_cls = RayPPOTrainer

    def before_setup(self, config) -> None:
        del config

    def prepare_runner_datasets(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        del config, tokenizer
        return train_dataset, val_dataset

    def after_init_workers(
        self,
        config,
        trainer,
        train_dataset,
        val_dataset,
    ) -> None:
        del config, trainer, train_dataset, val_dataset

    def train_reward_num_examine(self, config, default: int) -> int:
        method_debug = config.get("method_debug", {}) or {}
        return int(method_debug.get("train_reward_num_examine", default))

    def run(self, config):
        self.before_setup(config)
        print(
            f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}"
        )
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = (
            self.add_actor_rollout_worker(config)
        )
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(
            local_path, trust_remote_code=trust_remote_code
        )
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=self.train_reward_num_examine(config, 0),
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_dataset, val_dataset = self.prepare_runner_datasets(
            config,
            tokenizer,
            train_dataset,
            val_dataset,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = self.trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        self.after_init_workers(config, trainer, train_dataset, val_dataset)
        trainer.fit()
