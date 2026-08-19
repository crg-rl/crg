from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from mllm_crl.task import ReasoningGymRunner


def load_and_merge_task_config(config: DictConfig) -> DictConfig:
    path = config.get("task_config", None)
    if not path:
        return config
    task_config = OmegaConf.load(path)

    was_struct = OmegaConf.is_struct(config)
    OmegaConf.set_struct(config, value=False)
    config = OmegaConf.merge(config, task_config)
    OmegaConf.set_struct(config, value=was_struct)
    return config


ppo_config_path = (
    Path(run_ppo.__code__.co_filename).resolve().parent / "config"
)


@hydra.main(config_path=str(ppo_config_path), version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training
        parameters.
    """
    config = load_and_merge_task_config(config)
    auto_set_device(config)
    task_runner_class = ray.remote(num_cpus=1)(ReasoningGymRunner)
    run_ppo(config, task_runner_class)


if __name__ == "__main__":
    main()
