from __future__ import annotations

from continual_rlvr_algorithms.method.prompt_replay.offpolicy_replay import (
    OffPolicyReplayTrainer,
    validate_offpolicy_replay_config,
)
from continual_rlvr_algorithms.method.prompt_replay.prompt_replay import (
    PromptReplayConfig,
    PromptReplayDataset,
)
from continual_rlvr_algorithms.method.task_runner import (
    MethodHookReasoningGymRunner,
)


class PromptReplayReasoningGymRunner(MethodHookReasoningGymRunner):
    """Reasoning Gym runner with algorithm-owned prompt replay logic."""

    def before_setup(self, config) -> None:
        replay_config = PromptReplayConfig.from_data_config(config.get("data", {}))
        if replay_config.off_policy:
            validate_offpolicy_replay_config(config)
            self.trainer_cls = OffPolicyReplayTrainer

    def prepare_runner_datasets(
        self,
        config,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        del config, tokenizer
        train_dataset = PromptReplayDataset(train_dataset)
        return train_dataset, val_dataset
