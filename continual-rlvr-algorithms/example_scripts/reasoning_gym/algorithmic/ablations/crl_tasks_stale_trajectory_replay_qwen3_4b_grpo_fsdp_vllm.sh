#!/usr/bin/env bash
# Stale-trajectory sample-replay ablation for the 10-task Reasoning Gym
# algorithmic stream.
set -euo pipefail

export PYTHONUNBUFFERED=1
export VLLM_ASCEND_ENABLE_NZ=0
export HYDRA_FULL_ERROR=1
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-./tiktoken_cache}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
release_root="$(cd "$script_dir/../../../../.." && pwd)"

project_name="continual_rlvr_algorithms"
exp_name="qwen3_4b_algorithmic_stale_trajectory_sample_replay"
exp_dir="${OUTPUT_DIR:-${exp_name}_$(date +%Y-%m-%d-%H-%M-%S)}"
mkdir -p "$exp_dir"

n_gpu="${N_GPU:-8}"
n_cpu="${N_CPU:-120}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"
total_training_steps=500
steps_per_task=50
total_epochs=13

task_config_dir="$($PYTHON_BIN - <<'PY'
from pathlib import Path

import mllm_crl

print(
    Path(mllm_crl.__file__).resolve().parent
    / "task"
    / "reasoning_gym"
    / "config"
)
PY
)"

DATA=(
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.train_batch_size=512
    data.filter_overlong_prompts=True
    data.truncation="error"
    data.shuffle=False
    data.dataloader_num_workers=0
)

MODEL=(
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_shm=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}"
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.001
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="vllm"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
    actor_rollout_ref.rollout.load_format="safetensors"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.calculate_log_probs=False
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.use_torch_compile=False
)

ALGORITHM=(
    algorithm.adv_estimator="grpo"
    algorithm.rollout_correction.bypass_mode=False
    algorithm.rollout_correction.rollout_is="token"
    algorithm.rollout_correction.rollout_is_threshold=2.0
)

TRAINER=(
    trainer.n_gpus_per_node="$n_gpu"
    trainer.project_name="$project_name"
    trainer.experiment_name="$exp_name"
    trainer.val_before_train=False
    trainer.default_local_dir="$exp_dir"
    trainer.total_epochs="$total_epochs"
    trainer.total_training_steps="$total_training_steps"
    trainer.save_freq="$steps_per_task"
    trainer.test_freq=0
    trainer.logger="['console','tensorboard']"
)

MISCS=(
    custom_reward_function.path=pkg://mllm_crl.task.reasoning_gym.reward
    ray_kwargs.ray_init.num_cpus="$n_cpu"
    "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
    hydra.run.dir="$exp_dir"
)

METHOD=(
    ++data.sampler.class_path=pkg://continual_rlvr_algorithms.method.prompt_replay.prompt_replay
    ++data.sampler.class_name=PromptReplaySampler
    ++data.sampler.enabled=True
    ++data.sampler.replay_scope=previous_tasks
    ++data.sampler.replay_fraction=0.5
    ++data.sampler.cooldown_steps=5
    ++data.sampler.min_pass_rate=0.24
    ++data.sampler.max_pass_rate=0.7
    ++data.sampler.seed=1
    ++data.sampler.off_policy=True
)

"$PYTHON_BIN" -m continual_rlvr_algorithms.train \
    --config-name ppo_trainer \
    -- \
    +task_runner_cls=continual_rlvr_algorithms.method.prompt_replay.task_runner:PromptReplayReasoningGymRunner \
    ++task_config="$task_config_dir/crl_tasks_algorithmic.yaml" \
    +crl.steps_per_task="$steps_per_task" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "${MISCS[@]}" \
    "${METHOD[@]}" \
    "$@" 2>&1 | tee "$exp_dir/train_log.txt"
