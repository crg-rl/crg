#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export VLLM_ASCEND_ENABLE_NZ=0
export HYDRA_FULL_ERROR=1
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-./tiktoken_cache}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

project_name="continual_rlvr_algorithms"
exp_name="qwen3_4b_crl_tasks_algorithmic_osft_grpo_fsdp_vllm_4_910b"
exp_dir="${exp_name}_$(date +%Y-%m-%d-%H-%M-%S)"
mkdir "$exp_dir"

n_gpu="${N_GPU:-4}"
n_cpu="${N_CPU:-96}"
model_path="${MODEL_PATH:-Qwen/Qwen3-4B}"

total_training_steps="${TOTAL_TRAINING_STEPS:-500}"
steps_per_task="${STEPS_PER_TASK:-50}"
total_epochs="${TOTAL_EPOCHS:-13}"
osft_rank_ratio="${OSFT_RANK_RATIO:-0.5}"
osft_target_preset="${OSFT_TARGET_PRESET:-default}"
osft_fsdp2_lazy_init="${OSFT_FSDP2_LAZY_INIT:-true}"
osft_initialize="${OSFT_INITIALIZE:-true}"
osft_reinit_on_task_switch="${OSFT_REINIT_ON_TASK_SWITCH:-true}"
osft_reset_optimizer_on_reinit="${OSFT_RESET_OPTIMIZER_ON_REINIT:-true}"
osft_sync_rollout_on_reinit="${OSFT_SYNC_ROLLOUT_ON_REINIT:-true}"
actor_lr="${ACTOR_LR:-1e-6}"

target_patterns_override=""
case "$osft_target_preset" in
    default)
        ;;
    attn_only)
        target_patterns_override="[self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj]"
        ;;
    qkv_only)
        target_patterns_override="[self_attn.q_proj,self_attn.k_proj,self_attn.v_proj]"
        ;;
    *)
        echo "Unsupported OSFT_TARGET_PRESET: $osft_target_preset" >&2
        exit 1
        ;;
esac

task_config_dir="$(
  "$PYTHON_BIN" - <<'PY'
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
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_shm=True
    ++actor_rollout_ref.osft.enabled=true
    ++actor_rollout_ref.osft.rank_ratio=$osft_rank_ratio
    ++actor_rollout_ref.osft.initialize_osft=$osft_initialize
    ++actor_rollout_ref.osft.fsdp2_lazy_init=$osft_fsdp2_lazy_init
    ++actor_rollout_ref.osft.reinit_on_task_switch=$osft_reinit_on_task_switch
    ++actor_rollout_ref.osft.reset_optimizer_on_reinit=$osft_reset_optimizer_on_reinit
    ++actor_rollout_ref.osft.sync_rollout_on_reinit=$osft_sync_rollout_on_reinit
    ++actor_rollout_ref.osft.upcast_dtype=float32
    ++actor_rollout_ref.osft.output_dtype=bfloat16
)

if [[ -n "$target_patterns_override" ]]; then
    MODEL+=("++actor_rollout_ref.osft.target_patterns=$target_patterns_override")
fi

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=$actor_lr
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.001
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
    actor_rollout_ref.rollout.load_format="safetensors"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name="vllm"
    actor_rollout_ref.rollout.n=5
)

REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

ALGORITHM=(
    algorithm.adv_estimator="grpo"
)

TRAINER=(
    ++trainer.use_legacy_worker_impl=enable
    trainer.n_gpus_per_node=$n_gpu
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.val_before_train=False
    trainer.default_local_dir=$exp_dir
    trainer.total_epochs=$total_epochs
    trainer.total_training_steps=$total_training_steps
    trainer.save_freq=$steps_per_task
    trainer.test_freq=$steps_per_task
    trainer.logger="['console','tensorboard']"
)

MISCS=(
    custom_reward_function.path=pkg://mllm_crl.task.reasoning_gym.reward
    ray_kwargs.ray_init.num_cpus=$n_cpu
    "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
    hydra.run.dir=$exp_dir
)

FSDP=(
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
)

"$PYTHON_BIN" -m continual_rlvr_algorithms.train \
    --config-name ppo_trainer \
    -- \
    +task_runner_cls=continual_rlvr_algorithms.method.osft.task_runner:OSFTReasoningGymRunner \
    ++task_config="$task_config_dir"/crl_tasks_algorithmic.yaml \
    +crl.steps_per_task=$steps_per_task \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "${MISCS[@]}" \
    "${FSDP[@]}" \
    "$@" 2>&1 | tee "$exp_dir"/train_log.txt
