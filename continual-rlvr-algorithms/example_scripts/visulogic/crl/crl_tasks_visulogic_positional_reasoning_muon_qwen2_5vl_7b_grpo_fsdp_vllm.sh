#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-./tiktoken_cache}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

project_name="continual_rlvr_algorithms_vlm"
n_gpu="${N_GPU:-8}"
accel_label="${ACCELERATOR_LABEL:-${n_gpu}card}"
exp_name="${EXP_NAME:-qwen2_5vl_7b_crl_tasks_visulogic_positional_reasoning_muon_grpo_fsdp_vllm_${accel_label}}"
exp_dir="${EXP_DIR:-${exp_name}_$(date +%Y-%m-%d-%H-%M-%S)}"
mkdir -p "$exp_dir"

n_cpu="${N_CPU:-128}"
model_path="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

# CRG sequence over VisuLogic reasoning subtypes.
domain_name="positional_reasoning"
domain_tag="Positional Reasoning"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
workspace_code_root="${WORKSPACE_CODE_ROOT:-$(cd "$repo_root/.." && pwd)}"
mllm_crl_root="${MLLM_CRL_ROOT:-$workspace_code_root/mllm-crl}"
VERL_CONFIG_ROOT="${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}"
verl_root="${VERL_SOURCE_ROOT:-${VERL_ROOT:-$workspace_code_root/.deps/verl}}"

setting_name="${domain_name%_reasoning}"
data_root="${VISULOGIC_DATA_ROOT:-$workspace_code_root/data/visulogic/${setting_name}}"
train_file="${data_root}/train.parquet"
val_file="${data_root}/val.parquet"
task_field="subcategory"
tasks="['Translation','Rotation','Comparative','Flip']"
num_tasks=4
total_training_steps=500
steps_per_task=125
train_batch_size=512
val_batch_size=128
train_dataset_size=51200
val_dataset_size=256
# total_training_steps stops training; a large epoch budget avoids early exit.
total_epochs="${TOTAL_EPOCHS:-700}"

if [[ ! -f $train_file || ! -f $val_file ]]; then
  echo "[train] Missing VisuLogic CRL parquet:"
  echo "  - $train_file"
  echo "  - $val_file"
  echo "[train] Prepare domain parquet with extra_info.subcategory labels before running."
  exit 1
fi

train_file_abs=$(readlink -f "$train_file")
val_file_abs=$(readlink -f "$val_file")

############################ Parameter Groups ############################

DATA=(
    data.train_files="['$train_file_abs']"
    data.val_files="['$val_file_abs']"
    data.train_batch_size=$train_batch_size
    data.val_batch_size=$val_batch_size
    data.train_max_samples=-1
    data.val_max_samples=-1
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation="error"
    data.image_key=images
    data.return_multi_modal_inputs=True
    data.dataloader_num_workers=0
    data.shuffle=False
    data.custom_cls.path=pkg://mllm_crl.task.visulogic.crl_dataset
    data.custom_cls.name=VisuLogicCrlTaskSwitchingDataset
    +data.visulogic_crl.steps_per_task=$steps_per_task
    +data.visulogic_crl.train_dataset_size=$train_dataset_size
    +data.visulogic_crl.val_dataset_size=$val_dataset_size
    +data.visulogic_crl.task_field="$task_field"
    +data.visulogic_crl.tasks="$tasks"
)

MODEL=(
    actor_rollout_ref.model.path="$model_path"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
    ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.001
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}
    actor_rollout_ref.rollout.load_format="safetensors"
    actor_rollout_ref.rollout.max_model_len=3072
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
)

TRAINER=(
    trainer.n_gpus_per_node=$n_gpu
    trainer.nnodes=1
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.val_before_train=False
    trainer.default_local_dir=$exp_dir
    trainer.total_epochs=$total_epochs
    trainer.total_training_steps=$total_training_steps
    trainer.save_freq=$steps_per_task
    trainer.test_freq=-1
    trainer.logger=console
)

MISCS=(
    custom_reward_function.path=pkg://mllm_crl.task.visulogic.reward
    custom_reward_function.name=compute_score
    ray_kwargs.ray_init.num_cpus=$n_cpu
    "hydra.searchpath=[file://${VERL_CONFIG_ROOT:?Set VERL_CONFIG_ROOT to the installed VERL config directory}]"
    hydra.run.dir=$exp_dir
    "++ray_kwargs.ray_init.runtime_env.env_vars.TIKTOKEN_ENCODINGS_BASE=$TIKTOKEN_ENCODINGS_BASE"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$repo_root:$mllm_crl_root:$verl_root:${PYTHONPATH:-}"
    "++ray_kwargs.ray_init.runtime_env.env_vars.VISULOGIC_DOMAIN_TAG=$domain_tag"
)

FSDP=(
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

METHOD=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.actor.fsdp_config.offload_policy=false
    actor_rollout_ref.actor.fsdp_config.use_orig_params=false
    ++actor_rollout_ref.actor.optim.optimizer=Muon
    ++actor_rollout_ref.actor.optim.optimizer_impl=continual_rlvr_algorithms.method.muon.muon
    ++actor_rollout_ref.actor.optim.override_optimizer_config.momentum=0.95
    ++actor_rollout_ref.actor.optim.override_optimizer_config.ns_steps=5
    ++actor_rollout_ref.actor.optim.override_optimizer_config.nesterov=true
    ++actor_rollout_ref.actor.optim.override_optimizer_config.betas="[0.9,0.999]"
    ++actor_rollout_ref.actor.optim.override_optimizer_config.adamw_eps=1e-8
    ++actor_rollout_ref.actor.optim.override_optimizer_config.adjust_lr_fn=match_rms_adamw
    ++actor_rollout_ref.actor.optim.override_optimizer_config.adamw_lr_ratio=1.0
    ++actor_rollout_ref.actor.optim.override_optimizer_config.adamw_weight_decay_ratio=1.0
    ++actor_rollout_ref.actor.optim.override_optimizer_config.max_muon_aspect_ratio=8.0
    ++actor_rollout_ref.actor.optim.override_optimizer_config.max_muon_dim=65536
)


FUSED=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=160
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160
    ++actor_rollout_ref.model.use_fused_kernels=true
)


OVERLONG=(
    ++reward_model.reward_manager=dapo
    ++reward_model.overlong_buffer.enable=true
    ++reward_model.overlong_buffer.len=256
    ++reward_model.overlong_buffer.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=256
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=1024
)

############################ Launch ############################

PYTHONPATH="$repo_root:$mllm_crl_root:$verl_root:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m continual_rlvr_algorithms.vlm_train \
    --config-name ppo_trainer \
    -- \
    +task_runner_cls=continual_rlvr_algorithms.method.visulogic.task_runner:VisuLogicMuonRunner \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ALGORITHM[@]}" \
    "${TRAINER[@]}" \
    "${MISCS[@]}" \
    "${FSDP[@]}" \
    "${METHOD[@]}" \
    "${FUSED[@]}" \
    "${OVERLONG[@]}" \
    "$@" 2>&1 | tee "$exp_dir"/train_log.txt
