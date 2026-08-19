from __future__ import annotations

import asyncio
import contextlib
import os
import time
import traceback
import warnings

import torch
import torch.distributed as dist
import transformers
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from packaging import version
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
)
from transformers.dynamic_module_utils import custom_object_save

from ..auto_model_patch import patch_auto_model_classes_for_osft
from ..fsdp_gradient_projection import wrap_optimizer_with_osft_projection
from ..osft_config import load_osft_config
from ..reprotect import apply_osft_task_switch_reprotect
from ..upstream import ensure_upstream_paths, ensure_verl_config_compatibility

ensure_upstream_paths()
ensure_verl_config_compatibility()

from mini_trainer.osft_utils import create_svd_dict
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.device import (
    get_device_id,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.fs import copy_to_local, local_mkdir_safe
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (
    convert_weight_keys,
    get_generation_config,
    print_model_size,
    update_model_config,
)
from verl.utils.profiler import (
    DistProfiler,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import (
    reduce_timing,
    topk_reduce_ratio_min_max,
)
from verl.utils.ray_utils import get_event_loop
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from verl.workers.config.optimizer import build_optimizer
from verl.workers.fsdp_workers import (
    AsyncActorRolloutRefWorker,
    get_sharding_strategy,
    get_vl_model_vision_tower,
    logger,
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.hf_rollout import HFRollout


class EvalHFRollout(HFRollout):
    def __init__(self, module, config, model_config, device_mesh):
        BaseRollout.__init__(
            self,
            config=config,
            model_config=model_config,
            device_mesh=device_mesh,
        )
        self.config = config
        self.module = module

    async def resume(self, tags: list[str]):
        return None

    async def release(self):
        return None

    async def update_weights(self, weights, **kwargs):
        return None


def _get_auto_model_class_from_architecture(model_config):
    architecture = model_config.architectures[0]
    if "ForTokenClassification" in architecture:
        return AutoModelForTokenClassification
    if "ForCausalLM" in architecture:
        return AutoModelForCausalLM
    if "ForConditionalGeneration" in architecture:
        if version.parse(transformers.__version__) >= version.parse("4.54.0"):
            return AutoModelForImageTextToText
        return AutoModelForVision2Seq
    raise NotImplementedError(
        f"Unknown architecture {model_config.architectures!r}"
    )


def get_meta_tensor_names(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            names.append(name)
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            names.append(f"buffer:{name}")
    return names


def materialize_lazy_osft_actor(
    actor_module: torch.nn.Module,
) -> torch.nn.Module:
    if not getattr(actor_module, "requires_fsdp2_initialization", False):
        return actor_module

    original_state_dict = actor_module.eject_og_state_dict()
    if dist.get_rank() == 0:
        if not original_state_dict:
            raise ValueError(
                "Rank 0 must keep the original state dict for OSFT lazy materialization"
            )
    else:
        original_state_dict = {}

    actor_module.post_fsdp2_wrap_synchronize_state_dict_across_procs(
        actor_module, original_state_dict
    )
    actor_module.compute_distributed_svd(actor_module, original_state_dict)
    actor_module.mark_fsdp2_initialized()

    meta_names = get_meta_tensor_names(actor_module)
    if meta_names:
        preview = ", ".join(meta_names[:8])
        suffix = "" if len(meta_names) <= 8 else " ..."
        raise ValueError(
            f"OSFT lazy actor still has meta tensors after materialization: {preview}{suffix}"
        )

    dist.barrier()
    return actor_module


def should_skip_osft_svd_for_resume(config) -> bool:
    osft_config = load_osft_config(config)
    if not osft_config.enabled:
        return False
    if not osft_config.fsdp2_lazy_init:
        return False
    if not osft_config.skip_svd_on_resume:
        return False
    if osft_config.resume_mode == "disable":
        return False
    if not osft_config.resume_from_path:
        return False
    return True


def build_placeholder_osft_state_dict(
    actor_module: torch.nn.Module,
    rank0_osft_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    placeholder_state_dict: dict[str, torch.Tensor] = {}
    for logical_key, spec in actor_module.orig_param_registry.items():
        if spec.role != "osft_target":
            continue
        if logical_key not in rank0_osft_state_dict:
            raise ValueError(
                f"Missing OSFT target parameter in resume placeholder build: {logical_key}"
            )
        if logical_key not in actor_module.osft_paramspec_registry:
            raise ValueError(
                f"Missing OSFT factor spec for resume placeholder build: {logical_key}"
            )

        param = rank0_osft_state_dict.pop(logical_key)
        svd_dict = create_svd_dict(
            param,
            top_k=actor_module.osft_config[logical_key],
            decompose_existing=False,
            upcast_dtype=actor_module.upcast_dtype,
            output_dtype=param.dtype,
            use_meta=False,
        )
        osft_spec = actor_module.osft_paramspec_registry[logical_key]
        placeholder_state_dict[osft_spec.U_high] = svd_dict["U_high"]
        placeholder_state_dict[osft_spec.S_high] = svd_dict["S_high"]
        placeholder_state_dict[osft_spec.V_high] = svd_dict["V_high"]
        placeholder_state_dict[osft_spec.U_low] = svd_dict["U_low"]
        placeholder_state_dict[osft_spec.S_low] = svd_dict["S_low"]
        placeholder_state_dict[osft_spec.V_low] = svd_dict["V_low"]
        del param
    return placeholder_state_dict


def materialize_lazy_osft_actor_for_resume(
    actor_module: torch.nn.Module,
) -> torch.nn.Module:
    if not getattr(actor_module, "requires_fsdp2_initialization", False):
        return actor_module

    original_state_dict = actor_module.eject_og_state_dict()
    if dist.get_rank() == 0:
        if not original_state_dict:
            raise ValueError(
                "Rank 0 must keep the original state dict for OSFT resume materialization"
            )
    else:
        original_state_dict = {}

    actor_module.post_fsdp2_wrap_synchronize_state_dict_across_procs(
        actor_module, original_state_dict
    )

    if dist.get_rank() == 0:
        placeholder_state_dict = build_placeholder_osft_state_dict(
            actor_module, original_state_dict
        )
    else:
        placeholder_state_dict = {}

    set_model_state_dict(
        model=actor_module,
        model_state_dict=placeholder_state_dict,
        options=StateDictOptions(
            broadcast_from_rank0=True,
            strict=False,
            full_state_dict=True,
        ),
    )
    actor_module.mark_fsdp2_initialized()

    meta_names = get_meta_tensor_names(actor_module)
    if meta_names:
        preview = ", ".join(meta_names[:8])
        suffix = "" if len(meta_names) <= 8 else " ..."
        raise ValueError(
            f"OSFT resume actor still has meta tensors after materialization: {preview}{suffix}"
        )

    dist.barrier()
    return actor_module


def prepare_osft_state_dict_for_rollout_sync(
    actor_module_fsdp: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], torch.nn.Module]:
    unwrap_model = getattr(
        actor_module_fsdp, "_fsdp_wrapped_module", actor_module_fsdp
    )
    if not hasattr(unwrap_model, "prepare_state_dict_for_save"):
        raise ValueError(
            "OSFT actor model is missing prepare_state_dict_for_save during rollout sync"
        )

    state_dict = actor_module_fsdp.state_dict()
    state_dict = unwrap_model.prepare_state_dict_for_save(state_dict)
    leftover_osft_keys = [
        name
        for name in state_dict
        if ".osft_" in name or ".osft_params." in name
    ]
    if leftover_osft_keys:
        preview = ", ".join(leftover_osft_keys[:8])
        suffix = "" if len(leftover_osft_keys) <= 8 else " ..."
        raise ValueError(
            f"OSFT rollout sync still contains factor keys after reconstruction: {preview}{suffix}"
        )

    return state_dict, unwrap_model


class OSFTAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Actor worker wrapper that injects OSFT only for actor model construction."""

    def is_eval_wrapper_mode(self) -> bool:
        if getattr(self, "eval_wrapper_mode", False):
            return True
        if bool(
            OmegaConf.select(self.config, "eval_wrapper_mode", default=False)
        ):
            return True
        return bool(
            OmegaConf.select(
                self.config, "trainer.eval_wrapper_mode", default=False
            )
        )

    def uses_async_rollout_backend(self) -> bool:
        if not hasattr(self, "rollout"):
            return False
        if hasattr(self.rollout, "inference_engine"):
            return True
        rollout_name = OmegaConf.select(
            self.config, "rollout.name", default=None
        )
        return rollout_name in {"vllm", "sglang"}

    def register_eval_hf_rollout_dispatch(self) -> None:
        if not hasattr(self, "_register_dispatch_collect_info"):
            return
        if getattr(self, "ulysses_device_mesh", None) is not None:
            dp_rank = self.ulysses_device_mesh["dp"].get_local_rank()
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
        else:
            dp_rank = self.rank
            is_collect = True
        self._register_dispatch_collect_info(
            "rollout", dp_rank=dp_rank, is_collect=is_collect
        )

    async def wait_for_rollout_engine(
        self, timeout_seconds: float = 60.0
    ) -> None:
        if not self.is_eval_wrapper_mode():
            return
        if not hasattr(self, "rollout"):
            return
        if not hasattr(self.rollout, "inference_engine"):
            return

        remaining = timeout_seconds
        while getattr(self.rollout, "inference_engine", None) is None:
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for rollout inference_engine initialization"
                )
            await asyncio.sleep(0.1)
            remaining -= 0.1

    def _build_rollout(self, *args, **kwargs):
        model_cfg = getattr(self.config, "model", None)
        removed_osft_cfg = None
        had_osft_cfg = model_cfg is not None and "osft" in model_cfg
        was_struct = (
            OmegaConf.is_struct(model_cfg) if model_cfg is not None else None
        )

        if had_osft_cfg:
            OmegaConf.set_struct(model_cfg, False)
            removed_osft_cfg = model_cfg.pop("osft")
            OmegaConf.set_struct(model_cfg, was_struct)

        try:
            rollout_name = OmegaConf.select(
                self.config, "rollout.name", default=None
            )
            if rollout_name == "hf":
                self.rollout = EvalHFRollout(
                    self.actor_module_fsdp,
                    self.config.rollout,
                    self.config.model,
                    self.device_mesh,
                )
                self.register_eval_hf_rollout_dispatch()
                return
            return super()._build_rollout(*args, **kwargs)
        finally:
            if had_osft_cfg:
                OmegaConf.set_struct(model_cfg, False)
                model_cfg["osft"] = removed_osft_cfg
                OmegaConf.set_struct(model_cfg, was_struct)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        try:
            return super().init_model()
        except Exception as exc:
            role_flags = {
                "actor": getattr(self, "_is_actor", False),
                "rollout": getattr(self, "_is_rollout", False),
                "ref": getattr(self, "_is_ref", False),
                "rank": getattr(self, "rank", "unknown"),
                "local_rank": getattr(self, "local_rank", "unknown"),
            }
            model_cfg = getattr(self.config, "model", None)
            actor_cfg = getattr(self.config, "actor", None)
            print("[OSFT] init_model failed in worker", flush=True)
            print(f"[OSFT] role_flags={role_flags}", flush=True)
            if model_cfg is not None:
                print(
                    "[OSFT] model_cfg="
                    f"path={model_cfg.get('path', None)}, "
                    f"use_remove_padding={model_cfg.get('use_remove_padding', None)}, "
                    f"use_shm={model_cfg.get('use_shm', None)}, "
                    f"enable_gradient_checkpointing={model_cfg.get('enable_gradient_checkpointing', None)}, "
                    f"mtp={model_cfg.get('mtp', None)}",
                    flush=True,
                )
            if actor_cfg is not None:
                print(
                    "[OSFT] actor_cfg="
                    f"use_torch_compile={actor_cfg.get('use_torch_compile', None)}, "
                    f"use_fused_kernels={actor_cfg.get('use_fused_kernels', None)}, "
                    f"use_prefix_grouper={actor_cfg.get('use_prefix_grouper', None)}",
                    flush=True,
                )
            traceback.print_exc()
            print(f"[OSFT] init_model exception repr: {exc!r}", flush=True)
            raise

    async def rollout_mode(self):
        osft_config = load_osft_config(self.config)
        if not osft_config.enabled or not getattr(self, "_is_actor", False):
            await super().rollout_mode()
            return

        aggressive_empty_cache(force_sync=True)
        await self.wait_for_rollout_engine()

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_model = getattr(
            self.actor_module_fsdp,
            "_fsdp_wrapped_module",
            self.actor_module_fsdp,
        )
        if hasattr(peft_model, "peft_config"):
            raise ValueError(
                "OSFT rollout sync does not support LoRA actor models"
            )

        params, unwrap_model = prepare_osft_state_dict_for_rollout_sync(
            self.actor_module_fsdp
        )
        params = convert_weight_keys(params, unwrap_model)

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        # Rank 0 may spend much longer exporting the merged HF model than other ranks.
        # The outer Ray checkpoint fan-in is enough synchronization here.
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)
        device = get_device_id()
        per_tensor_param = (
            (
                name,
                param.to(device, non_blocking=True).full_tensor()
                if isinstance(param, DTensor)
                else param,
            )
            for name, param in params.items()
        )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        await self.rollout.update_weights(
            per_tensor_param,
            peft_config=None,
            base_sync_done=self.base_sync_done,
        )
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self, local_path, hdfs_path=None, del_local_after_load=False
    ):
        super().load_checkpoint(
            local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        if not self.is_eval_wrapper_mode():
            return
        self.eval_wrapper_mode = True
        if not getattr(self, "_is_actor", False) or not getattr(
            self, "_is_rollout", False
        ):
            return
        if not self.uses_async_rollout_backend():
            export_path = (
                None
                if local_path is None
                else os.path.join(local_path, "huggingface_osft_merged")
            )
            if export_path is not None and not os.path.isdir(export_path):
                self._save_osft_hf_merged_checkpoint(local_path)
            self.eval_rollout_mode_task = None
            self.eval_rollout_mode_error = None
            self.eval_rollout_mode_active = True
            return

        loop = get_event_loop()
        if loop.is_running():
            self.eval_rollout_mode_error = None
            self.eval_rollout_mode_active = False
            task = loop.create_task(self.rollout_mode())

            def on_done(done_task):
                exception = done_task.exception()
                if exception is not None:
                    self.eval_rollout_mode_error = exception
                    return
                self.eval_rollout_mode_active = True

            task.add_done_callback(on_done)
            self.eval_rollout_mode_task = task
            return

        loop.run_until_complete(self.rollout_mode())
        self.eval_rollout_mode_active = True

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(
            mesh_name="rollout"
        )
    )
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts):
        if not self.is_eval_wrapper_mode():
            return super().generate_sequences(prompts)

        if getattr(self, "_is_actor", False) and getattr(
            self, "_is_rollout", False
        ):
            if not self.uses_async_rollout_backend():
                return super().generate_sequences(prompts)

            if self.uses_async_rollout_backend():
                task = getattr(self, "eval_rollout_mode_task", None)
                if task is not None and not task.done():
                    raise RuntimeError(
                        "Eval generate_sequences called before rollout priming completed"
                    )
                error = getattr(self, "eval_rollout_mode_error", None)
                if error is not None:
                    raise RuntimeError(
                        "Eval rollout priming failed"
                    ) from error
                if not getattr(self, "eval_rollout_mode_active", False):
                    raise RuntimeError(
                        "Eval generate_sequences called before rollout priming"
                    )
            elif not getattr(self, "eval_rollout_mode_active", False):
                raise RuntimeError(
                    "Eval generate_sequences called before rollout activation"
                )

            assert self._is_rollout
            prompts = prompts.to(get_device_id())

            meta_info = {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
            prompts.meta_info.update(meta_info)

            timing_generate = {}
            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            (
                timing_generate_topk_ratio,
                timing_generate_min,
                timing_generate_max,
            ) = topk_reduce_ratio_min_max(
                timing_generate["generate_sequences"]
            )
            timing_generate = reduce_timing(timing_generate)
            timing_generate.update(
                {
                    "generation_timing/max": timing_generate_max,
                    "generation_timing/min": timing_generate_min,
                    "generation_timing/topk_ratio": timing_generate_topk_ratio,
                }
            )
            output.meta_info["timing"] = timing_generate
            output = output.to("cpu")
            get_torch_device().empty_cache()
            return output

        return super().generate_sequences(prompts)

    def _build_model_optimizer(
        self, *args, role="actor", model_path=None, **kwargs
    ):
        osft_config = load_osft_config(self.config)
        rank = getattr(self, "rank", 0)

        if not osft_config.enabled:
            return super()._build_model_optimizer(
                *args,
                role=role,
                model_path=model_path,
                **kwargs,
            )

        if role != "actor":
            if rank == 0 and osft_config.verify_actor_only:
                print(
                    "[OSFT] bypassing non-actor model construction; "
                    f"role={role} continues to use the upstream path"
                )
            return super()._build_model_optimizer(
                *args,
                role=role,
                model_path=model_path,
                **kwargs,
            )

        if model_path is None:
            raise ValueError("OSFT actor path requires a non-empty model_path")

        if rank == 0:
            print(
                "[OSFT] enabling actor-side OSFT wrapper "
                f"(rank_ratio={osft_config.rank_ratio}, "
                f"target_patterns={osft_config.target_patterns or 'auto'})"
            )
            if not osft_config.initialize_osft:
                print(
                    "[OSFT] warning: initialize_osft=false; "
                    "this run only verifies the wrapper path and will not "
                    "perform real OSFT decomposition"
                )

        with patch_auto_model_classes_for_osft(
            model_path=model_path,
            osft_config=osft_config,
        ) as patch_state:
            if osft_config.fsdp2_lazy_init:
                result = self.build_actor_model_optimizer_with_lazy_osft(
                    *args,
                    role=role,
                    model_path=model_path,
                    **kwargs,
                )
            else:
                result = super()._build_model_optimizer(
                    *args,
                    role=role,
                    model_path=model_path,
                    **kwargs,
                )

        if patch_state is None:
            raise RuntimeError(
                "OSFT patch state was not created for actor model construction"
            )

        if osft_config.require_patch_hit and patch_state.hit_count == 0:
            raise RuntimeError(
                "OSFT patch did not intercept any AutoModel.from_pretrained call during "
                "actor model construction"
            )

        if rank == 0 and osft_config.log_patch_summary:
            loader_names = (
                ",".join(patch_state.requested_loader_classes) or "none"
            )
            print(
                "[OSFT] actor patch summary "
                f"(actual={patch_state.actual_model_class_name}, "
                f"osft={patch_state.osft_model_class_name}, "
                f"hits={patch_state.hit_count}, "
                f"loaders={loader_names}, "
                f"model_path={patch_state.model_path})"
            )

        return result

    def build_actor_model_optimizer_with_lazy_osft(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        use_prefix_grouper=False,
        use_tiled_mlp=False,
        tiled_mlp_shards=4,
    ):
        if role != "actor":
            raise ValueError("Lazy OSFT build path only supports actor role")
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError(
                "TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`."
            )

        log_gpu_memory_usage(
            f"Before init {role} from HF AutoModel", logger=logger
        )
        local_path = model_path

        self.tokenizer = hf_tokenizer(
            local_path, trust_remote_code=trust_remote_code
        )
        self.processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code
        )
        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = (
                    self.config.model.custom_chat_template
                )
            else:
                self.tokenizer.chat_template = (
                    self.config.model.custom_chat_template
                )

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        attn_implementation = override_model_config.get(
            "attn_implementation", "flash_attention_2"
        )
        actor_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )
        if self.ulysses_sequence_parallel_size > 1 and hasattr(
            actor_model_config, "vision_config"
        ):
            actor_model_config.vision_config._attn_implementation = "eager"
        if (
            getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
            and attn_implementation == "flash_attention_3"
            and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = (
                "flash_attention_2"
            )
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in value
                for value in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    key
                    for key, value in actor_model_config.auto_map.items()
                    if actor_model_config.architectures[0] in value
                )
                if auto_class == "AutoModelForVision2Seq":
                    actor_module_class = AutoModelForVision2Seq
                elif auto_class == "AutoModelForCausalLM":
                    actor_module_class = AutoModelForCausalLM
                elif auto_class == "AutoModelForImageTextToText":
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel
            elif (
                type(actor_model_config)
                in AutoModelForVision2Seq._model_mapping.keys()
            ):
                actor_module_class = AutoModelForVision2Seq
            elif (
                type(actor_model_config)
                in AutoModelForCausalLM._model_mapping.keys()
            ):
                actor_module_class = AutoModelForCausalLM
            elif (
                type(actor_model_config)
                in AutoModelForImageTextToText._model_mapping.keys()
            ):
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            if use_liger:
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get(
                "fused_kernel_options", None
            )
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None)
                if fused_kernel_options is not None
                else None
            )
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        if self._is_lora:
            actor_module.enable_input_require_grads()
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                local_adapter_path = copy_to_local(
                    lora_adapter_path,
                    use_shm=self.config.model.get("use_shm", False),
                )
                from peft import PeftModel

                actor_module = PeftModel.from_pretrained(
                    actor_module, local_adapter_path, is_trainable=True
                )
                peft_config = actor_module.peft_config["default"]
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM
            else:
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": self.config.model.target_modules,
                    "exclude_modules": self.config.model.exclude_modules,
                    "bias": "none",
                }
                actor_module = get_peft_model(
                    actor_module, LoraConfig(**lora_config)
                )

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print(
                        "[actor model] Vision tower is set to not trainable."
                    )

        if should_skip_osft_svd_for_resume(self.config):
            if self.rank == 0:
                print(
                    "[OSFT] resume detected; materializing placeholder OSFT factors before checkpoint load"
                )
            actor_module = materialize_lazy_osft_actor_for_resume(actor_module)
        else:
            if self.rank == 0:
                print("[OSFT] materializing lazy OSFT actor before FSDP wrap")
            actor_module = materialize_lazy_osft_actor(actor_module)
        # Cast actor parameters to the configured dtype before FSDP wrapping.
        actor_module.to(torch_dtype)

        dist.barrier()
        if self.rank == 0:
            print_model_size(actor_module)
        log_gpu_memory_usage(
            f"After init {role} from HF AutoModel", logger=logger
        )

        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )
        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        sharding_strategy = get_sharding_strategy(
            self.device_mesh, fsdp_config.reshard_after_forward
        )
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=None,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            use_orig_params=self.use_orig_params,
            forward_prefetch=fsdp_config.get("forward_prefetch", False),
        )
        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        actor_optimizer = None
        actor_lr_scheduler = None
        if optim_config is not None:
            actor_optimizer = build_optimizer(
                actor_module_fsdp.parameters(), optim_config
            )
            actor_optimizer = wrap_optimizer_with_osft_projection(
                actor_optimizer,
                actor_module_fsdp,
                fsdp_cls=FSDP,
            )
            if self.rank == 0:
                print(
                    "[OSFT] wrapped actor optimizer with FSDP-safe OSFT gradient projection"
                )
            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get(
                "lr_scheduler_type", "constant"
            )
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get(
                    "lr_warmup_steps_ratio", 0.0
                )
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(
                    f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}"
                )

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(
                    f"LR scheduler type {lr_scheduler_type} is not supported"
                )

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def osft_actor_model(self):
        actor_module = getattr(self, "actor_module", None)
        if actor_module is not None:
            return actor_module
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return None
        return getattr(
            actor_module_fsdp, "_fsdp_wrapped_module", actor_module_fsdp
        )

    def osft_actor_fsdp_config(self):
        actor_config = getattr(self.config, "actor", None)
        if actor_config is None and hasattr(self.config, "get"):
            actor_config = self.config.get("actor")
        if actor_config is None:
            return None
        fsdp_config = getattr(actor_config, "fsdp_config", None)
        if fsdp_config is None and hasattr(actor_config, "get"):
            fsdp_config = actor_config.get("fsdp_config")
        return fsdp_config

    def osft_fsdp_param_offload_enabled(self) -> bool:
        fsdp_config = self.osft_actor_fsdp_config()
        if fsdp_config is None:
            return False
        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("param_offload", False))
        return bool(getattr(fsdp_config, "param_offload", False))

    def osft_fsdp_optimizer_offload_enabled(self) -> bool:
        fsdp_config = self.osft_actor_fsdp_config()
        if fsdp_config is None:
            return False
        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("optimizer_offload", False))
        return bool(getattr(fsdp_config, "optimizer_offload", False))

    @contextlib.contextmanager
    def osft_reprotect_context(self, logical_key: str | None = None):
        del logical_key
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None or not isinstance(
            actor_module_fsdp, FSDP
        ):
            yield
            return

        load_params = self.osft_fsdp_param_offload_enabled()
        load_optimizer = (
            getattr(self, "actor_optimizer", None) is not None
            and self.osft_fsdp_optimizer_offload_enabled()
        )
        if load_params:
            load_fsdp_model_to_gpu(actor_module_fsdp)
        if load_optimizer:
            load_fsdp_optimizer(
                self.actor_optimizer, device_id=get_device_id()
            )
        try:
            with FSDP.summon_full_params(
                actor_module_fsdp,
                writeback=True,
                recurse=True,
                offload_to_cpu=False,
            ):
                yield
        finally:
            if load_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            if load_params:
                offload_fsdp_model_to_cpu(actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_osft_task_switch_reprotect(
        self,
        previous_task: str,
        current_task: str,
    ) -> dict[str, object] | None:
        if not getattr(self, "_is_actor", False):
            return None
        osft_config = load_osft_config(self.config)
        if not osft_config.enabled:
            return None
        if not osft_config.reinit_on_task_switch:
            return {
                "applied": False,
                "reason": "reinit_on_task_switch=false",
                "previous_task": previous_task,
                "current_task": current_task,
            }

        actor_model = self.osft_actor_model()
        if actor_model is None:
            return {
                "applied": False,
                "reason": "missing actor model",
                "previous_task": previous_task,
                "current_task": current_task,
            }

        summary = apply_osft_task_switch_reprotect(
            actor_model=actor_model,
            optimizer=getattr(self, "actor_optimizer", None),
            previous_task=previous_task,
            current_task=current_task,
            reset_optimizer=osft_config.reset_optimizer_on_reinit,
            target_context=self.osft_reprotect_context,
        )
        if getattr(self, "rank", 0) == 0:
            print(
                "[OSFT] task switch re-SVD/re-protect "
                f"{previous_task} -> {current_task}: {summary}"
            )
        return summary

    def _save_osft_hf_merged_checkpoint(self, local_path: str) -> None:
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        state_dict = get_fsdp_full_state_dict(
            self.actor_module_fsdp,
            offload_to_cpu=True,
            rank0_only=True,
        )

        if self.rank == 0:
            unwrap_model = (
                self.actor_module_fsdp._fsdp_wrapped_module
                if fsdp_version(self.actor_module_fsdp) == 1
                else self.actor_module_fsdp
            )
            model_config = unwrap_model.config
            if hasattr(unwrap_model, "prepare_state_dict_for_save"):
                state_dict = unwrap_model.prepare_state_dict_for_save(
                    state_dict
                )

            export_path = os.path.join(local_path, "huggingface_osft_merged")
            local_mkdir_safe(export_path)

            generation_config = None
            if unwrap_model.can_generate() and getattr(
                model_config, "name_or_path", None
            ):
                try:
                    generation_config = GenerationConfig.from_pretrained(
                        model_config.name_or_path
                    )
                    generation_config.save_pretrained(export_path)
                except Exception:
                    generation_config = None

            if (
                hasattr(model_config, "auto_map")
                and None in model_config.auto_map
            ):
                model_config.auto_map = {
                    k: v
                    for k, v in model_config.auto_map.items()
                    if k is not None
                }

            model_config.save_pretrained(export_path)
            processing_class = (
                self.processor
                if getattr(self, "processor", None) is not None
                else self.tokenizer
            )
            if processing_class is not None:
                processing_class.save_pretrained(export_path)
            if hasattr(model_config, "auto_map"):
                custom_object_save(
                    unwrap_model, export_path, config=model_config
                )

            auto_model_cls = _get_auto_model_class_from_architecture(
                model_config
            )
            with init_empty_weights():
                save_model = auto_model_cls.from_config(
                    model_config,
                    torch_dtype=getattr(
                        model_config, "torch_dtype", torch.bfloat16
                    ),
                )
            save_model.to_empty(device="cpu")

            if save_model.can_generate() and generation_config is not None:
                save_model.generation_config = generation_config

            save_model.save_pretrained(export_path, state_dict=state_dict)
            print(
                f"[OSFT] saved merged hf_model to: {os.path.abspath(export_path)}"
            )
            del save_model
            del state_dict

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        super().save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        osft_config = load_osft_config(self.config)
        if not osft_config.enabled or not getattr(self, "_is_actor", False):
            return

        self._save_osft_hf_merged_checkpoint(local_path)
