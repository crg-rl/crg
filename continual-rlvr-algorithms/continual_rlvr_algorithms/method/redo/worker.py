from __future__ import annotations

import contextlib

import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from continual_rlvr_algorithms.method.redo.redo import (
    RedoConfig,
    apply_redo_reset_from_masks,
    build_redo_masks,
    build_redo_target_blocks,
    capture_redo_scores,
    extract_model_inputs,
    move_model_inputs_to_device,
)


class REDOActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.redo_config = RedoConfig.from_mapping(self.config.get("redo", {}))

    def redo_actor_model(self):
        actor_module = getattr(self, "actor_module", None)
        if actor_module is not None:
            return actor_module

        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return None
        return getattr(
            actor_module_fsdp,
            "_fsdp_wrapped_module",
            actor_module_fsdp,
        )

    def redo_forward_module(self):
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is not None:
            return actor_module_fsdp
        actor_module = getattr(self, "actor_module", None)
        if actor_module is None:
            raise ValueError("redo missing actor module")
        return actor_module

    def redo_param_context(self):
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return contextlib.nullcontext()
        if isinstance(actor_module_fsdp, FullyShardedDataParallel):
            return self.redo_full_param_context(
                actor_module_fsdp,
                load_model_on_entry=True,
            )
        return contextlib.nullcontext()

    def redo_model_runtime_context(self):
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return contextlib.nullcontext()
        if not isinstance(actor_module_fsdp, FullyShardedDataParallel):
            return contextlib.nullcontext()
        return self.redo_loaded_model_context(actor_module_fsdp)

    @contextlib.contextmanager
    def redo_loaded_model_context(self, actor_module_fsdp):
        load_on_entry = self.redo_fsdp_param_offload_enabled()
        if load_on_entry:
            load_fsdp_model_to_gpu(actor_module_fsdp)
        try:
            yield
        finally:
            if load_on_entry:
                offload_fsdp_model_to_cpu(actor_module_fsdp)

    def redo_reset_context(self, *, model_already_loaded: bool):
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return contextlib.nullcontext()
        if isinstance(actor_module_fsdp, FullyShardedDataParallel):
            return self.redo_full_param_context(
                actor_module_fsdp,
                load_model_on_entry=not model_already_loaded,
            )
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def redo_full_param_context(
        self,
        actor_module_fsdp,
        *,
        load_model_on_entry: bool,
    ):
        model_context = (
            self.redo_loaded_model_context(actor_module_fsdp)
            if load_model_on_entry
            else contextlib.nullcontext()
        )
        with (
            model_context,
            self.redo_optimizer_context(),
            FullyShardedDataParallel.summon_full_params(
                actor_module_fsdp,
                writeback=True,
                recurse=True,
                offload_to_cpu=True,
            ),
        ):
            yield

    @contextlib.contextmanager
    def redo_optimizer_context(self):
        optimizer = getattr(self, "actor_optimizer", None)
        load_on_entry = (
            optimizer is not None
            and self.redo_fsdp_optimizer_offload_enabled()
        )
        if load_on_entry:
            load_fsdp_optimizer(
                optimizer=optimizer,
                device_id=get_device_id(),
            )
        try:
            yield
        finally:
            if load_on_entry:
                offload_fsdp_optimizer(optimizer)

    def redo_fsdp_param_offload_enabled(self) -> bool:
        config = getattr(self, "config", None)
        if config is None:
            return False

        actor_config = getattr(config, "actor", None)
        if actor_config is None and hasattr(config, "get"):
            actor_config = config.get("actor")
        if actor_config is None:
            return False

        fsdp_config = getattr(actor_config, "fsdp_config", None)
        if fsdp_config is None and hasattr(actor_config, "get"):
            fsdp_config = actor_config.get("fsdp_config")
        if fsdp_config is None:
            return False

        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("param_offload", False))
        return bool(getattr(fsdp_config, "param_offload", False))

    def redo_fsdp_optimizer_offload_enabled(self) -> bool:
        fsdp_config = self.redo_actor_fsdp_config()
        if fsdp_config is None:
            return False

        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("optimizer_offload", False))
        return bool(getattr(fsdp_config, "optimizer_offload", False))

    def redo_actor_fsdp_config(self):
        config = getattr(self, "config", None)
        if config is None:
            return None

        actor_config = getattr(config, "actor", None)
        if actor_config is None and hasattr(config, "get"):
            actor_config = config.get("actor")
        if actor_config is None:
            return None

        fsdp_config = getattr(actor_config, "fsdp_config", None)
        if fsdp_config is None and hasattr(actor_config, "get"):
            fsdp_config = actor_config.get("fsdp_config")
        return fsdp_config

    def redo_compute_device(self, actor_model):
        device_name = get_device_name()
        if device_name != "cpu":
            return torch.device(f"{device_name}:{get_device_id()}")

        for parameter in actor_model.parameters():
            if parameter.device.type != "cpu":
                return parameter.device
        return next(actor_model.parameters()).device

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_redo_task_switch(
        self,
        previous_task: str,
        current_task: str,
        batch,
    ) -> dict[str, object] | None:
        if not getattr(self, "_is_actor", False):
            return None
        actor_model = self.redo_actor_model()
        if actor_model is None:
            return {
                "applied": False,
                "reason": "missing actor model",
                "previous_task": previous_task,
                "current_task": current_task,
            }

        with self.redo_model_runtime_context():
            targets = build_redo_target_blocks(actor_model)
            model_inputs = extract_model_inputs(
                batch,
                max_examples=self.redo_config.calibration_examples,
            )
            device = self.redo_compute_device(actor_model)
            model_inputs = move_model_inputs_to_device(
                model_inputs,
                device=device,
            )
            activations = capture_redo_scores(
                self.redo_forward_module(),
                targets,
                model_inputs,
            )
            masks = build_redo_masks(activations, tau=self.redo_config.tau)

        with self.redo_reset_context(model_already_loaded=False):
            summary = apply_redo_reset_from_masks(
                actor_model=actor_model,
                optimizer=self.actor_optimizer,
                config=self.redo_config,
                masks=masks,
            )
        summary["previous_task"] = previous_task
        summary["current_task"] = current_task
        if getattr(self, "rank", 0) == 0:
            print(
                f"[REDO] task switch "
                f"{previous_task} -> {current_task}: {summary}"
            )
        return summary
