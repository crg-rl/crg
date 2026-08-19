from __future__ import annotations

import contextlib

from torch.distributed.fsdp import FullyShardedDataParallel
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from continual_rlvr_algorithms.method.fire.fire import (
    FIREConfig,
    apply_fire_reset,
    clear_optimizer_state,
)


class FIREActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.fire_config = FIREConfig.from_mapping(self.config.get("fire", {}))

    def fire_actor_model(self):
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

    def fire_param_context(self):
        actor_module_fsdp = getattr(self, "actor_module_fsdp", None)
        if actor_module_fsdp is None:
            return contextlib.nullcontext()
        if isinstance(actor_module_fsdp, FullyShardedDataParallel):
            return self.fire_full_param_context(actor_module_fsdp)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def fire_reset_context(self):
        with self.fire_param_context(), self.fire_optimizer_context():
            yield

    @contextlib.contextmanager
    def fire_full_param_context(self, actor_module_fsdp):
        load_on_entry = self.fire_fsdp_param_offload_enabled()
        if load_on_entry:
            load_fsdp_model_to_gpu(actor_module_fsdp)
        try:
            with FullyShardedDataParallel.summon_full_params(
                actor_module_fsdp,
                writeback=True,
                recurse=True,
                offload_to_cpu=False,
            ):
                yield
        finally:
            if load_on_entry:
                offload_fsdp_model_to_cpu(actor_module_fsdp)

    def fire_fsdp_param_offload_enabled(self) -> bool:
        fsdp_config = self.fire_actor_fsdp_config()
        if fsdp_config is None:
            return False

        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("param_offload", False))
        return bool(getattr(fsdp_config, "param_offload", False))

    def fire_fsdp_optimizer_offload_enabled(self) -> bool:
        fsdp_config = self.fire_actor_fsdp_config()
        if fsdp_config is None:
            return False

        if hasattr(fsdp_config, "get"):
            return bool(fsdp_config.get("optimizer_offload", False))
        return bool(getattr(fsdp_config, "optimizer_offload", False))

    def fire_actor_fsdp_config(self):
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

    def fire_optimizer_context(self):
        optimizer = getattr(self, "actor_optimizer", None)
        config = getattr(self, "fire_config", FIREConfig())
        if (
            optimizer is None
            or not config.reset_optimizer
            or not self.fire_fsdp_optimizer_offload_enabled()
        ):
            return contextlib.nullcontext()
        return self.fire_offloaded_optimizer_context(optimizer)

    @contextlib.contextmanager
    def fire_offloaded_optimizer_context(self, optimizer):
        load_fsdp_optimizer(optimizer, device_id=get_device_id())
        try:
            yield
        finally:
            offload_fsdp_optimizer(optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def apply_fire_task_switch(
        self,
        previous_task: str,
        current_task: str,
    ) -> dict[str, object] | None:
        if not getattr(self, "_is_actor", False):
            return None
        actor_model = self.fire_actor_model()
        if actor_model is None:
            return {
                "applied": False,
                "reason": "missing actor model",
                "previous_task": previous_task,
                "current_task": current_task,
            }

        with self.fire_reset_context():
            summary = apply_fire_reset(actor_model, self.fire_config)
            if self.fire_config.reset_optimizer:
                summary["optimizer_state"] = clear_optimizer_state(
                    getattr(self, "actor_optimizer", None)
                )

        summary["previous_task"] = previous_task
        summary["current_task"] = current_task
        if getattr(self, "rank", 0) == 0:
            print(
                f"[FIRE] task switch "
                f"{previous_task} -> {current_task}: {summary}"
            )
        return summary
