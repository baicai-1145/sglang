# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

from dataclasses import dataclass, replace
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_compiler_backend, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if get_exec().moe.kt_weight_path is None:
        return None

    num_layers = getattr(
        model_config_of(server_args).hf_config, "num_hidden_layers", None
    )

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=get_exec().moe.kt_num_gpu_experts,
        cpuinfer_threads=get_exec().moe.kt_cpuinfer,
        threadpool_count=get_exec().moe.kt_threadpool_count,
        weight_path=get_exec().moe.kt_weight_path,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        method=get_exec().moe.kt_method,
        max_deferred_experts_per_token=get_exec().moe.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    return topk_ids.masked_fill(topk_ids >= num_gpu_experts, -1)


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank
        self.is_npu = is_npu()
        self.disk_offload = self.is_npu and os.environ.get("KT_NPU_DISK_EXPERTS") == "1"
        self.disk_engine = None
        if self.disk_offload:
            from sglang.srt.runtime_context import get_spec
            if get_spec().speculative_algorithm is not None:
                raise ValueError("Native disk expert baseline does not support speculative decoding/MTP")
        self.npu_dispatcher = None
        if self.is_npu:
            from sglang.srt.layers.moe.utils import get_moe_a2a_backend
            from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
                NPUW4A8MXFP4FusedMoEMethod,
            )
            if (get_parallel().tp_size != 1 or get_parallel().moe_ep_size != 1
                    or not get_moe_a2a_backend().is_none()):
                raise ValueError("KT native FP4 on NPU currently requires TP=EP=1 and no A2A")
            if kt_config.method != "MXFP4" or not isinstance(gpu_method, NPUW4A8MXFP4FusedMoEMethod):
                raise ValueError("KT NPU bring-up supports native MXFP4 experts only")
            if self.num_gpu_experts <= 0:
                raise ValueError("KT NPU bring-up requires at least one resident expert")
            if get_exec().moe.enable_eplb:
                raise ValueError("KT native FP4 NPU does not support EPLB")
            if kt_config.max_deferred_experts_per_token:
                raise ValueError("KT NPU bring-up does not support deferred experts")

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        if self.is_npu and getattr(layer.moe_runner_config, "num_fused_shared_experts", 0):
            raise ValueError("KT native FP4 NPU does not support fused shared experts")
        if self.is_npu and self.num_gpu_experts >= num_experts:
            raise ValueError("KT NPU requires fewer resident experts than total experts")
        cpu_kwargs = {}
        if self.is_npu:
            # Current KT inference API uses a mask; num_gpu_experts is SFT-only.
            cpu_kwargs = {
                "gpu_experts_mask": torch.arange(num_experts, device="cpu") < self.num_gpu_experts,
                "swiglu_limit": float(layer.moe_runner_config.swiglu_limit or 0.0),
            }
        else:
            cpu_kwargs = {"num_gpu_experts": self.num_gpu_experts}
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.disk_offload:
            from resident_policy import resident_ids
            ids = resident_ids(self.kt_config.layer_idx, self.num_gpu_experts, num_experts)
            self.resident_slot_by_id = {expert: slot for slot, expert in enumerate(ids)}
            self.logical_to_resident = torch.full((num_experts,), -1, dtype=torch.int64, device=layer.w13_weight.device)
            self.logical_to_resident[torch.tensor(ids, device=layer.w13_weight.device)] = torch.arange(self.num_gpu_experts, device=layer.w13_weight.device)
            from disk_experts import StreamingFP4Experts, shared_disk_store
            self.disk_engine = StreamingFP4Experts(
                shared_disk_store(self.kt_config.weight_path),
                layer=self.kt_config.layer_idx, num_experts=num_experts,
                device=str(layer.w13_weight.device),
                swiglu_limit=float(layer.moe_runner_config.swiglu_limit or 0.0),
                num_layers=self.kt_config.num_layers or 43,
                num_resident=self.num_gpu_experts,
            )
        elif self.tp_rank == 0:
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                **cpu_kwargs,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        if self.is_npu:
            from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata
            metadata = get_global_expert_location_metadata()
            if metadata is not None:
                mapping = metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx].reshape(-1)
                identity = torch.arange(self.global_num_experts, device="cpu", dtype=mapping.dtype)
                if mapping.shape != identity.shape or not torch.equal(mapping.cpu(), identity):
                    raise ValueError("KT native FP4 NPU requires identity expert placement (no EPLB)")
            for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, name)
                if (scale.reshape(self.num_gpu_experts, -1).amax(dim=1) == 0).any().item():
                    raise ValueError(f"Unloaded resident expert scales in {name}")
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            if self.is_npu:
                torch.npu.synchronize()
            else:
                torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            if self.is_npu and metadata is None:
                physical_to_logical_map_cpu = torch.arange(self.global_num_experts, dtype=torch.int64)
            else:
                physical_to_logical_map_cpu = metadata.physical_to_logical_map_cpu[self.kt_config.layer_idx].contiguous()
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)
        if self.is_npu:
            from sglang.srt.layers.moe.token_dispatcher import AscendTPDispatcher
            resident_config = replace(moe_runner_config, num_experts=self.num_gpu_experts)
            self.npu_dispatcher = AscendTPDispatcher(resident_config)
            self.npu_dispatcher.set_quant_config({"dispatcher_output_dtype": "bf16"})

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights,
            (torch.npu.current_stream(x.device).npu_stream if self.is_npu
             else torch.cuda.current_stream(x.device).cuda_stream)
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, (torch.npu.current_stream(x.device).npu_stream if self.is_npu
                else torch.cuda.current_stream(x.device).cuda_stream)
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        if self.is_npu and torch.npu.is_current_stream_capturing():
            raise RuntimeError("KT native FP4 NPU baseline requires graphs disabled")
        if self.is_npu and x.shape[0] == 0:
            return StandardCombineInput(hidden_states=torch.zeros_like(x))

        # Step 1: Submit CPU expert computation (non-blocking)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
        topk_ids = topk_output.topk_ids
        if self.disk_offload:
            masked_topk_ids = self.logical_to_resident[topk_ids.clamp_min(0).long()].masked_fill(topk_ids < 0, -1).to(topk_ids.dtype)
        else:
            masked_topk_ids = (
                topk_ids.masked_fill(topk_ids >= self.num_gpu_experts, -1)
                if self.is_npu else mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)
            )

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        if self.is_npu:
            masked_topk_output = masked_topk_output._replace(
                topk_weights=topk_output.topk_weights.masked_fill(masked_topk_ids < 0, 0)
            )
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        if self.is_npu:
            if x.shape[0] == 0 or not (masked_topk_ids >= 0).any().item():
                output = torch.zeros_like(x)
            else:
                resident_dispatch = self.npu_dispatcher.dispatch(x, masked_topk_output)
                resident_combine = self.gpu_method.apply(layer, resident_dispatch)
                # Routing allocates tokens*top_k rows, but GMM writes only sum(counts).
                # Invalid routes can reference the uninitialized tail in finalize.
                # Zero both their weights and their output (NaN * 0 is still NaN).
                active_rows = int(resident_dispatch.expert_tokens.sum().item())
                resident_combine.hidden_states[active_rows:].zero_()
                output = self.npu_dispatcher.combine(resident_combine)
        else:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states

        # Step 4: Synchronize CPU results and merge with GPU results
        if self.disk_offload:
            if os.environ.get("KT_NPU_DISK_STATS_PATH"):
                ids_cpu = topk_ids.detach().to(device="cpu", dtype=torch.int64)
                self.disk_engine.all_route_counts += torch.bincount(ids_cpu[ids_cpu >= 0], minlength=self.global_num_experts)
            cold_ids = topk_ids.masked_fill(masked_topk_ids >= 0, -1)
            cold_output = self.disk_engine.forward(x, cold_ids, topk_output.topk_weights)
            output = output + cold_output
        elif self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
