from typing import Optional, Tuple, Union

import torch
from sgl_kernel_npu.fla.fused_gdn_gating import (
    fused_gdn_gating_kernel_without_sigmoid,
    fused_gdn_gating_npu,
)
from sgl_kernel_npu.fla.l2norm import l2norm_fwd
from sgl_kernel_npu.fla.utils import prepare_chunk_indices

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

fused_gdn_gating = fused_gdn_gating_npu


class AscendGDNAttnBackend(AscendMambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = torch.Size(
            (
                *model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[
                    :-2
                ],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-1],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2],
            )
        )
        backends = model_runner.linear_attn_backends
        decode_backend = backends.decode
        prefill_backend = backends.prefill
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

    def _prepare_mamba_track_metadata(self, forward_batch: ForwardBatch):
        if self.forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mamba_track_mask_indices]
            )

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accept_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            # indices
            self.ssm_state_indices = torch.arange(
                cache_indices.shape[0] * seq_len,
                dtype=torch.int32,
                device=cache_indices.device,
            )
        else:
            self.ssm_state_indices = cache_indices
        if forward_mode.is_prefill():
            self.actual_seq_lengths = torch.diff(
                self.forward_metadata.query_start_loc
            ).to(torch.int32)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = False

    def _get_conv_weights_t(self, layer: RadixLinearAttention) -> torch.Tensor:
        w = getattr(layer, "_conv_weights_t", None)
        if w is None:
            w = layer.conv_weights.transpose(0, 1).contiguous()
            layer._conv_weights_t = w
        return w

    def _torch_conv1d_prefill(
        self, layer, x, query_start_loc, cache_indices, has_initial_states, conv_states
    ):
        """Exact causal-conv1d + SiLU for prefill, pure torch (aclnn-free).

        x: [total_tokens, dim]; per-request windows from query_start_loc.
        Also writes the trailing (k-1) taps into conv_states for decode.
        """
        w = self._get_conv_weights_t(layer).to(torch.float32)  # [k, dim]
        k = w.shape[0]
        x32 = x.to(torch.float32)
        out = torch.empty_like(x32)
        starts = query_start_loc.tolist()
        for i in range(len(starts) - 1):
            s, e = int(starts[i]), int(starts[i + 1])
            seg = x32[s:e]  # [L, dim]
            L = seg.shape[0]
            acc = seg * w[k - 1]
            for j in range(k - 1):
                # taps from previous tokens (zero-padded at sequence start)
                src_row = s + j
                if src_row >= 0 and (i == 0 or src_row >= s):
                    hist = (
                        x32[src_row : src_row + max(L - 1 - j, 0)]
                        if (L - 1 - j) > 0
                        else x32[s : s]
                    )
                    acc[j + 1 :] = acc[j + 1 :] + hist * w[j]
            if layer.bias is not None:
                acc = acc + layer.bias.to(torch.float32)
            out[s:e] = torch.nn.functional.silu(acc)
        # update conv states: last (k-1) inputs of each request
        for i in range(len(starts) - 1):
            s, e = int(starts[i]), int(starts[i + 1])
            slot = cache_indices[i]
            if slot < 0:
                continue
            tail = x32[max(e - (k - 1), s) : e]
            if tail.shape[0] < (k - 1):
                tail = torch.cat(
                    [torch.zeros((k - 1 - tail.shape[0], x.shape[-1]),
                                  dtype=torch.float32, device=x.device), tail]
                )
            conv_states[slot, :, :] = tail.to(conv_states.dtype)
        return out.to(x.dtype)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)

        import os as _os

        if _os.environ.get("DRAMA3_DEBUG_FRAMES"):
            cs = conv_states[cache_indices]
            bad = int((~torch.isfinite(cs)).sum())
            if bad:
                print(
                    f"[drama3] conv-in L{layer.layer_id}: slot={cache_indices.tolist()} "
                    f"nonfinite={bad}/{cs.numel()} "
                    f"absmax={float(cs[torch.isfinite(cs)].abs().max()) if (cs==cs).any() else -1:.1f} "
                    f"x_max={float(mixed_qkv.abs().max()):.1f}",
                    flush=True,
                )

        # Ascend 950 (CANN 9.1): the aclnn causal_conv1d AICPU path
        # intermittently emits NaN on run_mode=1 single-token decode with
        # fully-finite inputs (GetAscendCoreSyncAddr ret 207000 precedes it),
        # corrupting conv_states and the recurrent state. Replace the fused
        # op with an exact PyTorch computation: for a single decode token the
        # causal conv is just a dot of the last (kernel_size) inputs with the
        # weight taps, then SiLU (activation_mode=1).
        import os as _os

        if _os.environ.get("SGLANG_NPU_GDN_NATIVE_CONV_DECODE"):
            w = self._get_conv_weights_t(layer).to(torch.float32)  # [k, dim]
            k = w.shape[0]
            slots = cache_indices
            # NPU mamba pool conv_states layout per slot: [k-1, dim] (the
            # window axis is the FIRST one; observed at runtime).
            # Avoid torch.cat here: aclnnCat chokes when operands carry
            # mixed FRACTAL formats. Direct tap accumulation instead.
            hist = conv_states[slots].to(torch.float32)  # [bs, k-1, dim]
            x = mixed_qkv.to(torch.float32)  # [bs, dim]
            wf = w  # [k, dim]
            # y = sum_{j=0..k-2} hist[:,j,:] * wf[j] + x * wf[k-1]
            y = x * wf[k - 1]
            for j in range(k - 1):
                y = y + hist[:, j, :] * wf[j]
            if layer.bias is not None:
                y = y + layer.bias.to(torch.float32)
            y = torch.nn.functional.silu(y).to(mixed_qkv.dtype)
            # roll the cache: drop oldest row, append current token
            new_hist = torch.stack(
                [hist[:, jj, :] for jj in range(1, k - 1)] + [x], dim=1
            ).to(conv_states.dtype)
            conv_states[slots] = new_hist
            mixed_qkv = y
        else:
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        import os as _os

        if _os.environ.get("DRAMA3_DEBUG_FRAMES"):
            print(
                f"[drama3] decode L{layer.layer_id}: q_nan={bool(query.isnan().any())} "
                f"v_nan={bool(value.isnan().any())} a_nan={bool(a.isnan().any())} "
                f"b_nan={bool(b.isnan().any())} "
                f"cidx={cache_indices.tolist()} "
                f"st_nan={bool(ssm_states[cache_indices].isnan().any())} "
                f"out_nan={bool(core_attn_out.isnan().any())} "
                f"out_max={float(core_attn_out.abs().max()):.2f}",
                flush=True,
            )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0
        if is_target_verify:
            num_token_padding = mixed_qkv.shape[0]
            if (
                not self.graph_mode
                and forward_batch.global_num_token_non_padded_cpu != num_token_padding
            ):
                mixed_qkv = mixed_qkv[: forward_batch.global_num_token_non_padded_cpu]
                a = a[: forward_batch.global_num_token_non_padded_cpu]
                b = b[: forward_batch.global_num_token_non_padded_cpu]
                seq_len = forward_batch.global_num_token_non_padded_cpu

            batch_size = cache_indices.shape[0]
            draft_token_num = forward_batch.spec_info.draft_token_num
            num_accepted_tokens = torch.full(
                (batch_size,),
                draft_token_num,
                dtype=torch.int32,
                device=mixed_qkv.device,
            )
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[forward_metadata.track_conv_indices]
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )
            kernel_size = layer.conv_weights.shape[-1]
            conv_states_for_prefill = conv_states[
                :, -(kernel_size - 1) :, :
            ].contiguous()
            # Prefill keeps the aclnn causal_conv1d FN path: production logs
            # (v9) show healthy prefill activations (chunk_gdn v_max≈46,
            # exact 3.72s audio) — only the decode/update branch of the
            # sgl-kernel-npu 2026.6.1 op is numerically broken (stress test:
            # 30/30 wrong @ dim=8192, no state update, NaN leakage).
            import os as _os

            keep_native_hist = bool(
                _os.environ.get("SGLANG_NPU_GDN_NATIVE_CONV_DECODE")
            )
            if keep_native_hist:
                # The op does NOT write final taps back into the passed
                # conv_states slice (offline: copy_updated 0/20), leaving
                # stale garbage that the native decode would read as
                # history. Snapshot the raw input rows now; write the true
                # trailing taps after the op returns.
                x_rows_cache = mixed_qkv
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states_for_prefill,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_states,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
            import os as _os

            if _os.environ.get("DRAMA3_CONV_PREFILL_SAFE"):
                # The aclnn FN path (run_mode=0) intermittently emits NaN on
                # multi-request batches even with finite inputs/states (seen
                # at 217-token 7-request prefill; single-request batches are
                # healthy). Detect and fall back to an exact torch conv.
                if bool(mixed_qkv.isnan().any()):
                    mixed_qkv = self._torch_conv1d_prefill(
                        layer, mixed_qkv, query_start_loc, cache_indices,
                        has_initial_states, conv_states,
                    )
            if keep_native_hist:
                k = kernel_size
                qsl_list = (
                    query_start_loc.tolist()
                    if query_start_loc is not None
                    else [0, x_rows_cache.shape[0]]
                )
                for si in range(len(qsl_list) - 1):
                    s, e = int(qsl_list[si]), int(qsl_list[si + 1])
                    if e - s < k - 1:
                        continue
                    slot = (
                        int(cache_indices[si]) if si < len(cache_indices) else si
                    )
                    # last (k-1) input rows of this sequence, [k-1, dim],
                    # oldest-to-newest: the exact history decode needs.
                    conv_states[slot, -(k - 1) :, :] = x_rows_cache[
                        e - (k - 1) : e
                    ]
            else:
                conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            g, beta = fused_gdn_gating_kernel_without_sigmoid(
                layer.A_log, a, b, layer.dt_bias
            )
            beta = beta.unsqueeze(0)
            num_heads, head_k_dim = layer.num_q_heads, layer.head_q_dim
            num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim

            mixed_qkv_last_dim = mixed_qkv.shape[-1]

            mixed_qkv = mixed_qkv.view(batch_size, -1, mixed_qkv_last_dim)
            beta = beta.view(batch_size, -1, num_value_heads)
            g = g.view(batch_size, -1, num_value_heads)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update(
                mixed_qkv,
                num_heads,
                num_value_heads,
                head_k_dim,
                head_v_dim,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            core_attn_out = core_attn_out.view(-1, num_value_heads, head_v_dim)
            if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                core_attn_out = torch.cat(
                    [
                        core_attn_out,
                        core_attn_out.new_zeros(
                            num_token_padding - core_attn_out.shape[0],
                            *core_attn_out.shape[1:],
                        ),
                    ],
                    dim=0,
                )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )

            actual_seq_len = query.shape[0]
            query = query.view(
                actual_seq_len, layer.num_q_heads, layer.head_q_dim
            ).contiguous()
            key = key.view(
                actual_seq_len, layer.num_k_heads, layer.head_k_dim
            ).contiguous()
            value = value.view(
                actual_seq_len, layer.num_v_heads, layer.head_v_dim
            ).contiguous()
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

            import os as _os

            if _os.environ.get("DRAMA3_DEBUG_FRAMES"):
                w = self._get_conv_weights_t(layer)
                print(
                    f"[drama3] gdn gating: g_nan={bool(g.isnan().any())} "
                    f"g_max={float(g.max()):.2f} g_min={float(g.min()):.2f} "
                    f"beta_nan={bool(beta.isnan().any())} "
                    f"q_nan={bool(query.isnan().any())} v_nan={bool(value.isnan().any())} "
                    f"state_nan={bool(ssm_states[cache_indices].isnan().any())} "
                    f"convw_nan={bool(w.isnan().any())} "
                    f"convst_nan={bool(conv_states[cache_indices].isnan().any())}",
                    flush=True,
                )

            core_attn_out, last_recurrent_state, h = self.chunk_gdn(
                query, key, value, g, beta, ssm_states[cache_indices], query_start_loc
            )

            if last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        import os as _os

        if _os.environ.get("DRAMA3_DEBUG_FRAMES"):
            print(
                f"[drama3] gdn extend out: nan={bool(core_attn_out.isnan().any())} "
                f"absmax={float(core_attn_out.abs().max()):.3f} "
                f"shape={tuple(core_attn_out.shape)}",
                flush=True,
            )
        return core_attn_out

    def chunk_gdn(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
    ):
        q = l2norm_fwd(q)  # [T, Nk, Dk]
        k = l2norm_fwd(k)  # [T, Nk, Dk]
        g = g.to(torch.float32)  # [T, Nv]
        scale = k.shape[-1] ** -0.5

        # Ascend 950: the aclnn ChunkGatedDeltaRule binary is missing and
        # the sgl-kernel-npu triton chunk port diverges on head-repeated
        # inputs, producing NaN in recurrent state. Run prefill token-by-token
        # using aclnnRecurrentGatedDeltaRule, which is numerically exact.
        Nk = k.shape[-2]
        Nv = v.shape[-2]
        head_ratio = Nv // Nk
        if head_ratio > 1:
            q = q.repeat_interleave(head_ratio, dim=-2)
            k = k.repeat_interleave(head_ratio, dim=-2)

        qf = q.reshape(-1, q.shape[-2], q.shape[-1]).to(torch.bfloat16)
        kf = k.reshape(-1, k.shape[-2], k.shape[-1]).to(torch.bfloat16)
        vf = v.reshape(-1, v.shape[-2], v.shape[-1]).to(torch.bfloat16)
        betaf = beta.reshape(-1, beta.shape[-1]).to(torch.bfloat16)
        gf = g.reshape(-1, g.shape[-1]).to(torch.float32)

        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            seq_bounds = [
                (int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
                for i in range(len(cu_seqlens) - 1)
            ]
        else:
            seq_bounds = [(0, qf.shape[0])]

        batch = len(seq_bounds)
        if initial_state is not None:
            state = initial_state.to(torch.bfloat16).clone()
        else:
            state = torch.zeros(
                batch,
                Nv,
                vf.shape[-1],
                kf.shape[-1],
                dtype=torch.bfloat16,
                device=qf.device,
            )

        asl = torch.ones(1, dtype=torch.int32, device=qf.device)
        outs = []
        for b, (s, e) in enumerate(seq_bounds):
            idx = torch.tensor([b], dtype=torch.int32, device=qf.device)
            for t in range(s, e):
                o = torch.ops.npu.npu_recurrent_gated_delta_rule(
                    qf[t : t + 1].contiguous(),
                    kf[t : t + 1].contiguous(),
                    vf[t : t + 1].contiguous(),
                    state,
                    beta=betaf[t : t + 1].contiguous(),
                    scale=scale,
                    actual_seq_lengths=asl,
                    ssm_state_indices=idx,
                    g=gf[t : t + 1].contiguous(),
                )
                outs.append(o)

        import os as _os

        if _os.environ.get("DRAMA3_DEBUG_FRAMES"):
            print(
                f"[drama3] chunk_gdn: q_max={float(qf.abs().max()):.3f} "
                f"k_max={float(kf.abs().max()):.3f} v_max={float(vf.abs().max()):.3f} "
                f"beta_max={float(betaf.abs().max()):.3f} g_max={float(gf.max()):.3f} "
                f"outs0_max={float(outs[0].abs().max()):.3f} "
                f"outs_len={len(outs)} state_max={float(state.abs().max()):.3f}",
                flush=True,
            )

        core_attn_out = torch.cat(outs, dim=0).to(v.dtype)
        last_recurrent_state = state.to(v.dtype)
        return core_attn_out, last_recurrent_state, None

    def fused_recurrent_gated_delta_rule_update(
        self,
        mix_qkv: torch.Tensor,
        num_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.float32)
        batch_size = mix_qkv.shape[0]
        seq_len = mix_qkv.shape[1]
        scale = 1 / (head_k_dim**0.5)

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )

        if self.graph_mode:
            num_accept_tokens = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            actual_seq_lengths = torch.full(
                [batch_size], seq_len, dtype=torch.int32, device=cache_indices.device
            )
            ssm_state_indices = self.forward_metadata.mamba_cache_indices_gdn
        else:
            num_accept_tokens = self.num_accept_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.ssm_state_indices

        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out
