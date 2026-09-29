from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch_npu
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


def _chunk_gated_delta_rule_native(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
):
    initial_dtype = query.dtype
    # query/key: [B, T, H, K], value: [B, T, H, V], beta/g: [B, T, H]
    query = query.transpose(1, 2).contiguous().to(torch.float32)
    key = key.transpose(1, 2).contiguous().to(torch.float32)
    value = value.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))

    scale = 1 / (k_head_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    num_chunks = total_sequence_length // chunk_size
    query = query.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    key = key.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    value = value.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    k_beta = k_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    v_beta = v_beta.reshape(batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value_chunk = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.transpose(-1, -2).contiguous().to(torch.float32)
    )
    core_attn_out = torch.zeros_like(value_chunk)
    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, num_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value_chunk[:, :, i]
        attn_block = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_diag, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_block @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_sequence_length, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # Restore physical (V, K) layout to match SGLang temporal state pool
    last_recurrent_state = last_recurrent_state.transpose(-1, -2).contiguous()
    return core_attn_out, last_recurrent_state


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
            conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            draft_token_num = forward_batch.spec_info.draft_token_num
            batch_size = forward_batch.batch_size
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(batch_size, draft_token_num, layer.num_q_heads, layer.head_q_dim)
            key = key.view(batch_size, draft_token_num, layer.num_k_heads, layer.head_k_dim)
            value = value.view(batch_size, draft_token_num, layer.num_v_heads, layer.head_v_dim)
            a_v = a.view(batch_size, draft_token_num, layer.num_v_heads)
            b_v = b.view(batch_size, draft_token_num, layer.num_v_heads)

            from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
                fused_sigmoid_gating_delta_rule_update_npu,
            )

            core_attn_out = fused_sigmoid_gating_delta_rule_update_npu(
                A_log=layer.A_log,
                a=a_v,
                dt_bias=layer.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=query,
                k=key,
                v=value,
                b=b_v,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices[:batch_size],
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                cache_steps=draft_token_num,
            )
            core_attn_out = core_attn_out.view(-1, layer.num_v_heads, layer.head_v_dim)
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

        chunk_size = 64
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None
            else None
        )
        h = k.new_empty(
            len(chunk_indices) if chunk_indices is not None else 1,
            v.shape[-2],
            v.shape[-1],
            k.shape[-1],
        )  # [NT, NH, DV, Dk]

        # Ensure g and beta are 2D [T, Nv]
        if g.dim() == 3 and g.shape[0] == 1:
            g = g.squeeze(0)
        if beta.dim() == 3 and beta.shape[0] == 1:
            beta = beta.squeeze(0)

        num_heads = q.shape[-2]
        num_value_heads = v.shape[-2]
        if num_value_heads // num_heads > 1:
            q = q.repeat_interleave(num_value_heads // num_heads, dim=-2)
            k = k.repeat_interleave(num_value_heads // num_heads, dim=-2)

        op_fn = getattr(torch.ops.npu, "npu_chunk_gated_delta_rule", None) or getattr(
            torch.ops.npu, "chunk_gated_delta_rule", None
        )
        if op_fn is not None:
            try:
                core_attn_out, last_recurrent_state = op_fn(
                    q,
                    k,
                    v,
                    beta=beta,
                    initial_state=initial_state,
                    actual_seq_lengths=self.actual_seq_lengths,
                    scale=scale,
                    g=g,
                )
                return core_attn_out, last_recurrent_state, h
            except Exception as e:
                pass

        # Fallback to native implementation on NPU
        if cu_seqlens is None:
            batch_size = initial_state.shape[0] if initial_state is not None else 1
            cu_seqlens = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
        else:
            batch_size = len(cu_seqlens) - 1

        out_list = []
        final_states = []
        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            q_b = q[start:end].unsqueeze(0)
            k_b = k[start:end].unsqueeze(0)
            v_b = v[start:end].unsqueeze(0)
            g_b = g[start:end].unsqueeze(0)
            beta_b = beta[start:end].unsqueeze(0)
            s_b = initial_state[b : b + 1] if initial_state is not None else None

            o_b, s_next = _chunk_gated_delta_rule_native(
                q_b, k_b, v_b, g_b, beta_b, chunk_size=chunk_size, initial_state=s_b
            )
            out_list.append(o_b.squeeze(0))
            final_states.append(s_next)

        core_attn_out = torch.cat(out_list, dim=0)
        last_recurrent_state = torch.cat(final_states, dim=0)
        return core_attn_out, last_recurrent_state, h

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
        beta = beta.contiguous()
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

        # Split mix_qkv into query, key, value
        mix_qkv_flat = mix_qkv.view(-1, mix_qkv.shape[-1])
        q_dim = num_heads * head_k_dim
        k_dim = num_heads * head_k_dim
        v_dim = num_value_heads * head_v_dim
        q_flat, k_flat, v_flat = torch.split(mix_qkv_flat, [q_dim, k_dim, v_dim], dim=-1)
        # Vendor op (custom_transformer, RGDR_FUSED_GDN=1 default) applies
        # l2norm(q,k) + sigmoid(beta) in-kernel in full fp32, and maps GQA
        # heads natively (kHeadIdx = vHeadIdx / (NV/NK)) — pass raw tensors.
        q_flat = q_flat.view(-1, num_heads, head_k_dim).contiguous()
        k_flat = k_flat.view(-1, num_heads, head_k_dim).contiguous()
        v_flat = v_flat.view(-1, num_value_heads, head_v_dim).contiguous()

        beta_flat = beta.view(-1, num_value_heads).contiguous()
        g_flat = g.view(-1, num_value_heads).to(torch.float32).contiguous()
        ssm_indices_flat = ssm_state_indices.view(-1).contiguous()

        op_recurrent = getattr(torch_npu, "npu_recurrent_gated_delta_rule", None) or getattr(
            torch.ops.npu, "npu_recurrent_gated_delta_rule", None
        )
        assert op_recurrent is not None, "npu_recurrent_gated_delta_rule operator not found"

        attn_core_out = op_recurrent(
            q_flat,
            k_flat,
            v_flat,
            recurrent_state,
            beta=beta_flat,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_indices_flat,
            num_accepted_tokens=num_accept_tokens,
            g=g_flat,
        )

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out
