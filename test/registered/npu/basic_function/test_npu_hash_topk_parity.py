"""Parity: NPU fused hash gating (`custom.npu_moe_gating_top_k`) vs the torch reference.

``HashTopK._forward_npu`` replaces the ``_forward_torch`` fallback for the three
hash-routed DeepSeek-V4 layers on NPU. The fused op takes ``input_ids``/``tid2eid``
and selects experts by hash instead of by value, fusing
softplus+sqrt+hash-gather+renorm+cast into one kernel.

This test pins the two things that could silently drift:
  * ``topk_ids`` must be *exactly* equal (the lookup table is a pure gather);
  * the scoring/normalisation constants must line up -- ``norm_type=2``
    (sqrtsoftplus), ``eps == _RENORMALIZE_SUM_EPSILON`` (1e-20), and
    ``routed_scaling_factor`` applied *once* (the op always multiplies, so the
    caller must pass 1.0 and let ``forward`` scale, matching ``_forward_torch``).

The fused op has no slot for the fused shared expert (whose weight is the routed
sum divided by ``routed_scaling_factor``), so that configuration must raise
rather than silently mis-route.
"""

import unittest

import torch
import torch_npu  # noqa: F401

import custom_ops  # noqa: F401  (mounts torch.ops.custom.*)
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=60, suite="nightly-1-npu-a3", nightly=True)

VOCAB, N_EXPERTS, TOPK_ROUTED = 129280, 160, 6


class TestHashTopKNpuParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # HashTopK.__init__ reads get_exec().moe.enable_waterfill, which requires
        # the 'exec' config bag to be published.
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(7)

    def _build(self, num_fused_shared_experts):
        m = HashTopK(
            topk=TOPK_ROUTED + num_fused_shared_experts,
            num_experts=N_EXPERTS,
            num_fused_shared_experts=num_fused_shared_experts,
            vocab_size=VOCAB,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=False,
            layer_id=0,
        )
        m.tid2eid = torch.nn.Parameter(
            torch.randint(
                0, N_EXPERTS, (VOCAB, TOPK_ROUTED), dtype=torch.int32, device="npu"
            ),
            requires_grad=False,
        )
        return m

    def _assert_parity(self, m, logits, ids, label):
        w_ref, id_ref = m._forward_torch(logits, ids)
        w_new, id_new = m._forward_npu(logits, ids)
        torch.npu.synchronize()

        self.assertEqual(
            list(id_ref.shape), list(id_new.shape), f"{label}: topk_ids shape"
        )
        self.assertTrue(
            torch.equal(id_ref, id_new),
            f"{label}: topk_ids differ ({(id_ref != id_new).float().mean().item():.3f} frac)",
        )
        self.assertEqual(
            list(w_ref.shape), list(w_new.shape), f"{label}: topk_weights shape"
        )
        self.assertTrue(
            torch.allclose(w_ref.float(), w_new.float(), rtol=1e-5, atol=1e-6),
            f"{label}: topk_weights differ, max_abs="
            f"{(w_ref.float() - w_new.float()).abs().max().item():.3e}",
        )

    def test_parity_no_shared_expert(self):
        """The shipping config (--disable-shared-experts-fusion)."""
        m = self._build(0)
        for num_tokens in (1, 4, 64, 257):
            with self.subTest(num_tokens=num_tokens):
                self._assert_parity(
                    m,
                    torch.randn(
                        num_tokens, N_EXPERTS, dtype=torch.float32, device="npu"
                    )
                    * 2.0,
                    torch.randint(
                        0, VOCAB, (num_tokens,), dtype=torch.int64, device="npu"
                    ),
                    f"tokens={num_tokens}",
                )

    def test_parity_degenerate_ids(self):
        """Repeated and strided token ids: table gather + per-row renorm."""
        m = self._build(0)
        for label, ids in (
            ("repeated", torch.zeros(8, dtype=torch.int64, device="npu")),
            ("strided", (torch.arange(8, device="npu") * 16000) % VOCAB),
        ):
            with self.subTest(label=label):
                self._assert_parity(
                    m,
                    torch.randn(ids.shape[0], N_EXPERTS, dtype=torch.float32, device="npu"),
                    ids,
                    label,
                )

    def test_shared_expert_is_refused(self):
        """No silent mis-routing when a fused shared-expert slot is configured."""
        m = self._build(1)
        with self.assertRaises(NotImplementedError):
            m._forward_npu(
                torch.randn(2, N_EXPERTS, dtype=torch.float32, device="npu"),
                torch.randint(0, VOCAB, (2,), dtype=torch.int64, device="npu"),
            )


if __name__ == "__main__":
    unittest.main()
