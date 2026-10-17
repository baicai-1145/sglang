"""Lazy wrapper around the SiluConvDecode custom ops.

Two variants are built from the same kernel family:

* ``silu_conv_decode2`` — conv + SiLU **and** the history roll (preferred: the
  caller then spends only a weight cast and an indexed pool write per layer).
* ``silu_conv_decode``  — conv + SiLU only; the roll is done in torch.

``fused_conv`` hides which one is loaded so the call site stays identical, and
everything is opt-in via SGLANG_NPU_GDN_FUSED_CONV: if the extension fails to
build or load, ``fused_conv`` returns None and callers keep the previously
validated torch path.
"""
import os

_IMPL = None
_TRIED = False


def _load():
    global _IMPL, _TRIED
    if _TRIED:
        return _IMPL
    _TRIED = True
    if not os.environ.get("SGLANG_NPU_GDN_FUSED_CONV"):
        return None
    try:
        import torch
        from torch.utils.cpp_extension import load

        cann = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")
        # The tracked tree first: .tmp is gitignored and may be cleaned, which
        # would silently drop the fused path back to the torch conv.
        base = os.environ.get("SGLANG_NPU_GDN_FUSED_CONV_SRC") or next(
            (
                d
                for d in (
                    "/workspace/drama3-work/custom-ops",
                    "/workspace/drama3-work/.tmp/customop",
                )
                if os.path.exists(os.path.join(d, "silu_conv4_bind.cpp"))
                or os.path.exists(os.path.join(d, "silu_conv3_bind.cpp"))
            ),
            "/workspace/drama3-work/custom-ops",
        )
        inc = [f"{cann}/include", f"{cann}/include/aclnn"]
        ld = [
            f"-L{cann}/lib64",
            "-lascendcl",
            "-lnnopbase",
            f"-L{cann}/opp/vendors/customize/op_api/lib",
            "-lcust_opapi",
            f"-Wl,-rpath,{cann}/opp/vendors/customize/op_api/lib",
        ]

        # v4: takes the mamba pool and its slot indices, so the kernel does the
        # gather and the in-place roll itself. One device kernel per GDN layer.
        src4 = os.path.join(base, "silu_conv4_bind.cpp")
        if os.path.exists(src4):
            mod = load(
                name="silu_conv4_ext",
                sources=[src4],
                extra_include_paths=inc,
                extra_ldflags=ld,
                verbose=False,
            )
            _IMPL = ("v4", mod.silu_conv_decode4)
            print("[siluconv] fused conv v4 (pool in-place) loaded", flush=True)
            return _IMPL

        # v3: same math, weights widened bf16->fp32 inside the kernel, so the
        # caller stops paying a 128 KiB cast per layer per step.
        src3 = os.path.join(base, "silu_conv3_bind.cpp")
        if os.path.exists(src3):
            mod = load(
                name="silu_conv3_ext",
                sources=[src3],
                extra_include_paths=inc,
                extra_ldflags=ld,
                verbose=False,
            )
            _IMPL = ("v3", mod.silu_conv_decode3)
            print("[siluconv] fused conv v3 (bf16 weights, in-kernel widen) loaded", flush=True)
            return _IMPL

        # Prefer the two-output variant: it also performs the history roll.
        src2 = os.path.join(base, "silu_conv2_bind.cpp")
        if os.path.exists(src2):
            mod = load(
                name="silu_conv2_ext",
                sources=[src2],
                extra_include_paths=inc,
                extra_ldflags=ld,
                verbose=False,
            )
            _IMPL = ("two_out", mod.silu_conv_decode2)
            print("[siluconv] fused conv v2 (conv+silu+roll) loaded", flush=True)
            return _IMPL

        src1 = os.path.join(base, "silu_conv_bind.cpp")
        if os.path.exists(src1):
            mod = load(
                name="silu_conv_ext",
                sources=[src1],
                extra_include_paths=inc,
                extra_ldflags=ld,
                verbose=False,
            )
            _IMPL = ("one_out", mod.silu_conv_decode)
            print("[siluconv] fused conv v1 (conv+silu) loaded", flush=True)
            return _IMPL

        print(f"[siluconv] no binding source under {base}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[siluconv] load failed, keeping torch conv: {e}", flush=True)
        _IMPL = None
    return _IMPL


def is_available():
    return _load() is not None


def fused_conv(x, pool, indices, w, bias, stream):
    """Return y, updating the history window in ``pool``; None if unavailable.

    ``pool`` is the [S, k-1, D] mamba conv state, ``indices`` the [B] slots to
    use. v4 does the gather and the roll inside the kernel; older variants get
    the same treatment emulated here so the call site never changes.
    """
    impl = _load()
    if impl is None:
        return None
    kind, fn = impl
    if kind == "v4":
        # The kernel widens bf16 weights itself and rolls the window in place.
        return fn(x, pool, w, bias, indices, stream)

    import torch

    hist = pool[indices]
    if kind == "v3":
        y, hist_out = fn(x, hist, w, bias, stream)
    else:
        # v2/v1 take fp32 weights; bf16 -> fp32 is a lossless widening.
        w32 = w if w.dtype == torch.float32 else w.to(torch.float32)
        b32 = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
        if kind == "two_out":
            y, hist_out = fn(x, hist, w32, b32, stream)
        else:
            y = fn(x, hist, w32, b32, stream)
            hist_out = torch.cat([hist[:, 1:, :], x.unsqueeze(1)], dim=1)
    pool[indices] = hist_out
    return y


def conv_stream():
    import torch

    return int(torch.npu.current_stream().npu_stream)
