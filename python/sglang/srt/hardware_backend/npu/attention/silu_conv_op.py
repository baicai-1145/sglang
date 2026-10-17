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
        base = os.environ.get(
            "SGLANG_NPU_GDN_FUSED_CONV_SRC", "/workspace/drama3-work/.tmp/customop"
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


def fused_conv(x, hist, w, bias, stream):
    """Return (y, hist_out) or None when the fused op is unavailable.

    ``hist`` is the [B, k-1, D] history window, ``bias`` an fp32 [D] tensor
    (callers pass a zero vector when the conv has no bias).
    """
    impl = _load()
    if impl is None:
        return None
    kind, fn = impl
    if kind == "two_out":
        y, hist_out = fn(x, hist, w, bias, stream)
        return y, hist_out

    import torch

    y = fn(x, hist, w, bias, stream)
    hist_out = torch.cat([hist[:, 1:, :], x.unsqueeze(1)], dim=1)
    return y, hist_out


def conv_stream():
    import torch

    return int(torch.npu.current_stream().npu_stream)
