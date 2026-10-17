"""Lazy wrapper around the SiluConvDecode custom op.

The op is a CANN custom package (installed under vendors/customize) driven
through its aclnn entry points by a small torch extension. It replaces the
~17-kernel torch causal-conv in GDN decode with one kernel:

    y = silu(sum_j hist[:, j, :] * w[j] + x * w[k-1] + bias)

Everything is opt-in via SGLANG_NPU_GDN_FUSED_CONV so the validated torch path
stays available as the fallback, and any failure to build/load simply returns
None (callers then keep the old path).
"""
import os

_OP = None
_TRIED = False


def _load():
    global _OP, _TRIED
    if _TRIED:
        return _OP
    _TRIED = True
    if not os.environ.get("SGLANG_NPU_GDN_FUSED_CONV"):
        return None
    try:
        from torch.utils.cpp_extension import load

        cann = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.0")
        src = os.environ.get(
            "SGLANG_NPU_GDN_FUSED_CONV_SRC",
            "/workspace/drama3-work/.tmp/customop/silu_conv_bind.cpp",
        )
        if not os.path.exists(src):
            print(f"[siluconv] binding source missing: {src}", flush=True)
            return None
        mod = load(
            name="silu_conv_ext",
            sources=[src],
            extra_include_paths=[f"{cann}/include", f"{cann}/include/aclnn"],
            extra_ldflags=[
                f"-L{cann}/lib64",
                "-lascendcl",
                "-lnnopbase",
                f"-L{cann}/opp/vendors/customize/op_api/lib",
                "-lcust_opapi",
                f"-Wl,-rpath,{cann}/opp/vendors/customize/op_api/lib",
            ],
            verbose=False,
        )
        _OP = mod.silu_conv_decode
        print("[siluconv] custom fused conv loaded", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[siluconv] load failed, keeping torch conv: {e}", flush=True)
        _OP = None
    return _OP


def get_op():
    return _load()


def conv_stream():
    import torch

    return int(torch.npu.current_stream().npu_stream)
