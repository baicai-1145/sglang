"""Attribute hot small ops to a Python call site by counting aten dispatches.

The Ascend profiler cannot do this here: the LM's torch DB carries 754 tasks
with no shapes, and `record_shapes`/`with_stack` silently fail to export on
this torch_npu (traced twice). A TorchDispatchMode sees every aten op with the
live Python stack, which is exactly the missing link between "306 Casts per
frame" and the module that emits them.

Usage (from forward_decode, first line):
    from .opcount import maybe_count
    maybe_count(self)

Enable with DRAMA3_OPCOUNT=1, tune the window with DRAMA3_OPCOUNT_STEPS
(default 2 decode steps). Output is a histogram of aten op x call site.
"""
import os
import traceback
from collections import Counter

try:  # torch_npu may import a different torch build; stay optional
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:  # noqa: BLE001
    TorchDispatchMode = None

_SKIP = {
    "aten::empty",
    "aten::empty_strided",
    "aten::as_strided",
    "aten::detach",
    "aten::t",
    "aten::slice",
    "aten::select",
    "aten::view",
    "aten::reshape",
    "aten::unsqueeze",
    "aten::squeeze",
    "aten::_unsafe_view",
    "aten::lift_fresh",
    "aten::lift_fresh_copy",
    "aten::is_nonzero",
    "aten::item",
    "aten::equal",
    "aten::_local_scalar_dense",
}


class _Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.hits = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name not in _SKIP:
            # The call stack at this point is:
            #   0 this method
            #   1 torch dispatch plumbing
            #   2.. user code -> keep the first frame inside sglang
            # Keep three innermost sglang frames: a single frame lands on
            # call-site lines (e.g. a bare `return some_call(...)`) because
            # torch_npu inserts frames between dispatch and user code.
            frames = [
                f"{os.path.basename(fr.filename)}:{fr.lineno}"
                for fr in reversed(traceback.extract_stack()[:-2])
                if "sglang" in fr.filename and "opcount" not in fr.filename
            ]
            site = " <- ".join(frames[:3]) if frames else "?"
            self.hits[(name, site)] += 1
        return func(*args, **(kwargs or {}))


_STATE = {"mode": None, "steps": 0, "left": 0}


def maybe_count(backend):
    """Install the counter on the first decode call; report after N steps."""
    if not os.environ.get("DRAMA3_OPCOUNT"):
        return
    if TorchDispatchMode is None:
        return
    if _STATE["left"] == 0:
        _STATE["left"] = int(os.environ.get("DRAMA3_OPCOUNT_STEPS", "2"))
        _STATE["mode"] = _Counter()
        _STATE["mode"].__enter__()
        _STATE["steps"] = 0
        return
    _STATE["steps"] += 1
    if _STATE["steps"] < _STATE["left"]:
        return
    mode = _STATE["mode"]
    _STATE["mode"] = None
    _STATE["left"] = 0
    try:
        mode.__exit__(None, None, None)
    except Exception:  # noqa: BLE001
        pass
    total = sum(mode.hits.values())
    print(f"\n[opcount] {total} aten 调用 / {_STATE['steps']} 步", flush=True)
    for (name, site), n in mode.hits.most_common(30):
        print(f"  {n:6d}  {name:<26} {site}", flush=True)
