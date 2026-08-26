#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cuda/allreduce.cu").read_text(encoding="utf-8")


copy_impl = SOURCE.index("static bool ggml_cuda_ar_allreduce_copy_impl")
copy_outer = SOURCE.index("static bool ggml_cuda_ar_allreduce_copy_outer", copy_impl)
body = SOURCE[copy_impl:copy_outer]


def after(needle: str, position: int, reason: str) -> int:
    found = body.find(needle, position)
    if found < 0:
        raise AssertionError(f"{reason}: missing {needle!r}")
    return found


# Each rank's transport stream first copies its local contribution to host,
# then imports its peer contribution, and finally publishes an H2D-complete
# event to the compute stream. The in-place add cannot start until that event,
# so it also cannot overwrite the local source before its D2H has completed.
local_d2h = after("cudaMemcpyDeviceToHost, p->streams[i]", 0, "local D2H must use the rank transport stream")
peer_h2d = after("cudaMemcpyHostToDevice, p->streams[i]", local_d2h, "peer H2D must follow local D2H")
handoff = after("cudaEventRecord(p->ev_pool[i][slot].h2d, p->streams[i])", peer_h2d,
                "the transport stream must publish completion")
compute_wait = after("cudaStreamWaitEvent(cuda_ctx[i]->stream(), p->ev_pool[i][slot].h2d)", handoff,
                     "the compute stream must acquire the transport result")
after("ggml_cuda_ar_add_kernel<T_dst, T_src><<<", compute_wait,
      "the in-place add must follow the transport handoff")

print("CUDA internal AllReduce copy-engine source-lifetime checks passed")
