#!/usr/bin/env python3
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    hip = read("ggml/src/ggml-cuda/vendors/hip.h")

    disable_pos = hip.find("#define HIP_DISABLE_WARP_SYNC_BUILTINS 1")
    runtime_pos = hip.find("#include <hip/hip_runtime.h>")
    require(disable_pos >= 0, "HIP sync warp builtins must be disabled for ROCm compatibility")
    require(runtime_pos >= 0, "HIP runtime include not found")
    require(disable_pos < runtime_pos, "HIP_DISABLE_WARP_SYNC_BUILTINS must be defined before hip_runtime.h")

    required_helpers = [
        "GGML_HIP_SHFL_SYNC_3",
        "GGML_HIP_SHFL_SYNC_4",
        "GGML_HIP_SHFL_UP_SYNC_3",
        "GGML_HIP_SHFL_UP_SYNC_4",
        "GGML_HIP_SHFL_DOWN_SYNC_3",
        "GGML_HIP_SHFL_DOWN_SYNC_4",
        "GGML_HIP_SHFL_XOR_SYNC_3",
        "GGML_HIP_SHFL_XOR_SYNC_4",
    ]
    for helper in required_helpers:
        require(helper in hip, f"missing HIP shuffle helper {helper}")

    required_shims = [
        "__shfl_sync(...)",
        "__shfl_up_sync(...)",
        "__shfl_down_sync(...)",
        "__shfl_xor_sync(...)",
    ]
    for shim in required_shims:
        require(shim in hip, f"missing variadic HIP compatibility shim for {shim}")

    require("__shfl(var, srcLane)" in hip, "3-argument __shfl_sync must map to legacy HIP default width")
    require("__shfl(var, srcLane, width)" in hip, "4-argument __shfl_sync must preserve explicit width")
    require("__shfl_up(var, delta)" in hip, "3-argument __shfl_up_sync must map to legacy HIP default width")
    require("__shfl_up(var, delta, width)" in hip, "4-argument __shfl_up_sync must preserve explicit width")
    require("__shfl_down(var, delta)" in hip, "3-argument __shfl_down_sync must map to legacy HIP default width")
    require("__shfl_down(var, delta, width)" in hip, "4-argument __shfl_down_sync must preserve explicit width")
    require("__shfl_xor(var, laneMask)" in hip, "3-argument __shfl_xor_sync must map to legacy HIP default width")
    require("__shfl_xor(var, laneMask, width)" in hip, "4-argument __shfl_xor_sync must preserve explicit width")

    portable = read("ggml/src/ggml-cuda/fattn-kvarn-portable.cuh")
    sync_helper = re.search(
        r"static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_portable_sync_32\(\)\s*\{\s*"
        r"#if defined\(GGML_USE_HIP\)\s*__syncthreads\(\);\s*"
        r"#else\s*__syncwarp\(\);\s*#endif\s*\}",
        portable,
    )
    require(sync_helper is not None,
            "portable KVarN sync helper must use a HIP block barrier and a CUDA warp barrier")
    portable_calls = portable[:sync_helper.start()] + portable[sync_helper.end():]
    require(re.search(r"\b__syncwarp\s*\(", portable_calls) is None,
            "portable KVarN must not call raw __syncwarp outside the CUDA-only helper branch")
    require(len(re.findall(r"\bggml_cuda_fattn_kvarn_portable_sync_32\s*\(\s*\)\s*;", portable_calls)) == 9,
            "all nine portable KVarN shared-state synchronization sites must use the helper")

    loader = read("ggml/src/ggml-cuda/fattn-mma-kvarn-load.cuh")
    mma_sync = re.search(
        r"static __device__ __forceinline__ void ggml_cuda_fattn_kvarn_mma_syncwarp\(\)\s*\{(.*?)\n\}",
        loader, re.S,
    )
    require(mma_sync is not None, "MMA KVarN must define a warp-local synchronization helper")
    hip_sync, cuda_sync = mma_sync.group(1).split("#else", 1)
    require("#if defined(GGML_USE_HIP)" in hip_sync
            and hip_sync.index("__ATOMIC_RELEASE") < hip_sync.index("__builtin_amdgcn_wave_barrier")
            < hip_sync.index("__ATOMIC_ACQUIRE"),
            "HIP MMA scratch publication must use release/wave-barrier/acquire ordering")
    require("__syncthreads" not in hip_sync and "__syncwarp();" in cuda_sync,
            "per-warp MMA loops must not acquire a whole-block barrier or change CUDA sync")
    for name, count in (("fattn-mma-kvarn-impl.cuh", 14), ("fattn-mma-kvarn-decode.cuh", 1)):
        body = read("ggml/src/ggml-cuda/" + name)
        require(re.search(r"\b__syncwarp\s*\(", body) is None,
                f"{name} must not reference HIP-disabled CUDA warp-sync builtins")
        require(body.count("ggml_cuda_fattn_kvarn_mma_syncwarp();") == count,
                f"all {count} warp-local synchronization sites in {name} must use the helper")


if __name__ == "__main__":
    main()
