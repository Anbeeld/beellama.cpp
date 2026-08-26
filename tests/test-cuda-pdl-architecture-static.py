from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cuda/common.cuh").read_text(encoding="utf-8")

start = SOURCE.index("static bool ggml_cuda_kernel_can_use_pdl")
end = SOURCE.index("#endif //defined(GGML_CUDA_USE_PDL)", start)
helper = SOURCE[start:end]

required = "ggml_cuda_info().devices[device].cc >= GGML_CUDA_CC_HOPPER"
if required not in helper:
    raise AssertionError(
        "PDL launch eligibility must require a Hopper-or-newer physical device, "
        "not only Hopper-era PTX"
    )

if "attr.ptxVersion >= 90" not in helper:
    raise AssertionError("PDL launch eligibility must retain the loaded-kernel PTX guard")

print("CUDA PDL architecture contract checks passed")
