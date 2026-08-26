from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "ggml/src/ggml-cuda/getrows.cu").read_text(encoding="utf-8")


if "static __global__ void k_get_rows_float_vec(" not in SOURCE:
    raise AssertionError("same-type CUDA GET_ROWS must retain the upstream int4 vector kernel")

start = SOURCE.index("static void get_rows_cuda_float(")
end = SOURCE.index("template <typename dst_t>", start)
route = SOURCE[start:end]

required_guards = (
    "if constexpr (std::is_same<src0_t, dst_t>::value)",
    "enough_blocks",
    "ne00 % VEC == 0",
    "nb01 % 16 == 0",
    "nb02 % 16 == 0",
    "nb03 % 16 == 0",
    "nb1  % 16 == 0",
    "nb2  % 16 == 0",
    "nb3  % 16 == 0",
    "((uintptr_t) src0_d) % 16 == 0",
    "((uintptr_t) dst_d) % 16 == 0",
)
for guard in required_guards:
    if guard not in route:
        raise AssertionError(f"CUDA GET_ROWS vector route is missing guard: {guard}")

vector_launch = "ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>"
scalar_launch = "ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>"
if vector_launch not in route or scalar_launch not in route:
    raise AssertionError("CUDA GET_ROWS must retain guarded vector and scalar fallback launches")
if route.index(vector_launch) >= route.index(scalar_launch):
    raise AssertionError("CUDA GET_ROWS vector selection must precede the scalar fallback")

print("CUDA GET_ROWS guarded vector route checks passed")
