if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT DEFINED CMAKE_HIP_FLAGS_DEBUG)
    set(CMAKE_HIP_FLAGS_DEBUG "-g -O2")
endif()

# CMake on Windows doesn't support the HIP language yet
if (WIN32)
    set(CXX_IS_HIPCC TRUE)
else()
    string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
endif()

if (CXX_IS_HIPCC)
    if (LINUX)
        if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
            message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
        endif()

        message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                " Prefer setting the HIP compiler directly. See README for details.")
    endif()
else()
    # Forward (AMD)GPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
    if(AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if(GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    cmake_minimum_required(VERSION 3.21)
    enable_language(HIP)
endif()

find_package(hip     REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocblas REQUIRED)

if (GGML_HIP_RCCL)
    find_package(rccl REQUIRED)
endif()

if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "At least ROCM/HIP V6.1 is required")
endif()

message(STATUS "HIP and hipBLAS found")

file(GLOB   GGML_HEADERS_ROCM "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_ROCM "../../include/ggml-cuda.h")

file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
ggml_cuda_select_kvarn_fast_decode_sources(SRCS "../ggml-cuda/template-instances" ${SRCS})
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})
file(GLOB   SRCS "../ggml-cuda/template-instances/mmf*.cu")
list(APPEND GGML_SOURCES_ROCM ${SRCS})

    macro(ggml_add_fattn_vec_pair_rocm OUT_VAR PREFIX K_TYPE V_TYPE)
        set(_FILE "${PREFIX}/fattn-vec-instance-${K_TYPE}-${V_TYPE}.cu")
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}")
            message(FATAL_ERROR "Missing FA vec template instance: ${_FILE}")
        endif()
        list(APPEND ${OUT_VAR} "${_FILE}")
    endmacro()

    if (GGML_CUDA_FA_ALL_QUANTS AND GGML_CUDA_FA_HALF_QUANTS)
        message(FATAL_ERROR
            "GGML_CUDA_FA_ALL_QUANTS and GGML_CUDA_FA_HALF_QUANTS are mutually exclusive. "
            "Use GGML_CUDA_FA_ALL_QUANTS for the full K/V pair matrix, or GGML_CUDA_FA_HALF_QUANTS for only K>=V pairs.")
    endif()

    set(FA_VEC_PREFIX "../ggml-cuda/template-instances")

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "${FA_VEC_PREFIX}/fattn-vec-instance-*.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    elseif (GGML_CUDA_FA_HALF_QUANTS)
        set(GGML_CUDA_FA_KV_TYPES_ORDERED
            f16
            bf16
            q8_0
            q6_1
            q6_0
            q5_1
            q5_0
            q4_1
            turbo4_tcq
            turbo4_0
            q4_0
            q3_1
            turbo3_tcq
            turbo3_0
            q3_0
            q2_1
            turbo2_tcq
            turbo2_0
            q2_0
        )

        function(ggml_cuda_fa_kv_rank_rocm OUT TYPE)
            if (TYPE STREQUAL "f16")
                set(${OUT} 0 PARENT_SCOPE)
            elseif (TYPE STREQUAL "bf16")
                set(${OUT} 1 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q8_0")
                set(${OUT} 2 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q6_1")
                set(${OUT} 3 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q6_0")
                set(${OUT} 4 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q5_1")
                set(${OUT} 5 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q5_0")
                set(${OUT} 6 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q4_1" OR TYPE STREQUAL "turbo4_tcq" OR TYPE STREQUAL "turbo4_0")
                set(${OUT} 7 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q4_0")
                set(${OUT} 8 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q3_1" OR TYPE STREQUAL "turbo3_tcq" OR TYPE STREQUAL "turbo3_0")
                set(${OUT} 9 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q3_0")
                set(${OUT} 10 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q2_1" OR TYPE STREQUAL "turbo2_tcq" OR TYPE STREQUAL "turbo2_0")
                set(${OUT} 11 PARENT_SCOPE)
            elseif (TYPE STREQUAL "q2_0")
                set(${OUT} 12 PARENT_SCOPE)
            else()
                message(FATAL_ERROR "Unknown HIP/ROCm FA K/V rank type: ${TYPE}")
            endif()
        endfunction()

        list(LENGTH GGML_CUDA_FA_KV_TYPES_ORDERED GGML_CUDA_FA_KV_TYPES_LEN)
        math(EXPR GGML_CUDA_FA_KV_TYPES_LAST "${GGML_CUDA_FA_KV_TYPES_LEN} - 1")

    set(GGML_CUDA_FA_HALF_PAIR_COUNT 0)

        foreach(KI RANGE 0 ${GGML_CUDA_FA_KV_TYPES_LAST})
            list(GET GGML_CUDA_FA_KV_TYPES_ORDERED ${KI} K_TYPE)

            foreach(VI RANGE 0 ${GGML_CUDA_FA_KV_TYPES_LAST})
                list(GET GGML_CUDA_FA_KV_TYPES_ORDERED ${VI} V_TYPE)

                ggml_cuda_fa_kv_rank_rocm(K_RANK "${K_TYPE}")
                ggml_cuda_fa_kv_rank_rocm(V_RANK "${V_TYPE}")

                if (K_RANK LESS_EQUAL V_RANK OR K_TYPE STREQUAL "f16" OR V_TYPE STREQUAL "f16")
                    ggml_add_fattn_vec_pair_rocm(GGML_SOURCES_ROCM "${FA_VEC_PREFIX}" "${K_TYPE}" "${V_TYPE}")
                    math(EXPR GGML_CUDA_FA_HALF_PAIR_COUNT "${GGML_CUDA_FA_HALF_PAIR_COUNT} + 1")
                endif()
            endforeach()
        endforeach()

        if (NOT GGML_CUDA_FA_HALF_PAIR_COUNT EQUAL 217)
            message(FATAL_ERROR "GGML_CUDA_FA_HALF_QUANTS expected 217 FA vec pairs, got ${GGML_CUDA_FA_HALF_PAIR_COUNT}")
        endif()

        add_compile_definitions(GGML_CUDA_FA_HALF_QUANTS)
    else()
        set(GGML_CUDA_FA_DEFAULT_KV_PAIRS
            f16:f16
            f16:q8_0
            f16:q6_1
            f16:q6_0
            f16:q5_1
            f16:q5_0
            bf16:bf16
            bf16:q8_0
            bf16:q6_1
            bf16:q6_0
            bf16:q5_1
            bf16:q5_0
            q8_0:q8_0
            q8_0:q6_1
            q8_0:q6_0
            q8_0:q5_1
            q8_0:q5_0
            q8_0:q4_1
            q8_0:q4_0
            q6_1:q6_1
            q6_1:q6_0
            q6_1:q5_1
            q6_1:q5_0
            q6_1:q4_1
            q6_1:q4_0
            q6_0:q6_0
            q6_0:q5_1
            q6_0:q5_0
            q6_0:q4_1
            q6_0:q4_0
            q5_1:q5_1
            q5_1:q5_0
            q5_1:q4_1
            q5_1:q4_0
            q5_1:q3_1
            q5_1:q3_0
            q5_0:q5_0
            q5_0:q4_1
            q5_0:q4_0
            q5_0:q3_1
            q5_0:q3_0
            q4_1:q4_1
            q4_1:q4_0
            q4_1:q3_1
            q4_1:q3_0
            q4_1:q2_1
            q4_1:q2_0
            q4_0:q4_0
            q4_0:q3_1
            q4_0:q3_0
            q4_0:q2_1
            q4_0:q2_0
            q3_1:q3_1
            q3_1:q3_0
            q3_1:q2_1
            q3_1:q2_0
            q3_0:q3_0
            q3_0:q2_1
            q3_0:q2_0
            q2_1:q2_1
            q2_1:q2_0
            q2_0:q2_0
        )

        list(LENGTH GGML_CUDA_FA_DEFAULT_KV_PAIRS GGML_CUDA_FA_DEFAULT_PAIR_COUNT)
        if (NOT GGML_CUDA_FA_DEFAULT_PAIR_COUNT EQUAL 62)
            message(FATAL_ERROR "Default HIP FA vec policy expected 62 pairs, got ${GGML_CUDA_FA_DEFAULT_PAIR_COUNT}")
        endif()

        foreach(PAIR ${GGML_CUDA_FA_DEFAULT_KV_PAIRS})
            string(REPLACE ":" ";" PARTS "${PAIR}")
            list(GET PARTS 0 K_TYPE)
            list(GET PARTS 1 V_TYPE)
            ggml_add_fattn_vec_pair_rocm(GGML_SOURCES_ROCM "${FA_VEC_PREFIX}" "${K_TYPE}" "${V_TYPE}")
        endforeach()
    endif()

ggml_add_backend_library(ggml-hip
                         ${GGML_HEADERS_ROCM}
                         ${GGML_SOURCES_ROCM}
                        )

# TODO: do not use CUDA definitions for HIP
if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()

add_compile_definitions(GGML_USE_HIP)

if (GGML_CUDA_FORCE_MMQ)
    add_compile_definitions(GGML_CUDA_FORCE_MMQ)
endif()

if (GGML_CUDA_FORCE_CUBLAS)
    add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
endif()

if (GGML_CUDA_NO_PEER_COPY)
    add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
endif()

if (GGML_HIP_GRAPHS)
    add_compile_definitions(GGML_HIP_GRAPHS)
endif()

if (GGML_HIP_NO_VMM)
    add_compile_definitions(GGML_HIP_NO_VMM)
endif()

if (GGML_HIP_ROCWMMA_FATTN)
    add_compile_definitions(GGML_HIP_ROCWMMA_FATTN)
endif()

if (NOT GGML_HIP_MMQ_MFMA)
    add_compile_definitions(GGML_HIP_NO_MMQ_MFMA)
endif()

if (GGML_HIP_RCCL)
    add_compile_definitions(GGML_USE_NCCL) # RCCL has the same interface as NCCL.
endif()

if (GGML_HIP_EXPORT_METRICS)
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Rpass-analysis=kernel-resource-usage --save-temps")
endif()

if (NOT GGML_CUDA_FA)
    add_compile_definitions(GGML_CUDA_NO_FA)
endif()

if (CXX_IS_HIPCC)
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
    if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        # CMake on Windows doesn't support the HIP language yet.
        # Therefore we workaround debug build's failure on HIP backend this way.
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES COMPILE_FLAGS "-O2 -g")
    endif()
    target_link_libraries(ggml-hip PRIVATE hip::device)
else()
    set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
endif()

if (GGML_STATIC)
    message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
endif()

if (GGML_HIP_RCCL)
    target_link_libraries(ggml-hip PRIVATE ggml-base roc::rccl)
endif()

target_link_libraries(ggml-hip PRIVATE ggml-base hip::host roc::rocblas roc::hipblas)
