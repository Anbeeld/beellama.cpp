if (NOT EXISTS $ENV{MUSA_PATH})
    if (NOT EXISTS /opt/musa)
        set(MUSA_PATH /usr/local/musa)
    else()
        set(MUSA_PATH /opt/musa)
    endif()
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

set(CMAKE_C_COMPILER "${MUSA_PATH}/bin/clang")
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_COMPILER "${MUSA_PATH}/bin/clang++")
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

find_package(MUSAToolkit)

if (MUSAToolkit_FOUND)
    message(STATUS "MUSA Toolkit found")

    if (NOT DEFINED MUSA_ARCHITECTURES)
        set(MUSA_ARCHITECTURES "21;22;31")
    endif()
    message(STATUS "Using MUSA architectures: ${MUSA_ARCHITECTURES}")

    file(GLOB   GGML_HEADERS_MUSA "../ggml-cuda/*.cuh")
    list(APPEND GGML_HEADERS_MUSA "../../include/ggml-cuda.h")
    list(APPEND GGML_HEADERS_MUSA "../ggml-musa/mudnn.cuh")

    file(GLOB   GGML_SOURCES_MUSA "../ggml-cuda/*.cu")
    if (NOT GGML_CUDA_KVARN)
        list(FILTER GGML_SOURCES_MUSA EXCLUDE REGEX "kvarn(-wht)?[.]cu$")
    endif()
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_MUSA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/fattn-mma*.cu")
    ggml_cuda_select_kvarn_fast_decode_sources(SRCS "../ggml-cuda/template-instances" ${SRCS})
    list(APPEND GGML_SOURCES_MUSA ${SRCS})
    file(GLOB   SRCS "../ggml-cuda/template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_MUSA ${SRCS})

    if (GGML_MUSA_MUDNN_COPY)
        file(GLOB   SRCS "../ggml-musa/*.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
        add_compile_definitions(GGML_MUSA_MUDNN_COPY)
    endif()

    macro(ggml_add_fattn_vec_pair_musa OUT_VAR PREFIX K_TYPE V_TYPE)
        set(_FILE "${PREFIX}/fattn-vec-instance-${K_TYPE}-${V_TYPE}.cu")
        if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}")
            message(FATAL_ERROR "Missing FA vec template instance: ${_FILE}")
        endif()
        list(APPEND ${OUT_VAR} "${_FILE}")
    endmacro()

    set(FA_VEC_PREFIX "../ggml-cuda/template-instances")

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "${FA_VEC_PREFIX}/fattn-vec-instance-*.cu")
        list(APPEND GGML_SOURCES_MUSA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        ggml_cuda_get_fattn_vec_default_pairs(GGML_CUDA_FA_DEFAULT_KV_PAIRS)
        foreach(PAIR ${GGML_CUDA_FA_DEFAULT_KV_PAIRS})
            string(REPLACE ":" ";" PARTS "${PAIR}")
            list(GET PARTS 0 K_TYPE)
            list(GET PARTS 1 V_TYPE)
            ggml_add_fattn_vec_pair_musa(GGML_SOURCES_MUSA "${FA_VEC_PREFIX}" "${K_TYPE}" "${V_TYPE}")
        endforeach()
    endif()

    set_source_files_properties(${GGML_SOURCES_MUSA} PROPERTIES LANGUAGE CXX)
    foreach(SOURCE ${GGML_SOURCES_MUSA})
        set(COMPILE_FLAGS "-Od3 -fno-strict-aliasing -ffast-math -fsigned-char -x musa -mtgpu -fmusa-flush-denormals-to-zero")
        foreach(ARCH ${MUSA_ARCHITECTURES})
            set(COMPILE_FLAGS "${COMPILE_FLAGS} --cuda-gpu-arch=mp_${ARCH}")
        endforeach()
        set_property(SOURCE ${SOURCE} PROPERTY COMPILE_FLAGS ${COMPILE_FLAGS})
    endforeach()

    ggml_add_backend_library(ggml-musa
                             ${GGML_HEADERS_MUSA}
                             ${GGML_SOURCES_MUSA}
                            )

    # TODO: do not use CUDA definitions for MUSA
    if (NOT GGML_BACKEND_DL)
        target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
    endif()

    add_compile_definitions(GGML_USE_MUSA)
    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_MUSA_GRAPHS)
        add_compile_definitions(GGML_MUSA_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        target_link_libraries(ggml-musa PRIVATE MUSA::musart_static MUSA::mublas_static)
        # TODO: mudnn has not provided static libraries yet
        # if (GGML_MUSA_MUDNN_COPY)
        #     target_link_libraries(ggml-musa PRIVATE mudnn_static)
        # endif()
    else()
        target_link_libraries(ggml-musa PRIVATE MUSA::musart MUSA::mublas)
        if (GGML_MUSA_MUDNN_COPY)
            target_link_libraries(ggml-musa PRIVATE mudnn)
        endif()
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the musa driver lib (libmusa.so)
    else()
        target_link_libraries(ggml-musa PRIVATE MUSA::musa_driver)
    endif()
else()
    message(FATAL_ERROR "MUSA Toolkit not found")
endif()
