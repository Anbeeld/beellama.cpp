# Offline bridge is deliberately excluded from installed libllama dependencies.
option(LLAMA_BUILD_SINQ_OPTIMIZER "Build the optional offline SINQ native bridge" OFF)
if (LLAMA_BUILD_SINQ_OPTIMIZER)
    add_executable(llama-sinq-native native-codec.cpp ../../src/sinq/sinq-bridge-config.cpp)
    target_include_directories(llama-sinq-native PRIVATE ../../src ../../vendor ../../vendor/nlohmann)
    target_compile_features(llama-sinq-native PRIVATE cxx_std_17)
    target_link_libraries(llama-sinq-native PRIVATE llama-sinq-optimizer)
    add_executable(llama-sinq-model model-check.cpp)
    target_compile_features(llama-sinq-model PRIVATE cxx_std_17)
    target_link_libraries(llama-sinq-model PRIVATE llama)
    if (LLAMA_BUILD_TESTS)
        find_package(Python3 COMPONENTS Interpreter QUIET)
    endif()
    if (LLAMA_BUILD_TESTS AND Python3_Interpreter_FOUND)
        add_test(NAME test-sinq-native-bridge COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/test-sinq-native-bridge.py --bridge $<TARGET_FILE:llama-sinq-native>)
        add_test(NAME test-sinq-optimizer-e2e COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/test-sinq-optimizer-e2e.py --bin $<TARGET_FILE_DIR:llama-sinq-native>)
        set_tests_properties(test-sinq-native-bridge test-sinq-optimizer-e2e PROPERTIES
            SKIP_RETURN_CODE 77
            ENVIRONMENT "OMP_NUM_THREADS=24;MKL_NUM_THREADS=24;HF_HUB_OFFLINE=1;TRANSFORMERS_OFFLINE=1;CUDA_VISIBLE_DEVICES=0")
    endif()
    install(TARGETS llama-sinq-native llama-sinq-model RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
