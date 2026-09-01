if (NOT DEFINED GGML_METAL_EMBED_SOURCE OR NOT EXISTS "${GGML_METAL_EMBED_SOURCE}")
    message(FATAL_ERROR "GGML_METAL_EMBED_SOURCE must name an existing file")
endif()
if (NOT DEFINED GGML_METAL_EMBED_ASM)
    message(FATAL_ERROR "GGML_METAL_EMBED_ASM must be set")
endif()

file(MD5 "${GGML_METAL_EMBED_SOURCE}" source_md5)
file(APPEND "${GGML_METAL_EMBED_ASM}" "/* src-md5: ${source_md5} */\n")
