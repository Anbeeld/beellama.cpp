get_filename_component(DEST_DIR "${DEST}" DIRECTORY)
file(MAKE_DIRECTORY "${DEST_DIR}")

# Independent CTest fixtures can request the same file concurrently.
file(LOCK "${DEST}.lock" GUARD PROCESS TIMEOUT 120)
if(EXISTS "${DEST}" AND HASH MATCHES "^([A-Za-z0-9]+)=(.+)$")
    set(HASH_ALGORITHM "${CMAKE_MATCH_1}")
    set(EXPECTED_HASH "${CMAKE_MATCH_2}")
    file(${HASH_ALGORITHM} "${DEST}" ACTUAL_HASH)
    if(ACTUAL_HASH STREQUAL EXPECTED_HASH)
        return()
    endif()
endif()

if(URL)
    set(MODEL_URL "${URL}")
elseif(REPO)
    set(MODEL_URL "https://huggingface.co/${REPO}/resolve/main/${NAME}?download=true")
else()
    set(MODEL_URL "https://huggingface.co/ggml-org/models/resolve/main/${NAME}?download=true")
endif()

if(NOT EXISTS "${DEST}")
    message(STATUS "Downloading ${NAME} from ${MODEL_URL}...")
endif()

file(DOWNLOAD
    "${MODEL_URL}"
    "${DEST}"
    TLS_VERIFY ON
    EXPECTED_HASH ${HASH}
    STATUS status
)

list(GET status 0 code)

if(NOT code EQUAL 0)
    list(GET status 1 msg)
    message(FATAL_ERROR "Failed to download ${NAME}: ${msg}")
endif()
