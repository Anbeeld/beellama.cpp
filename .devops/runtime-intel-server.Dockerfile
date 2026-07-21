# syntax=docker/dockerfile:1.7

ARG ONEAPI_VERSION=2025.3.3-0-devel-ubuntu24.04

FROM intel/deep-learning-essentials:${ONEAPI_VERSION} AS server

# 25.x causes ~21% SYCL regression: https://github.com/ggml-org/llama.cpp/issues/23160
# 25.x is needed for multi-GPU: https://github.com/ggml-org/llama.cpp/issues/21747
#ARG IGC_VERSION=v2.20.5
#ARG IGC_VERSION_FULL=2_2.20.5+19972
#ARG COMPUTE_RUNTIME_VERSION=25.40.35563.10
#ARG COMPUTE_RUNTIME_VERSION_FULL=25.40.35563.10-0
#ARG IGDGMM_VERSION=22.8.2
ARG IGC_VERSION=v2.34.4
ARG IGC_VERSION_FULL=2_2.34.4+21428
ARG COMPUTE_RUNTIME_VERSION=26.18.38308.1
ARG COMPUTE_RUNTIME_VERSION_FULL=26.18.38308.1-0
ARG IGDGMM_VERSION=22.10.0

RUN mkdir /tmp/neo/ && cd /tmp/neo/ \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && dpkg --install *.deb \
    && rm -rf /tmp/neo

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg libgomp1 \
    && rm -rf /tmp/* /var/tmp/*

WORKDIR /app

COPY app/ /app/

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/Anbeeld/beellama.cpp
ARG IMAGE_SOURCE=https://github.com/Anbeeld/beellama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="BeeLlama.cpp" \
      org.opencontainers.image.description="BeeLlama.cpp GGUF inference with upstream DFlash and KVarN KV-cache support" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

ENV LLAMA_ARG_HOST=0.0.0.0

RUN test -x /app/llama-server \
    && test -s /app/libggml-sycl.so \
    && ldd /app/libggml-sycl.so \
    && mv /app/libggml-sycl.so /app/libggml-sycl.so.smoke-disabled \
    && /app/llama-server --version \
    && mv /app/libggml-sycl.so.smoke-disabled /app/libggml-sycl.so

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
