# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS server

ARG RUNTIME_PACKAGES="ca-certificates curl ffmpeg libgomp1"
ARG SMOKE_DISABLED_BACKEND=""

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends ${RUNTIME_PACKAGES} \
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
    && if [ -n "${SMOKE_DISABLED_BACKEND}" ]; then \
         test -s "/app/${SMOKE_DISABLED_BACKEND}" \
         && mv "/app/${SMOKE_DISABLED_BACKEND}" "/app/${SMOKE_DISABLED_BACKEND}.smoke-disabled" \
         && /app/llama-server --version \
         && mv "/app/${SMOKE_DISABLED_BACKEND}.smoke-disabled" "/app/${SMOKE_DISABLED_BACKEND}"; \
       else \
         /app/llama-server --version; \
       fi

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
