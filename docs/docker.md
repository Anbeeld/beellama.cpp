# Docker

BeeLlama Docker images are published by the push/manual Docker workflow to
`ghcr.io/anbeeld/beellama.cpp`. CUDA images are built from the stock
`.devops/cuda.Dockerfile`; pass `CUDA_DOCKER_ARCH` when building locally for a
specific GPU generation, for example `120` for Blackwell or `86` for RTX 3090.

## Prerequisites
* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (ex. /llama/models)

## Images
The CI workflow publishes `llama-server` images to
`ghcr.io/anbeeld/beellama.cpp`:

- `server` / `server-cpu`: CPU backend. (`linux/amd64`, `linux/arm64`)
- `server-cuda` / `server-cuda12`: CUDA 12.4 backend. (`linux/amd64`)
- `server-cuda13`: CUDA 13.3 backend. (`linux/amd64`)
- `server-rocm`: ROCm backend. (`linux/amd64`)
- `server-vulkan`: Vulkan backend. (`linux/amd64`)
- `server-sycl`: SYCL backend for Intel GPUs. (`linux/amd64`)

Release tags are published as both floating tags such as `server-sycl` and
versioned tags such as `server-sycl-v0.3.0`. Branch builds use development tags
such as `server-sycl-v0.3.0-dev` and commit-specific tags such as
`server-sycl-v0.3.0-<short-sha>`.

The SYCL image is built as a generic Intel SYCL target. It intentionally does
not set `GGML_SYCL_DEVICE_ARCH`, so device code is selected by the oneAPI
runtime for the host Intel GPU instead of being pinned to one GPU generation.

## Usage

Replace `/path/to/models` below with the actual path where you downloaded the models.

```bash
docker run --rm -it \
  -v /path/to/models:/models \
  -p 8080:8080 \
  ghcr.io/anbeeld/beellama.cpp:server \
  -m /models/model.gguf --port 8080 --host 0.0.0.0 -n 512
```

Use the backend-specific server tag for GPU acceleration. For example:

```bash
docker run --gpus all \
  -v /path/to/models:/models \
  -p 8080:8080 \
  ghcr.io/anbeeld/beellama.cpp:server-cuda13 \
  -m /models/model.gguf --port 8080 --host 0.0.0.0 -ngl 999
```

## Docker With CUDA

Assuming one has the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) properly installed on Linux, or is using a GPU enabled cloud, `cuBLAS` should be accessible inside the container.

## Docker With SYCL

The SYCL image targets Intel GPUs through oneAPI and Level Zero. The host still
needs a working Intel GPU driver stack, and the container needs access to the
DRI devices:

```bash
docker run --rm -it \
  --device /dev/dri \
  -v /path/to/models:/models \
  -p 8080:8080 \
  ghcr.io/anbeeld/beellama.cpp:server-sycl \
  -m /models/model.gguf --port 8080 --host 0.0.0.0 -ngl 999
```

Set `ONEAPI_DEVICE_SELECTOR=level_zero:<index>` if the host has multiple Intel
GPU devices and you need to choose one explicitly.

## Building Docker locally

```bash
docker build -t local/beellama.cpp:full-cuda --target full -f .devops/cuda.Dockerfile .
docker build -t local/beellama.cpp:light-cuda --target light -f .devops/cuda.Dockerfile .
docker build -t local/beellama.cpp:server-cuda --target server -f .devops/cuda.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the CUDA environment supported by your container host, as well as the GPU architecture.

The defaults are:

- `CUDA_VERSION` set to `12.4.1`
- `UBUNTU_VERSION` set to `22.04`
- `CUDA_DOCKER_ARCH` set to the cmake build default, which includes all the supported architectures

The resulting images, are essentially the same as the non-CUDA images:

1. `local/beellama.cpp:full-cuda`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/beellama.cpp:light-cuda`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/beellama.cpp:server-cuda`: This image only includes the `llama-server` executable.

## Usage

After building locally, Usage is similar to the non-CUDA examples, but you'll need to add the `--gpus` flag. You will also want to use the `--n-gpu-layers` flag.

```bash
docker run --gpus all -v /path/to/models:/models local/beellama.cpp:full-cuda --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/beellama.cpp:light-cuda -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run --gpus all -v /path/to/models:/models local/beellama.cpp:server-cuda -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 1
```

## Docker With MUSA

Assuming one has the [mt-container-toolkit](https://developer.mthreads.com/musa/native) properly installed on Linux, `muBLAS` should be accessible inside the container.

## Building Docker locally

```bash
docker build -t local/beellama.cpp:full-musa --target full -f .devops/musa.Dockerfile .
docker build -t local/beellama.cpp:light-musa --target light -f .devops/musa.Dockerfile .
docker build -t local/beellama.cpp:server-musa --target server -f .devops/musa.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the MUSA environment supported by your container host, as well as the GPU architecture.

The defaults are:

- `MUSA_VERSION` set to `rc4.3.0`

The resulting images, are essentially the same as the non-MUSA images:

1. `local/beellama.cpp:full-musa`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/beellama.cpp:light-musa`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/beellama.cpp:server-musa`: This image only includes the `llama-server` executable.

## Usage

After building locally, Usage is similar to the non-MUSA examples, but you'll need to set `mthreads` as default Docker runtime. This can be done by executing `(cd /usr/bin/musa && sudo ./docker setup $PWD)` and verifying the changes by executing `docker info | grep mthreads` on the host machine. You will also want to use the `--n-gpu-layers` flag.

```bash
docker run -v /path/to/models:/models local/beellama.cpp:full-musa --run -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run -v /path/to/models:/models local/beellama.cpp:light-musa -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 1
docker run -v /path/to/models:/models local/beellama.cpp:server-musa -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 1
```

## Docker With SYCL

## Building Docker locally

```bash
docker build -t local/llama.cpp:full-intel --target full -f .devops/intel.Dockerfile .
docker build -t local/llama.cpp:light-intel --target light -f .devops/intel.Dockerfile .
docker build -t local/llama.cpp:server-intel --target server -f .devops/intel.Dockerfile .
```

You may want to pass in some different `ARGS`, depending on the SYCL environment supported by your container host, as well as the GPU architecture.
Refer to [.devops/intel.Dockerfile](../.devops/intel.Dockerfile) for the available `ARGS` and their defaults.

The resulting images, are essentially the same as the non-SYCL images:

1. `local/llama.cpp:full-intel`: This image includes both the `llama-cli` and `llama-completion` executables and the tools to convert LLaMA models into ggml and convert into 4-bit quantization.
2. `local/llama.cpp:light-intel`: This image only includes the `llama-cli` and `llama-completion` executables.
3. `local/llama.cpp:server-intel`: This image only includes the `llama-server` executable.

## Usage

After building locally, usage is similar to the non-SYCL examples, but you'll need to add the `--device` flag.

```bash
# First, find all the DRI cards
ls -la /dev/dri
# Then, pick the card that you want to use (here for e.g. /dev/dri/card0).
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:full-intel -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:light-intel -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 99
docker run --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 -v /path/to/models:/models local/llama.cpp:server-intel -m /models/7B/ggml-model-q4_0.gguf --port 8080 --host 0.0.0.0 -n 512 --n-gpu-layers 99
```

*Notes:*
- Docker has been tested successfully on native Linux. WSL support has not been verified yet.
- You may need to install Intel GPU driver on the **host** machine *(Please refer to the [Linux configuration](./backend/SYCL.md#linux) for details)*.
