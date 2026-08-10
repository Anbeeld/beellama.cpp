# Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts [thousands of models](https://huggingface.co/models?library=gguf&sort=trending) in the upstream-compatible GGUF format used by BeeLlama:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)

You can use a compatible model from [Hugging Face](https://huggingface.co/) with the CLI argument `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

You can use the same CLI invocation to download from other sites, by pointing the `MODEL_ENDPOINT` environment variable to an endpoint compatible with the Hugging Face API.
BeeLlama can also run models downloaded to the local filesystem.

After downloading a model, use the CLI tools to run it locally - see below.

BeeLlama requires the model to use the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other formats can be converted to GGUF using the `convert_*.py` Python scripts in this repository.
To learn more about model quantization, [read this documentation](../tools/quantize/README.md)

The Hugging Face platform provides upstream tools for converting, quantizing, and hosting compatible models:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)
