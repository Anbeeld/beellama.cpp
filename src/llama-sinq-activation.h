#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct llama_sinq_activation_entry {
    std::string name, site, reference_module, data_file, sha256;
    int64_t input_dim = 0, output_dim = 0;
    int count = 0;
};
struct llama_sinq_activation_manifest {
    std::string source_sha256, source_revision, manifest_sha256;
    std::map<std::string,llama_sinq_activation_entry> tensors;
};

// Throws on unsupported schema/source/site, malformed data or resource bounds.
// All entries are validated in bounded memory; samples are loaded per tensor.
llama_sinq_activation_manifest llama_sinq_load_activations(const std::string & path,
                                                          const std::string & original_gguf);
std::vector<float> llama_sinq_load_activation_samples(const llama_sinq_activation_entry & entry);
