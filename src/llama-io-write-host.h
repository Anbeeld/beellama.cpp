#pragma once

#include "llama-io.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        if (winfos.empty()) {
            return;
        }

        // KVarN state interleaves thousands of small, regularly-strided slices.
        // A native batch read can still issue one transfer per slice (CUDA).
        // Sort only the read descriptors, not the serialized destination layout.
        std::vector<const write_info *> sorted;
        sorted.reserve(winfos.size());
        for (const auto & winfo : winfos) {
            sorted.push_back(&winfo);
        }
        std::sort(sorted.begin(), sorted.end(), [](const write_info * a, const write_info * b) {
            if (a->tensor != b->tensor) {
                return std::less<ggml_tensor *>{}(a->tensor, b->tensor);
            }
            if (a->size != b->size) {
                return a->size < b->size;
            }
            if (a->offset != b->offset) {
                return a->offset < b->offset;
            }
            return a->data_offset < b->data_offset;
        });

        // Keep unmatched slices in a single batch, retaining the upstream API.
        std::vector<const ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        size_t i = 0;
        while (i < sorted.size()) {
            size_t group_end = i + 1;
            while (group_end < sorted.size() &&
                    sorted[group_end]->tensor == sorted[i]->tensor &&
                    sorted[group_end]->size == sorted[i]->size) {
                ++group_end;
            }

            while (i < group_end) {
                const write_info & first = *sorted[i];
                size_t run = 1;
                size_t source_stride = 0;
                size_t destination_stride = 0;
                if (first.size > 0 && i + 1 < group_end &&
                        sorted[i + 1]->offset > first.offset &&
                        sorted[i + 1]->data_offset > first.data_offset) {
                    source_stride = sorted[i + 1]->offset - first.offset;
                    destination_stride = sorted[i + 1]->data_offset - first.data_offset;
                    if (source_stride >= first.size && destination_stride >= first.size) {
                        run = 2;
                        while (i + run < group_end &&
                                sorted[i + run]->offset - sorted[i + run - 1]->offset == source_stride &&
                                sorted[i + run]->data_offset > sorted[i + run - 1]->data_offset &&
                                sorted[i + run]->data_offset - sorted[i + run - 1]->data_offset == destination_stride) {
                            ++run;
                        }
                    }
                }

                if (run > 1) {
                    ggml_backend_tensor_get_2d(first.tensor, base_ptr + first.data_offset,
                            first.offset, first.size, run, source_stride, destination_stride);
                } else {
                    tensors.push_back(first.tensor);
                    offsets.push_back(first.offset);
                    sizes.push_back(first.size);
                    data_offsets.push_back(first.data_offset);
                }
                i += run;
            }
        }
        ggml_backend_tensor_get_n(tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), tensors.size());
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};
