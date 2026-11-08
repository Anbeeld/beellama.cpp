#include "../src/llama-io-write-host.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

// Exercise the real backend dispatch, but count transfers instead of measuring time.
struct transfer {
    const ggml_tensor * tensor;
    void * data;
    size_t offset;
    size_t size;
    size_t copies;
    size_t stride_tensor;
    size_t stride_data;
};

struct recording_buffer {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_type buft = {};
    ggml_backend_buffer_t buffer = nullptr;
    std::array<std::array<uint8_t, 512>, 3> storage = {};
    std::array<ggml_tensor *, 3> tensors = {};
    std::vector<transfer> scalar;
    std::vector<transfer> native_2d;
    std::vector<std::vector<transfer>> batches;

    static recording_buffer & self(ggml_backend_buffer_t buffer) {
        return *static_cast<recording_buffer *>(buffer->context);
    }

    static void copy(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
        GGML_ASSERT(offset <= ggml_nbytes(tensor));
        GGML_ASSERT(size <= ggml_nbytes(tensor) - offset);
        if (size != 0) {
            std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
        }
    }

    static void get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                    void * data, size_t offset, size_t size) {
        self(buffer).scalar.push_back({tensor, data, offset, size, 1, 0, 0});
        copy(tensor, data, offset, size);
    }

    static void get_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                       void * data, size_t offset, size_t size, size_t copies,
                       size_t stride_tensor, size_t stride_data) {
        GGML_ASSERT(copies > 1);
        GGML_ASSERT(size > 0);
        GGML_ASSERT(stride_tensor > 0);
        GGML_ASSERT(stride_data >= size);
        self(buffer).native_2d.push_back({tensor, data, offset, size, copies, stride_tensor, stride_data});
        for (size_t i = 0; i < copies; ++i) {
            copy(tensor, static_cast<uint8_t *>(data) + i * stride_data,
                 offset + i * stride_tensor, size);
        }
    }

    static void get_n(ggml_backend_buffer_t buffer, const ggml_tensor * const * tensors,
                      void * const * data, const size_t * offsets, const size_t * sizes, size_t count) {
        std::vector<transfer> batch;
        for (size_t i = 0; i < count; ++i) {
            batch.push_back({tensors[i], data[i], offsets[i], sizes[i], 1, 0, 0});
            copy(tensors[i], data[i], offsets[i], sizes[i]);
        }
        self(buffer).batches.push_back(batch);
    }

    recording_buffer(bool with_2d, bool with_batch) {
        ggml_init_params params = {};
        params.mem_size = 4 * ggml_tensor_overhead();
        params.no_alloc = true;
        ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);

        ggml_backend_buffer_i iface = {};
        iface.get_tensor = get;
        iface.get_tensor_2d = with_2d ? get_2d : nullptr;
        iface.get_tensor_n = with_batch ? get_n : nullptr;
        buffer = ggml_backend_buffer_init(&buft, iface, this, sizeof(storage));
        GGML_ASSERT(buffer != nullptr);
        for (size_t t = 0; t < tensors.size(); ++t) {
            for (size_t i = 0; i < storage[t].size(); ++i) {
                storage[t][i] = uint8_t((i * 37 + t * 83 + i / 7) & 255);
            }
            tensors[t] = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, storage[t].size());
            tensors[t]->buffer = buffer;
            tensors[t]->data = storage[t].data();
        }
    }

    ~recording_buffer() {
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
    }

    void reset() {
        scalar.clear();
        native_2d.clear();
        batches.clear();
    }
};

struct write_op {
    ggml_tensor * tensor;
    size_t offset;
    size_t size;
    std::vector<uint8_t> metadata;
};

struct writes {
    std::vector<write_op> ops;
    size_t size = 0;

    size_t tensor(ggml_tensor * tensor, size_t offset, size_t count) {
        const size_t destination = size;
        ops.push_back({tensor, offset, count, {}});
        size += count;
        return destination;
    }

    void metadata(size_t count, uint8_t seed) {
        std::vector<uint8_t> bytes(count);
        for (size_t i = 0; i < count; ++i) {
            bytes[i] = uint8_t(seed + i);
        }
        ops.push_back({nullptr, 0, count, bytes});
        size += count;
    }
};

// Independently serialize each operation immediately through the public API.
// Guard bytes also catch writes beyond the writer's exact output range.
static std::vector<uint8_t> serialize(recording_buffer & backend, const writes & input) {
    constexpr size_t guard = 16;
    std::vector<uint8_t> expected(input.size + 2 * guard, 0xa5);
    size_t position = guard;
    for (const auto & op : input.ops) {
        if (op.tensor) {
            ggml_backend_tensor_get(op.tensor, expected.data() + position, op.offset, op.size);
        } else if (op.size != 0) {
            std::memcpy(expected.data() + position, op.metadata.data(), op.size);
        }
        position += op.size;
    }
    backend.reset();

    std::vector<uint8_t> actual(expected.size(), 0xa5);
    {
        llama_io_write_host writer(actual.data() + guard, input.size);
        position = guard;
        for (const auto & op : input.ops) {
            if (op.tensor) {
                writer.write_tensor(op.tensor, op.offset, op.size);
            } else {
                // Supply a valid pointer even for a zero-length metadata write.
                const uint8_t empty = 0;
                writer.write(op.size ? op.metadata.data() : &empty, op.size);
                GGML_ASSERT(std::memcmp(actual.data() + position,
                                        expected.data() + position, op.size) == 0);
            }
            position += op.size;
            GGML_ASSERT(writer.n_bytes() == position - guard);
        }
    }
    GGML_ASSERT(actual == expected);
    return actual;
}

static void expect_run(const recording_buffer & backend, const std::vector<uint8_t> & output,
                       const ggml_tensor * tensor, size_t destination, size_t offset,
                       size_t size, size_t copies, size_t stride_tensor, size_t stride_data) {
    size_t matches = 0;
    for (const auto & call : backend.native_2d) {
        if (call.tensor == tensor && call.data == output.data() + 16 + destination &&
            call.offset == offset && call.size == size && call.copies == copies &&
            call.stride_tensor == stride_tensor && call.stride_data == stride_data) {
            ++matches;
        }
    }
    GGML_ASSERT(matches == 1);
}

static void test_interleaved(bool with_2d, bool with_batch) {
    recording_buffer backend(with_2d, with_batch);
    writes input;
    input.metadata(7, 0x10);
    input.tensor(backend.tensors[2], 9, 5); // An unmatched write must remain batched.
    size_t a = 0;
    size_t b = 0;
    for (size_t row = 0; row < 4; ++row) {
        input.metadata(5, uint8_t(0x40 + row));
        const size_t da = input.tensor(backend.tensors[0], 3 + row * 16, 4);
        const size_t db = input.tensor(backend.tensors[1], 7 + row * 24, 6);
        if (row == 0) {
            a = da;
            b = db;
        }
    }
    input.tensor(backend.tensors[2], 41, 7);
    input.metadata(9, 0xe0);
    const auto output = serialize(backend, input);

    if (with_2d) {
        // Issue 196: get_n must not hide the two strided runs, even when the
        // backend also provides a native batch hook and metadata interleaves them.
        GGML_ASSERT(backend.native_2d.size() == 2);
        expect_run(backend, output, backend.tensors[0], a, 3, 4, 4, 16, 15);
        expect_run(backend, output, backend.tensors[1], b, 7, 6, 4, 24, 15);
    } else {
        GGML_ASSERT(backend.native_2d.empty());
    }
    if (with_batch) {
        GGML_ASSERT(backend.batches.size() == 1);
        size_t singles = 0;
        for (const auto & batch : backend.batches) {
            for (const auto & call : batch) {
                if (with_2d) {
                    GGML_ASSERT(call.tensor == backend.tensors[2]);
                }
                if (call.tensor == backend.tensors[2]) {
                    GGML_ASSERT((call.offset == 9 && call.size == 5) ||
                                (call.offset == 41 && call.size == 7));
                    ++singles;
                }
            }
        }
        GGML_ASSERT(singles == 2);
    } else {
        GGML_ASSERT(backend.batches.empty());
        GGML_ASSERT(!backend.scalar.empty());
    }
}

static void test_broken_runs(bool with_2d, bool with_batch) {
    recording_buffer backend(with_2d, with_batch);
    writes input;
    input.metadata(3, 0x70);
    const size_t a0 = input.size;
    for (size_t offset : {size_t(0), size_t(16), size_t(32)}) {
        input.tensor(backend.tensors[0], offset, 4);
    }
    // The source stride changes, requiring another run of the same tensor.
    const size_t a1 = input.size;
    for (size_t offset : {size_t(80), size_t(96), size_t(112)}) {
        input.tensor(backend.tensors[0], offset, 4);
    }
    // A size change must also end the previous run.
    const size_t a2 = input.size;
    for (size_t offset : {size_t(160), size_t(176), size_t(192)}) {
        input.tensor(backend.tensors[0], offset, 8);
    }
    const size_t b0 = input.size;
    for (size_t offset : {size_t(0), size_t(16), size_t(32)}) {
        input.tensor(backend.tensors[1], offset, 5);
    }
    input.metadata(11, 0xa0); // Only the destination stride changes here.
    const size_t b1 = input.size;
    for (size_t offset : {size_t(48), size_t(64), size_t(80)}) {
        input.tensor(backend.tensors[1], offset, 5);
    }
    // Duplicate/descending source offsets cannot form a positive-stride run.
    for (size_t offset : {size_t(100), size_t(100), size_t(80), size_t(60)}) {
        input.tensor(backend.tensors[2], offset, 3);
    }
    input.tensor(backend.tensors[2], 512, 0);
    input.metadata(0, 0);
    input.tensor(backend.tensors[2], 12, 1);
    input.tensor(backend.tensors[2], 27, 2);
    input.metadata(4, 0xf0);
    const auto output = serialize(backend, input);
    if (with_2d) {
        expect_run(backend, output, backend.tensors[0], a0, 0, 4, 3, 16, 4);
        expect_run(backend, output, backend.tensors[0], a1, 80, 4, 3, 16, 4);
        expect_run(backend, output, backend.tensors[0], a2, 160, 8, 3, 16, 8);
        expect_run(backend, output, backend.tensors[1], b0, 0, 5, 3, 16, 5);
        expect_run(backend, output, backend.tensors[1], b1, 48, 5, 3, 16, 5);
        GGML_ASSERT(backend.native_2d.size() == 5);
    } else {
        GGML_ASSERT(backend.native_2d.empty());
    }
    if (!with_batch) {
        GGML_ASSERT(backend.batches.empty());
    }
}

static void test_views_and_buffers() {
    recording_buffer first(true, true);
    recording_buffer second(true, true);
    // A view resolves its backend through view_src, not its own buffer.
    auto * view = ggml_view_1d(first.ctx, first.tensors[0], 256, 64);
    GGML_ASSERT(view->buffer == nullptr && view->view_src == first.tensors[0]);
    writes input;
    for (size_t row = 0; row < 4; ++row) {
        input.tensor(view, row * 16, 4);
        input.metadata(3, uint8_t(row));
        input.tensor(second.tensors[0], row * 24, 8);
    }
    serialize(first, input);
    GGML_ASSERT(first.native_2d.size() == 1 && second.native_2d.size() == 1);
    GGML_ASSERT(first.native_2d[0].copies == 4 && second.native_2d[0].copies == 4);

    first.reset();
    writes overlapping;
    overlapping.tensor(first.tensors[0], 0, 8);
    overlapping.tensor(first.tensors[0], 4, 8);
    serialize(first, overlapping);
    GGML_ASSERT(first.native_2d.empty());
    GGML_ASSERT(first.batches.size() == 1 && first.batches[0].size() == 2);
}

static void test_no_tensor_writes() {
    recording_buffer backend(true, true);
    writes empty;
    serialize(backend, empty);
    GGML_ASSERT(backend.scalar.empty() && backend.native_2d.empty() && backend.batches.empty());
    writes metadata;
    metadata.metadata(13, 0x21);
    metadata.metadata(0, 0);
    metadata.metadata(19, 0x92);
    serialize(backend, metadata);
    GGML_ASSERT(backend.scalar.empty() && backend.native_2d.empty() && backend.batches.empty());
}

int main() {
    test_no_tensor_writes();
    test_views_and_buffers();
    test_interleaved(false, true);
    test_interleaved(false, false);
    test_interleaved(true, false);
    test_interleaved(true, true);
    test_broken_runs(false, true);
    test_broken_runs(false, false);
    test_broken_runs(true, false);
    test_broken_runs(true, true);
    std::puts("test-io-write-host: OK");
    return 0;
}
