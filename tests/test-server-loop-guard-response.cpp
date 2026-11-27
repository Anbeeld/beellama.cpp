#include "../tools/server/server-task.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    for (bool include_usage : {false, true}) {
        server_task_result_cmpl_final result{};
        result.stop = STOP_TYPE_EOS;
        result.include_usage = include_usage;
        result.reasoning_output_tokens = 12;
        result.visible_output_tokens = 4;
        result.n_decoded = 16;

        assert(!result.to_json_oaicompat_chat().contains("loop_guard"));
        for (const auto & chunk : result.to_json_oaicompat_chat_stream()) {
            assert(!chunk.contains("loop_guard"));
        }

        auto & event = result.loop_guard_event;
        event.triggered = true;
        event.region = "reasoning";
        event.detector = "ngram_dominance";
        event.action = "force-close";
        event.coverage = 768;
        event.interventions = 1;
        event.decoded_token_index = 12;

        const auto native = result.to_json_non_oaicompat();
        const auto chat = result.to_json_oaicompat_chat();
        if (!chat.contains("loop_guard")) {
            std::fprintf(stderr, "chat response is missing loop_guard telemetry\n");
            return 1;
        }
        assert(chat.at("loop_guard") == native.at("loop_guard"));
        assert(chat.at("loop_guard").at("action") == "force-close");
        assert(chat.at("usage").at("completion_tokens_details").at("reasoning_tokens") == 12);

        const auto chunks = result.to_json_oaicompat_chat_stream();
        int events = 0;
        for (const auto & chunk : chunks) {
            if (chunk.contains("loop_guard")) {
                ++events;
                assert(chunk.at("loop_guard") == native.at("loop_guard"));
                assert(chunk.at("choices").at(0).at("finish_reason") == "stop");
            }
        }
        assert(events == 1);
        if (include_usage) {
            assert(chunks.back().at("usage").at("completion_tokens_details").at("reasoning_tokens") == 12);
        }
    }
}
