/* Spark's reply format: the checkpoint's template writes these tags around each call. */
#pragma once
#include "rad_builder.h"

namespace spark25_bf16 {

inline const RadChatFormat* spark_chat_format() {
    static const char* const breaks[] = {"<tool_call>"};
    static const char* const triggers[] = {"<tool_call>"};
    static const RadChatFormat f = [] {
        RadChatFormat x{};
        x.struct_size = sizeof(x);
        x.name = "plugin:spark25";
        x.reasoning_open = "<think>";
        x.reasoning_close = "</think>";
        x.reasoning_breaks = breaks;
        x.n_reasoning_breaks = 1;
        x.reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT;
        /* There is no delimiter between the call opener and the function name. */
        x.name_prefix = "<tool_call>";
        x.call_end = "</tool_call>";
        x.arg_prefix = "<arg_key>";
        x.arg_name_suffix = "</arg_key>";
        x.arg_value_prefix = "<arg_value>";
        x.arg_value_suffix = "</arg_value>";
        x.string_value = RAD_CHAT_VALUE_RAW;
        x.other_value = RAD_CHAT_VALUE_JSON;
        x.triggers = triggers;
        x.n_triggers = 1;
        return x;
    }();
    return &f;
}

}  /* namespace spark25_bf16 */
