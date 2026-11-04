/* tests/stream_test.cpp -- a streamed answer is the buffered answer, in every API's shape.
 *
 * A streaming client assembles the reply a buffered client receives. OpenAI's chunks, Anthropic's
 * content blocks and the Responses API's output items are three state machines over one shared
 * reader (ReplyStream), and the server holds back bytes that may still turn out to be a stop
 * string or half a character. Each stream here runs with the scheduler in manual mode
 * (server_fakes.h): the test hands the reply over in pieces, one drain per piece, so it decides
 * every boundary a drain ends on. Each reply is cut in two at every byte and also sent one byte at
 * a time, and the assembly must equal the buffered body built by the same server.
 */
#include "server_fakes.h"
#include "work_counter.h"

#include <cstdio>
#include <cstring>

namespace {

/* Streams `reply` through `f` with its scheduler in manual mode: the reply in pieces ending at
 * `cuts`, one drain after each, then the sink finished with `fin` and the stream run to its end.
 * Returns the body a client read. */
std::string stream_pieces(Fixture& f, const char* path, const std::string& body,
                          const std::string& reply, const std::vector<size_t>& cuts, Finish fin) {
    f.sched.manual = true;
    HttpResponse r = f.post(path, body);
    f.sched.manual = false;
    CHECK_EQ(r.status, 200);
    if (!r.streaming()) return r.body;
    const size_t i = f.sched.seen.size() - 1;
    std::string out, c;
    r.next(c);
    out += c;
    size_t at = 0;
    std::vector<size_t> ends = cuts;
    ends.push_back(reply.size());
    for (size_t e : ends) {
        if (e <= at) continue;
        f.sched.push(i, reply.substr(at, e - at));
        c.clear();
        r.next(c);
        out += c;
        at = e;
    }
    f.sched.finish(i, fin);
    for (int k = 0; k < 64; ++k) {
        c.clear();
        const bool more = r.next(c);
        out += c;
        if (!more) break;
    }
    r.on_close(true);
    return out;
}

/* Every way the test cuts a reply: in two at each byte, and into single bytes. */
std::vector<std::vector<size_t>> all_cuts(size_t n) {
    std::vector<std::vector<size_t>> out;
    out.push_back({});
    for (size_t k = 1; k < n; ++k) out.push_back({ k });
    std::vector<size_t> bytes;
    for (size_t k = 1; k < n; ++k) bytes.push_back(k);
    out.push_back(bytes);
    return out;
}

bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

std::string describe(const std::vector<size_t>& cuts) {
    if (cuts.size() > 1) return "one byte a drain";
    return cuts.empty() ? "whole" : "cut at " + std::to_string(cuts[0]);
}

/* ------------------------------------------------------------------ OpenAI's chunks */

struct ChatReply {
    std::string content, reasoning, finish;
    std::vector<std::pair<std::string, std::string>> calls;   /* name, arguments */
    bool operator==(const ChatReply& o) const {
        return content == o.content && reasoning == o.reasoning && finish == o.finish &&
               calls == o.calls;
    }
};

ChatReply chat_stream(const std::string& body) {
    ChatReply a;
    int bad = 0;
    for (const std::string& p : sse_payloads(body, &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        if (j["choices"].empty()) continue;
        const json& ch = j["choices"][0];
        if (!ch["finish_reason"].is_null()) a.finish = ch["finish_reason"].get<std::string>();
        const json& d = ch["delta"];
        if (d.contains("content") && d["content"].is_string())
            a.content += d["content"].get<std::string>();
        if (d.contains("reasoning_content")) a.reasoning += d["reasoning_content"].get<std::string>();
        if (!d.contains("tool_calls")) continue;
        for (const json& tc : d["tool_calls"]) {
            const size_t k = tc["index"].get<size_t>();
            if (k >= a.calls.size()) a.calls.resize(k + 1);
            const json& fn = tc["function"];
            if (fn.contains("name")) a.calls[k].first = fn["name"].get<std::string>();
            if (fn.contains("arguments")) a.calls[k].second += fn["arguments"].get<std::string>();
        }
    }
    CHECK_EQ(bad, 0);
    return a;
}

ChatReply chat_buffered(const std::string& body) {
    ChatReply a;
    const json c = json::parse(body)["choices"][0];
    a.finish = c["finish_reason"].get<std::string>();
    const json& m = c["message"];
    if (m["content"].is_string()) a.content = m["content"].get<std::string>();
    if (m.contains("reasoning_content")) a.reasoning = m["reasoning_content"].get<std::string>();
    if (m.contains("tool_calls"))
        for (const json& tc : m["tool_calls"])
            a.calls.emplace_back(tc["function"]["name"].get<std::string>(),
                                 tc["function"]["arguments"].get<std::string>());
    return a;
}

/* ------------------------------------------------------------------ Anthropic's blocks */

struct Block {
    std::string type, text, name;   /* text: a text or thinking block's, or a tool_use's input */
    bool operator==(const Block& o) const {
        return type == o.type && text == o.text && name == o.name;
    }
};
struct AnthropicReply {
    std::vector<Block> blocks;
    std::string stop_reason;
    json stop_sequence, usage;
};

/* Folds the events into blocks, checking the protocol on the way: blocks open in index order,
 * deltas go to the open block and match its type, each block is stopped before the next opens,
 * and the stream ends with message_delta then message_stop. A tool_use's input is its
 * partial_json concatenated, re-serialised when it parses (the buffered body carries an object). */
AnthropicReply anthropic_stream(const std::string& body) {
    AnthropicReply a;
    bool open = false, stopped = false;
    std::string partial;
    std::vector<std::string> order;
    for (const json& ev : sse_events(body)) {
        const std::string t = ev.value("type", "");
        order.push_back(t);
        if (t == "content_block_start") {
            CHECK(!open);
            CHECK_EQ(ev["index"].get<size_t>(), a.blocks.size());
            const json& cb = ev["content_block"];
            Block b;
            b.type = cb["type"].get<std::string>();
            if (b.type == "tool_use") b.name = cb["name"].get<std::string>();
            a.blocks.push_back(b);
            partial.clear();
            open = true;
        } else if (t == "content_block_delta") {
            CHECK(open);
            CHECK_EQ(ev["index"].get<size_t>() + 1, a.blocks.size());
            const json& d = ev["delta"];
            const std::string dt = d["type"].get<std::string>();
            Block& b = a.blocks.back();
            if (dt == "thinking_delta") { CHECK_EQ(b.type, std::string("thinking")); b.text += d["thinking"].get<std::string>(); }
            else if (dt == "text_delta") { CHECK_EQ(b.type, std::string("text")); b.text += d["text"].get<std::string>(); }
            else if (dt == "input_json_delta") { CHECK_EQ(b.type, std::string("tool_use")); partial += d["partial_json"].get<std::string>(); }
            else CHECK(dt == "signature_delta" && b.type == "thinking");
        } else if (t == "content_block_stop") {
            CHECK(open);
            CHECK_EQ(ev["index"].get<size_t>() + 1, a.blocks.size());
            open = false;
            Block& b = a.blocks.back();
            if (b.type == "tool_use") {
                const json in = json::parse(partial, nullptr, false);
                b.text = in.is_discarded() ? partial : in.dump();
            }
        } else if (t == "message_delta") {
            CHECK(!open);
            a.stop_reason = ev["delta"]["stop_reason"].get<std::string>();
            a.stop_sequence = ev["delta"]["stop_sequence"];
            a.usage = ev["usage"];
        } else if (t == "message_stop") {
            stopped = true;
        }
    }
    CHECK(!order.empty() && order.front() == "message_start");
    CHECK(stopped && order.back() == "message_stop");
    return a;
}

AnthropicReply anthropic_buffered(const std::string& body) {
    AnthropicReply a;
    const json m = json::parse(body);
    for (const json& cb : m["content"]) {
        Block b;
        b.type = cb["type"].get<std::string>();
        if (b.type == "text") b.text = cb["text"].get<std::string>();
        else if (b.type == "thinking") b.text = cb["thinking"].get<std::string>();
        else { b.name = cb["name"].get<std::string>(); b.text = cb["input"].dump(); }
        a.blocks.push_back(b);
    }
    a.stop_reason = m["stop_reason"].get<std::string>();
    a.stop_sequence = m["stop_sequence"];
    a.usage = m["usage"];
    return a;
}

/* ------------------------------------------------------------------ the Responses API's items */

struct Item {
    std::string type, status, text, name, arguments;
    bool operator==(const Item& o) const {
        return type == o.type && status == o.status && text == o.text && name == o.name &&
               arguments == o.arguments;
    }
};

std::vector<Item> items_of(const json& output) {
    std::vector<Item> v;
    for (const json& o : output) {
        Item it;
        it.type = o["type"].get<std::string>();
        it.status = o.value("status", "");
        if (it.type == "message" || it.type == "reasoning")
            for (const json& p : o["content"]) it.text += p.value("text", "");
        if (it.type == "function_call") {
            it.name = o["name"].get<std::string>();
            it.arguments = o["arguments"].get<std::string>();
        }
        v.push_back(it);
    }
    return v;
}

/* The terminal response object of a Responses stream, after checking the stream itself: sequence
 * numbers count from 0 without a gap; every item added is done once at the same output_index, in
 * order; each done item's text or arguments is its deltas concatenated; and the terminal output
 * is exactly the done items. */
json responses_stream(const std::string& body) {
    json terminal;
    int64_t seq = 0;
    std::vector<json> done;
    std::vector<std::string> deltas;
    int64_t open = -1;
    for (const json& ev : sse_events(body)) {
        CHECK_EQ(ev.value("sequence_number", (int64_t)-1), seq);
        ++seq;
        const std::string t = ev.value("type", "");
        if (t == "response.output_item.added") {
            CHECK_EQ(open, (int64_t)-1);
            open = ev["output_index"].get<int64_t>();
            CHECK_EQ(open, (int64_t)done.size());
            deltas.emplace_back();
        } else if (t == "response.output_text.delta" || t == "response.reasoning_text.delta" ||
                   t == "response.function_call_arguments.delta") {
            CHECK_EQ(ev["output_index"].get<int64_t>(), open);
            if (open < 0) continue;
            deltas.back() += ev["delta"].get<std::string>();
        } else if (t == "response.output_item.done") {
            CHECK_EQ(ev["output_index"].get<int64_t>(), open);
            if (open < 0) continue;
            const json& item = ev["item"];
            const std::vector<Item> one = items_of(json::array({ item }));
            CHECK_EQ(one[0].type == "function_call" ? one[0].arguments : one[0].text, deltas.back());
            done.push_back(item);
            open = -1;
        } else if (t == "response.completed" || t == "response.incomplete") {
            terminal = ev["response"];
        }
    }
    CHECK_EQ(open, (int64_t)-1);
    CHECK(terminal.is_object());
    if (!terminal.is_object()) return json{ { "output", json::array() }, { "status", "" } };
    CHECK(items_of(terminal["output"]) == items_of(json(done)));
    return terminal;
}

/* ------------------------------------------------------------------ the replies */

const json& bash_tools() {
    static const json t = json::parse(R"([{"type":"function","function":{"name":"bash",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},
                      "required":["command"]}}}])");
    return t;
}

const std::string kCall1 = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                           "</function>\n</tool_call>\n";
const std::string kCall2 = "<tool_call>\n<function=bash>\n<parameter=command>\npwd\n</parameter>\n"
                           "</function>\n</tool_call>\n";
const std::string kOpen = "<tool_call>\n<function=bash>\n<parameter=command>\ncat /a";

/* The buffered text of a reply that ended inside a call, as the stream shows it: without the
 * call's bytes and the line break before them, which the reader takes as the call's separator
 * once the call has opened. The buffered body restores both as text. */
void drop_open_call(std::string& text) {
    CHECK(ends_with(text, kOpen));
    if (!ends_with(text, kOpen)) return;
    text.resize(text.size() - kOpen.size());
    while (!text.empty() && text.back() == '\n') text.pop_back();
}

/* Replies in Qwen's format, read by the real parser: an answer after reasoning, closed calls, a
 * reply that ends inside a call, and two cut short by max_tokens. */
struct Reply {
    const char* name;
    std::string text;
    Finish      fin;
    bool        open_call;
};
const std::vector<Reply>& replies() {
    static const std::vector<Reply> r = {
        { "an answer after reasoning", "Two and two.\n</think>\n\nIt is 4.", Finish::Stop, false },
        { "one call", "Look first.\n</think>\n\nChecking.\n" + kCall1, Finish::Stop, false },
        { "two calls", "Plan.\n</think>\n\n" + kCall1 + kCall2, Finish::Stop, false },
        { "a reply ending inside a call", "Look first.\n</think>\n\nChecking.\n" + kCall1 + kOpen,
          Finish::Stop, true },
        { "cut while thinking", "Still thinking about i", Finish::Length, false },
        { "cut in the answer", "Done.\n</think>\n\nThe answ", Finish::Length, false },
    };
    return r;
}

/* max_tokens: the reply's length when it is cut there (the threaded fake stops at the cap), else
 * room to spare. */
int64_t cap_for(const Reply& r) { return r.fin == Finish::Length ? (int64_t)r.text.size() : 1000; }

}  /* namespace */

/* ================================================================== OpenAI's chat stream */

/* The chat stream assembles to the buffered message at every drain boundary: reasoning, content,
 * each call's name and arguments, and the finish reason. A reply ending inside a call is the one
 * difference the API has: its call's deltas went out before the reply ended, so the stream shows
 * them under tool_calls and the buffered body as text; both finish "stop", which is what tells a
 * client not to run it. */
TEST(the_chat_stream_assembles_to_the_buffered_message_at_every_boundary) {
    Fixture f;
    f.tmpl.real_parser = qwen_reply_parser(bash_tools());
    REQUIRE(f.tmpl.real_parser != nullptr);
    for (const Reply& rp : replies()) {
        const std::string body = R"({"messages":[{"role":"user","content":"hi"}],"tools":)" +
                                 bash_tools().dump() + ",\"max_tokens\":" +
                                 std::to_string(cap_for(rp));
        f.sched.reply = rp.text;
        HttpResponse b = f.post("/v1/chat/completions", body + "}");
        REQUIRE(b.status == 200);
        f.sched.join_all();
        ChatReply want = chat_buffered(b.body);
        if (rp.open_call) {
            CHECK_EQ(want.finish, std::string("stop"));
            drop_open_call(want.content);
        }
        for (const auto& cuts : all_cuts(rp.text.size())) {
            ChatReply got = chat_stream(stream_pieces(
                f, "/v1/chat/completions", body + ",\"stream\":true}", rp.text, cuts, rp.fin));
            if (rp.open_call) {
                /* The open call went out named, with arguments that are no whole JSON value. */
                const bool open = !got.calls.empty() && got.calls.back().first == "bash" &&
                                  !json::accept(got.calls.back().second);
                CHECK(open);
                if (open) got.calls.pop_back();
            }
            if (got == want) continue;
            CHECK(got == want);
            fprintf(stderr, "    %s, %s: content \"%s\" vs \"%s\", finish %s vs %s\n", rp.name,
                    describe(cuts).c_str(), got.content.c_str(), want.content.c_str(),
                    got.finish.c_str(), want.finish.c_str());
            break;
        }
    }
}

/* ================================================================== Anthropic's stream */

/* Anthropic's stream folds into the buffered message's blocks (thinking, text, tool_use with its
 * input), stop_reason and stop_sequence, at every drain boundary. A reply ending inside a call
 * leaves a tool_use block whose input does not parse where the buffered body has text; both say
 * end_turn, so neither asks the client to run a tool. */
TEST(the_messages_stream_assembles_to_the_buffered_message_at_every_boundary) {
    Fixture f;
    f.tmpl.real_parser = qwen_reply_parser(bash_tools());
    REQUIRE(f.tmpl.real_parser != nullptr);
    const std::string tools = R"([{"name":"bash","input_schema":{"type":"object",
        "properties":{"command":{"type":"string"}},"required":["command"]}}])";
    for (const Reply& rp : replies()) {
        const std::string body = R"({"model":"m","messages":[{"role":"user","content":"hi"}],)"
                                  R"("tools":)" + tools + ",\"max_tokens\":" +
                                  std::to_string(cap_for(rp));
        f.sched.reply = rp.text;
        HttpResponse b = f.post("/v1/messages", body + "}");
        REQUIRE(b.status == 200);
        f.sched.join_all();
        AnthropicReply want = anthropic_buffered(b.body);
        CHECK_EQ(want.stop_reason, std::string(rp.fin == Finish::Length ? "max_tokens"
                                               : rp.open_call || !has(rp.text, "<tool_call>")
                                                   ? "end_turn" : "tool_use"));
        if (rp.open_call) {
            /* thinking, the text with the open call's bytes at its end, the closed call */
            REQUIRE(want.blocks.size() == 3 && want.blocks[1].type == "text");
            std::string& text = want.blocks[1].text;
            drop_open_call(text);
        }
        for (const auto& cuts : all_cuts(rp.text.size())) {
            AnthropicReply got = anthropic_stream(stream_pieces(
                f, "/v1/messages", body + ",\"stream\":true}", rp.text, cuts, rp.fin));
            if (rp.open_call) {
                /* The open call's block went out, named, with input that does not parse. */
                const bool open = !got.blocks.empty() && got.blocks.back().type == "tool_use" &&
                                  got.blocks.back().name == "bash" &&
                                  !json::accept(got.blocks.back().text);
                CHECK(open);
                if (open) got.blocks.pop_back();
            }
            const bool same = got.blocks == want.blocks && got.stop_reason == want.stop_reason &&
                              got.stop_sequence == want.stop_sequence;
            if (same) continue;
            CHECK(same);
            fprintf(stderr, "    %s, %s: %zu blocks vs %zu, stop_reason %s vs %s\n", rp.name,
                    describe(cuts).c_str(), got.blocks.size(), want.blocks.size(),
                    got.stop_reason.c_str(), want.stop_reason.c_str());
            for (size_t k = 0; k < got.blocks.size(); ++k)
                fprintf(stderr, "      got %s [%s]\n", got.blocks[k].type.c_str(),
                        got.blocks[k].text.c_str());
            break;
        }
    }
}

/* ================================================================== the Responses stream */

/* response.completed carries the buffered response's items (type, status, text, call name and
 * arguments) at every drain boundary, and the events agree with it (responses_stream). A reply
 * ending inside a call is the one difference: the stream already announced the call, so its item
 * closes with status "incomplete", never "completed" with arguments that do not parse; the
 * buffered body has the call's text in its message. */
TEST(the_responses_stream_assembles_to_the_buffered_response_at_every_boundary) {
    Fixture f;
    f.tmpl.real_parser = qwen_reply_parser(bash_tools());
    REQUIRE(f.tmpl.real_parser != nullptr);
    const std::string tools = R"([{"type":"function","name":"bash","parameters":{"type":"object",
        "properties":{"command":{"type":"string"}},"required":["command"]}}])";
    for (const Reply& rp : replies()) {
        const std::string body = R"({"model":"m","input":"hi","tools":)" + tools +
                                 ",\"max_output_tokens\":" + std::to_string(cap_for(rp));
        f.sched.reply = rp.text;
        HttpResponse b = f.post("/v1/responses", body + "}");
        REQUIRE(b.status == 200);
        f.sched.join_all();
        const json buffered = json::parse(b.body);
        std::vector<Item> want = items_of(buffered["output"]);
        CHECK_EQ(buffered["status"].get<std::string>(),
                 std::string(rp.fin == Finish::Length ? "incomplete" : "completed"));
        if (rp.open_call) {
            /* reasoning, the message with the open call's bytes at its end, the closed call */
            REQUIRE(want.size() == 3 && want[1].type == "message");
            std::string& text = want[1].text;
            drop_open_call(text);
        }
        for (const auto& cuts : all_cuts(rp.text.size())) {
            const json got = responses_stream(stream_pieces(
                f, "/v1/responses", body + ",\"stream\":true}", rp.text, cuts, rp.fin));
            std::vector<Item> items = items_of(got["output"]);
            for (const Item& it : items)
                if (it.type == "function_call" && it.status == "completed")
                    CHECK(json::accept(it.arguments));
            if (rp.open_call) {
                /* The open call's item, already announced, closes incomplete. */
                const bool open = !items.empty() && items.back().type == "function_call" &&
                                  items.back().name == "bash" &&
                                  items.back().status == "incomplete" &&
                                  !json::accept(items.back().arguments);
                CHECK(open);
                if (open) items.pop_back();
            }
            const bool same = items == want && got["status"] == buffered["status"];
            if (same) continue;
            CHECK(same);
            fprintf(stderr, "    %s, %s: %zu items vs %zu\n", rp.name, describe(cuts).c_str(),
                    items.size(), want.size());
            for (const Item& it : items)
                fprintf(stderr, "      got %s %s [%s%s]\n", it.type.c_str(), it.status.c_str(),
                        it.text.c_str(), it.arguments.c_str());
            break;
        }
    }
}

/* With an assistant prefill, Anthropic's API answers only what follows it: a reply that repeats
 * the prefill loses it, and one that starts like it and then differs keeps every byte, buffered
 * and streamed alike at every drain boundary. The stream sees the reply a piece at a time, so it
 * cannot drop a matching start until it knows the whole prefill follows. */
TEST(a_prefilled_messages_stream_drops_only_a_repeated_prefill) {
    Fixture f;
    const std::string body = R"({"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},
                                {"role":"assistant","content":"The"}])";
    for (const char* reply : { "The cat sat", "Tea time", "Th", "T", "The", "Other" }) {
        f.sched.reply = reply;
        const AnthropicReply want = anthropic_buffered(f.post("/v1/messages", body + "}").body);
        f.sched.join_all();
        for (const auto& cuts : all_cuts(std::strlen(reply))) {
            const AnthropicReply got = anthropic_stream(stream_pieces(
                f, "/v1/messages", body + ",\"stream\":true}", reply, cuts, Finish::Stop));
            if (got.blocks == want.blocks) continue;
            CHECK(got.blocks == want.blocks);
            fprintf(stderr, "    \"%s\", %s: %zu blocks [%s] vs %zu [%s]\n", reply,
                    describe(cuts).c_str(), got.blocks.size(),
                    got.blocks.empty() ? "" : got.blocks[0].text.c_str(), want.blocks.size(),
                    want.blocks.empty() ? "" : want.blocks[0].text.c_str());
            break;
        }
    }
}

/* ================================================================== hold-back at every boundary */

namespace {

/* The text and finish of a body in one endpoint's shape, buffered or streamed. */
struct Text {
    std::string text, finish;
};

Text text_buffered(const std::string& path, const std::string& body) {
    const json j = json::parse(body);
    if (path == "/completion") return { j["content"].get<std::string>(), j["stop_type"].get<std::string>() };
    const json& c = j["choices"][0];
    const json& t = path == "/v1/chat/completions" ? c["message"]["content"] : c["text"];
    return { t.is_string() ? t.get<std::string>() : std::string(), c["finish_reason"].get<std::string>() };
}

Text text_streamed(const std::string& path, const std::string& body) {
    Text t;
    int bad = 0;
    for (const std::string& p : sse_payloads(body, &bad)) {
        if (p == "[DONE]") continue;
        const json j = json::parse(p);
        if (path == "/completion") {
            t.text += j.value("content", "");
            if (j.value("stop", false)) t.finish = j["stop_type"].get<std::string>();
            continue;
        }
        if (j["choices"].empty()) continue;
        const json& c = j["choices"][0];
        if (!c["finish_reason"].is_null()) t.finish = c["finish_reason"].get<std::string>();
        if (path == "/v1/chat/completions") {
            if (c["delta"].contains("content") && c["delta"]["content"].is_string())
                t.text += c["delta"]["content"].get<std::string>();
        } else {
            t.text += c["text"].get<std::string>();
        }
    }
    CHECK_EQ(bad, 0);
    return t;
}

}  /* namespace */

/* What the server holds back while a reply runs (a tail that may still become a stop string, an
 * unfinished character, a half-written escape) and releases at the end: for every split into two
 * drains and one byte a drain, the streamed text of chat, completions and llama-server's native
 * completion is the buffered text, and the finish agrees. A stop string's bytes never reach the
 * wire, or the concatenation would hold them. On chat a reply that starts inside its reasoning
 * block applies its stop strings only after the closer. */
TEST(a_streamed_text_is_the_buffered_text_at_every_drain_boundary) {
    struct Case {
        const char* name;
        std::string reply, stop;
        bool        think;   /* the reply starts inside a block `|` closes */
    };
    const Case cases[] = {
        { "a stop string", "abENDcd", "END", false },
        { "a stop string's prefix that never completes", "abENxEN", "END", false },
        { "a stop deferred past the reasoning closer", "aENDb|cENDd", "END", true },
        { "a two-byte character", "x\xC3\xA9y\xC3\xA9", "", false },
        { "characters around a stop", "\xC3\xA9STOP\xC3\xA9", "STOP", false },
        { "escapes", "q\\u12\\\\\"x\\", "", false },
        { "a stop longer than the reply's tail", "abcZZ", "ZZZ", false },
        { "a tail that stops being a stop's start", "abSTxSTOPy", "STOP", false },
        { "a stop's start that starts it again", "xSTASTARy", "STAR", false },
    };
    for (const char* path : { "/v1/chat/completions", "/v1/completions", "/completion" }) {
        for (const Case& c : cases) {
            Fixture f;
            if (c.think) {
                f.tmpl.think_close = "|";
                f.tmpl.think_starts_open = true;
            }
            const std::string stop = c.stop.empty() ? "" : ",\"stop\":[\"" + c.stop + "\"]";
            const std::string body = std::string(path) == "/v1/chat/completions"
                ? R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1000)" + stop
                : R"({"prompt":"hi","max_tokens":1000)" + stop;
            f.sched.reply = c.reply;
            HttpResponse b = f.post(path, body + "}");
            REQUIRE(b.status == 200);
            f.sched.join_all();
            const Text want = text_buffered(path, b.body);
            for (const auto& cuts : all_cuts(c.reply.size())) {
                const Text got = text_streamed(path, stream_pieces(f, path, body + ",\"stream\":true}",
                                                                   c.reply, cuts, Finish::Stop));
                if (got.text == want.text && got.finish == want.finish) continue;
                CHECK_EQ(got.text, want.text);
                CHECK_EQ(got.finish, want.finish);
                fprintf(stderr, "    %s %s, %s\n", path, c.name, describe(cuts).c_str());
                break;
            }
        }
    }
}

/* A running choice holds back only a tail that may still become a stop string: what precedes it is
 * on the wire after the drain that produced it, on chat, completions and llama-server's native
 * completion. The hold-back this replaced kept the longest stop string less a byte off the wire
 * whatever the tail was, so with a ten-byte stop every delta trailed by nine bytes and a short
 * reply showed nothing before its end. */
TEST(a_running_choice_holds_back_only_what_may_still_become_a_stop_string) {
    struct Step {
        const char* piece;   /* the drain's new bytes */
        const char* wire;    /* everything on the wire after it */
    };
    const Step steps[] = {
        { "Hello", "Hello" },
        { " ENDOF", "Hello " },              /* "ENDOF" may still become "ENDOFREPLY" */
        { "X\n", "Hello ENDOFX" },           /* it cannot now, but "\n" may become "\n\n" */
        { "ENDO", "Hello ENDOFX\n" },        /* "\nE" cannot, "ENDO" may become "ENDOFREPLY" */
        { "FREPLY tail", "Hello ENDOFX\n" }, /* the stop is cut, and nothing of it went out */
    };
    for (const char* path : { "/v1/chat/completions", "/v1/completions", "/completion" }) {
        Fixture f;
        const std::string stop = R"(,"stop":["ENDOFREPLY","\n\n"],"stream":true})";
        const std::string body = std::string(path) == "/v1/chat/completions"
            ? R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":1000)" + stop
            : R"({"prompt":"hi","max_tokens":1000)" + stop;
        f.sched.manual = true;
        HttpResponse r = f.post(path, body);
        f.sched.manual = false;
        REQUIRE(r.status == 200);
        REQUIRE(r.streaming());
        const size_t i = f.sched.seen.size() - 1;
        std::string out, c;
        r.next(c);
        out += c;
        for (const Step& s : steps) {
            f.sched.push(i, s.piece);
            c.clear();
            r.next(c);
            out += c;
            const Text got = text_streamed(path, out);
            if (got.text == s.wire) continue;
            CHECK_EQ(got.text, std::string(s.wire));
            fprintf(stderr, "    %s, after \"%s\"\n", path, s.piece);
        }
        f.sched.finish(i, Finish::Stop);
        for (int k = 0; k < 64; ++k) {
            c.clear();
            const bool more = r.next(c);
            out += c;
            if (!more) break;
        }
        r.on_close(true);
        const Text all = text_streamed(path, out);
        CHECK_EQ(all.text, std::string("Hello ENDOFX\n"));
        CHECK_EQ(all.finish, std::string(std::string(path) == "/completion" ? "word" : "stop"));
    }
}

namespace {

/* The work of 24 drains of a streamed completion, in radtest::Work's unit, each after a short
 * piece of text, once the text is longer than any stop string: what each drain holds back is then
 * looked for in as many tails as the longest stop string is long. 20 pieces of 300 bytes, each
 * the next byte of `fill`, come first. Also checks the text arrives. */
uint64_t drains_work(const std::string& stops, const std::string& fill = "the quick brown ") {
    Fixture f;
    f.sched.manual = true;
    HttpResponse r = f.post("/v1/completions", R"({"prompt":"hi","max_tokens":100000,"stream":true,)"
                                               R"("stop":)" + stops + "}");
    f.sched.manual = false;
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());
    if (r.status != 200 || !r.streaming()) return 0;
    const size_t i = f.sched.seen.size() - 1;
    std::string out, c;
    r.next(c);
    out += c;
    std::string sent;
    for (int k = 0; k < 20; ++k) {
        const std::string part(300, fill[(size_t)k % fill.size()]);
        sent += part;
        f.sched.push(i, part);
        c.clear();
        r.next(c);
        out += c;
    }
    uint64_t w = 0;
    for (int k = 0; k < 24; ++k) {
        const std::string piece = "word " + std::to_string(k) + ", ";
        sent += piece;
        f.sched.push(i, piece);
        c.clear();
        w += radtest::work().of([&] { r.next(c); });
        out += c;
    }
    f.sched.finish(i, Finish::Stop);
    for (int k = 0; k < 64; ++k) {
        c.clear();
        const bool more = r.next(c);
        out += c;
        if (!more) break;
    }
    r.on_close(true);
    CHECK_EQ(text_streamed("/v1/completions", out).text, sent);
    return w;
}

}  /* namespace */

/* A request may send 64 stop strings of 4 KiB, and a running choice's hold-back looks at the tails
 * of what it has not sent, up to the longest stop string, on each drain. A tail is compared only
 * where it starts with a byte some stop string starts with, so stop strings the text cannot start
 * cost a drain a lookup per tail: such a drain costs about as much as one with a single two-byte
 * stop. Comparing every tail with every stop string, 262 080 calls a drain, cost three hundred
 * times. */
TEST(a_drain_looks_for_long_stop_strings_only_where_they_can_start) {
    json many = json::array();
    for (int k = 0; k < 64; ++k) many.push_back("~" + std::string(4095, (char)('a' + k % 26)));
    const uint64_t one = drains_work(R"(["~~"])"), all = drains_work(many.dump());
    std::fprintf(stderr, "    one stop %llu, 64 of 4 KiB %llu %s, ratio %.2f\n",
                 (unsigned long long)one, (unsigned long long)all, radtest::work().unit(),
                 (double)all / (double)one);
    CHECK((double)all / (double)one < (radtest::work().exact() ? 20.0 : 40.0));
}

/* The client sets the stop strings and steers the text: 64 stop strings of 4 KiB that all begin
 * with 4093 'a', and a reply that is thousands of 'a' and then something else, matches them for
 * as long as there are 'a' and then no more, at every one of the 4095 tails of every one of the
 * 64 strings, which is 500 million bytes compared on a drain with no bound. The comparisons share
 * a budget past which a drain holds back all it has not sent, so the cost of a drain is bounded
 * whatever the text. */
TEST(a_drain_that_almost_matches_long_stop_strings_costs_a_bounded_amount) {
    json many = json::array();
    for (int k = 0; k < 64; ++k)
        many.push_back(std::string(4093, 'a') + (char)('0' + k / 8) + (char)('0' + k % 8));
    const uint64_t one = drains_work(R"(["~~"])");
    const uint64_t all = drains_work(many.dump(), "aaaaaaaaaaaaaaaaaaax");
    std::fprintf(stderr, "    one stop %llu, 64 of 4 KiB almost matched %llu %s, ratio %.2f\n",
                 (unsigned long long)one, (unsigned long long)all, radtest::work().unit(),
                 (double)all / (double)one);
    CHECK((double)all / (double)one < (radtest::work().exact() ? 40.0 : 80.0));
}

/* ================================================================== Anthropic's stop reasons */

/* Anthropic's stop_reason and stop_sequence say why a reply ended, buffered and streamed alike: a
 * client stop sequence is stop_sequence with the word, the format's own end-of-turn marker is
 * end_turn (no client asked for it), max_tokens is max_tokens, and a complete tool call is
 * tool_use. usage splits the prompt into what was computed and what the cache served (the fake
 * reports 4 cached), and counts the tokens generated. */
TEST(anthropic_stop_reasons_and_usage_say_why_a_reply_ended_and_what_it_cost) {
    Fixture f;
    f.tmpl.additional_stops = { "<|end|>" };
    struct Case {
        std::string reply, extra;
        int         max_tokens;
        const char* reason;
        json        sequence;
        int64_t     out;
    };
    const Case cases[] = {
        { "abENDcd", R"(,"stop_sequences":["END"])", 64, "stop_sequence", "END", -1 },
        { "ab<|end|>cd", "", 64, "end_turn", nullptr, -1 },
        { "abcdefgh", "", 3, "max_tokens", nullptr, 3 },
        { R"(TOOL:get:{"c":1})", R"(,"tools":[{"name":"get","input_schema":{"type":"object"}}])",
          64, "tool_use", nullptr, 16 },
        { "hello", "", 64, "end_turn", nullptr, 5 },
    };
    for (const Case& c : cases) {
        const std::string body = R"({"model":"m","messages":[{"role":"user","content":"hi"}],)"
                                 R"("max_tokens":)" + std::to_string(c.max_tokens) + c.extra;
        const int64_t n_prompt = json::parse(f.post("/v1/messages/count_tokens", body + "}").body)
                                     .value("input_tokens", (int64_t)-1);
        REQUIRE(n_prompt > 4);
        f.sched.reply = c.reply;
        const AnthropicReply b = anthropic_buffered(f.post("/v1/messages", body + "}").body);
        f.sched.join_all();
        HttpResponse sr = f.post("/v1/messages", body + ",\"stream\":true}");
        const AnthropicReply s = anthropic_stream(drain(sr));
        f.sched.join_all();
        for (const AnthropicReply* a : { &b, &s }) {
            CHECK_EQ(a->stop_reason, std::string(c.reason));
            CHECK(a->stop_sequence == c.sequence);
            CHECK_EQ(a->usage.value("input_tokens", (int64_t)-1), n_prompt - 4);
            CHECK_EQ(a->usage.value("cache_read_input_tokens", (int64_t)-1), (int64_t)4);
            if (c.out >= 0) CHECK_EQ(a->usage.value("output_tokens", (int64_t)-1), c.out);
        }
        if (std::string(c.reason) == "tool_use") {
            REQUIRE(b.blocks.size() == 1 && s.blocks.size() == 1);
            CHECK(b.blocks[0] == s.blocks[0]);
            CHECK_EQ(b.blocks[0].text, std::string(R"({"c":1})"));
        }
    }
}

RAD_TEST_MAIN()
