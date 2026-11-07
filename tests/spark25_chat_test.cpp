/* Spark's reasoning and XML/KV tool calls, including streaming at every byte. */
#include "rad_test.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "../arch/spark25_bf16/chat.h"

#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace rad;
using json = nlohmann::ordered_json;

static const json& tools() {
    static const json t = json::parse(R"([
      {"type":"function","function":{"name":"weather","parameters":{"type":"object",
       "properties":{"city":{"type":"string"},"days":{"type":"integer"},
                     "verbose":{"type":"boolean"},"units":{"type":"array","items":{"type":"string"}}}}}},
      {"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}
    ])");
    return t;
}

static ChatFormat format() {
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(spark25_bf16::spark_chat_format(), &f, &why));
    CHECK_OK(chat_format_check(f, &why));
    return f;
}

static std::shared_ptr<const ReplyParser> parser(const std::string& prefix, bool with_tools = true) {
    std::string why;
    ChatFormat f = format();
    std::shared_ptr<const ReplyFormat> rf;
    std::shared_ptr<const ReplyParser> p;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    ReplyOptions opts;
    opts.generation_prompt = prefix;
    opts.tools = with_tools ? &tools() : nullptr;
    if (rf) CHECK_OK(ReplyParser::create(rf, opts, &p, &why));
    return p;
}

TEST(thinking_and_typed_calls_survive_one_byte_streaming) {
    auto p = parser("<｜start▁of▁sentence｜><|Bot|><think>");
    REQUIRE(p != nullptr);
    const std::string s = "I should look.</think>Checking."
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
        "<arg_key>verbose</arg_key><arg_value>true</arg_value>"
        "<arg_key>units</arg_key><arg_value>[\"C\",\"F\"]</arg_value></tool_call>"
        "<tool_call>ping</tool_call>";
    const ReplyMessage expected = reply_parse(*p, s);
    CHECK_EQ(expected.reasoning, std::string("I should look."));
    CHECK_EQ(expected.content, std::string("Checking."));
    REQUIRE_EQ(expected.calls.size(), (size_t)2);
    CHECK_EQ(expected.calls[0].name, std::string("weather"));
    CHECK(expected.calls[0].closed);
    const json args = json::parse(expected.calls[0].arguments);
    CHECK_EQ(args["city"].get<std::string>(), std::string("Paris"));
    CHECK_EQ(args["days"].get<int>(), 3);
    CHECK_EQ(args["verbose"].get<bool>(), true);
    CHECK(args["units"] == json::array({"C", "F"}));
    CHECK_EQ(expected.calls[1].name, std::string("ping"));
    CHECK(json::parse(expected.calls[1].arguments) == json::object());
    auto stream = p->open();
    for (char c : s) stream->feed(std::string_view(&c, 1), nullptr);
    stream->finish(nullptr);
    const ReplyMessage& got = stream->message();
    CHECK_EQ(got.reasoning, expected.reasoning);
    CHECK_EQ(got.content, expected.content);
    REQUIRE_EQ(got.calls.size(), expected.calls.size());
    for (size_t i = 0; i < got.calls.size(); ++i) {
        CHECK_EQ(got.calls[i].name, expected.calls[i].name);
        CHECK_EQ(got.calls[i].arguments, expected.calls[i].arguments);
        CHECK(got.calls[i].closed);
    }
}

TEST(thinking_off_and_an_unclosed_thought_before_a_call) {
    auto off = parser("<｜start▁of▁sentence｜><|Bot|></think>", false);
    REQUIRE(off != nullptr);
    const ReplyMessage a = reply_parse(*off, "Hello.");
    CHECK(a.reasoning.empty());
    CHECK_EQ(a.content, std::string("Hello."));
    auto on = parser("<｜start▁of▁sentence｜><|Bot|><think>");
    REQUIRE(on != nullptr);
    const ReplyMessage b = reply_parse(*on, "Plan.<tool_call>ping</tool_call>");
    CHECK_EQ(b.reasoning, std::string("Plan."));
    REQUIRE_EQ(b.calls.size(), (size_t)1);
    CHECK_EQ(b.calls[0].name, std::string("ping"));
}

TEST(grammar_starts_on_the_real_call_opener) {
    const ChatFormat f = format();
    ChatGrammarRequest req;
    req.tools = &tools();
    req.generation_prompt = "<think>";
    ChatGrammar g;
    std::string why;
    CHECK_OK(chat_format_grammar(f, req, &g, &why));
    CHECK(g.lazy);
    CHECK(!g.text.empty());
    REQUIRE_EQ(g.trigger_words.size(), (size_t)1);
    CHECK_EQ(g.trigger_words[0], std::string("<tool_call>"));
}

/* Set this to the checkpoint's actual chat_template.jinja to exercise it without redistributing
 * model files in the repository. Ordinary CI still exercises the reply format above. */
TEST(reference_template_defaults_to_thinking_and_allows_disabling_it) {
    const char* path = std::getenv("SPARK25_REFERENCE_TEMPLATE");
    if (!path || !*path) SKIP("set SPARK25_REFERENCE_TEMPLATE to the checkpoint's template");
    std::ifstream file(path);
    REQUIRE((bool)file);
    std::ostringstream s;
    s << file.rdbuf();
    ChatTemplate t;
    REQUIRE_EQ(t.load(s.str(), "<｜start▁of▁sentence｜>", "<｜end▁of▁sentence｜>"), RAD_OK);
    const json messages = json::array({{{"role", "user"}, {"content", "Hello"}}});
    for (bool think : {true, false}) {
        ChatOptions opt;
        opt.enable_thinking = think;
        ChatPrompt prompt;
        REQUIRE_EQ(t.apply(messages, json::array(), opt, &prompt), RAD_OK);
        const std::string suffix = think ? "<think>" : "</think>";
        CHECK(prompt.prompt.size() >= suffix.size());
        CHECK(prompt.prompt.ends_with(suffix));
        if (think) CHECK(prompt.prompt.find("thinking_effort=high") != std::string::npos);
    }
    ChatOptions low;
    low.template_kwargs["reasoning_effort"] = "\"low\"";
    ChatPrompt lp;
    REQUIRE_EQ(t.apply(messages, json::array(), low, &lp), RAD_OK);
    CHECK(lp.prompt.find("thinking_effort=low") != std::string::npos);

    /* OpenAI tool arguments arrive as a JSON string; the checkpoint template requires a mapping. */
    const json history = json::parse(R"([
      {"role":"user","content":"Weather?"},
      {"role":"assistant","content":"","reasoning_content":"Check.","tool_calls":[
        {"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]},
      {"role":"tool","tool_call_id":"call_1","content":"Sunny"}
    ])");
    ChatPrompt hp;
    ChatOptions options;
    REQUIRE_EQ(t.apply(history, tools(), options, &hp), RAD_OK);
    CHECK(hp.prompt.find("<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>") != std::string::npos);
    CHECK(hp.prompt.find("<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>") != std::string::npos);
    CHECK(hp.prompt.find("<tool_response>Sunny</tool_response>") != std::string::npos);
}

RAD_TEST_MAIN()
