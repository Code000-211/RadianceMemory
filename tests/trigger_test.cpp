/* trigger_test.cpp -- the lazy grammar's trigger: the word scanner, and the grammar around it.
 *
 * The scanner is checked against its plain definition (after each piece, search everything fed
 * so far for every word and take the leftmost start) over many random streams. A small alphabet
 * makes words overlap, prefix one another and repeat inside themselves, so the failure-link
 * construction is exercised. The grammar is checked the same way one level up: a lazy grammar
 * that has fired must reach exactly the state (compared by mask) an eager grammar reaches when
 * handed the text from the word onward. Waiting on a trigger runs on the scheduler thread for
 * every token, so it must cost as much on the last token of a long reply as on the first; that is
 * counted in work (tests/work_counter.h). Cases named perf_ hold wall-time floors for the serial
 * performance tier.
 */
#include "rad_test.h"
#include "work_counter.h"

#include "sample/gbnf.h"
#include "sample/grammar.h"
#include "sample/vocab_view.h"
#include "sample/word_scan.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace rad;

/* ================================================================== the scanner */

namespace {

struct Hit {
    long     piece = -1;
    uint64_t start = WordScanner::npos;
    bool operator==(const Hit& o) const { return piece == o.piece && start == o.start; }
};

/* What the scanner is defined to compute, by brute force over the whole text so far. */
Hit oracle(const std::vector<std::string>& words, const std::vector<std::string>& pieces) {
    std::string buf;
    for (size_t i = 0; i < pieces.size(); ++i) {
        buf += pieces[i];
        uint64_t best = WordScanner::npos;
        for (const std::string& w : words) {
            const size_t p = buf.find(w);
            if (p != std::string::npos && p < best) best = p;
        }
        if (best != WordScanner::npos) return { (long)i, best };
    }
    return {};
}

Hit scan(const WordScanner& sc, const std::vector<std::string>& pieces) {
    WordScanner::State st;
    for (size_t i = 0; i < pieces.size(); ++i) {
        const uint64_t s = sc.feed(st, pieces[i].data(), pieces[i].size());
        if (s != WordScanner::npos) return { (long)i, s };
    }
    return {};
}

WordScanner built(const std::vector<std::string>& words) {
    WordScanner sc;
    std::string err;
    const int st = sc.build(words, &err);
    if (st < 0) radtest::fail(__FILE__, __LINE__, "build failed: " + err);
    return sc;
}

}  /* namespace */

TEST(scanner_finds_a_word_split_across_pieces) {
    const WordScanner sc = built({ "<tool_call>\n" });
    CHECK(scan(sc, { "Sure, ", "<tool", "_call", ">", "\n", "{" }) == (Hit{ 4, 6 }));
    CHECK(scan(sc, { "<tool_call>" }) == Hit{});
    CHECK(scan(sc, { "x<tool_call>\ny" }) == (Hit{ 0, 1 }));
}

/* Of the occurrences completing in one piece the leftmost start wins, and an occurrence that has
 * not completed yet does not count however early it began. */
TEST(scanner_reports_the_leftmost_start_among_completed_words) {
    const WordScanner sc = built({ "bc", "abcd" });
    CHECK(scan(sc, { "abcd" }) == (Hit{ 0, 0 }));
    CHECK(scan(sc, { "abc", "d" }) == (Hit{ 0, 1 }));
    CHECK(scan(sc, { "a", "b", "c" }) == (Hit{ 2, 1 }));
}

TEST(scanner_on_the_classic_overlapping_set) {
    const WordScanner sc = built({ "he", "she", "his", "hers" });
    CHECK(scan(sc, { "ushers" }) == (Hit{ 0, 1 }));
    CHECK(scan(sc, { "us", "hers" }) == (Hit{ 1, 1 }));   /* "she" spans the split */
    CHECK(scan(sc, { "ahishers" }) == (Hit{ 0, 1 }));
}

TEST(scanner_on_words_that_overlap_themselves) {
    CHECK(scan(built({ "aab" }), { "aaab" }) == (Hit{ 0, 1 }));
    CHECK(scan(built({ "abab" }), { "abaabab" }) == (Hit{ 0, 3 }));
    CHECK(scan(built({ "aaa" }), { "a", "a", "a", "a" }) == (Hit{ 2, 0 }));
}

TEST(scanner_on_multibyte_and_zero_bytes) {
    const std::string arrow = "\xE2\x86\x92";
    const WordScanner sc = built({ arrow + "call", std::string("\0x", 2) });
    CHECK(scan(sc, { "go \xE2\x86", "\x92" "ca", "ll" }) == (Hit{ 2, 3 }));
    CHECK(scan(sc, { std::string("ab\0", 3), "x" }) == (Hit{ 1, 2 }));
    CHECK(scan(sc, { "\xE2\x86\x93" "call" }) == Hit{});
}

TEST(scanner_refuses_an_empty_word_and_matches_nothing_without_words) {
    WordScanner sc;
    std::string err;
    CHECK_EQ(sc.build({ "ok", "" }, &err), RAD_E_INVAL);
    CHECK(!err.empty());
    const WordScanner none = built({});
    CHECK(none.empty());
    CHECK(scan(none, { "anything", "at all" }) == Hit{});
}

/* A copied state continues exactly as the original: the speculative window depends on it. */
TEST(scanner_state_copies_are_exact) {
    const WordScanner sc = built({ "<tool_call>\n", "[TOOL" });
    WordScanner::State a;
    CHECK_EQ(sc.feed(a, "text <tool_c", 12), WordScanner::npos);
    WordScanner::State b = a;
    CHECK_EQ(sc.feed(b, "all>x", 5), WordScanner::npos);
    CHECK_EQ(sc.feed(a, "all>\n", 5), 5u);
    CHECK_EQ(b.off, 17u);
}

TEST(scanner_agrees_with_the_definition_on_random_streams) {
    /* Five symbols, one of them the lead byte of a multi-byte sequence, so that most random words
     * share prefixes and suffixes with one another and with themselves. */
    const char alpha[] = { 'a', 'b', 'c', '\xE2', '\n' };
    std::mt19937 rng(20240917u);
    auto pick = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    int matched = 0;
    for (int iter = 0; iter < 20000; ++iter) {
        std::vector<std::string> words((size_t)pick(1, 6));
        for (auto& w : words) {
            w.resize((size_t)pick(1, 7));
            for (char& c : w) c = alpha[pick(0, 4)];
        }
        std::vector<std::string> pieces((size_t)pick(0, 60));
        for (auto& p : pieces) {
            p.resize((size_t)pick(0, 6));
            for (char& c : p) c = alpha[pick(0, 4)];
        }
        const WordScanner sc = built(words);
        const Hit want = oracle(words, pieces);
        const Hit got  = scan(sc, pieces);
        if (!(got == want)) {
            std::string ws;
            for (auto& w : words) ws += "'" + w + "' ";
            radtest::fail(__FILE__, __LINE__, "iteration " + std::to_string(iter) + " words " +
                          ws + "want piece " + std::to_string(want.piece) + " start " +
                          std::to_string((long long)want.start) + ", got piece " +
                          std::to_string(got.piece) + " start " +
                          std::to_string((long long)got.start));
            return;
        }
        matched += want.piece >= 0;
    }
    CHECK(matched > 5000);   /* the generator reaches matches, not only misses */
}

namespace {

/* 8 MiB of pseudo-random printable text. */
const std::string& scan_text() {
    static const std::string t = [] {
        std::mt19937 rng(7u);
        std::string text(size_t(8) << 20, ' ');
        for (char& c : text) c = (char)(' ' + (int)(rng() % 90));
        return text;
    }();
    return t;
}

/* Scans the first `n` bytes of scan_text() in 4-byte pieces, as tokens arrive. */
uint64_t scan_prefix(const WordScanner& sc, size_t n) {
    const std::string& text = scan_text();
    WordScanner::State st;
    uint64_t sink = 0;
    for (size_t i = 0; i + 4 <= n; i += 4) sink += sc.feed(st, text.data() + i, 4);
    return sink;
}

}  /* namespace */

/* One table load a byte, at any length: the whole stream is twice the work of its first half. */
TEST(scanner_work_is_linear_in_the_stream) {
    const WordScanner sc = built({ "<tool_call>\n", "<function=" });
    const size_t n = scan_text().size();
    uint64_t sink = 0;
    const uint64_t half = radtest::work().of([&] { sink += scan_prefix(sc, n / 2); });
    const uint64_t full = radtest::work().of([&] { sink += scan_prefix(sc, n); });
    CHECK(sink != 1);
    std::printf("  scanner: full/half %.3f in %s\n", (double)full / (double)half,
                radtest::work().unit());
    CHECK((double)full / (double)half > 1.8);
    CHECK((double)full / (double)half < 2.2);
}

/* The scan's throughput floor in wall time, best of three. 40 MB/s is 100 ns for a four-byte
 * token. Not held under a sanitizer, which slows the loop by an order of magnitude. */
TEST(perf_scanner_throughput_has_a_floor) {
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    std::printf("  SKIP: the floor is for an unsanitised build\n");
    return;
#endif
    const WordScanner sc = built({ "<tool_call>\n", "<function=" });
    const size_t n = scan_text().size();
    double best = 1e30;
    for (int i = 0; i < 3; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(scan_prefix(sc, n) != 1);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const double mb_s = (double)n / (1 << 20) / best;
    std::printf("  scanner: %.0f MB/s in 4-byte pieces\n", mb_s);
    CHECK(mb_s > 40.0);
}

/* ================================================================== the lazy grammar */

namespace {

/* Characters, the pieces a tokeniser might split the trigger into, the marker as one control
 * token, and an end-of-generation token. */
struct Vocab {
    SimpleVocab v;
    int32_t marker = -1, nl = -1, close = -1, eog = -1;
    std::vector<int32_t> letters, prose;
    Vocab() {
        for (char c = 'a'; c <= 'z'; ++c) letters.push_back(v.add(std::string(1, c)));
        prose = letters;
        for (const char* p : { " ", ",", ".", "<", ">", "/", "_", "tool", "call", "xx<tool",
                               "_call>\n", "<tool_c", "all>", "ok ", "\n\n" })
            prose.push_back(v.add(p));
        nl     = v.add("\n");
        marker = v.add("<tool_call>");
        close  = v.add("</tool_call>");
        eog    = v.add("<|end|>", true);
        prose.push_back(nl);
        prose.push_back(marker);
    }
    int64_t n() const { return v.n_tokens(); }
};

const char* kCall = R"(root ::= "<tool_call>\n" [a-z]+ ("," [a-z]+)* "\n</tool_call>")";

std::unique_ptr<GbnfGrammar> make(const Vocab& vb, bool lazy,
                                  const std::vector<std::string>& words,
                                  const std::vector<int32_t>& tokens) {
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    if (GbnfGrammar::create(kCall, "root", &vb.v, lazy, words, tokens, &g, &err) < 0)
        radtest::fail(__FILE__, __LINE__, "create: " + err);
    return g;
}

std::vector<uint32_t> mask_of(const GbnfGrammar& g, int64_t n) {
    std::vector<uint32_t> m((size_t)bitmask_words(n), 0u);
    const int st = g.fill_mask(m.data(), (int64_t)m.size());
    if (st < 0) m.assign(m.size(), 0xDEADBEEFu);
    return m;
}

bool admits(const std::vector<uint32_t>& m, int32_t id) {
    return (m[(size_t)id >> 5] >> ((uint32_t)id & 31u)) & 1u;
}

}  /* namespace */

TEST(lazy_grammar_waits_then_constrains_from_the_word) {
    Vocab vb;
    auto g = make(vb, true, { "<tool_call>\n" }, {});
    for (int32_t t : { vb.letters[7], vb.letters[8], vb.prose[26] }) CHECK_OK(g->accept_token(t));
    CHECK(g->awaiting_trigger());
    CHECK_OK(g->accept_token(vb.marker));
    CHECK(g->awaiting_trigger());          /* the word ends with the newline */
    CHECK_OK(g->accept_token(vb.nl));
    CHECK(!g->awaiting_trigger());
    const auto m = mask_of(*g, vb.n());
    CHECK(admits(m, vb.letters[0]));
    CHECK(!admits(m, vb.marker));
    CHECK(!admits(m, vb.nl));
}

/* The word may begin inside a token and end inside another; only its own bytes are replayed. */
TEST(lazy_grammar_replays_from_inside_a_token) {
    Vocab vb;
    auto lazy  = make(vb, true, { "<tool_call>\n" }, {});
    auto eager = make(vb, false, {}, {});
    int32_t xx = -1, rest = -1;
    for (int32_t i = 0; i < (int32_t)vb.n(); ++i) {
        if (vb.v.token_piece(i) == "xx<tool")  xx = i;
        if (vb.v.token_piece(i) == "_call>\n") rest = i;
    }
    REQUIRE(xx >= 0 && rest >= 0);
    CHECK_OK(lazy->accept_token(xx));
    CHECK_OK(lazy->accept_token(rest));
    CHECK(!lazy->awaiting_trigger());
    CHECK_OK(eager->accept_str("<tool_call>\n"));
    CHECK(mask_of(*lazy, vb.n()) == mask_of(*eager, vb.n()));
}

TEST(lazy_grammar_fires_on_a_trigger_token_id) {
    Vocab vb;
    auto g = make(vb, true, {}, { vb.marker });
    CHECK_OK(g->accept_token(vb.letters[0]));
    CHECK(g->awaiting_trigger());
    CHECK_OK(g->accept_token(vb.marker));
    CHECK(!g->awaiting_trigger());
    const auto m = mask_of(*g, vb.n());
    CHECK(admits(m, vb.nl));
    CHECK(!admits(m, vb.letters[0]));
}

/* A partial word seen inside a speculative window is rewound with the window. */
TEST(lazy_grammar_rollback_forgets_a_partial_word) {
    Vocab vb;
    auto g = make(vb, true, { "<tool_call>\n" }, {});
    g->push_state();
    CHECK_OK(g->accept_token(vb.marker));
    CHECK(g->awaiting_trigger());
    CHECK_OK(g->rollback());
    CHECK_OK(g->accept_token(vb.nl));      /* "\n" alone: the marker was rewound */
    CHECK(g->awaiting_trigger());

    g->push_state();
    CHECK_OK(g->accept_token(vb.marker));
    CHECK_OK(g->accept_token(vb.nl));
    CHECK(!g->awaiting_trigger());         /* fired inside the window */
    CHECK_OK(g->rollback());
    CHECK(g->awaiting_trigger());          /* and rewound with it */
    CHECK_OK(g->accept_token(vb.marker));
    CHECK_OK(g->accept_token(vb.nl));
    CHECK(!g->awaiting_trigger());
}

/* Grammar-level differential. Random prose from pieces that spell the word in several splits,
 * with speculative windows walked and rewound as the sampler does, until the word appears; then
 * tokens are drawn from the lazy grammar's own mask. At every position after the trigger the
 * lazy grammar must admit exactly what an eager grammar admits after the same text from the word
 * onward. */
TEST(lazy_grammar_agrees_with_an_eager_grammar_on_random_streams) {
    Vocab vb;
    std::mt19937 rng(424242u);
    auto pick = [&](size_t n) { return (size_t)(rng() % n); };
    int fired = 0, refused = 0;
    for (int iter = 0; iter < 400; ++iter) {
        auto lazy = make(vb, true, { "<tool_call>\n" }, {});
        std::string text;
        int last = RAD_OK;
        for (int k = 0; k < 400 && lazy->awaiting_trigger(); ++k) {
            if (pick(4) == 0) {
                lazy->push_state();
                for (int d = 0; d < 3; ++d) lazy->accept_token(vb.prose[pick(vb.prose.size())]);
                CHECK_OK(lazy->rollback());
            }
            const int32_t t = vb.prose[pick(vb.prose.size())];
            text += vb.v.token_piece(t);
            last = lazy->accept_token(t);
        }
        if (lazy->awaiting_trigger()) continue;
        ++fired;
        const size_t at = text.find("<tool_call>\n");
        REQUIRE(at != std::string::npos);
        auto eager = make(vb, false, {}, {});
        /* The token that completed the word may carry bytes past it that the grammar refuses
         * ("<tool_call>" then "\n\n"); the lazy grammar must refuse exactly when the eager one
         * refuses the same text. */
        const int est = eager->accept_str(text.substr(at));
        CHECK_EQ(last < 0, est < 0);
        if (last < 0 || est < 0) { ++refused; continue; }
        for (int k = 0; k < 12; ++k) {
            const auto ml = mask_of(*lazy, vb.n());
            const auto me = mask_of(*eager, vb.n());
            if (ml != me) {
                radtest::fail(__FILE__, __LINE__, "masks differ at iteration " +
                              std::to_string(iter) + " step " + std::to_string(k));
                return;
            }
            std::vector<int32_t> ok;
            for (int32_t i = 0; i < (int32_t)vb.n(); ++i) if (admits(ml, i)) ok.push_back(i);
            if (ok.empty() || ml[0] == 0xDEADBEEFu) break;
            const int32_t t = ok[pick(ok.size())];
            if (vb.v.is_eog(t)) break;
            REQUIRE(lazy->accept_token(t) >= 0);
            REQUIRE(eager->accept_token(t) >= 0);
        }
    }
    CHECK(fired - refused > 100);
}

namespace {

/* `n` speculative windows over a waiting lazy grammar: push, three drafts, rewind, one token. */
void wait_windows(GbnfGrammar& g, const Vocab& vb, int n) {
    const int32_t w = vb.letters[22], sp = vb.prose[26];
    for (int i = 0; i < n; ++i) {
        g.push_state();
        g.accept_token(w);
        g.accept_token(sp);
        g.accept_token(w);
        g.rollback();
        g.accept_token(i % 3 ? w : sp);
    }
}

}  /* namespace */

/* Waiting costs the same at any length. A lazy grammar sees every token of the reply until its
 * word appears, and each speculative window copies its state twice; the last tokens of a long
 * reply must cost what the first did. Counted in work, 20,000 windows at the start and again after
 * 200,000 more, each the least of three runs: under the CPU-time fallback (a container build, tests
 * beside it) one run alone came out 1.65x the other. */
TEST(lazy_grammar_waiting_work_does_not_grow_with_the_reply) {
    Vocab vb;
    auto g = make(vb, true, { "<tool_call>\n" }, {});
    auto least = [&] {
        uint64_t m = UINT64_MAX;
        for (int k = 0; k < 3; ++k)
            m = std::min(m, radtest::work().of([&] { wait_windows(*g, vb, 20000); }));
        return m;
    };
    const uint64_t first = least();
    wait_windows(*g, vb, 200000);
    const uint64_t last = least();
    CHECK(g->awaiting_trigger());
    std::printf("  lazy wait: %llu then %llu %s per 20k windows\n", (unsigned long long)first,
                (unsigned long long)last, radtest::work().unit());
    CHECK((double)last < (double)first * 1.25);
}

/* The wait's cost in wall time, for the serial performance tier: flat, and under 5 us a token. */
TEST(perf_lazy_grammar_waiting_cost_has_a_ceiling) {
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    std::printf("  SKIP: the ceiling is for an unsanitised build\n");
    return;
#endif
    Vocab vb;
    auto g = make(vb, true, { "<tool_call>\n" }, {});
    auto window = [&](int n) {
        const auto t0 = std::chrono::steady_clock::now();
        wait_windows(*g, vb, n);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto best = [&]() { return std::min({ window(20000), window(20000), window(20000) }); };
    const double first = best();
    window(200000);
    const double last = best();
    CHECK(g->awaiting_trigger());
    std::printf("  lazy wait: %.3f us/token first, %.3f us/token after 320k\n",
                first * 1e6 / 20000, last * 1e6 / 20000);
    CHECK(last < first * 2.0 + 0.001);
    CHECK(last * 1e6 / 20000 < 5.0);
}

RAD_TEST_MAIN()
