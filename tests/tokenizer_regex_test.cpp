/* tokenizer_regex_test.cpp -- the pre-tokeniser pattern compiler, construct by construct.
 *
 * The expected splits are Oniguruma's: each row was run through HuggingFace tokenizers' Split
 * (behaviour and invert as given) and its pieces recorded, since that is the engine tokenizer.json
 * patterns are written for and the only definition of what one means. There is a row per
 * construct the dialect covers (escapes, classes and set operations, every shorthand and
 * property, groups, greedy and lazy quantifiers, flags and their scope, anchors, one-codepoint
 * lookaround, iteration over empty matches), so a regression names the construct.
 *
 * Every construct outside the dialect must fail to compile with a message naming it, since a
 * pattern that does not compile is a vocabulary that does not load.
 */
#include "rad_test.h"

#include "rad_internal.h"
#include "text/tokenizer_regex.h"

#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace rad;
using namespace std::literals;

namespace {

std::vector<std::string> split(const TokRegex& re, std::string_view text, int behavior = TOK_SPLIT_ISOLATED,
                               bool invert = false) {
    std::vector<TokSpan> out, scratch;
    tok_split(re, text, behavior, invert, out, scratch);
    std::vector<std::string> pieces;
    for (const TokSpan& s : out) pieces.emplace_back(text.substr(s.begin, s.end - s.begin));
    return pieces;
}

std::string show(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o += ", ";
        o += "'";
        for (unsigned char c : v[i]) o += (c < 0x20 || c == 0x7F) ? fmt("\\x%02x", c) : std::string(1, (char)c);
        o += "'";
    }
    return o + "]";
}

std::unique_ptr<TokRegex> compile(std::string_view pat) {
    std::unique_ptr<TokRegex> re;
    std::string why;
    if (TokRegex::compile(pat, &re, &why) < 0) {
        std::fprintf(stderr, "    compile failed: %s\n", why.c_str());
        return nullptr;
    }
    return re;
}

struct Case {
    std::string_view name, pattern, text;
    std::vector<std::string_view> pieces;
};

/* Recorded from HuggingFace tokenizers 0.23.2 (Oniguruma), Split(behavior="isolated"). */
const std::vector<Case> k_cases = {
    { "literal"sv,
      "ab|."sv, "xabab"sv,
      { "x"sv, "ab"sv, "ab"sv } },
    { "escapes"sv,
      "\\t|\\n|\\r|\\f|\\v|\\a|\\e|\\x41|\\x{1F600}|\xc3""\xa9""|\\0|."sv, "\x09""\x0a""\x0d""\x0c""\x0b""\x07""\x1b""A\xf0""\x9f""\x98""\x80""\xc3""\xa9""\x00""z"sv,
      { "\x09"""sv, "\x0a"""sv, "\x0d"""sv, "\x0c"""sv, "\x0b"""sv, "\x07"""sv, "\x1b"""sv, "A"sv, "\xf0""\x9f""\x98""\x80"""sv, "\xc3""\xa9"""sv, "\x00"""sv, "z"sv } },
    { "escaped metachar"sv,
      "\\.|\\*|\\(|\\[|\\{|\\||\\\\|\\$|\\^|\\?|\\+|\\)|."sv, ".*([{|\\$^?+)x"sv,
      { "."sv, "*"sv, "("sv, "["sv, "{"sv, "|"sv, "\\"sv, "$"sv, "^"sv, "?"sv, "+"sv, ")"sv, "x"sv } },
    { "dot excludes newline"sv,
      ".+"sv, "ab\x0a""cd\x0d""\x0a""ef"sv,
      { "ab"sv, "\x0a"""sv, "cd\x0d"""sv, "\x0a"""sv, "ef"sv } },
    { "(?m) dot includes newline"sv,
      "(?m:.)+"sv, "ab\x0a""cd"sv,
      { "ab\x0a""cd"sv } },
    { "control escape"sv,
      "\\cA|."sv, "\x01""b"sv,
      { "\x01"""sv, "b"sv } },
    { "class range"sv,
      "[a-c]+|."sv, "abcdcba"sv,
      { "abc"sv, "d"sv, "cba"sv } },
    { "negated class"sv,
      "[^a-c]+|."sv, "xyzabc"sv,
      { "xyz"sv, "a"sv, "b"sv, "c"sv } },
    { "class with first ]"sv,
      "[]a]+|."sv, "a]b"sv,
      { "a]"sv, "b"sv } },
    { "class with trailing -"sv,
      "[a-]+|."sv, "a-b"sv,
      { "a-"sv, "b"sv } },
    { "class range from -"sv,
      "[--/]+|."sv, "-./a"sv,
      { "-./"sv, "a"sv } },
    { "class with escapes"sv,
      "[\\]\\[]+|."sv, "a[]b"sv,
      { "a"sv, "[]"sv, "b"sv } },
    { "class with shorthands"sv,
      "[a\\d\\s]+|."sv, "a1 b"sv,
      { "a1 "sv, "b"sv } },
    { "nested class union"sv,
      "[a-c[x-z]]+|."sv, "abxyzd"sv,
      { "abxyz"sv, "d"sv } },
    { "class intersection"sv,
      "[a-z&&[^aeiou]]+|."sv, "bcdaefg"sv,
      { "bcd"sv, "a"sv, "e"sv, "fg"sv } },
    { "property intersection"sv,
      "[\\p{L}&&\\p{Lu}]+|."sv, "aBC"sv,
      { "a"sv, "BC"sv } },
    { "negated intersection"sv,
      "[^\\p{L}\\p{N}&&[^\\s]]+|."sv, "ab ,."sv,
      { "a"sv, "b"sv, " ,."sv } },
    { "bloom class"sv,
      " ?[^(\\s|[.,!?\xe2""\x80""\xa6""\xe3""\x80""\x82""\xef""\xbc""\x8c""\xe3""\x80""\x81""\xe0""\xa5""\xa4""\xdb""\x94""\xd8""\x8c""])]+"sv, "hello, world. (x|y)"sv,
      { "hello"sv, ","sv, " world"sv, ". ("sv, "x"sv, "|"sv, "y"sv, ")"sv } },
    { "\\d \\D"sv,
      "\\d+|\\D+"sv, "ab12\xd9""\xa3""\xe0""\xa5""\xaa""x"sv,
      { "ab"sv, "12\xd9""\xa3""\xe0""\xa5""\xaa"""sv, "x"sv } },
    { "\\s \\S"sv,
      "\\s+|\\S+"sv, "a \x09""\xc2""\xa0""\xe3""\x80""\x80""b\xe2""\x80""\xa8""c"sv,
      { "a"sv, " \x09""\xc2""\xa0""\xe3""\x80""\x80"""sv, "b"sv, "\xe2""\x80""\xa8"""sv, "c"sv } },
    { "\\w atom counts the Latin-1 fractions"sv,
      "\\w+"sv, "a\xc2""\xbd""b\xc2""\xb2""c\xe2""\x91""\xa0""d"sv,
      { "a\xc2""\xbd""b\xc2""\xb2""c"sv, "\xe2""\x91""\xa0"""sv, "d"sv } },
    { "\\w in a class does not"sv,
      "[\\w]+"sv, "a\xc2""\xbd""b\xc2""\xb2""c\xe2""\x91""\xa0""d"sv,
      { "a"sv, "\xc2""\xbd"""sv, "b"sv, "\xc2""\xb2"""sv, "c"sv, "\xe2""\x91""\xa0"""sv, "d"sv } },
    { "\\h hex digit"sv,
      "\\h+|."sv, "ab 09 xyz"sv,
      { "ab"sv, " "sv, "09"sv, " "sv, "x"sv, "y"sv, "z"sv } },
    { "letter property"sv,
      "\\p{L}+|."sv, "h\xc3""\xa9""llo \xd0""\xbc""\xd0""\xb8""\xd1""\x80"" \xe4""\xb8""\x96""\xe7""\x95""\x8c"" 1"sv,
      { "h\xc3""\xa9""llo"sv, " "sv, "\xd0""\xbc""\xd0""\xb8""\xd1""\x80"""sv, " "sv, "\xe4""\xb8""\x96""\xe7""\x95""\x8c"""sv, " "sv, "1"sv } },
    { "negated property"sv,
      "\\P{L}+|."sv, "ab12"sv,
      { "a"sv, "b"sv, "12"sv } },
    { "caret negated property"sv,
      "\\p{^L}+|."sv, "ab12"sv,
      { "a"sv, "b"sv, "12"sv } },
    { "long property name"sv,
      "\\p{Uppercase_Letter}+|."sv, "aBCd"sv,
      { "a"sv, "BC"sv, "d"sv } },
    { "property name normalisation"sv,
      "\\p{ lu }+|."sv, "aBCd"sv,
      { "a"sv, "BC"sv, "d"sv } },
    { "subcategories"sv,
      "\\p{Nd}+|\\p{No}+|\\p{Nl}+|."sv, "12\xc2""\xbd""\xe2""\x85""\xab""\xd9""\xa3"""sv,
      { "12"sv, "\xc2""\xbd"""sv, "\xe2""\x85""\xab"""sv, "\xd9""\xa3"""sv } },
    { "marks"sv,
      "[\\p{Mn}\\p{Mc}\\p{Me}]+|."sv, "e\xcc""\x81""\xe0""\xa4""\x83""\xe2""\x83""\x9d"" x"sv,
      { "e"sv, "\xcc""\x81""\xe0""\xa4""\x83""\xe2""\x83""\x9d"""sv, " "sv, "x"sv } },
    { "script"sv,
      "\\p{Han}+|\\p{Hiragana}+|\\p{Katakana}+|."sv, "\xe6""\xbc""\xa2""\xe5""\xad""\x97""\xe3""\x81""\xb2""\xe3""\x82""\x89""\xe3""\x81""\x8c""\xe3""\x81""\xaa""\xe3""\x82""\xab""\xe3""\x82""\xbf""\xe3""\x82""\xab""\xe3""\x83""\x8a""a"sv,
      { "\xe6""\xbc""\xa2""\xe5""\xad""\x97"""sv, "\xe3""\x81""\xb2""\xe3""\x82""\x89""\xe3""\x81""\x8c""\xe3""\x81""\xaa"""sv, "\xe3""\x82""\xab""\xe3""\x82""\xbf""\xe3""\x82""\xab""\xe3""\x83""\x8a"""sv, "a"sv } },
    { "separators and others"sv,
      "[\\p{Zs}]+|[\\p{Cf}\\p{Co}\\p{Cn}]+|."sv, "a \xc2""\xa0""\xe2""\x80""\x8b""\xee""\x80""\x80""\xcd""\xb8""b"sv,
      { "a"sv, " \xc2""\xa0"""sv, "\xe2""\x80""\x8b""\xee""\x80""\x80""\xcd""\xb8"""sv, "b"sv } },
    { "leftmost-first alternation"sv,
      "a|ab|abc"sv, "abc"sv,
      { "a"sv, "bc"sv } },
    { "alternation order inside groups"sv,
      "(a|ab)(c|bcd)(d*)"sv, "abcd"sv,
      { "abcd"sv } },
    { "capturing and named groups"sv,
      "(a)|(?:b)|(?<n>c)|."sv, "abcd"sv,
      { "a"sv, "b"sv, "c"sv, "d"sv } },
    { "comment group"sv,
      "(?#comment)a|."sv, "ab"sv,
      { "a"sv, "b"sv } },
    { "empty alternative"sv,
      "a|"sv, "bab"sv,
      { "b"sv, "a"sv, "b"sv } },
    { "empty first alternative"sv,
      "|a"sv, "bab"sv,
      { "b"sv, "a"sv, "b"sv } },
    { "empty group"sv,
      "(?:)"sv, "ab"sv,
      { "a"sv, "b"sv } },
    { "greedy star"sv,
      "a*"sv, "baab"sv,
      { "b"sv, "aa"sv, "b"sv } },
    { "lazy star"sv,
      "a*?"sv, "baab"sv,
      { "b"sv, "a"sv, "a"sv, "b"sv } },
    { "lazy plus"sv,
      "a+?b|."sv, "aaab"sv,
      { "aaab"sv } },
    { "lazy interval"sv,
      "a{2,3}?|."sv, "aaaaa"sv,
      { "aa"sv, "aa"sv, "a"sv } },
    { "interval"sv,
      "a{2}|."sv, "aaa"sv,
      { "aa"sv, "a"sv } },
    { "open interval"sv,
      "a{2,}|."sv, "aaa b"sv,
      { "aaa"sv, " "sv, "b"sv } },
    { "interval without minimum"sv,
      "a{,2}|."sv, "aaa"sv,
      { "aa"sv, "a"sv } },
    { "zero interval"sv,
      "a{0}|."sv, "ab"sv,
      { "a"sv, "b"sv } },
    { "literal brace"sv,
      "a{x}|{|."sv, "a{x}{"sv,
      { "a{x}"sv, "{"sv } },
    { "counted group"sv,
      "(?:a{2}){2,3}|."sv, "aaaaaaa"sv,
      { "aaaaaa"sv, "a"sv } },
    { "optional inside counted"sv,
      "(a?){3}b|."sv, "aab"sv,
      { "aab"sv } },
    { "star over alternation"sv,
      "(?:a|ab)*c|."sv, "ababc"sv,
      { "ababc"sv } },
    { "digit groups"sv,
      "\\p{N}{1,3}"sv, "1234567 12"sv,
      { "123"sv, "456"sv, "7"sv, " "sv, "12"sv } },
    { "(?i) group"sv,
      "(?i:[a-c])+|."sv, "ABCabcd"sv,
      { "ABCabc"sv, "d"sv } },
    { "(?i) Kelvin sign"sv,
      "(?i:k)+|."sv, "kK\xe2""\x84""\xaa"""sv,
      { "kK\xe2""\x84""\xaa"""sv } },
    { "(?i) long s"sv,
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|\\p{L}+|."sv, "it'\xc5""\xbf""x 'S '\xc5""\xbf"" 'K 'LL 'Re"sv,
      { "it"sv, "'\xc5""\xbf"""sv, "x"sv, " "sv, "'S"sv, " "sv, "'\xc5""\xbf"""sv, " "sv, "'"sv, "K"sv, " "sv, "'LL"sv, " "sv, "'Re"sv } },
    { "(?i) negated class"sv,
      "(?i:[^a-z])+|."sv, "aA1B"sv,
      { "a"sv, "A"sv, "1"sv, "B"sv } },
    { "(?i) to the end of the group"sv,
      "a(?i)b|c"sv, "aBcC ab"sv,
      { "aB"sv, "cC "sv, "ab"sv } },
    { "(?i) then (?-i)"sv,
      "(?i)a(?-i:b)|."sv, "AbAB"sv,
      { "Ab"sv, "A"sv, "B"sv } },
    { "(?i) scoped"sv,
      "(?i:a)b|."sv, "AbAB"sv,
      { "Ab"sv, "A"sv, "B"sv } },
    { "(?x) extended"sv,
      "(?x: a b # comment\x0a"" )+|."sv, "abab"sv,
      { "abab"sv } },
    { "line anchors"sv,
      "^a|b$|\\n"sv, "ab\x0a""ab\x0a""a\x0a""b"sv,
      { "a"sv, "b"sv, "\x0a"""sv, "a"sv, "b"sv, "\x0a"""sv, "a"sv, "\x0a"""sv, "b"sv } },
    { "line start and end"sv,
      "^.|.$"sv, "xy\x0a""zw"sv,
      { "x"sv, "y"sv, "\x0a"""sv, "z"sv, "w"sv } },
    { "dollar before newline"sv,
      "\\s+$|\\S+|\\s+"sv, "a  \x0a""  b  "sv,
      { "a"sv, "  "sv, "\x0a""  "sv, "b"sv, "  "sv } },
    { "text anchors"sv,
      "\\A.|.\\z"sv, "abc"sv,
      { "a"sv, "b"sv, "c"sv } },
    { "empty dollar matches"sv,
      "$"sv, "a\x0a""b"sv,
      { "a"sv, "\x0a""b"sv } },
    { "empty caret matches"sv,
      "^"sv, "ab\x0a""cd"sv,
      { "ab\x0a"""sv, "cd"sv } },
    { "word boundary"sv,
      "\\ba|."sv, "a ba a"sv,
      { "a"sv, " "sv, "b"sv, "a"sv, " "sv, "a"sv } },
    { "not word boundary"sv,
      "\\B.|x"sv, "ab cd"sv,
      { "a"sv, "b"sv, " c"sv, "d"sv } },
    { "empty word boundaries"sv,
      "\\b"sv, "ab cd"sv,
      { "ab"sv, " "sv, "cd"sv } },
    { "word boundary over a fraction"sv,
      "\\b"sv, "a\xc2""\xbd"" b"sv,
      { "a\xc2""\xbd"""sv, " "sv, "b"sv } },
    { "negative lookahead"sv,
      "x(?!y)|."sv, "xyxz x"sv,
      { "x"sv, "y"sv, "x"sv, "z"sv, " "sv, "x"sv } },
    { "the \\s+(?!\\S) idiom"sv,
      "\\s+(?!\\S)|\\s+"sv, "a   b   "sv,
      { "a"sv, "  "sv, " "sv, "b"sv, "   "sv } },
    { "positive lookahead"sv,
      "(?=[\\p{L}])[^a-z]+|."sv, "ABcD\xc3""\xa9"""sv,
      { "AB"sv, "c"sv, "D\xc3""\xa9"""sv } },
    { "empty lookahead matches"sv,
      "(?=a)"sv, "baab"sv,
      { "b"sv, "a"sv, "ab"sv } },
    { "lookbehind"sv,
      "(?<=a)b|."sv, "abcb"sv,
      { "a"sv, "b"sv, "c"sv, "b"sv } },
    { "negative lookbehind"sv,
      "(?<!a)b|."sv, "abcb"sv,
      { "a"sv, "b"sv, "c"sv, "b"sv } },
    { "lookbehind of a class"sv,
      "(?<=\\s)\\S+|\\s+|\\S+"sv, "ab cd  ef"sv,
      { "ab"sv, " "sv, "cd"sv, "  "sv, "ef"sv } },
    { "empty match after a match is skipped"sv,
      "x*|b"sv, "abc"sv,
      { "a"sv, "b"sv, "c"sv } },
};

struct Behaviour {
    std::string_view pattern, text;
    int behavior;
    bool invert;
    std::vector<std::string_view> pieces;
};

/* The same, for every SplitDelimiterBehavior with and without invert. */
const std::vector<Behaviour> k_behaviours = {
    { "\\s+"sv, "  hello  big world "sv, 0, false,
      { "hello"sv, "big"sv, "world"sv } },
    { "\\s+"sv, "  hello  big world "sv, 0, true,
      { "  "sv, "  "sv, " "sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 1, false,
      { "  "sv, "hello"sv, "  "sv, "big"sv, " "sv, "world"sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 1, true,
      { "  "sv, "hello"sv, "  "sv, "big"sv, " "sv, "world"sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 2, false,
      { "  "sv, "hello  "sv, "big "sv, "world "sv } },
    { "\\s+"sv, "  hello  big world "sv, 2, true,
      { "  hello"sv, "  big"sv, " world"sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 3, false,
      { "  hello"sv, "  big"sv, " world"sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 3, true,
      { "  "sv, "hello  "sv, "big "sv, "world "sv } },
    { "\\s+"sv, "  hello  big world "sv, 4, false,
      { "  "sv, "hello"sv, "  "sv, "big"sv, " "sv, "world"sv, " "sv } },
    { "\\s+"sv, "  hello  big world "sv, 4, true,
      { "  "sv, "hello"sv, "  "sv, "big"sv, " "sv, "world"sv, " "sv } },
    { "[,.]"sv, "a,b..c,"sv, 0, false,
      { "a"sv, "b"sv, "c"sv } },
    { "[,.]"sv, "a,b..c,"sv, 0, true,
      { ","sv, "."sv, "."sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 1, false,
      { "a"sv, ","sv, "b"sv, "."sv, "."sv, "c"sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 1, true,
      { "a"sv, ","sv, "b"sv, "."sv, "."sv, "c"sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 2, false,
      { "a,"sv, "b."sv, "."sv, "c,"sv } },
    { "[,.]"sv, "a,b..c,"sv, 2, true,
      { "a"sv, ",b"sv, "."sv, ".c"sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 3, false,
      { "a"sv, ",b"sv, "."sv, ".c"sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 3, true,
      { "a,"sv, "b."sv, "."sv, "c,"sv } },
    { "[,.]"sv, "a,b..c,"sv, 4, false,
      { "a"sv, ","sv, "b"sv, ".."sv, "c"sv, ","sv } },
    { "[,.]"sv, "a,b..c,"sv, 4, true,
      { "a"sv, ","sv, "b"sv, ".."sv, "c"sv, ","sv } },
    { "\\d+"sv, "12ab345c6"sv, 0, false,
      { "ab"sv, "c"sv } },
    { "\\d+"sv, "12ab345c6"sv, 0, true,
      { "12"sv, "345"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 1, false,
      { "12"sv, "ab"sv, "345"sv, "c"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 1, true,
      { "12"sv, "ab"sv, "345"sv, "c"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 2, false,
      { "12"sv, "ab345"sv, "c6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 2, true,
      { "12ab"sv, "345c"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 3, false,
      { "12ab"sv, "345c"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 3, true,
      { "12"sv, "ab345"sv, "c6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 4, false,
      { "12"sv, "ab"sv, "345"sv, "c"sv, "6"sv } },
    { "\\d+"sv, "12ab345c6"sv, 4, true,
      { "12"sv, "ab"sv, "345"sv, "c"sv, "6"sv } },
    { "x*"sv, "axxbx"sv, 0, false,
      { "a"sv, "b"sv } },
    { "x*"sv, "axxbx"sv, 0, true,
      { "xx"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 1, false,
      { "a"sv, "xx"sv, "b"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 1, true,
      { "a"sv, "xx"sv, "b"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 2, false,
      { "axx"sv, "bx"sv } },
    { "x*"sv, "axxbx"sv, 2, true,
      { "a"sv, "xxb"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 3, false,
      { "a"sv, "xxb"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 3, true,
      { "axx"sv, "bx"sv } },
    { "x*"sv, "axxbx"sv, 4, false,
      { "a"sv, "xx"sv, "b"sv, "x"sv } },
    { "x*"sv, "axxbx"sv, 4, true,
      { "a"sv, "xx"sv, "b"sv, "x"sv } },
    { "$"sv, "a\x0a""b"sv, 0, false,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 0, true,
      {  } },
    { "$"sv, "a\x0a""b"sv, 1, false,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 1, true,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 2, false,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 2, true,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 3, false,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 3, true,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 4, false,
      { "a"sv, "\x0a""b"sv } },
    { "$"sv, "a\x0a""b"sv, 4, true,
      { "a"sv, "\x0a""b"sv } },
};

}  /* namespace */

TEST(every_construct_splits_as_oniguruma_does) {
    for (const Case& c : k_cases) {
        auto re = compile(c.pattern);
        REQUIRE(re != nullptr);
        const std::vector<std::string> got = split(*re, c.text);
        std::vector<std::string> want(c.pieces.begin(), c.pieces.end());
        CHECK(got == want);
        if (got != want)
            std::fprintf(stderr, "    %.*s: /%.*s/\n      got  %s\n      want %s\n", (int)c.name.size(),
                         c.name.data(), (int)c.pattern.size(), c.pattern.data(), show(got).c_str(),
                         show(want).c_str());
    }
}

TEST(every_split_behaviour_and_invert_folds_as_huggingface_does) {
    for (const Behaviour& b : k_behaviours) {
        auto re = compile(b.pattern);
        REQUIRE(re != nullptr);
        const std::vector<std::string> got = split(*re, b.text, b.behavior, b.invert);
        std::vector<std::string> want(b.pieces.begin(), b.pieces.end());
        CHECK(got == want);
        if (got != want)
            std::fprintf(stderr, "    /%.*s/ behaviour %d invert %d\n      got  %s\n      want %s\n",
                         (int)b.pattern.size(), b.pattern.data(), b.behavior, b.invert ? 1 : 0,
                         show(got).c_str(), show(want).c_str());
    }
}

/* match_at is the primitive find_iter is built on: the leftmost-first match anchored at a
 * position, which is not necessarily the longest. */
TEST(match_at_is_leftmost_first_not_longest) {
    auto re = compile("a|ab|abc");
    REQUIRE(re != nullptr);
    CHECK_EQ(re->match_at("abc", 0), (int64_t)1);
    auto greedy = compile("a+");
    REQUIRE(greedy != nullptr);
    CHECK_EQ(greedy->match_at("aaab", 0), (int64_t)3);
    CHECK_EQ(greedy->match_at("aaab", 3), (int64_t)-1);
    auto lazy = compile("a+?");
    REQUIRE(lazy != nullptr);
    CHECK_EQ(lazy->match_at("aaab", 0), (int64_t)1);
    /* The lookahead is decided by the codepoint after the match, including the end of the text. */
    auto ws = compile("\\s+(?!\\S)");
    REQUIRE(ws != nullptr);
    CHECK_EQ(ws->match_at("   x", 0), (int64_t)2);
    CHECK_EQ(ws->match_at("   ", 0), (int64_t)3);
    CHECK_EQ(ws->match_at(" x", 0), (int64_t)-1);
}

/* Every construct outside the dialect is refused by name. The checked phrase is what a reader of
 * the load error needs to find the construct in a 200-character pattern. */
/* A possessive repetition is taken only where the greedy reading gives the same matches: one class
 * at the top level, its tail classes that cannot take its codepoints. Ling-3.0's pattern is such a
 * case, and it splits as its greedy spelling does. */
TEST(possessive_quantifiers_with_a_disjoint_tail_read_as_greedy) {
    const char* ling = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}"
                       "| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    const char* greedy = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
                         "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    auto a = compile(ling), b = compile(greedy);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    const char* texts[] = {
        "Hello, world! It's 2026... \"quoted\"   spaces\n\nnew lines\r\n--dashes--",
        "Die Patientin (67) hat seit 2 h Thoraxschmerzen; RR 158/94 mmHg, SpO\xe2\x82\x82 93 %.",
        "  ...!!!??? x\t\ty  \xe4\xb8\xad\xe6\x96\x87\xe5\xad\x97 ,,; 'LL 'Ve",
    };
    for (const char* t : texts) {
        std::vector<rad::TokSpan> x, y;
        a->find_iter(t, x);
        b->find_iter(t, y);
        REQUIRE_EQ(x.size(), y.size());
        for (size_t i = 0; i < x.size(); ++i) {
            CHECK_EQ(x[i].begin, y[i].begin);
            CHECK_EQ(x[i].end, y[i].end);
        }
    }
    auto plus = compile("a++"), star = compile("[^b]*+b?");
    REQUIRE(plus != nullptr);
    REQUIRE(star != nullptr);
    CHECK_EQ(plus->match_at("aaab", 0), (int64_t)3);
    CHECK_EQ(star->match_at("xyzb", 0), (int64_t)4);
}

TEST(constructs_outside_the_dialect_are_refused_by_name) {
    const struct { const char* pattern; const char* says; } refused[] = {
        { "(a)\\1",                 "backreference" },
        { "\\k<n>",                 "backreference" },
        { "(?>a+)",                 "atomic group" },
        { "a++a",                   "possessive" },
        { "[ab]*+b",                "possessive" },
        { "(?:a++)",                "possessive" },
        { "(a*+)b",                 "possessive" },
        { "(?:ab)++",               "possessive" },
        { "\\s++(?!\\S)",            "possessive" },
        { "a{2}{3}",                "quantifier applied to a quantifier" },
        { "a{1,2}+",                "quantifier applied to a quantifier" },
        { "a**",                    "quantifier applied to a quantifier" },
        { "\\d{1,3}(?=(?:\\d{3})*\\b)", "lookaround wider than one codepoint" },
        { "(?=ab)",                 "lookaround wider than one codepoint" },
        { "(?<=ab)c",               "lookaround wider than one codepoint" },
        { "(?i:\xc3\x9f)",          "multi-codepoint" },       /* U+00DF matches "ss" */
        { "(?i:[a-\xc3\xbf])",      "multi-codepoint" },       /* the range holds U+00DF */
        { "(?i:[a-z&&[^k]])",       "intersection under (?i)" },
        { "(?i:[\\p{Lu}])",         "property inside a class under (?i)" },
        { "(?i:[a[b]])",            "nested class under (?i)" },
        { "(a*)*",                  "can match empty" },
        { "(?:|a)+",                "can match empty" },
        { "(?:^)*",                 "quantifier on an assertion" },
        { "^*",                     "quantifier on an assertion" },
        { "[[:alpha:]]",            "POSIX bracket" },
        { "\\p{Alphabetic}",        "property the tables do not carry" },
        { "\\p{InBasicLatin}",      "property the tables do not carry" },
        { "\\p{Klingon}",           "property the tables do not carry" },
        { "\\pL",                   "without braces" },
        { "\\Z",                    "\\Z" },
        { "\\G",                    "\\G" },
        { "\\R",                    "\\R" },
        { "\\X",                    "\\X" },
        { "\\K",                    "\\K" },
        { "\\Qa.b\\E",              "\\Q" },
        { "\\y",                    "unknown escape" },
        { "(?s:.)",                 "unknown group option" },
        { "(?~abc)",                "absent operator" },
        { "a{1001}",                "above 1000" },
        { "a{3,2}",                 "minimum exceeds its maximum" },
        { "*a",                     "nothing to repeat" },
        { "(ab",                    "unterminated group" },
        { "ab)",                    "unmatched ')'" },
        { "[ab",                    "unterminated character class" },
        { "[b-a]",                  "out of order" },
        { "\\x{110000}",            "above U+10FFFF" },
        { "\xff",                   "not UTF-8" },
    };
    for (const auto& r : refused) {
        std::unique_ptr<TokRegex> re;
        std::string why;
        const int st = TokRegex::compile(r.pattern, &re, &why);
        CHECK_EQ(st, RAD_E_UNSUPPORTED);
        CHECK(why.find(r.says) != std::string::npos);
        if (why.find(r.says) == std::string::npos)
            std::fprintf(stderr, "    /%s/ refused with \"%s\", expected it to name \"%s\"\n",
                         r.pattern, why.c_str(), r.says);
    }
}

/* Shipped patterns compile to small machines: a pre-tokeniser DFA with thousands of states would
 * mean the construction had gone wrong, not that the pattern needs it. */
TEST(shipped_patterns_compile_to_small_dfas) {
    const char* shipped[] = {
        /* Qwen3.5 / 3.6 / 3.8 */
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* MiniCPM5 */
        "\\p{N}{1,3}",
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* Llama 3 */
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* GPT-2 */
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
        /* o200k */
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* Kimi K2 */
        "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* DeepSeek V3 */
        "[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        /* DeepSeek Coder */
        "\\s+$", "[\\r\\n]",
        /* BLOOM */
        " ?[^(\\s|[.,!?\xe2\x80\xa6\xe3\x80\x82\xef\xbc\x8c\xe3\x80\x81\xe0\xa5\xa4\xdb\x94\xd8\x8c])]+",
        /* llama.cpp's gpt-4o adaptation */
        "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "[^\\n]+|[\\n]+", "\\S+", "[\\p{P}\\$\\+<=>\\^~\\|]+", "[0-9][0-9][0-9]",
    };
    for (const char* p : shipped) {
        auto re = compile(p);
        CHECK(re != nullptr);
        if (!re) continue;
        CHECK(re->n_states() < 200);
        CHECK(re->n_classes() < 64);
    }
}

/* ================================================================== the text a split sees */

TEST(canonical_utf8_is_exactly_what_round_trips) {
    const struct { std::string_view s; bool canonical; } cases[] = {
        { ""sv, true }, { "abc"sv, true }, { "\xc3\xa9"sv, true }, { "\xe4\xb8\xad"sv, true },
        { "\xf0\x9f\x98\x80"sv, true }, { "\xf4\x8f\xbf\xbf"sv, true },
        { "\xed\xa0\x80"sv, true },                 /* a surrogate decodes and re-encodes as itself */
        { "\xc3"sv, false }, { "\x80"sv, false }, { "\xc0\x80"sv, false }, { "\xc1\xbf"sv, false },
        { "\xe0\x80\x80"sv, false }, { "\xe0\x9f\xbf"sv, false }, { "\xf0\x80\x80\x80"sv, false },
        { "\xf0\x8f\xbf\xbf"sv, false }, { "\xf4\x90\x80\x80"sv, false }, { "\xf5\x80\x80\x80"sv, false },
        { "\xf8\x88\x80\x80\x80"sv, false }, { "\xff"sv, false }, { "ab\xe4\xb8"sv, false },
        { "abcdefgh\xc3"sv, false },                /* past the eight-byte ASCII stride */
    };
    for (const auto& c : cases) {
        CHECK_EQ(utf8_is_canonical(c.s), c.canonical);
        std::string out;
        utf8_canonicalize(c.s, out);
        CHECK_EQ(out == c.s, c.canonical);
        CHECK(utf8_is_canonical(out));
    }
    /* The decoder's rules: one U+FFFD per byte that starts nothing decodable, overlong forms
     * shortened, surrogates kept. */
    const std::string fffd = "\xef\xbf\xbd";
    const struct { std::string_view in; std::string out; } fix[] = {
        { "\xc3"sv, fffd },
        { "\xe4\xb8" "a"sv, fffd + fffd + "a" },
        { "\xc0\x80"sv, std::string(1, '\0') },
        { "\xe0\x80\xaf"sv, "/" },
        { "\xf0\x82\x82\xac"sv, "\xe2\x82\xac" },
        { "\xf4\x90\x80\x80"sv, fffd + fffd + fffd + fffd },
        { "\xf8\x88\x80\x80\x80"sv, fffd + fffd + fffd + fffd + fffd },
        { "\xed\xa0\x80"sv, "\xed\xa0\x80" },
    };
    for (const auto& f : fix) {
        std::string out;
        utf8_canonicalize(f.in, out);
        CHECK(out == f.out);
    }
}

/* Replace follows HuggingFace: the content is inserted verbatim, `$` included, at every match,
 * empty ones too. */
TEST(replace_inserts_the_content_verbatim) {
    auto re = compile("\\s+");
    REQUIRE(re != nullptr);
    std::string out;
    tok_replace(*re, "a  b\t\nc ", "$1_", out);
    CHECK(out == "a$1_b$1_c$1_");
    auto empty = compile("x*");
    REQUIRE(empty != nullptr);
    out.clear();
    tok_replace(*empty, "ab", "-", out);
    CHECK(out == "-a-b-");
}

RAD_TEST_MAIN()
