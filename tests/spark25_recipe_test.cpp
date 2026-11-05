/* Regression tests for the shipped Spark IQ8R recipe allocations. */
#include "rad_test.h"
#include "format/recipe.h"

#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

using namespace rad;

static bool load_recipe(const char* name, Recipe* recipe) {
    const std::string path = std::string(RAD_SPARK25_RECIPE_DIR) + "/" + name;
    std::ifstream in(path);
    if (!in) return false;
    std::ostringstream text;
    text << in.rdbuf();
    std::string why;
    return recipe->add_text(text.str(), path, &why) == RAD_OK;
}

static std::string option(const RecipeRule& rule, const std::string& key) {
    for (const auto& [k, v] : rule.options)
        if (k == key) return v;
    return {};
}

static void check_iq8r_rule(const RecipeRule* rule, const char* quantizer, bool calibrated) {
    REQUIRE(rule != nullptr);
    CHECK_EQ(rule->quantizer, std::string(quantizer));
    CHECK_EQ(option(*rule, "codes"), std::string("i8"));
    CHECK_EQ(option(*rule, "group"), std::string("128"));
    CHECK_EQ(option(*rule, "scale"), std::string("bf16"));
    CHECK_EQ(option(*rule, "clamp"), std::string("sym"));
    CHECK_EQ(option(*rule, "rule"), std::string("search"));
    if (calibrated) {
        CHECK_EQ(option(*rule, "transform"), std::string("fwht128"));
        CHECK_EQ(option(*rule, "calib"), std::string("/test/spark-grams"));
    } else {
        CHECK(option(*rule, "transform").empty());
        CHECK(option(*rule, "calib").empty());
    }
}

TEST(spark_iq8r_recipes_allocate_the_quality_and_full_variants_explicitly) {
    REQUIRE_EQ(::setenv("CALIB", "/test/spark-grams", 1), 0);
    const char* projections[] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"};
    Recipe keep12, full;
    REQUIRE(load_recipe("spark-x26-4b-iq8r-keep12.recipe", &keep12));
    REQUIRE(load_recipe("spark-x26-4b-iq8r.recipe", &full));

    for (int layer = 0; layer < 36; ++layer) {
        for (const char* projection : projections) {
            const std::string name = "blk." + std::to_string(layer) + "." + projection + ".weight";
            const RecipeRule* hybrid = keep12.match(name);
            if (layer < 12) {
                CHECK(hybrid == nullptr);  // unmatched tensors retain source BF16
            } else {
                check_iq8r_rule(hybrid, "gptq", true);
            }
            check_iq8r_rule(full.match(name), "gptq", true);
        }
    }
    check_iq8r_rule(keep12.match("output.weight"), "rtn", false);
    check_iq8r_rule(full.match("output.weight"), "rtn", false);
    CHECK(keep12.match("blk.0.attn_gate.weight") == nullptr);
    CHECK(keep12.match("token_embd.weight") == nullptr);
    CHECK(full.match("blk.35.input_norm.weight") == nullptr);
}

RAD_TEST_MAIN()
