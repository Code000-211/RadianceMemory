/* tests/shapes_test.cpp -- Hunyuan3D-2's API over a shape generator (core/server/shapes.cpp).
 *
 * The generator is faked here: it finishes every request on submit with a sphere's field on the
 * request's grid, so these pin what the server owns: which requests are refused and under which
 * field, the request it hands the generator (the prepared picture, the steps, the resolution, the
 * guidance, the seed), the mesh file /generate answers with, and the background job seen through
 * /send and /status.
 */
#include "server_fakes.h"

#include "gen/shape.h"
#include "mm/media.h"
#include "server/oai.h"

#include <chrono>
#include <thread>

namespace {

struct FakeShapeGen : IShapeGen {
    gen::ShapeGenInfo inf;
    int         finish_status = RAD_OK;
    std::string finish_why;
    /* What each request asked, kept after the request itself goes. */
    struct Seen {
        int32_t steps, resolution;
        double guidance;
        uint64_t seed;
        bool dense;
        size_t n;
    };
    std::vector<Seen> got;
    mutable std::mutex got_m;

    FakeShapeGen() {
        RadShapeGenDecl d{};
        d.canvas = 32;
        d.border_ratio = 0.15;
        d.enc_size = 28;
        d.enc_patch = 14;
        for (int c = 0; c < 3; ++c) { d.pix_mean[c] = 0.5; d.pix_std[c] = 0.25; }
        d.default_steps = 50;
        d.max_steps = 200;
        d.default_guidance = 5.0;
        d.bounds = 1.01;
        d.level = 0.0;
        d.default_resolution = 24;
        d.min_resolution = 8;
        d.max_resolution = 48;
        d.max_lanes = 2;
        inf = gen::shape_info(d, "fake-shape");
    }
    const gen::ShapeGenInfo& info() const override { return inf; }
    int submit(std::unique_ptr<gen::ShapeRequest> r, std::string*) override {
        std::lock_guard<std::mutex> lk(got_m);
        const int32_t res = r->resolution > 0 ? r->resolution : (int32_t)inf.default_resolution;
        got.push_back({ r->steps, r->resolution, r->guidance, r->seed, r->dense,
                        r->patches.size() });
        gen::ShapeTicket& t = *r->ticket;
        t.steps = (r->steps > 0 ? r->steps : 50) - 1;
        t.resolution = res;
        if (finish_status == RAD_OK) {
            const std::vector<float> ax = gen::shape_axis(inf.bounds, res);
            const int64_t n = (int64_t)res + 1;
            std::vector<float> f((size_t)(n * n * n));
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t k = 0; k < n; ++k) {
                        const float x = ax[(size_t)i], y = ax[(size_t)j], z = ax[(size_t)k];
                        f[(size_t)((i * n + j) * n + k)] = 0.5f - std::sqrt(x * x + y * y + z * z);
                    }
            t.set_field(std::move(f), res);
        }
        t.finish(finish_status, finish_why);
        return RAD_OK;
    }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
    Seen last() const {
        std::lock_guard<std::mutex> lk(got_m);
        return got.back();
    }
};

/* A 16 x 16 PNG, opaque inside [4, 12)^2 and transparent around it, as base64. */
std::string picture_b64(bool opaque_anywhere = true) {
    std::vector<uint8_t> px((size_t)16 * 16 * 4, 0);
    for (int y = 4; y < 12 && opaque_anywhere; ++y)
        for (int x = 4; x < 12; ++x) {
            uint8_t* p = &px[((size_t)y * 16 + x) * 4];
            p[0] = 30; p[1] = 120; p[2] = 220; p[3] = 255;
        }
    std::vector<uint8_t> file;
    std::string why;
    if (mm::encode_image(px.data(), 16, 16, 4, "png", 0, &file, &why) < 0) return "";
    return base64_encode(file.data(), file.size());
}

struct Shapes {
    FakeShapeGen gen;
    std::unique_ptr<Fixture> f;
    Shapes() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.shapes = &gen;
            o.shape_model = true;
        } }));
    }
    static json body(json extra = json::object()) {
        json b = { { "image", picture_b64() } };
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
        return b;
    }
    HttpResponse generate(const json& b) { return f->post("/generate", b.dump()); }
    std::string refused(const json& b, int status = 400) {
        HttpResponse r = generate(b);
        CHECK_EQ(r.status, status);
        return str_at(json::parse(r.body, nullptr, false)["error"], "param");
    }
};

}  /* namespace */

/* A request that cannot be served is a 400 naming its field before anything is queued. */
TEST(a_shape_request_that_cannot_be_served_names_its_field) {
    if (!mm::decoder_available()) return;
    Shapes s;
    CHECK_EQ(s.refused(json::object()), std::string("image"));
    CHECK_EQ(s.refused({ { "image", "not base64 !" } }), std::string("image"));
    CHECK_EQ(s.refused({ { "image", picture_b64(false) } }), std::string("image"));
    CHECK_EQ(s.refused(Shapes::body({ { "texture", true } })), std::string("texture"));
    CHECK_EQ(s.refused(Shapes::body({ { "text", "a penguin" } })), std::string("text"));
    CHECK_EQ(s.refused(Shapes::body({ { "mesh", "Z2xURg==" } })), std::string("mesh"));
    CHECK_EQ(s.refused(Shapes::body({ { "type", "fbx" } })), std::string("type"));
    CHECK_EQ(s.refused(Shapes::body({ { "num_inference_steps", 1 } })),
             std::string("num_inference_steps"));
    CHECK_EQ(s.refused(Shapes::body({ { "octree_resolution", 4096 } })),
             std::string("octree_resolution"));
    CHECK_EQ(s.refused(Shapes::body({ { "guidance_scale", -1 } })), std::string("guidance_scale"));
    CHECK_EQ(s.refused(Shapes::body({ { "volume_decoding", "flashvdm" } })),
             std::string("volume_decoding"));
    CHECK_EQ(s.refused(Shapes::body({ { "mc_algo", "dmc" } })), std::string("mc_algo"));
    CHECK_EQ(s.refused(Shapes::body({ { "temperature", 1 } })), std::string("temperature"));
    CHECK_EQ(s.gen.got.size(), (size_t)0);
}

/* The generator gets the prepared picture and the request's numbers; the answer is the mesh as
 * the file `type` names. */
TEST(generate_answers_with_the_mesh_file) {
    if (!mm::decoder_available()) return;
    Shapes s;
    HttpResponse r = s.generate(Shapes::body({ { "num_inference_steps", 30 },
                                               { "octree_resolution", 32 },
                                               { "guidance_scale", 7.5 }, { "seed", 99 },
                                               { "face_count", 40000 }, { "mc_algo", "mc" } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(r.content_type, std::string("model/gltf-binary"));
    REQUIRE(r.body.size() > 12);
    CHECK_EQ(r.body.substr(0, 4), std::string("glTF"));
    const FakeShapeGen::Seen q = s.gen.last();
    CHECK_EQ(q.steps, 30);
    CHECK_EQ(q.resolution, 32);
    CHECK_EQ(q.guidance, 7.5);
    CHECK_EQ(q.seed, (uint64_t)99);
    CHECK(!q.dense);
    CHECK_EQ(q.n, (size_t)(2 * 2 * 3 * 14 * 14));
    /* The reference's defaults: seed 1234, the model's steps and resolution. */
    r = s.generate(Shapes::body({ { "type", "obj" }, { "volume_decoding", "dense" } }));
    REQUIRE_EQ(r.status, 200);
    CHECK_EQ(r.content_type, std::string("model/obj"));
    CHECK(r.body.rfind("v ", 0) == 0);
    CHECK_EQ(s.gen.last().seed, (uint64_t)1234);
    CHECK_EQ(s.gen.last().steps, 0);
    CHECK(s.gen.last().dense);
}

/* A failed shape is the generator's reason, not a file. */
TEST(a_failed_shape_reports_the_reason) {
    if (!mm::decoder_available()) return;
    Shapes s;
    s.gen.finish_status = RAD_E_STATE;
    s.gen.finish_why = "the engine stopped";
    HttpResponse r = s.generate(Shapes::body());
    CHECK(r.status >= 400);
    CHECK(r.body.find("the engine stopped") != std::string::npos);
}

/* /send queues the shape and answers with its uid; /status reports it until the file is in
 * `model_base64`, and an unknown uid is a 404. */
TEST(send_and_status_make_the_shape_in_the_background) {
    if (!mm::decoder_available()) return;
    Shapes s;
    HttpResponse r = s.f->post("/send", Shapes::body({ { "type", "stl" } }).dump());
    REQUIRE_EQ(r.status, 200);
    const std::string uid = str_at(json::parse(r.body, nullptr, false), "uid");
    REQUIRE(!uid.empty());
    json o;
    for (int i = 0; i < 400; ++i) {
        o = json::parse(s.f->get(("/status/" + uid).c_str()).body, nullptr, false);
        if (str_at(o, "status") != "processing") break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE_EQ(str_at(o, "status"), std::string("completed"));
    std::vector<uint8_t> file;
    REQUIRE(base64_decode(str_at(o, "model_base64"), &file));
    CHECK(file.size() > 84);
    CHECK_EQ(s.f->get("/status/shape_nope").status, 404);
}

/* A model that makes no shapes answers the routes with a 501. */
TEST(the_shape_routes_need_a_shape_model) {
    Fixture f;
    CHECK_EQ(f.post("/generate", "{}").status, 501);
    CHECK_EQ(f.post("/send", "{}").status, 501);
    CHECK_EQ(f.get("/status/x").status, 501);
}

RAD_TEST_MAIN()
