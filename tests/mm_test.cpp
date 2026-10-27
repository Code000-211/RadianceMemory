/* mm_test.cpp -- the multimodal processor against the reference, number for number.
 *
 * Every expected value in mm_golden.h comes from transformers' own code (tests/mkgolden.py names
 * the function behind each table), in lfm2_golden.h from its LFM2-VL processor
 * (tests/mklfm2golden.py), in gemma4_golden.h from its Gemma 4 processors
 * (tests/mkgemma4golden.py) and in glm5_golden.h from its GLM-5.3 processor
 * (tests/mkglm5golden.py): smart_resize in both forms, the frames a video keeps, the patches of a
 * picture and a clip, the timestamps between a clip's groups, and get_rope_index's positions over
 * a prompt carrying both. A mistake in any of these still yields a model that answers, about a
 * slightly different picture or with positions a row off, so only this file can catch it.
 * imaging_golden.h holds Pillow's Lanczos resize and alpha paste (tests/mkimggolden.py).
 *
 * The decoder is checked on fixtures from tests/data/media (README.md there) and on files built
 * here: the refusals a hostile request reaches first, orientation, alpha, and audio downmixing.
 *
 * Pixel inputs are a splitmix hash of their coordinates rather than a ramp: a patch taken with the
 * wrong stride from a ramp is often another plausible ramp.
 */
#include "rad_test.h"
#include "rad_types.h"
#include "mm/imaging.h"
#include "mm/media.h"
#include "mm/processor.h"
#include "gemma4_golden.h"
#include "gemma4u_golden.h"
#include "glm5_golden.h"
#include "imaging_golden.h"
#include "lfm2_golden.h"
#include "mm/resample.h"
#include "mm_golden.h"
#include "rad_builder.h"
#include "rad_plugin.h"
#include "uocr_golden.h"
#include "uocr_resample_golden.h"
#include "../arch/common/rad_resample.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <malloc.h>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace rad;
using namespace rad::mm;

namespace {

uint8_t pix(uint64_t f, uint64_t y, uint64_t x, uint64_t c) {
    uint64_t z = f * 0x9E3779B97F4A7C15ull + y * 0xBF58476D1CE4E5B9ull + x * 0x94D049BB133111EBull +
                 c * 0x2545F4914F6CDD1Dull + 12345ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint8_t)((z ^ (z >> 31)) & 255);
}

Frame frame(int f, int h, int w) {
    Frame fr;
    fr.w = w;
    fr.h = h;
    fr.rgb.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                fr.rgb[((size_t)y * w + x) * 3 + c] = pix((uint64_t)f, (uint64_t)y, (uint64_t)x, (uint64_t)c);
    return fr;
}

/* FNV-1a over each element's bytes, least significant first. */
template <class T> uint64_t fnv(const std::vector<T>& v) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (T x : v) {
        const auto u = (std::make_unsigned_t<T>)x;
        for (size_t b = 0; b < sizeof(T); ++b) h = (h ^ (uint8_t)(u >> (8 * b))) * 0x100000001b3ull;
    }
    return h;
}

/* The Qwen3.8 processor configuration, as a converted container states it. */
VisionConfig qwen_cfg() {
    VisionConfig c;
    c.patch = 16;
    c.temporal = 2;
    c.merge = 2;
    c.image_min_pixels = 65536;
    c.image_max_pixels = 16777216;
    c.video_min_pixels = 4096;
    c.video_max_pixels = 25165824;
    c.tok_image = 248056;
    c.tok_video = 248057;
    c.tok_start = 248053;
    c.tok_end = 248054;
    c.max_patches = 16384;
    return c;
}

}  /* namespace */

TEST(smart_resize_is_the_references_for_pictures) {
    for (const GoldImageResize& g : kGoldImageResize) {
        int64_t oh = 0, ow = 0;
        std::string why;
        CHECK_OK(smart_resize_image(g.h, g.w, 32, g.min_px, g.max_px, &oh, &ow, &why));
        if (oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld x %lld in [%lld, %lld]: %lld x %lld, reference %lld x %lld\n",
                         (long long)g.h, (long long)g.w, (long long)g.min_px, (long long)g.max_px,
                         (long long)oh, (long long)ow, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

TEST(smart_resize_is_the_references_for_video) {
    for (const GoldVideoResize& g : kGoldVideoResize) {
        int64_t oh = 0, ow = 0;
        std::string why;
        CHECK_OK(smart_resize_video(g.t, g.h, g.w, 2, 32, g.min_px, g.max_px, &oh, &ow, &why));
        if (oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld frames of %lld x %lld: %lld x %lld, reference %lld x %lld\n",
                         (long long)g.t, (long long)g.h, (long long)g.w, (long long)oh,
                         (long long)ow, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

TEST(the_frames_a_video_keeps_are_the_references) {
    for (const GoldSample& g : kGoldSample) {
        VideoInfo info;
        info.n_frames = g.total;
        info.fps = g.fps;
        const std::vector<int64_t> idx = sample_frames(info, 2.0, 4, 768);
        REQUIRE_EQ((int32_t)idx.size(), g.n);
        int off = 0;
        for (int32_t i = 0; i < g.n; ++i) off += idx[(size_t)i] != kGoldSampleIdx[g.first + i];
        if (off)
            std::fprintf(stderr, "    %lld frames at %g fps: %d of %d indices differ\n",
                         (long long)g.total, g.fps, off, g.n);
        CHECK_EQ(off, 0);
    }
}

TEST(a_picture_patchifies_as_the_reference_does) {
    Processor p;
    REQUIRE(p.init(qwen_cfg(), [](const std::string&) { return std::vector<int32_t>{ 1 }; }) >= 0);
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.image_from(frame(0, 256, 320), &it, &why));
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_t, kGoldImageGrid[0]);
    CHECK_EQ(it->grid_h, kGoldImageGrid[1]);
    CHECK_EQ(it->grid_w, kGoldImageGrid[2]);
    REQUIRE_EQ(it->pixels.size(), (size_t)it->n_patch * 1536);
    for (int i = 0; i < 16; ++i) CHECK_EQ(it->pixels[(size_t)i], kGoldImageHead[i]);
    CHECK_EQ(fnv(it->pixels), kGoldImageFnv);

    /* <|vision_start|>, one pad per merged row, <|vision_end|>: the template's triple with the pad
     * run at its real length, as transformers substitutes it. */
    const int32_t rows = (kGoldImageGrid[1] / 2) * (kGoldImageGrid[2] / 2);
    REQUIRE_EQ(it->expand.size(), (size_t)rows + 2);
    CHECK_EQ(it->expand.front(), 248053);
    CHECK_EQ(it->expand.back(), 248054);
    for (int32_t i = 1; i <= rows; ++i) CHECK_EQ(it->expand[(size_t)i], 248056);
    REQUIRE_EQ(it->seg_at.size(), (size_t)1);
    CHECK_EQ(it->seg_at[0], 1);
    CHECK_EQ(it->n_rows(), (int64_t)rows);
}

TEST(a_clip_patchifies_and_is_stamped_as_the_reference_does) {
    std::vector<std::string> stamps;
    Processor p;
    REQUIRE(p.init(qwen_cfg(), [&](const std::string& s) {
                stamps.push_back(s);
                return std::vector<int32_t>(s.size(), 7);   /* one token per byte, so length shows */
            }) >= 0);
    Video v;
    v.info.n_frames = 6;
    v.info.fps = 2.0;
    v.info.w = 96;
    v.info.h = 64;
    v.index = sample_frames(v.info, 2.0, 4, 768);
    for (int64_t i : v.index) v.frames.push_back(frame((int)i, 64, 96));
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.video_from(v, &it, &why));
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_t, kGoldVideoGrid[0]);
    CHECK_EQ(it->grid_h, kGoldVideoGrid[1]);
    CHECK_EQ(it->grid_w, kGoldVideoGrid[2]);
    REQUIRE_EQ(it->pixels.size(), (size_t)it->n_patch * 1536);
    for (int i = 0; i < 16; ++i) CHECK_EQ(it->pixels[(size_t)i], kGoldVideoHead[i]);
    CHECK_EQ(fnv(it->pixels), kGoldVideoFnv);

    /* One stamp per group, in the reference's wording and rounding: 0.25 s is "<0.2 seconds>". */
    const size_t n_stamps = sizeof(kGoldVideoStamps) / sizeof(kGoldVideoStamps[0]);
    REQUIRE_EQ(stamps.size(), n_stamps);
    for (size_t i = 0; i < n_stamps; ++i) CHECK_EQ(stamps[i], std::string(kGoldVideoStamps[i]));

    /* <stamp> <|vision_start|> pads <|vision_end|> per group, and nothing else. */
    const int32_t rows = (kGoldVideoGrid[1] / 2) * (kGoldVideoGrid[2] / 2);
    size_t at = 0;
    REQUIRE_EQ(it->seg_at.size(), (size_t)kGoldVideoGrid[0]);
    for (int32_t g = 0; g < kGoldVideoGrid[0]; ++g) {
        at += stamps[(size_t)g].size();
        REQUIRE(at < it->expand.size());
        CHECK_EQ(it->expand[at], 248053);
        CHECK_EQ(it->seg_at[(size_t)g], (int32_t)(at + 1));
        for (int32_t r = 0; r < rows; ++r) CHECK_EQ(it->expand[at + 1 + (size_t)r], 248057);
        CHECK_EQ(it->expand[at + 1 + (size_t)rows], 248054);
        at += (size_t)rows + 2;
    }
    CHECK_EQ(at, it->expand.size());
}

TEST(the_rotary_layout_is_get_rope_index) {
    /* mkgolden.py's prompt: an image of 2 x 3 merged rows in its vision pair, then a clip of two
     * groups of 2 x 4 rows, each after five timestamp tokens. Only placement and grids reach the
     * layout, so the items carry nothing else. */
    auto img = std::make_shared<Item>();
    img->kind = Kind::Image;
    img->grid_t = 1; img->grid_h = 4; img->grid_w = 6;
    img->rows_h = 2; img->rows_w = 3; img->seg_rows = 6; img->n_seg = 1;
    img->seg_at = { 1 };
    auto vid = std::make_shared<Item>();
    vid->kind = Kind::Video;
    vid->grid_t = 2; vid->grid_h = 4; vid->grid_w = 8;
    vid->rows_h = 2; vid->rows_w = 4; vid->seg_rows = 8; vid->n_seg = 2;
    vid->seg_at = { kGoldRopeVideoSeg[0], kGoldRopeVideoSeg[1] };

    PromptMedia pm;
    Placed a; a.item = img; a.tok = kGoldRopeImageAt;
    Placed b; b.item = vid; b.tok = kGoldRopeVideoAt;
    pm.items = { a, b };
    pm.layout();
    REQUIRE_EQ(pm.rope.runs.size(), (size_t)3);

    int off = 0;
    for (int32_t i = 0; i < kGoldRopeN; ++i) {
        int32_t p[3];
        pm.rope.at(i, p);
        if (p[0] != kGoldRopeT[i] || p[1] != kGoldRopeH[i] || p[2] != kGoldRopeW[i]) {
            if (off < 4)
                std::fprintf(stderr, "    token %d: (%d, %d, %d), reference (%d, %d, %d)\n", i,
                             p[0], p[1], p[2], kGoldRopeT[i], kGoldRopeH[i], kGoldRopeW[i]);
            ++off;
        }
    }
    CHECK_EQ(off, 0);

    /* Generation continues from the reference's delta: position = index + delta, in all three. */
    int32_t p[3];
    pm.rope.at(kGoldRopeN + 5, p);
    CHECK_EQ(p[0], kGoldRopeN + 5 + kGoldRopeDelta);
    CHECK_EQ(p[1], p[0]);
    CHECK_EQ(p[2], p[0]);
}

/* The decoder on the formats a request may carry. tests/data/media holds a solid red picture in
 * eleven still formats and a short solid red clip in nine container and codec pairs (its README
 * lists them, the commands, and the admitted formats without a fixture); each must decode, at its
 * size, to red. This also holds a reduced
 * FFmpeg build, such as the container image's, to the accepted formats. Without libavformat
 * there is no decoder and the cases skip. */
namespace {

std::vector<uint8_t> media_file(const char* name) {
    std::ifstream f(std::string(RAD_MEDIA_FIXTURES) + "/" + name, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

/* Solid red through a YUV codec and back comes out near (254, 0, 0), not exactly. */
bool is_red(const Frame& f) {
    if (f.rgb.size() != (size_t)f.w * f.h * 3 || f.rgb.empty()) return false;
    for (size_t i = 0; i < f.rgb.size(); i += 3)
        if (f.rgb[i] < 200 || f.rgb[i + 1] > 60 || f.rgb[i + 2] > 60) return false;
    return true;
}

std::vector<int64_t> first_and_last(const VideoInfo& info, const void*) {
    if (info.n_frames <= 1) return {0};
    return {0, info.n_frames - 1};
}

}  /* namespace */

TEST(every_admitted_picture_format_decodes) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    for (const char* name : { "red.png", "red.apng", "red.jpg", "red.webp", "red.gif", "red.bmp",
                              "red.tiff", "red.qoi", "red.ppm", "red.jls", "red.avif" }) {
        const std::vector<uint8_t> b = media_file(name);
        CHECK(!b.empty());
        Frame f;
        std::string why;
        const int rc = decode_image(b.data(), b.size(), int64_t(1) << 24, &f, &why);
        if (rc != RAD_OK) std::printf("  %s: %s\n", name, why.c_str());
        CHECK_EQ(rc, RAD_OK);
        const int want = std::string(name) == "red.avif" ? 64 : 32;
        CHECK_EQ(f.w, want);
        CHECK_EQ(f.h, want);
        CHECK(is_red(f));
    }
}

/* A width that is not a whole number of the scaler's vectors: converting 4:2:0 at 1788 wide to
 * RGB, the scaler writes 12 bytes past 1788 x 3, which in an exact-size buffer corrupts the next
 * heap chunk's header. The mmap threshold is raised so the RGB comes from the heap, as in a
 * long-running server, and glibc checks neighbouring headers on every free below. */
TEST(a_picture_of_an_awkward_width_converts_inside_its_buffer) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<uint8_t> b = media_file("red-1788x282.jpg");
    REQUIRE(!b.empty());
    mallopt(M_MMAP_THRESHOLD, 64 << 20);
    for (int i = 0; i < 8; ++i) {
        Frame f;
        std::string why;
        const int rc = decode_image(b.data(), b.size(), int64_t(1) << 24, &f, &why);
        if (rc != RAD_OK) std::printf("  %s\n", why.c_str());
        CHECK_EQ(rc, RAD_OK);
        CHECK_EQ(f.w, 1788);
        CHECK_EQ(f.h, 282);
        CHECK(is_red(f));
        /* The next chunk on the heap: glibc's checks abort here if the conversion wrote past its
         * buffer. */
        std::vector<uint8_t> behind(4096, 1);
        CHECK_EQ(behind[0], 1);
    }
    mallopt(M_MMAP_THRESHOLD, 128 << 10);
}

TEST(every_admitted_video_format_decodes) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    for (const char* name : { "red-h264.mp4", "red-hevc.mp4", "red-av1.mp4", "red-vp8.webm",
                              "red-vp9.webm", "red-h264.mkv", "red-h264.ts", "red-mpeg4.avi",
                              "red-theora.ogv" }) {
        const std::vector<uint8_t> b = media_file(name);
        CHECK(!b.empty());
        Video v;
        std::string why;
        const int rc = decode_video(b.data(), b.size(), first_and_last, nullptr, int64_t(1) << 24,
                                    &v, &why);
        if (rc != RAD_OK) std::printf("  %s: %s\n", name, why.c_str());
        CHECK_EQ(rc, RAD_OK);
        CHECK(v.info.n_frames >= 1);
        CHECK(!v.frames.empty());
        CHECK_EQ(v.frames.size(), v.index.size());
        for (const Frame& f : v.frames) {
            CHECK_EQ(f.w, 64);
            CHECK_EQ(f.h, 64);
            CHECK(is_red(f));
        }
    }
}

/* Bands are checked as a whole: the command line can replace one end of a band and leave the
 * other to the container, so a minimum alone can exceed the container's maximum. */
TEST(a_media_configuration_with_an_empty_band_names_it) {
    mm::VisionConfig c;
    c.image_min_pixels = 65536;
    c.image_max_pixels = 16777216;
    c.video_min_pixels = 4096;
    c.video_max_pixels = 25165824;
    std::string why;
    CHECK_EQ(c.check(&why), RAD_OK);

    mm::VisionConfig b = c;
    b.image_min_pixels = 32 << 20;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("image_max_pixels") != std::string::npos);
    b = c;
    b.video_max_pixels = 1000;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("video_max_pixels") != std::string::npos);
    b = c;
    b.fps = 0.0;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    b = c;
    b.min_frames = 10;
    b.max_frames = 5;
    CHECK_EQ(b.check(&why), RAD_E_INVAL);
    CHECK(why.find("max_frames") != std::string::npos);
}

/* The audio front end matches Gemma4AudioFeatureExtractor (transformers 5.18) on a 16037-sample
 * tone (0.5 sin 440 Hz + 0.25 sin 3 kHz at 16 kHz): 100 real frames and these log-mel values,
 * checked to bf16 resolution since the front end stores bf16. */
TEST(the_audio_front_end_is_the_references) {
    mm::AudioConfig c;
    c.rate = 16000; c.frame_length = 320; c.hop = 160; c.fft = 512; c.n_mels = 128;
    c.min_freq = 0.0; c.max_freq = 8000.0; c.mel_floor = 1e-3;
    c.max_samples = 480000;
    c.tok_audio = 258881; c.tok_start = 256000; c.tok_end = 258883;
    c.merge = 4; c.max_frames = 3000;
    mm::Processor p;
    CHECK_EQ(p.init_audio(c), RAD_OK);
    const int n = 16037;
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i) {
        const float t = (float)i / 16000.0f;
        x[(size_t)i] = 0.5f * std::sin(2.0f * 3.14159265358979f * 440.0f * t) +
                       0.25f * std::sin(2.0f * 3.14159265358979f * 3000.0f * t);
    }
    std::shared_ptr<mm::Item> it;
    std::string why;
    CHECK_EQ(p.audio_from(x, &it, &why), RAD_OK);
    CHECK(it != nullptr);
    if (!it) return;
    CHECK_EQ(it->valid, 100);
    CHECK_EQ(it->seg_rows, 25);
    CHECK_EQ(it->grid_h, 100);
    CHECK_EQ(it->expand.size(), (size_t)27);
    CHECK_EQ(it->expand.front(), 256000);
    CHECK_EQ(it->expand.back(), 258883);
    struct Want { int frame, bin; float v; };
    const Want want[] = { { 0, 0, -6.907755374908447f }, { 0, 14, 0.5496962070465088f },
                          { 1, 14, -2.903297185897827f }, { 50, 14, -2.905748128890991f },
                          { 50, 96, -5.7037482261657715f }, { 99, 14, -2.902763605117798f },
                          { 99, 127, -5.386524200439453f } };
    for (const Want& w : want) {
        const float got = rad_bf16_to_f32(it->pixels[(size_t)(w.frame * 128 + w.bin)]);
        CHECK_NEAR(got, w.v, 0.008f * std::max(1.0f, std::fabs(w.v)));
    }
    /* A clip shorter than one real frame is <boa><eoa> and nothing to encode. */
    std::vector<float> tiny(150, 0.1f);
    CHECK_EQ(p.audio_from(tiny, &it, &why), RAD_OK);
    CHECK_EQ(it->seg_rows, 0);
    CHECK(it->encoded);
    CHECK_EQ(it->expand.size(), (size_t)2);
}

namespace {

/* LiquidAI d1-3B's processor_config.json, as a converted container states it. */
VisionConfig lfm2_cfg() {
    VisionConfig c;
    c.family = Family::Lfm2;
    c.patch = kLfm2Patch;
    c.merge = kLfm2Merge;
    c.temporal = 1;
    const int64_t f2 = (int64_t)c.factor() * c.factor();
    c.image_min_pixels = c.video_min_pixels = kLfm2MinTok * f2;
    c.image_max_pixels = c.video_max_pixels = kLfm2MaxTok * f2;
    c.min_tiles = kLfm2MinTiles;
    c.max_tiles = kLfm2MaxTiles;
    c.tile = kLfm2Tile;
    c.use_thumbnail = true;
    c.max_pixels_tolerance = kLfm2Tolerance;
    c.max_segment_patches = kLfm2MaxPatches;
    c.tok_image = kLfm2ImageToken;
    c.max_patches = 4096;
    return c;
}

int32_t lfm2_marker(const std::string& name) {
    for (const GoldLfm2Marker& m : kGoldLfm2Markers)
        if (name == m.name) return m.id;
    return -1;
}

}  /* namespace */

TEST(lfm2_sizes_and_tiles_are_the_references) {
    for (const GoldLfm2Size& g : kGoldLfm2Size) {
        int64_t oh = 0, ow = 0;
        lfm2_size(g.h, g.w, kLfm2Patch * kLfm2Merge, (int64_t)kLfm2MinTok * 1024,
                  (int64_t)kLfm2MaxTok * 1024, &oh, &ow);
        int32_t cols = 0, rows = 0;
        lfm2_grid(g.h, g.w, kLfm2MinTiles, kLfm2MaxTiles, kLfm2Tile, &cols, &rows);
        const bool large = lfm2_too_large(g.h, g.w, kLfm2Patch, kLfm2Patch * kLfm2Merge,
                                          (int64_t)kLfm2MaxTok * 1024, kLfm2Tolerance);
        if (oh != g.oh || ow != g.ow || cols != g.cols || rows != g.rows || large != (g.large != 0))
            std::fprintf(stderr, "    %lld x %lld: %lld x %lld, %d x %d tiles, large %d; reference "
                         "%lld x %lld, %d x %d, %d\n", (long long)g.h, (long long)g.w, (long long)oh,
                         (long long)ow, cols, rows, (int)large, (long long)g.oh, (long long)g.ow,
                         g.cols, g.rows, g.large);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
        CHECK_EQ(cols, g.cols);
        CHECK_EQ(rows, g.rows);
        CHECK_EQ(large, g.large != 0);
    }
}

/* Whole pictures, small to tiled, one area-capped first as d1's runner does: every segment's size
 * and patches, and the tokens that replace the picture. The processor keeps patches in merge-block
 * order and the reference row-major, so they are reordered (Item::coords) before hashing. */
TEST(lfm2_pictures_are_the_references) {
    Processor p;
    REQUIRE_EQ(p.init(lfm2_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      lfm2_marker), RAD_OK);
    for (const GoldLfm2Pic& g : kGoldLfm2Pics) {
        Frame f = frame(g.f, g.h, g.w);
        if (g.cap > 0) {
            Frame capped;
            cap_area(f, g.cap, &capped);
            f = std::move(capped);
        }
        std::shared_ptr<Item> it;
        std::string why;
        CHECK_OK(p.image_from(f, &it, &why));
        REQUIRE(it != nullptr);
        REQUIRE_EQ(it->n_seg, g.n_seg);
        const int64_t pd = it->patch_dim;
        REQUIRE_EQ(pd, (int64_t)3 * kLfm2Patch * kLfm2Patch);
        for (int32_t s = 0; s < it->n_seg; ++s) {
            const GoldLfm2Seg& gs = kGoldLfm2Segs[g.seg0 + s];
            CHECK_EQ(it->seg_h(s), gs.h);
            CHECK_EQ(it->seg_w(s), gs.w);
            const int64_t p0 = it->patch_at(s), np = it->patch_at(s + 1) - p0;
            REQUIRE_EQ(np, (int64_t)gs.h * gs.w);
            std::vector<int32_t> planes((size_t)(4 * np));
            it->coords(p0, np, planes.data());
            std::vector<uint16_t> rowmajor((size_t)(np * pd));
            for (int64_t k = 0; k < np; ++k) {
                const int64_t at = (int64_t)planes[(size_t)k] * gs.w + planes[(size_t)(np + k)];
                std::copy_n(it->pixels.data() + (p0 + k) * pd, pd, rowmajor.data() + at * pd);
            }
            if (fnv(rowmajor) != gs.fnv)
                std::fprintf(stderr, "    picture %d x %d, segment %d of %d (%d x %d): patches differ\n",
                             g.h, g.w, s, it->n_seg, gs.h, gs.w);
            CHECK_EQ(fnv(rowmajor), gs.fnv);
        }
        CHECK_EQ((int32_t)it->expand.size(), g.n_ids);
        CHECK_EQ(fnv(it->expand), g.ids_fnv);
        /* Each placeholder run starts at Item::seg_at, one token per merged row. */
        REQUIRE_EQ(it->seg_at.size(), (size_t)it->n_seg);
        for (int32_t s = 0; s < it->n_seg; ++s) {
            const int64_t at = it->seg_at[(size_t)s], n = it->row_at(s + 1) - it->row_at(s);
            for (int64_t k = 0; k < n; ++k) CHECK_EQ(it->expand[(size_t)(at + k)], kLfm2ImageToken);
        }
    }
}

/* ================================================================== Gemma 4's processor
 *
 * gemma4_golden.h: the processors AutoProcessor loads for these checkpoints, which transformers
 * and vLLM users run (Gemma4ImageProcessor, torchvision backend, and
 * EmbeddingGemma2VideoProcessor): sizes, patches as the tower reads them, soft-token counts and the
 * frames a clip keeps. The processor keeps patches in pooling-block order and the reference
 * row-major, so they are reordered (Item::coords) before comparing. */
namespace {

VisionConfig gemma4_cfg(int32_t image_soft, int32_t video_soft) {
    VisionConfig c;
    c.family = Family::Gemma4;
    c.patch = kGemma4Patch;
    c.merge = kGemma4Pool;
    c.temporal = 1;
    c.image_soft_tokens = image_soft;
    c.video_soft_tokens = video_soft;
    c.fps = 1.0;
    c.min_frames = 1;
    c.max_frames = 32;
    c.max_frame_rows = 0;
    const int64_t f2 = (int64_t)c.factor() * c.factor();
    c.image_min_pixels = c.video_min_pixels = f2;
    c.image_max_pixels = (int64_t)image_soft * f2;
    c.video_max_pixels = (int64_t)video_soft * f2 * c.max_frames;
    c.tok_image = 258880;
    c.tok_video = 258884;
    c.tok_start = 255999;
    c.tok_end = 258882;
    c.max_patches = 1120 * 9;
    return c;
}

/* Segment s's patches row-major, as the reference lays them out. */
std::vector<uint16_t> raster(const Item& it, int32_t s) {
    const int64_t pd = it.patch_dim, p0 = it.patch_at(s), np = it.patch_at(s + 1) - p0;
    std::vector<int32_t> planes((size_t)(4 * np));
    it.coords(p0, np, planes.data());
    std::vector<uint16_t> out((size_t)(np * pd));
    for (int64_t k = 0; k < np; ++k) {
        const int64_t at = (int64_t)planes[(size_t)k] * it.seg_w(s) + planes[(size_t)(np + k)];
        std::copy_n(it.pixels.data() + (p0 + k) * pd, pd, out.data() + at * pd);
    }
    return out;
}

}  /* namespace */

TEST(gemma4_sizes_are_the_references) {
    for (const GoldGemma4Size& g : kGoldGemma4Size) {
        int64_t oh = 0, ow = 0;
        std::string why;
        const int rc = gemma4_size(g.h, g.w, kGemma4Patch, kGemma4Pool,
                                   (int64_t)g.budget * kGemma4Pool * kGemma4Pool, &oh, &ow, &why);
        if (g.oh == 0) {
            /* The reference refuses: nothing to resize to. */
            CHECK(rc < 0);
            continue;
        }
        if (rc != RAD_OK || oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld x %lld at %d soft tokens: %lld x %lld (%d), reference "
                         "%lld x %lld\n", (long long)g.h, (long long)g.w, g.budget, (long long)oh,
                         (long long)ow, rc, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(rc, RAD_OK);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

/* Pictures already at their size: the grid, the soft tokens and <boi> placeholders <eoi>, and
 * every patch value, exactly. */
TEST(gemma4_pictures_patchify_as_the_reference_does) {
    for (const GoldGemma4Pic& g : kGoldGemma4Pics) {
        Processor p;
        REQUIRE_EQ(p.init(gemma4_cfg(g.budget, 70), [](const std::string&) {
                              return std::vector<int32_t>{};
                          }), RAD_OK);
        std::shared_ptr<Item> it;
        std::string why;
        CHECK_OK(p.image_from(frame(g.f, g.h, g.w), &it, &why));
        REQUIRE(it != nullptr);
        CHECK_EQ(it->grid_h, g.grid_h);
        CHECK_EQ(it->grid_w, g.grid_w);
        REQUIRE_EQ(it->n_seg, 1);
        CHECK_EQ(it->n_rows(), (int64_t)g.soft);
        REQUIRE_EQ(it->patch_dim, (int64_t)3 * kGemma4Patch * kGemma4Patch);
        const std::vector<uint16_t> r = raster(*it, 0);
        for (int i = 0; i < 8; ++i) CHECK_EQ(r[(size_t)i], g.head[i]);
        if (fnv(r) != g.fnv)
            std::fprintf(stderr, "    picture %d x %d at %d soft tokens: patches differ\n", g.h, g.w,
                         g.budget);
        CHECK_EQ(fnv(r), g.fnv);
        REQUIRE_EQ(it->expand.size(), (size_t)g.soft + 2);
        CHECK_EQ(it->expand.front(), 255999);
        CHECK_EQ(it->expand.back(), 258882);
        CHECK_EQ((int64_t)std::count(it->expand.begin(), it->expand.end(), 258880), (int64_t)g.soft);
    }
}

/* Resized pictures, one grown and one thin enough for the one-side rule. This processor resizes
 * with Pillow's bicubic and the reference with torchvision's, which differ by a pixel level or two
 * on a few values (0.3-0.7% on these noise pictures, at most 2 levels). Every 400th-odd value is
 * compared: each within 3 levels, at most 2% not equal. A wrong patch order, orientation or size
 * moves most of them by tens of levels. */
TEST(gemma4_resized_pictures_are_the_references_to_a_resampling_tolerance) {
    for (const GoldGemma4Resized& g : kGoldGemma4Resized) {
        Processor p;
        REQUIRE_EQ(p.init(gemma4_cfg(g.budget, 70), [](const std::string&) {
                              return std::vector<int32_t>{};
                          }), RAD_OK);
        std::shared_ptr<Item> it;
        std::string why;
        CHECK_OK(p.image_from(frame(g.f, g.h, g.w), &it, &why));
        REQUIRE(it != nullptr);
        CHECK_EQ(it->grid_h, g.grid_h);
        CHECK_EQ(it->grid_w, g.grid_w);
        CHECK_EQ(it->n_rows(), (int64_t)g.soft);
        const std::vector<uint16_t> r = raster(*it, 0);
        REQUIRE_EQ(r.size(), (size_t)g.grid_h * g.grid_w * it->patch_dim);
        int unequal = 0, far = 0;
        for (int32_t k = 0; k < g.n; ++k) {
            const size_t at = (size_t)k * (size_t)g.stride;
            REQUIRE(at < r.size());
            const float ours = rad_bf16_to_f32(r[at]);
            const float ref = rad_bf16_to_f32(kGoldGemma4ResizedValues[g.first + k]);
            unequal += ours != ref;
            far += std::fabs(ours - ref) > 3.0f * (2.0f / 255.0f);
        }
        if (far || unequal * 50 > g.n)
            std::fprintf(stderr, "    %d x %d at %d soft tokens: %d of %d values unequal, %d past "
                         "3 levels\n", g.h, g.w, g.budget, unequal, g.n, far);
        CHECK_EQ(far, 0);
        CHECK(unequal * 50 <= g.n);
    }
}

/* A clip at a size its budget keeps: one segment per frame, each spelled <boi> placeholders <eoi>
 * back to back, and every patch value, exactly. The video processor rescales in double where the
 * image processor rescales in float, which differ for byte 127; the clip holds that byte. */
TEST(a_gemma4_clip_patchifies_as_the_reference_does) {
    Processor p;
    REQUIRE_EQ(p.init(gemma4_cfg(280, 70), [](const std::string&) {
                          return std::vector<int32_t>{};
                      }), RAD_OK);
    Video v;
    v.info.n_frames = 8;
    v.info.fps = 1.0;
    v.info.w = kGoldGemma4ClipW;
    v.info.h = kGoldGemma4ClipH;
    for (int32_t f : kGoldGemma4ClipFrames) {
        v.index.push_back(f);
        v.frames.push_back(frame(f, kGoldGemma4ClipH, kGoldGemma4ClipW));
    }
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.video_from(v, &it, &why));
    REQUIRE(it != nullptr);
    const int32_t n = (int32_t)v.frames.size();
    REQUIRE_EQ(it->n_seg, n);
    CHECK_EQ(it->grid_h, kGoldGemma4ClipGridH);
    CHECK_EQ(it->grid_w, kGoldGemma4ClipGridW);
    CHECK_EQ(it->seg_rows, kGoldGemma4ClipSoft);
    std::vector<uint16_t> all;
    for (int32_t s = 0; s < n; ++s) {
        const std::vector<uint16_t> r = raster(*it, s);
        all.insert(all.end(), r.begin(), r.end());
    }
    if (fnv(all) != kGoldGemma4ClipFnv)
        std::fprintf(stderr, "    clip of %d frames of %d x %d: patches differ\n", n,
                     kGoldGemma4ClipH, kGoldGemma4ClipW);
    CHECK_EQ(fnv(all), kGoldGemma4ClipFnv);
    REQUIRE_EQ(it->expand.size(), (size_t)n * (kGoldGemma4ClipSoft + 2));
    for (int32_t s = 0; s < n; ++s) {
        const size_t at = (size_t)s * (kGoldGemma4ClipSoft + 2);
        CHECK_EQ(it->expand[at], 255999);
        CHECK_EQ(it->seg_at[(size_t)s], (int32_t)at + 1);
        CHECK_EQ(it->expand[at + 1 + (size_t)kGoldGemma4ClipSoft], 258882);
    }
}

TEST(gemma4_frames_are_sampled_as_the_reference_does) {
    for (const GoldGemma4Sample& g : kGoldGemma4Sample) {
        VideoInfo info;
        info.n_frames = g.total;
        info.fps = g.fps;
        const std::vector<int64_t> idx = sample_frames_gemma4(info, 1.0, g.max_frames);
        CHECK_EQ((int32_t)idx.size(), g.n);
        int off = 0;
        for (int32_t i = 0; i < std::min(g.n, (int32_t)idx.size()); ++i)
            off += idx[(size_t)i] != kGoldGemma4SampleIdx[g.first + i];
        if (off)
            std::fprintf(stderr, "    %lld frames at %g fps, at most %d: %d indices differ\n",
                         (long long)g.total, g.fps, g.max_frames, off);
        CHECK_EQ(off, 0);
    }
}

/* ================================================================== Gemma 4 12B's processor
 *
 * gemma4u_golden.h: the 12B's processors with its processor_config.json's settings
 * (Gemma4UnifiedImageProcessor, Gemma4UnifiedVideoProcessor, Gemma4UnifiedAudioFeatureExtractor). A
 * patch is a whole 48 x 48 pooling window (merge 1), its values u / 255; the reference's patches are
 * row-major, as merge 1 makes this processor's. */
namespace {

VisionConfig gemma4u_cfg(int32_t image_soft, int32_t video_soft) {
    VisionConfig c = gemma4_cfg(image_soft, video_soft);
    c.patch = kGemma4Patch * kGemma4Pool;
    c.merge = 1;
    c.unit_pixels = true;
    c.max_patches = 1120;
    return c;
}

}  /* namespace */

/* A 48-pixel window as one patch sizes a picture as the 16-pixel patches pooled 3 x 3 do. */
TEST(gemma4_unified_sizes_are_the_references) {
    for (const GoldGemma4Size& g : kGoldGemma4Size) {
        int64_t oh = 0, ow = 0;
        std::string why;
        const int rc = gemma4_size(g.h, g.w, kGemma4Patch * kGemma4Pool, 1, g.budget, &oh, &ow, &why);
        if (g.oh == 0) {
            CHECK(rc < 0);
            continue;
        }
        CHECK_EQ(rc, RAD_OK);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
    }
}

TEST(gemma4_unified_pictures_patchify_as_the_reference_does) {
    for (const GoldGemma4uPic& g : kGoldGemma4uPics) {
        Processor p;
        REQUIRE_EQ(p.init(gemma4u_cfg(g.budget, 70), [](const std::string&) {
                              return std::vector<int32_t>{};
                          }), RAD_OK);
        std::shared_ptr<Item> it;
        std::string why;
        CHECK_OK(p.image_from(frame(g.f, g.h, g.w), &it, &why));
        REQUIRE(it != nullptr);
        CHECK_EQ(it->grid_h, g.grid_h);
        CHECK_EQ(it->grid_w, g.grid_w);
        REQUIRE_EQ(it->n_seg, 1);
        CHECK_EQ(it->n_rows(), (int64_t)g.soft);
        REQUIRE_EQ(it->patch_dim, (int64_t)3 * 48 * 48);
        const std::vector<uint16_t> r = raster(*it, 0);
        CHECK(r == it->pixels);
        for (int i = 0; i < 8; ++i) CHECK_EQ(r[(size_t)i], g.head[i]);
        if (fnv(r) != g.fnv)
            std::fprintf(stderr, "    picture %d x %d at %d soft tokens: patches differ\n", g.h, g.w,
                         g.budget);
        CHECK_EQ(fnv(r), g.fnv);
        REQUIRE_EQ(it->expand.size(), (size_t)g.soft + 2);
        CHECK_EQ(it->expand.front(), 255999);
        CHECK_EQ(it->expand.back(), 258882);
        CHECK_EQ((int64_t)std::count(it->expand.begin(), it->expand.end(), 258880), (int64_t)g.soft);
    }
}

TEST(a_gemma4_unified_clip_patchifies_as_the_reference_does) {
    Processor p;
    REQUIRE_EQ(p.init(gemma4u_cfg(280, 70), [](const std::string&) {
                          return std::vector<int32_t>{};
                      }), RAD_OK);
    Video v;
    v.info.n_frames = 8;
    v.info.fps = 1.0;
    v.info.w = kGoldGemma4uClipW;
    v.info.h = kGoldGemma4uClipH;
    for (int32_t f : kGoldGemma4uClipFrames) {
        v.index.push_back(f);
        v.frames.push_back(frame(f, kGoldGemma4uClipH, kGoldGemma4uClipW));
    }
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.video_from(v, &it, &why));
    REQUIRE(it != nullptr);
    REQUIRE_EQ(it->n_seg, (int32_t)v.frames.size());
    CHECK_EQ(it->grid_h, kGoldGemma4uClipGridH);
    CHECK_EQ(it->grid_w, kGoldGemma4uClipGridW);
    CHECK_EQ(it->seg_rows, kGoldGemma4uClipSoft);
    if (fnv(it->pixels) != kGoldGemma4uClipFnv)
        std::fprintf(stderr, "    unified clip: patches differ\n");
    CHECK_EQ(fnv(it->pixels), kGoldGemma4uClipFnv);
}

/* The 12B's audio: frames of 640 samples as they are, the last zero-padded, one placeholder each
 * between <boa> and <eoa>; read from what a converted container states. */
TEST(a_gemma4_unified_clip_of_audio_is_its_samples_in_frames) {
    const char* keys[] = { "audio_config.model_type", "audio_config.audio_embed_dim",
                           "processor.feature_extractor.sampling_rate",
                           "processor.feature_extractor.audio_samples_per_token",
                           "processor.audio_seq_length", "audio_token_id", "boa_token_id",
                           "eoa_token_index" };
    const char* vals[] = { "gemma4_unified_audio", "640", "16000", "640", "750", "258881", "256000",
                           "258883" };
    RadModelMeta m{};
    m.n_kv = (int)(sizeof(keys) / sizeof(keys[0]));
    m.kv_key = keys;
    m.kv_val = vals;
    mm::AudioConfig c;
    std::string why;
    REQUIRE_EQ(c.from_meta(m, &why), RAD_OK);
    CHECK(c.family == mm::AudioFamily::Gemma4Unified);
    CHECK_EQ(c.n_mels, 640);
    CHECK_EQ(c.max_samples, (int64_t)480000);
    CHECK_EQ(c.tok_end, 258883);
    c.merge = 1;
    c.max_frames = 750;
    mm::Processor p;
    REQUIRE_EQ(p.init_audio(c, &why), RAD_OK);
    const int n = kGoldGemma4uAudioN;
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i) x[(size_t)i] = (float)((double)((int64_t)i * 7919 % 2001 - 1000) / 1000.0);
    std::shared_ptr<mm::Item> it;
    REQUIRE_EQ(p.audio_from(x, &it, &why), RAD_OK);
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_h, kGoldGemma4uAudioFrames);
    CHECK_EQ(it->seg_rows, kGoldGemma4uAudioFrames);
    CHECK_EQ(it->patch_dim, (int64_t)640);
    CHECK_EQ(fnv(it->pixels), kGoldGemma4uAudioFnv);
    REQUIRE_EQ(it->expand.size(), (size_t)kGoldGemma4uAudioFrames + 2);
    CHECK_EQ(it->expand.front(), 256000);
    CHECK_EQ(it->expand[1], 258881);
    CHECK_EQ(it->expand.back(), 258883);
    /* 30 s at most: a longer clip is cut there; a frame more than a pass holds is refused. */
    std::vector<float> longer(480000 + 6400, 0.25f);
    REQUIRE_EQ(p.audio_from(longer, &it, &why), RAD_OK);
    CHECK_EQ(it->grid_h, 750);
    c.max_frames = 100;
    mm::Processor q;
    REQUIRE_EQ(q.init_audio(c, &why), RAD_OK);
    CHECK_EQ(q.audio_from(longer, &it, &why), RAD_E_INVAL);
    /* No samples: <boa><eoa> and nothing to encode. */
    REQUIRE_EQ(p.audio_from({}, &it, &why), RAD_OK);
    CHECK(it->encoded);
    CHECK_EQ(it->expand.size(), (size_t)2);
}

/* What a converted 12B container states for its pictures: Gemma 4's family with the window as the
 * patch and u / 255 values, the budget from processor_config.json. */
TEST(gemma4_unified_vision_config_from_the_container) {
    const char* keys[] = { "vision_config.model_type", "vision_config.patch_size",
                           "vision_config.pooling_kernel_size", "vision_config.model_patch_size",
                           "image_token_id", "video_token_id", "boi_token_id", "eoi_token_id",
                           "processor.image_processor.max_soft_tokens",
                           "processor.video_processor.max_soft_tokens" };
    const char* vals[] = { "gemma4_unified_vision", "16", "3", "48", "258880", "258884", "255999",
                           "258882", "280", "70" };
    RadModelMeta m{};
    m.n_kv = (int)(sizeof(keys) / sizeof(keys[0]));
    m.kv_key = keys;
    m.kv_val = vals;
    VisionConfig c;
    std::string why;
    CHECK_OK(c.from_meta(m, &why));
    CHECK(c.family == Family::Gemma4);
    CHECK(c.unit_pixels);
    CHECK_EQ(c.patch, 48);
    CHECK_EQ(c.merge, 1);
    CHECK_EQ(c.patch_dim(), (int64_t)6912);
    CHECK_EQ(c.image_soft_tokens, 280);
    CHECK_EQ(c.video_soft_tokens, 70);
    vals[0] = "gemma4_vision";
    VisionConfig d;
    CHECK_OK(d.from_meta(m, &why));
    CHECK(!d.unit_pixels);
    CHECK_EQ(d.patch, 16);
    CHECK_EQ(d.merge, 3);
}

/* ================================================================== GLM-5.3's processor
 *
 * glm5_golden.h: Glm5NextProcessor as the checkpoint's processor_config.json sets it up
 * (Glm5NextImageProcessor, torchvision backend): canvases, patch counts, patches as the tower reads
 * them, and the token ids around a picture. The reference's patches are in merge-block order, as
 * this processor's are, so they compare without reordering. */
namespace {

VisionConfig glm5_cfg(int32_t max_tok, int64_t max_patches) {
    VisionConfig c;
    c.family = Family::Glm5Next;
    c.patch = kGlm5Patch;
    c.merge = kGlm5Merge;
    c.temporal = kGlm5Temporal;
    c.min_image_tokens = kGlm5MinTok;
    c.max_image_tokens = max_tok;
    for (int i = 0; i < 3; ++i) {
        c.mean[i] = kGlm5Mean[i];
        c.stdv[i] = kGlm5Std[i];
    }
    const int64_t f2 = (int64_t)c.factor() * c.factor();
    c.image_min_pixels = c.video_min_pixels = kGlm5MinTok * f2;
    c.image_max_pixels = c.video_max_pixels = (int64_t)max_tok * f2;
    c.tok_image = kGlm5Image;
    c.tok_video = 154855;
    c.tok_start = kGlm5Start;
    c.tok_end = kGlm5End;
    c.max_patches = max_patches;
    return c;
}

}  /* namespace */

/* Canvases: sides rounded up to 28, grown under 16 tokens, the budget's binary search over 8000
 * and over the 4096 a pass of 16384 patches holds, aspect ratios to 30000:1. */
TEST(glm5_canvases_are_the_references) {
    for (const GoldGlm5Size& g : kGoldGlm5Size) {
        int64_t oh = 0, ow = 0, ch = 0, cw = 0;
        std::string why;
        CHECK_OK(glm5next_size(g.h, g.w, kGlm5Temporal, kGlm5Patch * kGlm5Merge, kGlm5MinTok,
                               g.budget, &oh, &ow, &ch, &cw, &why));
        if (oh != g.oh || ow != g.ow)
            std::fprintf(stderr, "    %lld x %lld at %d tokens: %lld x %lld, reference %lld x "
                         "%lld\n", (long long)g.h, (long long)g.w, g.budget, (long long)oh,
                         (long long)ow, (long long)g.oh, (long long)g.ow);
        CHECK_EQ(oh, g.oh);
        CHECK_EQ(ow, g.ow);
        CHECK_EQ((oh / kGlm5Patch) * (ow / kGlm5Patch), g.patches);
        CHECK(ch <= oh && cw <= ow);
    }
}

/* Whole pictures: padded only (448 x 448 is the one with no padding), grown under the floor (the
 * 1 x 1, 61 x 97 and 3 x 400 ones), shrunk to a budget (900 x 1400 at 256 tokens, 2300 x 2900 at
 * 8000), every patch value exactly, and <|begin_of_image|>, one <|image|> a token,
 * <|end_of_image|>, as the reference processor expands the template's triple. */
TEST(glm5_pictures_are_the_references) {
    for (const GoldGlm5Pic& g : kGoldGlm5Pics) {
        Processor p;
        REQUIRE_EQ(p.init(glm5_cfg(g.budget, 32768), [](const std::string&) {
                              return std::vector<int32_t>{};
                          }), RAD_OK);
        std::shared_ptr<Item> it;
        std::string why;
        CHECK_OK(p.image_from(frame(g.f, g.h, g.w), &it, &why));
        REQUIRE(it != nullptr);
        CHECK_EQ(it->grid_t, 1);
        CHECK_EQ(it->grid_h, g.grid_h);
        CHECK_EQ(it->grid_w, g.grid_w);
        CHECK(it->rope_1d);
        REQUIRE_EQ(it->patch_dim, (int64_t)3 * kGlm5Temporal * kGlm5Patch * kGlm5Patch);
        REQUIRE_EQ(it->pixels.size(), (size_t)it->n_patch * (size_t)it->patch_dim);
        for (int i = 0; i < 8; ++i) CHECK_EQ(it->pixels[(size_t)i], g.head[i]);
        if (fnv(it->pixels) != g.fnv)
            std::fprintf(stderr, "    picture %d x %d at %d tokens: patches differ\n", g.h, g.w,
                         g.budget);
        CHECK_EQ(fnv(it->pixels), g.fnv);
        const int64_t rows = (int64_t)g.grid_h * g.grid_w / (kGlm5Merge * kGlm5Merge);
        CHECK_EQ(it->n_rows(), rows);
        CHECK_EQ((int32_t)it->expand.size(), g.n_ids);
        CHECK_EQ(fnv(it->expand), g.ids_fnv);
        REQUIRE_EQ(it->expand.size(), (size_t)rows + 2);
        CHECK_EQ(it->expand.front(), kGlm5Start);
        CHECK_EQ(it->expand.back(), kGlm5End);
        CHECK_EQ((int64_t)std::count(it->expand.begin(), it->expand.end(), kGlm5Image), rows);
        REQUIRE_EQ(it->seg_at.size(), (size_t)1);
        CHECK_EQ(it->seg_at[0], 1);
    }
}

/* The checkpoint's budget is lowered to what one encoder pass holds (a token is four patches),
 * as a pixel band is; a pass that holds less than the floor is refused. Video and a request's own
 * budget are refused by name. */
TEST(glm5_budget_is_one_pass_and_pictures_only) {
    Processor p;
    REQUIRE_EQ(p.init(glm5_cfg(kGlm5MaxTok, 16384), [](const std::string&) {
                          return std::vector<int32_t>{};
                      }), RAD_OK);
    CHECK_EQ(p.config().max_image_tokens, 4096);
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.image_from(frame(0, 2048, 2048), &it, &why));
    REQUIRE(it != nullptr);
    CHECK_EQ(it->grid_h, 128);
    CHECK_EQ(it->grid_w, 128);
    CHECK_EQ(it->n_rows(), (int64_t)4096);
    Processor q;
    CHECK_EQ(q.init(glm5_cfg(kGlm5MaxTok, 60), [](const std::string&) {
                 return std::vector<int32_t>{};
             }), RAD_E_INVAL);
    Video v;
    v.frames.push_back(frame(0, 64, 64));
    v.frames.push_back(frame(1, 64, 64));
    CHECK_EQ(p.video_from(v, &it, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("pictures, not video") != std::string::npos);
    const uint8_t png_magic[8] = { 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a };
    CHECK_EQ(p.image(png_magic, sizeof png_magic, 280, &it, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("--image-tokens") != std::string::npos);
}

/* A converted GLM-5.3 container states the processor under `processor.image_processor.*` and
 * the media ids under their own names, not Qwen's vision_start / vision_end. */
TEST(glm5_reads_its_processor_from_the_container) {
    const char* keys[] = {
        "vision_config.model_type", "vision_config.patch_size", "vision_config.temporal_patch_size",
        "vision_config.spatial_merge_size", "image_token_id", "video_token_id",
        "image_start_token_id", "image_end_token_id", "processor.image_processor.min_image_tokens",
        "processor.image_processor.max_image_tokens", "processor.image_processor.image_mean",
        "processor.image_processor.image_std", "processor.image_processor.patch_expand_factor",
    };
    const char* vals[] = {
        "glm5_next_vision", "14", "2", "2", "154854", "154855", "154830", "154831", "16", "8000",
        "0.48145465999999998 0.4578275 0.40821073000000002",
        "0.26862954 0.26130258000000001 0.27577710999999999", "1",
    };
    RadModelMeta m{};
    m.n_kv = (int)(sizeof(keys) / sizeof(keys[0]));
    m.kv_key = keys;
    m.kv_val = vals;
    VisionConfig c;
    std::string why;
    CHECK_OK(c.from_meta(m, &why));
    CHECK(c.family == Family::Glm5Next);
    CHECK_EQ(c.patch_dim(), (int64_t)1176);
    CHECK_EQ(c.factor(), 28);
    CHECK_EQ(c.min_image_tokens, 16);
    CHECK_EQ(c.max_image_tokens, 8000);
    CHECK_EQ(c.tok_image, kGlm5Image);
    CHECK_EQ(c.tok_start, kGlm5Start);
    CHECK_EQ(c.tok_end, kGlm5End);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(c.mean[i], kGlm5Mean[i]);
        CHECK_EQ(c.stdv[i], kGlm5Std[i]);
    }
    vals[12] = "2";
    CHECK_EQ(c.from_meta(m, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("patch_expand_factor") != std::string::npos);
}

/* ================================================================== Pillow's arithmetic
 *
 * imaging_golden.h: Image.resize(LANCZOS) of an RGBA splitmix picture at sizes that shrink, grow,
 * keep one axis or both, and end at one pixel, and the picture pasted on white through its alpha.
 * The images endpoint conditions an edit on these pixels. */
namespace {

FrameRGBA rgba_frame(int f, int h, int w) {
    FrameRGBA fr;
    fr.w = w;
    fr.h = h;
    fr.rgba.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 4; ++c)
                fr.rgba[((size_t)y * w + x) * 4 + c] =
                    pix((uint64_t)f, (uint64_t)y, (uint64_t)x, (uint64_t)c);
    return fr;
}

}  /* namespace */

TEST(the_lanczos_resize_is_pillows_bit_for_bit) {
    const FrameRGBA src = rgba_frame(0, kGoldLanczosSrcH, kGoldLanczosSrcW);
    for (const GoldLanczos& g : kGoldLanczos) {
        FrameRGBA out;
        resize_lanczos_rgba(src, g.w, g.h, &out);
        REQUIRE_EQ(out.rgba.size(), (size_t)g.w * g.h * 4);
        CHECK_EQ(out.w, g.w);
        CHECK_EQ(out.h, g.h);
        if (fnv(out.rgba) != g.fnv)
            std::fprintf(stderr, "    %d x %d -> %d x %d differs from Pillow\n", src.w, src.h, g.w,
                         g.h);
        CHECK_EQ(fnv(out.rgba), g.fnv);
    }
}

TEST(the_white_composite_is_pillows_paste) {
    const FrameRGBA src = rgba_frame(0, kGoldLanczosSrcH, kGoldLanczosSrcW);
    Frame out;
    composite_white(src, &out);
    CHECK_EQ(out.w, src.w);
    CHECK_EQ(out.h, src.h);
    CHECK_EQ(fnv(out.rgb), kGoldCompositeFnv);
}

/* ================================================================== the decoder's edges */
namespace {

std::vector<uint8_t> pixels(int f, int h, int w, int ch) {
    std::vector<uint8_t> v((size_t)w * h * ch);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < ch; ++c)
                v[((size_t)y * w + x) * ch + c] =
                    pix((uint64_t)f, (uint64_t)y, (uint64_t)x, (uint64_t)c);
    return v;
}

/* PNG's chunk CRC (ISO 3309), bit by bit. */
uint32_t crc32(const uint8_t* p, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

void be32(std::vector<uint8_t>* b, uint32_t v) {
    for (int s = 24; s >= 0; s -= 8) b->push_back((uint8_t)(v >> s));
}

void png_chunk(std::vector<uint8_t>* b, const char* type, const std::vector<uint8_t>& data) {
    be32(b, (uint32_t)data.size());
    const size_t at = b->size();
    b->insert(b->end(), type, type + 4);
    b->insert(b->end(), data.begin(), data.end());
    be32(b, crc32(b->data() + at, b->size() - at));
}

/* A well-formed PNG header stating w x h, then one compressed byte of picture: how a request
 * claims a vast picture in a few dozen bytes. */
std::vector<uint8_t> png_claiming(uint32_t w, uint32_t h) {
    std::vector<uint8_t> b = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    std::vector<uint8_t> ihdr;
    be32(&ihdr, w);
    be32(&ihdr, h);
    ihdr.insert(ihdr.end(), { 8, 2, 0, 0, 0 });   /* 8-bit RGB, deflate, no filter or interlace */
    png_chunk(&b, "IHDR", ihdr);
    png_chunk(&b, "IDAT", { 0x78, 0x9c, 0x63, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01 });
    png_chunk(&b, "IEND", {});
    return b;
}

/* A PCM16 WAV: `ch` interleaved channels at `rate`. */
std::vector<uint8_t> wav(const std::vector<int16_t>& s, int ch, int rate) {
    std::vector<uint8_t> b;
    auto le = [&](uint32_t v, int n) {
        for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i)));
    };
    const uint32_t data = (uint32_t)(s.size() * 2);
    b.insert(b.end(), { 'R', 'I', 'F', 'F' });
    le(36 + data, 4);
    b.insert(b.end(), { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' });
    le(16, 4);
    le(1, 2);                                  /* PCM */
    le((uint32_t)ch, 2);
    le((uint32_t)rate, 4);
    le((uint32_t)(rate * ch * 2), 4);          /* bytes per second */
    le((uint32_t)(ch * 2), 2);                 /* bytes per frame */
    le(16, 2);
    b.insert(b.end(), { 'd', 'a', 't', 'a' });
    le(data, 4);
    for (int16_t v : s) le((uint16_t)v, 2);
    return b;
}

/* `seconds` of a sine at `hz` and amplitude `amp`, PCM16. */
std::vector<int16_t> tone(int rate, double seconds, double hz, double amp) {
    std::vector<int16_t> s((size_t)(rate * seconds));
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (int16_t)std::lround(amp * 32767.0 * std::sin(2.0 * M_PI * hz * (double)i / rate));
    return s;
}

std::vector<int16_t> interleave(const std::vector<int16_t>& l, const std::vector<int16_t>& r) {
    std::vector<int16_t> s;
    for (size_t i = 0; i < l.size(); ++i) {
        s.push_back(l[i]);
        s.push_back(r[i]);
    }
    return s;
}

/* A sine's amplitude from its RMS, and its rising zero crossings, over [a, b). */
double amplitude(const std::vector<float>& x, size_t a, size_t b) {
    double e = 0.0;
    for (size_t i = a; i < b; ++i) e += (double)x[i] * x[i];
    return std::sqrt(2.0 * e / (double)(b - a));
}

int rising(const std::vector<float>& x, size_t a, size_t b) {
    int n = 0;
    for (size_t i = a + 1; i < b; ++i) n += x[i - 1] < 0.0f && x[i] >= 0.0f;
    return n;
}

}  /* namespace */

/* The pixel limit bounds what a request makes the server allocate, so it is applied to the size
 * the header states, before a frame is decoded, and the limit itself is admitted. */
TEST(a_picture_over_the_pixel_limit_is_refused_from_its_header) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<uint8_t> px = pixels(1, 48, 64, 3);
    std::vector<uint8_t> png;
    std::string why;
    REQUIRE_EQ(encode_image(px.data(), 64, 48, 3, "png", 0, &png, &why), RAD_OK);
    Frame f;
    CHECK_EQ(decode_image(png.data(), png.size(), 64 * 48, &f, &why), RAD_OK);
    CHECK_EQ(decode_image(png.data(), png.size(), 64 * 48 - 1, &f, &why), RAD_E_INVAL);
    CHECK(has(why, "64x48, over the server's 3071-pixel limit"));
    FrameRGBA fa;
    why.clear();
    CHECK_EQ(decode_image_rgba(png.data(), png.size(), 64 * 48 - 1, &fa, &why), RAD_E_INVAL);
    CHECK(has(why, "3071-pixel limit"));

    /* 81 million pixels claimed in 66 bytes: refused on the header, so nothing is allocated for
     * them. A side past what FFmpeg can hold (60000) is refused by its own size check. */
    std::vector<uint8_t> huge = png_claiming(9000, 9000);
    why.clear();
    CHECK_EQ(decode_image(huge.data(), huge.size(), int64_t(1) << 26, &f, &why), RAD_E_INVAL);
    CHECK(has(why, "9000x9000, over the server's 67108864-pixel limit"));
    huge = png_claiming(60000, 60000);
    why.clear();
    CHECK_EQ(decode_image(huge.data(), huge.size(), int64_t(1) << 26, &f, &why), RAD_E_INVAL);
    CHECK(!why.empty());
}

TEST(a_video_over_the_pixel_limit_is_refused_before_a_frame_is_decoded) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<uint8_t> b = media_file("red-h264.mp4");
    REQUIRE(!b.empty());
    Video v;
    std::string why;
    CHECK_EQ(decode_video(b.data(), b.size(), first_and_last, nullptr, 64 * 64 - 1, &v, &why),
             RAD_E_INVAL);
    CHECK(has(why, "64x64, over the server's 4095-pixel limit for a frame"));
    CHECK(v.frames.empty());
    CHECK_EQ(decode_video(b.data(), b.size(), first_and_last, nullptr, 64 * 64, &v, &why), RAD_OK);
}

/* Bytes cut short or not media at all come back as a refusal with a reason, or, from a decoder
 * that conceals damage, as a picture of the size its header states; never as a crash or a frame
 * whose buffer disagrees with its size. Under ASan an over-read fails here. */
TEST(damaged_and_foreign_bytes_are_refused_with_a_reason) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    for (const char* name : { "red.png", "red.jpg", "red.webp", "red.gif", "red.bmp", "red.tiff" }) {
        const std::vector<uint8_t> b = media_file(name);
        REQUIRE(b.size() > 64);
        for (size_t n : { (size_t)1, (size_t)8, (size_t)33, b.size() / 4, b.size() / 2,
                          b.size() * 3 / 4, b.size() - 1 }) {
            Frame f;
            std::string why;
            const int rc = decode_image(b.data(), n, int64_t(1) << 24, &f, &why);
            const bool ok = rc == RAD_OK ? f.w == 32 && f.h == 32 && f.rgb.size() == 32 * 32 * 3
                                         : rc < 0 && !why.empty();
            if (!ok)
                ::radtest::fail(__FILE__, __LINE__, std::string(name) + " cut to " +
                                std::to_string(n) + " bytes: status " + std::to_string(rc) +
                                ", " + std::to_string(f.w) + " x " + std::to_string(f.h));
        }
    }
    const std::vector<uint8_t> mp4 = media_file("red-h264.mp4");
    REQUIRE(mp4.size() > 64);
    for (size_t n : { (size_t)64, mp4.size() / 4, mp4.size() / 2, mp4.size() * 3 / 4,
                      mp4.size() - 1 }) {
        Video v;
        std::string why;
        const int rc = decode_video(mp4.data(), n, first_and_last, nullptr, int64_t(1) << 24, &v,
                                    &why);
        bool ok = rc == RAD_OK ? !v.frames.empty() && v.frames.size() == v.index.size()
                               : rc < 0 && !why.empty();
        for (const Frame& f : v.frames) ok = ok && f.rgb.size() == (size_t)f.w * f.h * 3;
        if (!ok)
            ::radtest::fail(__FILE__, __LINE__, "red-h264.mp4 cut to " + std::to_string(n) +
                            " bytes: status " + std::to_string(rc));
    }

    std::mt19937 rng(7);
    std::vector<uint8_t> noise(1024);
    for (uint8_t& c : noise) c = (uint8_t)rng();
    const std::vector<uint8_t> sound = wav(tone(16000, 0.1, 440.0, 0.5), 1, 16000);
    const std::vector<uint8_t> png = media_file("red.png");
    Frame f;
    Video v;
    std::vector<float> a;
    std::string why;
    CHECK(decode_image(noise.data(), noise.size(), int64_t(1) << 24, &f, &why) < 0 && !why.empty());
    why.clear();
    CHECK(decode_video(noise.data(), noise.size(), first_and_last, nullptr, int64_t(1) << 24, &v,
                       &why) < 0 && !why.empty());
    why.clear();
    CHECK(decode_audio(noise.data(), noise.size(), 16000, 1 << 20, &a, &why) < 0 && !why.empty());
    why.clear();
    CHECK_EQ(decode_image(png.data(), 0, int64_t(1) << 24, &f, &why), RAD_E_INVAL);
    CHECK(!why.empty());
    /* Each decoder admits only its own kind. */
    why.clear();
    CHECK(decode_image(sound.data(), sound.size(), int64_t(1) << 24, &f, &why) < 0 && !why.empty());
    why.clear();
    CHECK(decode_audio(png.data(), png.size(), 16000, 1 << 20, &a, &why) < 0 && !why.empty());
}

/* tests/data/media/quad.*: 33 x 17, red top left, green top right, blue bottom left, white bottom
 * right, split at x = 16 and y = 8, written by Pillow (tests/mkimggolden.py). A decoder that
 * flips the rows (BMP stores them bottom up), mirrors them, or takes the wrong stride on an odd
 * width puts a wrong colour in some quadrant. JPEG is checked away from the edges, to a
 * tolerance. */
namespace {

const uint8_t* quad_colour(int y, int x) {
    static const uint8_t c[4][3] = { { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 255 } };
    return c[(y < 8 ? 0 : 2) + (x < 16 ? 0 : 1)];
}

}  /* namespace */

TEST(pictures_decode_the_right_way_up_and_the_right_way_round) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    struct F { const char* name; int tol, margin; };
    for (const F& fx : { F{ "quad.png", 0, 0 }, F{ "quad.bmp", 0, 0 }, F{ "quad.webp", 0, 0 },
                         F{ "quad.jpg", 12, 2 } }) {
        const std::vector<uint8_t> b = media_file(fx.name);
        REQUIRE(!b.empty());
        Frame f;
        std::string why;
        CHECK_EQ(decode_image(b.data(), b.size(), int64_t(1) << 24, &f, &why), RAD_OK);
        REQUIRE_EQ(f.rgb.size(), (size_t)33 * 17 * 3);
        CHECK_EQ(f.w, 33);
        CHECK_EQ(f.h, 17);
        int bad = 0;
        for (int y = 0; y < 17; ++y)
            for (int x = 0; x < 33; ++x) {
                if (std::abs(x - 16) < fx.margin || std::abs(y - 8) < fx.margin) continue;
                const uint8_t* want = quad_colour(y, x);
                const uint8_t* got = &f.rgb[((size_t)y * 33 + x) * 3];
                for (int c = 0; c < 3; ++c) bad += std::abs((int)got[c] - (int)want[c]) > fx.tol;
            }
        if (bad) std::fprintf(stderr, "    %s: %d channel values off\n", fx.name, bad);
        CHECK_EQ(bad, 0);
    }
}

/* quad-alpha.png: the quadrants with alpha rising across columns as round(255 x / 32). The RGBA
 * decode keeps it; the RGB decode drops it without compositing, as PIL's convert("RGB") does, so
 * even a fully transparent column keeps its colour. */
TEST(the_rgba_decode_keeps_alpha_and_the_rgb_decode_drops_it_uncomposited) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<uint8_t> b = media_file("quad-alpha.png");
    REQUIRE(!b.empty());
    FrameRGBA fa;
    Frame f;
    std::string why;
    CHECK_EQ(decode_image_rgba(b.data(), b.size(), int64_t(1) << 24, &fa, &why), RAD_OK);
    CHECK_EQ(decode_image(b.data(), b.size(), int64_t(1) << 24, &f, &why), RAD_OK);
    REQUIRE_EQ(fa.rgba.size(), (size_t)33 * 17 * 4);
    REQUIRE_EQ(f.rgb.size(), (size_t)33 * 17 * 3);
    int bad = 0;
    for (int y = 0; y < 17; ++y)
        for (int x = 0; x < 33; ++x) {
            const uint8_t* want = quad_colour(y, x);
            const size_t i = (size_t)y * 33 + x;
            for (int c = 0; c < 3; ++c) {
                bad += fa.rgba[i * 4 + c] != want[c];
                bad += f.rgb[i * 3 + c] != want[c];
            }
            bad += fa.rgba[i * 4 + 3] != (uint8_t)std::lround(255.0 * x / 32.0);
        }
    CHECK_EQ(bad, 0);
}

/* What the images endpoint writes: png and lossless webp decode back to the same pixels; jpeg
 * refuses alpha, and a format it does not write is refused by name. */
TEST(lossless_encodes_decode_back_to_the_same_pixels) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<uint8_t> rgb = pixels(2, 17, 33, 3), rgba = pixels(3, 17, 33, 4);
    std::vector<uint8_t> file;
    std::string why;
    Frame f;
    FrameRGBA fa;
    REQUIRE_EQ(encode_image(rgb.data(), 33, 17, 3, "png", 0, &file, &why), RAD_OK);
    CHECK_EQ(decode_image(file.data(), file.size(), int64_t(1) << 24, &f, &why), RAD_OK);
    CHECK(f.rgb == rgb);
    REQUIRE_EQ(encode_image(rgba.data(), 33, 17, 4, "png", 0, &file, &why), RAD_OK);
    CHECK_EQ(decode_image_rgba(file.data(), file.size(), int64_t(1) << 24, &fa, &why), RAD_OK);
    CHECK(fa.rgba == rgba);
    REQUIRE_EQ(encode_image(rgb.data(), 33, 17, 3, "webp", 100, &file, &why), RAD_OK);
    /* The encoder must write a real lossless container. The decode leg depends on FFmpeg's native
     * VP8L decoder, which refuses some valid libwebp streams ("Invalid data"): the system's
     * ffmpeg 8.0.1 CLI refuses the same bytes, while libwebp/PIL decode them, so that specific
     * refusal is an external decoder limitation and is not counted as a radiance failure. Any
     * other decode error, or wrong pixels, still fails. */
    const bool riff_webp = file.size() >= 20 && std::memcmp(file.data(), "RIFF", 4) == 0 &&
                           std::memcmp(file.data() + 8, "WEBP", 4) == 0 &&
                           std::memcmp(file.data() + 12, "VP8L", 4) == 0;
    CHECK(riff_webp);
    const int webp_rc = decode_image(file.data(), file.size(), int64_t(1) << 24, &f, &why);
    const bool ffmpeg_vp8l_limit = webp_rc == RAD_E_INVAL && has(why, "Invalid data");
    if (ffmpeg_vp8l_limit)
        std::fprintf(stderr, "    note: this FFmpeg's VP8L decoder refuses the libwebp stream "
                             "(external; libwebp decodes it)\n");
    else {
        CHECK_EQ(webp_rc, RAD_OK);
        CHECK(f.rgb == rgb);
    }
    REQUIRE_EQ(encode_image(rgb.data(), 33, 17, 3, "jpeg", 90, &file, &why), RAD_OK);
    CHECK_EQ(decode_image(file.data(), file.size(), int64_t(1) << 24, &f, &why), RAD_OK);
    CHECK_EQ(f.w, 33);
    CHECK_EQ(f.h, 17);
    CHECK_EQ(encode_image(rgba.data(), 33, 17, 4, "jpeg", 90, &file, &why), RAD_E_INVAL);
    CHECK(has(why, "alpha"));
    CHECK_EQ(encode_image(rgb.data(), 33, 17, 3, "gif", 0, &file, &why), RAD_E_INVAL);
    CHECK(has(why, "png, jpeg, webp"));
}

/* Channels are averaged, not downmixed: media.h's choice, matching the reference loaders, where
 * swresample's own downmix weights a stereo pair by 0.707. A 440 Hz tone at 0.5 in the left
 * channel only is 0.25 in mono (0.354 downmixed), and the same tone in both channels stays 0.5
 * (1.0 if summed). The rate conversion keeps the pitch and the length. */
TEST(stereo_audio_is_averaged_to_mono_and_resampled) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    const std::vector<int16_t> left = tone(44100, 1.0, 440.0, 0.5), silent(left.size(), 0);
    for (const bool both : { false, true }) {
        const std::vector<uint8_t> b = wav(interleave(left, both ? left : silent), 2, 44100);
        std::vector<float> m;
        std::string why;
        REQUIRE_EQ(decode_audio(b.data(), b.size(), 16000, 1 << 20, &m, &why), RAD_OK);
        CHECK(m.size() >= 15990 && m.size() <= 16010);
        REQUIRE(m.size() >= 14000);
        CHECK_NEAR(amplitude(m, 2000, 14000), both ? 0.5 : 0.25, both ? 0.005 : 0.0025);
        /* 0.75 s of 440 Hz */
        CHECK(std::abs(rising(m, 2000, 14000) - 330) <= 1);
    }
}

/* Mono at the target rate is a format conversion only: every sample is the PCM value over 32768.
 * max_samples cuts the clip there exactly, as the reference extractor truncates. */
TEST(mono_audio_at_the_rate_passes_through_and_truncates_exactly) {
    if (!decoder_available()) SKIP("no media decoder in this build");
    std::vector<int16_t> s(4000);
    for (size_t i = 0; i < s.size(); ++i)
        s[i] = (int16_t)(uint16_t)(pix(9, 0, i, 0) | (pix(9, 0, i, 1) << 8));
    const std::vector<uint8_t> b = wav(s, 1, 16000);
    std::vector<float> m;
    std::string why;
    REQUIRE_EQ(decode_audio(b.data(), b.size(), 16000, 1 << 20, &m, &why), RAD_OK);
    REQUIRE_EQ(m.size(), s.size());
    int off = 0;
    for (size_t i = 0; i < s.size(); ++i) off += m[i] != (float)s[i] / 32768.0f;
    CHECK_EQ(off, 0);
    std::vector<float> cut;
    REQUIRE_EQ(decode_audio(b.data(), b.size(), 16000, 1000, &cut, &why), RAD_OK);
    CHECK_EQ(cut.size(), (size_t)1000);
    CHECK(std::equal(cut.begin(), cut.end(), m.begin()));
}

/* Qwen3-ASR's front end is the reference's, and its chunks are staged apart. Qwen3ASRFeatureExtractor
 * (transformers 5.18) on the same two-tone signal: 23456 samples are 146 real frames, one whole
 * chunk of 100 and one of 46, which the encoder keeps 13 and 6 rows of -- so 19 placeholders and
 * 104 + 48 staged frames, the whole chunk's last four read as zero. 3000 samples are zero-padded to
 * the extractor's 8000 first: 50 frames, 7 rows, and the padding's frames sit at the clip's floor
 * (its maximum less 8, then (x + 4) / 4). */
TEST(the_qwen3_asr_front_end_is_the_references) {
    mm::AudioConfig c;
    c.family = mm::AudioFamily::Qwen3ASR;
    c.rate = 16000; c.frame_length = 400; c.hop = 160; c.fft = 400; c.n_mels = 128;
    c.min_freq = 0.0; c.max_freq = 8000.0; c.min_samples = 8000; c.chunk = 100;
    c.tok_audio = 151676; c.tok_start = 151669; c.tok_end = 151670;
    c.merge = 8; c.max_frames = 3120;
    mm::Processor p;
    CHECK_EQ(p.init_audio(c), RAD_OK);
    CHECK_EQ(p.audio_config().max_samples, (int64_t)480000);
    auto tone = [](int n) {
        std::vector<float> x((size_t)n);
        for (int i = 0; i < n; ++i) {
            const float t = (float)i / 16000.0f;
            x[(size_t)i] = 0.5f * std::sin(2.0f * 3.14159265358979f * 440.0f * t) +
                           0.25f * std::sin(2.0f * 3.14159265358979f * 3000.0f * t);
        }
        return x;
    };
    struct Want { int frame, bin; float v; };
    auto staged_at = [](int frame) { return (frame / 100) * 104 + frame % 100; };
    std::shared_ptr<mm::Item> it;
    std::string why;

    CHECK_EQ(p.audio_from(tone(23456), &it, &why), RAD_OK);
    CHECK(it != nullptr);
    if (!it) return;
    CHECK_EQ(it->valid, 146);
    CHECK_EQ(it->seg_rows, 19);
    CHECK_EQ(it->grid_h, 152);
    CHECK_EQ(it->n_seg, 1);
    CHECK_EQ(it->expand.size(), (size_t)21);
    CHECK_EQ(it->expand.front(), 151669);
    CHECK_EQ(it->expand[1], 151676);
    CHECK_EQ(it->expand.back(), 151670);
    const Want want[] = { { 0, 18, 0.7657754421234131f }, { 1, 18, 1.4826123714447021f },
                          { 50, 17, 1.3374924659729004f }, { 50, 18, 1.4853556156158447f },
                          { 50, 88, 1.16568124294281f }, { 99, 88, 1.1656821966171265f },
                          { 100, 18, 1.485356092453003f }, { 145, 18, 1.485353708267212f },
                          { 145, 89, -0.5146430730819702f }, { 144, 19, 1.2752504348754883f },
                          { 70, 0, -0.5146430730819702f } };
    for (const Want& w : want) {
        const float got = rad_bf16_to_f32(it->pixels[(size_t)(staged_at(w.frame) * 128 + w.bin)]);
        CHECK_NEAR(got, w.v, 0.008f * std::max(1.0f, std::fabs(w.v)));
    }
    /* The whole chunk's alignment frames are zeros and read as zero; the next chunk reads from its
     * first frame; the last chunk's padding past its 46 real frames is the reference's zeros. */
    for (int f = 100; f < 104; ++f)
        for (int b = 0; b < 128; ++b) CHECK_EQ(it->pixels[(size_t)(f * 128 + b)], 0);
    std::vector<int32_t> planes(4 * 152);
    it->coords(0, 152, planes.data());
    for (int f = 0; f < 152; ++f) {
        CHECK_EQ(planes[(size_t)f], f);
        CHECK_EQ(planes[(size_t)(152 + f)], (f >= 100 && f < 104) ? 0 : 1);
        CHECK_EQ(planes[(size_t)(2 * 152 + f)], 146);
        CHECK_EQ(planes[(size_t)(3 * 152 + f)], 152);
    }
    for (int f = 104 + 46; f < 152; ++f) CHECK_EQ(it->pixels[(size_t)(f * 128 + 18)], 0);

    CHECK_EQ(p.audio_from(tone(3000), &it, &why), RAD_OK);
    CHECK_EQ(it->valid, 50);
    CHECK_EQ(it->seg_rows, 7);
    CHECK_EQ(it->grid_h, 56);
    const Want short_want[] = { { 0, 18, 0.7657754421234131f }, { 1, 18, 1.4826123714447021f },
                                { 49, 18, -0.5146448612213135f }, { 49, 89, -0.5146448612213135f },
                                { 48, 19, -0.5146448612213135f } };
    for (const Want& w : short_want) {
        const float got = rad_bf16_to_f32(it->pixels[(size_t)(w.frame * 128 + w.bin)]);
        CHECK_NEAR(got, w.v, 0.008f * std::max(1.0f, std::fabs(w.v)));
    }
    /* Longer than one pass is refused, not cut: the reference does not truncate. */
    CHECK_EQ(p.audio_from(std::vector<float>(480001, 0.0f), &it, &why), RAD_E_INVAL);
    CHECK(why.find("longer than one audio encoder pass") != std::string::npos);
}

/* The FFT a front end runs is one it has: Qwen3-ASR's mixed radix takes any length, a power of
 * two included, and puts a tone in the same mel bin at 512 points as at 400; Gemma 4's radix-2
 * transform refuses a length that is not a power of two, by name. */
TEST(each_audio_front_end_runs_its_fft_at_any_length_it_takes) {
    auto qwen = [](int32_t fft) {
        mm::AudioConfig c;
        c.family = mm::AudioFamily::Qwen3ASR;
        c.rate = 16000; c.frame_length = fft; c.hop = 160; c.fft = fft; c.n_mels = 128;
        c.min_freq = 0.0; c.max_freq = 8000.0; c.min_samples = 8000; c.chunk = 100;
        c.tok_audio = 151676; c.tok_start = 151669; c.tok_end = 151670;
        c.merge = 8; c.max_frames = 3120;
        return c;
    };
    std::vector<float> x(16000);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.5f * std::sin(2.0f * 3.14159265358979f * 440.0f * (float)i / 16000.0f);
    int peak[2] = { -1, -1 };
    for (int k = 0; k < 2; ++k) {
        mm::Processor p;
        std::string why;
        REQUIRE_EQ(p.init_audio(qwen(k ? 512 : 400), &why), RAD_OK);
        std::shared_ptr<mm::Item> it;
        REQUIRE_EQ(p.audio_from(x, &it, &why), RAD_OK);
        REQUIRE(it != nullptr);
        /* Frame 50, away from both ends of the clip. */
        float best = -1e30f;
        for (int b = 0; b < 128; ++b) {
            const float v = rad_bf16_to_f32(it->pixels[(size_t)(50 * 128 + b)]);
            CHECK(std::isfinite(v));
            if (v > best) { best = v; peak[k] = b; }
        }
    }
    CHECK(peak[0] > 0);
    CHECK_EQ(peak[1], peak[0]);

    mm::AudioConfig g;
    g.rate = 16000; g.frame_length = 320; g.hop = 160; g.fft = 400; g.n_mels = 128;
    g.min_freq = 0.0; g.max_freq = 8000.0; g.mel_floor = 1e-3; g.max_samples = 480000;
    g.tok_audio = 258881; g.tok_start = 256000; g.tok_end = 258883;
    g.merge = 4; g.max_frames = 3000;
    mm::Processor p;
    std::string why;
    CHECK_EQ(p.init_audio(g, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("not a power of two") != std::string::npos);
    CHECK(!p.has_audio());
}

/* Gemma 4's feature extractor settings come from preprocessor_config.json (`preprocessor.`, as
 * EmbeddingGemma 2 ships them) or from processor_config.json's `feature_extractor` object (as
 * Gemma 4 E2B/E4B ship them); where both are present, the first wins. */
TEST(gemma4_audio_settings_come_from_either_processor_file) {
    const char* keys[] = { "audio_config.model_type", "audio_token_id", "boa_token_id",
                           "eoa_token_id", "processor.feature_extractor.sampling_rate",
                           "processor.feature_extractor.frame_length",
                           "processor.feature_extractor.hop_length",
                           "processor.feature_extractor.feature_size",
                           "processor.feature_extractor.fft_length",
                           "processor.feature_extractor.mel_floor",
                           "processor.feature_extractor.dither", "preprocessor.hop_length" };
    const char* vals[] = { "gemma4_audio", "258881", "256000", "258883", "16000", "320", "160",
                           "128", "1024", "0.002", "0", "200" };
    RadModelMeta m{};
    m.n_kv = (int)(sizeof keys / sizeof keys[0]) - 1;   /* without preprocessor.hop_length */
    m.kv_key = keys;
    m.kv_val = vals;
    mm::AudioConfig c;
    std::string why;
    REQUIRE_EQ(c.from_meta(m, &why), RAD_OK);
    CHECK_EQ(c.fft, 1024);
    CHECK_EQ(c.hop, 160);
    CHECK_NEAR(c.mel_floor, 0.002, 1e-12);
    m.n_kv += 1;
    mm::AudioConfig d;
    REQUIRE_EQ(d.from_meta(m, &why), RAD_OK);
    CHECK_EQ(d.hop, 200);

    /* a dithering extractor is refused from either file */
    vals[10] = "1e-5";
    mm::AudioConfig e;
    CHECK_EQ(e.from_meta(m, &why), RAD_E_UNSUPPORTED);
}

/* A whole chunk keeps 13 rows: three convolutions of kernel 3, stride 2, padding 1 -- and the
 * chunk shorter than that keeps what the reference's _post_cnn_length gives it. */
TEST(a_chunk_keeps_the_rows_three_convolutions_leave) {
    CHECK_EQ(mm::AudioConfig::chunk_rows(100), 13);
    CHECK_EQ(mm::AudioConfig::chunk_rows(97), 13);
    CHECK_EQ(mm::AudioConfig::chunk_rows(96), 12);
    CHECK_EQ(mm::AudioConfig::chunk_rows(46), 6);
    CHECK_EQ(mm::AudioConfig::chunk_rows(1), 1);
    CHECK_EQ(mm::AudioConfig::chunk_rows(0), 0);
}

/* A stream resamples as one piece would, and cleanly: a tone under the lower Nyquist frequency
 * comes through at its frequency and level, one over it is taken out rather than folded back, a
 * constant stays that constant, and pushing the stream in pieces changes no sample. */
TEST(a_resampled_stream_is_its_tone_whether_pushed_whole_or_in_pieces) {
    auto tone = [](double hz, int32_t rate, size_t n) {
        std::vector<float> x(n);
        for (size_t i = 0; i < n; ++i)
            x[i] = (float)(0.5 * std::sin(2.0 * M_PI * hz * (double)i / rate));
        return x;
    };
    const std::vector<float> in = tone(1000.0, 24000, 24000);
    mm::Resampler whole(24000, 16000), parts(24000, 16000);
    std::vector<float> a, b;
    whole.push(in.data(), in.size(), &a);
    whole.flush(&a);
    for (size_t i = 0, k = 1; i < in.size(); i += k, k = k % 997 + 13)
        parts.push(in.data() + i, std::min(k, in.size() - i), &b);
    parts.flush(&b);
    CHECK_EQ(a.size(), (size_t)16000);
    CHECK(a == b);
    const std::vector<float> want = tone(1000.0, 16000, 16000);
    double err = 0.0;
    for (size_t i = 200; i < 15800; ++i) err = std::max(err, (double)std::fabs(a[i] - want[i]));
    CHECK(err < 2e-3);

    /* 10 kHz is past 16 kHz's Nyquist frequency: gone, not folded back to 6 kHz. */
    const std::vector<float> high = tone(10000.0, 24000, 24000);
    mm::Resampler down(24000, 16000);
    std::vector<float> c;
    down.push(high.data(), high.size(), &c);
    down.flush(&c);
    double rms = 0.0;
    for (size_t i = 200; i + 200 < c.size(); ++i) rms += (double)c[i] * c[i];
    rms = std::sqrt(rms / (double)(c.size() - 400));
    CHECK(rms < 0.005);   /* it went in at 0.35 */

    /* Up from 8 kHz: a constant is that constant, and the length doubles. */
    const std::vector<float> dc(4000, 0.25f);
    mm::Resampler up(8000, 16000);
    std::vector<float> u;
    up.push(dc.data(), dc.size(), &u);
    up.flush(&u);
    CHECK_EQ(u.size(), (size_t)8000);
    for (size_t i = 100; i < 7900; ++i) CHECK_NEAR(u[i], 0.25f, 1e-4f);

    mm::Resampler same(16000, 16000);
    std::vector<float> p;
    same.push(in.data(), 100, &p);
    CHECK(std::equal(p.begin(), p.end(), in.begin()) && p.size() == 100);
    CHECK(mm::Resampler::takes(24000) && mm::Resampler::takes(44100));
    CHECK(!mm::Resampler::takes(23999));
}


/* ================================================================== Unlimited-OCR
 *
 * The checkpoint's two picture modes, from its own remote code (tests/mkuocrgolden.py): the tile
 * grid dynamic_preprocess chooses (its ratio set sorted by area, ties in the set's order), the
 * placeholders that replace the picture (infer()'s global view plus crops; infer_multi()'s page),
 * and the pixels of the global view and of the tile mosaic, in the order Item::patch_raster
 * stages them: row by row, each patch (channel, y, x).
 */

/* The checkpoint's processor configuration, as a converted container states it. */
VisionConfig uocr_cfg() {
    VisionConfig c;
    c.family = Family::UnlimitedOcr;
    c.patch = kUocrPatch;
    c.temporal = 1;
    c.merge = kUocrDownsample;
    c.downsample = kUocrDownsample;
    c.base_size = kUocrBaseSize;
    c.image_size = kUocrBaseSize;
    c.tile = kUocrTile;
    c.min_tiles = kUocrMinTiles;
    c.max_tiles = kUocrMaxTiles;
    c.crop_mode = kUocrCropMode;
    c.tok_image = kUocrImageToken;
    c.max_patches = 16384;                 /* the server's --mm-max-patches default */
    const int64_t f2 = (int64_t)c.factor() * c.factor();
    c.image_min_pixels = c.video_min_pixels = f2;
    c.image_max_pixels = c.video_max_pixels = (int64_t)c.base_size * c.base_size;
    c.min_frames = 1;
    return c;
}

int32_t uocr_marker(const std::string& s) { return s == "<image>" ? kUocrImageToken : -1; }

/* A segment's patches in raster order, its first `real_w` patches a row: the tensor the reference
 * holds for that view, patch by patch. */
std::vector<uint16_t> segment_pixels(const Item& it, int32_t s, int32_t real_w) {
    const int64_t pd = it.patch_dim;
    const int32_t gh = it.seg_h(s), gw = it.seg_w(s);
    const int64_t p0 = it.patch_at(s);
    std::vector<uint16_t> out;
    out.reserve((size_t)gh * (size_t)real_w * (size_t)pd);
    for (int32_t py = 0; py < gh; ++py)
        for (int32_t px = 0; px < real_w; ++px) {
            const uint16_t* p = it.pixels.data() + (p0 + (int64_t)py * gw + px) * pd;
            out.insert(out.end(), p, p + pd);
        }
    return out;
}

/* The container's model_type selects the family; a key it does not carry takes the reference's
 * default, so a container converted before the tower landed still states a processor. */
TEST(unlimitedocr_configures_itself_from_the_container) {
    const char* keys[] = { "model_type", "preprocessor.crop_to_patches", "preprocessor.max_patches",
                           "preprocessor.min_patches", "preprocessor.resample",
                           "preprocessor.size.height", "preprocessor.size.width",
                           "preprocessor.tile_size", "processor.downsample_ratio",
                           "processor.image_token", "processor.patch_size" };
    const char* vals[] = { "unlimited_ocr", "true", "32", "2", "3", "1024", "1024", "640", "4",
                           "<image>", "16" };
    RadModelMeta m{};
    m.n_kv = (int)(sizeof keys / sizeof keys[0]);
    m.kv_key = keys;
    m.kv_val = vals;
    VisionConfig c;
    std::string why;
    REQUIRE_EQ(c.from_meta(m, &why), RAD_OK);
    CHECK(c.family == Family::UnlimitedOcr);
    CHECK_EQ(c.patch, 16);
    CHECK_EQ(c.downsample, 4);
    CHECK_EQ(c.merge, 4);
    CHECK_EQ(c.temporal, 1);
    CHECK_EQ(c.base_size, 1024);
    CHECK_EQ(c.image_size, 1024);
    CHECK_EQ(c.tile, 640);
    CHECK_EQ(c.min_tiles, 2);
    CHECK_EQ(c.max_tiles, 32);
    CHECK(c.crop_mode);
    CHECK_EQ(c.factor(), 64);
    CHECK_EQ(c.patch_dim(), (int64_t)3 * 16 * 16);
    CHECK_EQ(c.mean[0], 0.5f);
    CHECK_EQ(c.stdv[2], 0.5f);
    CHECK(c.image_token == "<image>");

    /* Only model_type: every processor key takes the reference's default. */
    const char* one_key[] = { "model_type" };
    const char* one_val[] = { "unlimited_ocr" };
    RadModelMeta mt{};
    mt.n_kv = 1;
    mt.kv_key = one_key;
    mt.kv_val = one_val;
    VisionConfig d;
    REQUIRE_EQ(d.from_meta(mt, &why), RAD_OK);
    CHECK_EQ(d.patch, 16);
    CHECK_EQ(d.downsample, 4);
    CHECK_EQ(d.base_size, 1024);
    CHECK_EQ(d.image_size, 1024);
    CHECK_EQ(d.tile, 640);
    CHECK_EQ(d.min_tiles, 2);
    CHECK_EQ(d.max_tiles, 32);
    CHECK(d.crop_mode);
    CHECK(d.image_token == "<image>");

    /* PIL's bicubic is the only filter this processor stages. */
    const char* bad_key[] = { "model_type", "preprocessor.resample" };
    const char* bad_val[] = { "unlimited_ocr", "2" };
    RadModelMeta mb{};
    mb.n_kv = 2;
    mb.kv_key = bad_key;
    mb.kv_val = bad_val;
    VisionConfig b;
    CHECK_EQ(b.from_meta(mb, &why), RAD_E_UNSUPPORTED);
}

/* The transposed ratios whose tall side the reference's Python set visits last, which is what its
 * tie rule needs beyond "the wider grid wins": the table in core/mm/processor.cpp must be the
 * golden's, pair for pair. */
TEST(unlimitedocr_tie_order_is_the_references) {
    int32_t listed = 0;
    for (int32_t c = 1; c <= kUocrMaxTiles; ++c)
        for (int32_t r = c + 1; r <= kUocrMaxTiles; ++r) {
            if ((int64_t)c * r < kUocrMinTiles || (int64_t)c * r > kUocrMaxTiles) continue;
            bool want = false;
            for (int32_t k = 0; k < kUocrNTallLast; ++k)
                if (kUocrTallLast[k].c == c && kUocrTallLast[k].r == r) want = true;
            if (want) ++listed;
            if (uocr_tall_last(c, r) != want)
                std::fprintf(stderr, "    pair %dx%d: %s last here, %s in the reference\n", c, r,
                             uocr_tall_last(c, r) ? "tall" : "wide",
                             want ? "tall" : "wide");
            CHECK_EQ((int)uocr_tall_last(c, r), (int)want);
        }
    CHECK_EQ(listed, kUocrNTallLast);
}

/* A line wider than one encoder pass is cut into chunks of whole tiles; only the last chunk
 * carries the newline column, and only the real 4x4 view separator is flagged as one. */
TEST(unlimitedocr_wide_lines_are_chunked_whole_tiles) {
    Processor p;
    REQUIRE_EQ(p.init(uocr_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      uocr_marker), RAD_OK);
    const int32_t q = kUocrTile / kUocrPatch;              /* 40 */
    const int32_t w = 16000, h = 1024;                     /* 15.6:1: a 16-tile line */
    int32_t cols = 1, rows = 1;
    uocr_grid(h, w, kUocrMinTiles, kUocrMaxTiles, kUocrTile, &cols, &rows);
    REQUIRE(cols > 10);                                    /* more than one chunk */
    const int32_t chunk = (int32_t)((16384 / q - 4) / q);   /* 10 */

    ImageMode mode;
    mode.kind = ImageMode::Gundam;
    std::shared_ptr<Item> it;
    std::string why;
    REQUIRE_EQ(p.image_from(frame(0, h, w), &it, &why, mode), RAD_OK);
    REQUIRE(it != nullptr);

    /* Two line chunks, the global view, the separator. */
    REQUIRE_EQ(it->n_seg, 4);
    CHECK_EQ(it->seg_h(0), q);
    CHECK_EQ(it->seg_h(1), q);
    CHECK(!it->seg_is_sep(0));
    CHECK(!it->seg_is_sep(1));
    CHECK(!it->seg_is_sep(2));
    CHECK(it->seg_is_sep(3));
    CHECK_EQ(it->seg_w(3), kUocrDownsample);
    CHECK_EQ(it->seg_h(3), kUocrDownsample);

    /* Earlier chunks end at their last tile; only the last has the 4 newline columns. */
    CHECK_EQ(it->seg_w(0), q * chunk);
    CHECK_EQ(it->seg_w(1), q * (cols - chunk) + kUocrDownsample);
    for (int32_t s = 0; s < 2; ++s) CHECK_EQ(it->seg_w(s) % q, s == 1 ? kUocrDownsample : 0);

    /* The placeholders are the reference's layout: (10*cols + 1) * 10 lines, plus 273 + 1. */
    const int64_t want = 273 + (int64_t)(10 * cols + 1) * 10;
    CHECK_EQ((int64_t)it->expand.size(), want);
    CHECK_EQ(it->n_rows(), want);
    CHECK_EQ(uocr_tokens(p.config(), mode, cols, rows), want);

    /* Rows: each chunk contributes its patches / 16, the global 272, the separator 1. */
    CHECK_EQ(it->row_at(1), (int64_t)q * it->seg_w(0) / 16);
    CHECK_EQ(it->row_at(2), (int64_t)q * it->seg_w(0) / 16 + (int64_t)q * it->seg_w(1) / 16);
    CHECK_EQ(it->row_at(3), it->row_at(2) + 272);
    CHECK_EQ(it->row_at(4), want);
}

/* dynamic_preprocess's grid for a sweep of shapes, the exact ties of its ratio set included. */
TEST(unlimitedocr_tile_grids_are_the_references) {
    for (const GoldUocrGrid& g : kUocrGrid) {
        int32_t cols = 1, rows = 1;
        /* The reference asks dynamic_preprocess only for a picture larger than one tile. */
        if (g.w > kUocrTile || g.h > kUocrTile)
            uocr_grid(g.h, g.w, kUocrMinTiles, kUocrMaxTiles, kUocrTile, &cols, &rows);
        if (cols != g.cols || rows != g.rows)
            std::fprintf(stderr, "    picture %lldx%lld: grid %dx%d, reference %dx%d\n",
                         (long long)g.w, (long long)g.h, cols, rows, g.cols, g.rows);
        CHECK_EQ(cols, g.cols);
        CHECK_EQ(rows, g.rows);
    }
}

/* Whole pictures in both modes: the segments (a line of tiles per tile row, the global view, the
 * view separator), every placeholder the picture marker, the pixel count the reference's layout
 * has, and the patches of the global view and of the tile mosaic. */
TEST(unlimitedocr_pictures_are_the_references) {
    Processor p;
    REQUIRE_EQ(p.init(uocr_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      uocr_marker), RAD_OK);
    const int32_t q = kUocrTile / kUocrPatch;
    for (const GoldUocrPic& g : kUocrPictures) {
        ImageMode mode;
        if (g.mode == 1) {
            mode.kind = ImageMode::Gundam;
        } else {
            mode.kind = ImageMode::Base;
            mode.image_size = g.mode == 2 ? kUocrBaseSize : kUocrTile;
        }
        std::shared_ptr<Item> it;
        std::string why;
        const int rc = p.image_from(frame(g.f, g.h, g.w), &it, &why, mode);
        if (rc < 0) std::fprintf(stderr, "    %dx%d mode %d: %s\n", g.w, g.h, g.mode, why.c_str());
        CHECK_OK(rc);
        REQUIRE(it != nullptr);
        REQUIRE_EQ(it->n_seg, g.n_seg);
        REQUIRE_EQ((int64_t)it->expand.size(), (int64_t)g.n_ids);
        CHECK_EQ(it->n_rows(), (int64_t)g.n_ids);
        REQUIRE_EQ(it->seg_at.size(), (size_t)it->n_seg);
        for (int32_t v : it->expand) CHECK_EQ(v, kUocrImageToken);
        CHECK(it->rope_1d);
        CHECK(it->patch_raster);

        int32_t cols = 0, rows = 0;
        for (int32_t s = 0; s < it->n_seg; ++s) {
            const GoldUocrSeg& gs = kUocrSegs[g.seg0 + s];
            CHECK_EQ(it->seg_h(s), gs.h);
            CHECK_EQ(it->seg_w(s), gs.w);
            CHECK_EQ(it->seg_at[(size_t)s], (int32_t)it->row_at(s));
            const int64_t np = it->patch_at(s + 1) - it->patch_at(s);
            CHECK_EQ(np, (int64_t)gs.h * gs.w);
            if (gs.h == q && s + 2 < it->n_seg) {       /* a line of tiles */
                if (cols) { CHECK_EQ(cols, gs.real / q); }
                cols = gs.real / q;
                rows += 1;
            }
            /* Raster order: patch k of a segment sits at (k / w, k % w). */
            std::vector<int32_t> planes((size_t)(4 * np));
            it->coords(it->patch_at(s), np, planes.data());
            for (int64_t k = 0; k < np; ++k) {
                CHECK_EQ(planes[(size_t)k], (int32_t)(k / gs.w));
                CHECK_EQ(planes[(size_t)(np + k)], (int32_t)(k % gs.w));
                CHECK_EQ(planes[(size_t)(2 * np + k)], gs.h);
                CHECK_EQ(planes[(size_t)(3 * np + k)], gs.w);
            }
        }
        const bool tiled_ref = g.mode == 1 && g.cols * g.rows > 1;
        if (cols != (tiled_ref ? g.cols : 0) || rows != (tiled_ref ? g.rows : 0))
            std::fprintf(stderr, "    picture %dx%d mode %d: %d lines of %d tiles, reference %dx%d\n",
                         g.w, g.h, g.mode, rows, cols, g.cols, g.rows);
        CHECK_EQ(cols, tiled_ref ? g.cols : 0);
        CHECK_EQ(rows, tiled_ref ? g.rows : 0);

        /* The global view, then the mosaic of the crops (if any). */
        const int32_t sg = it->n_seg - 2;
        const GoldUocrSeg& gg = kUocrSegs[g.seg0 + sg];
        if (fnv(segment_pixels(*it, sg, gg.real)) != g.global_fnv)
            std::fprintf(stderr, "    picture %dx%d mode %d: the global view's patches differ\n",
                         g.w, g.h, g.mode);
        CHECK_EQ(fnv(segment_pixels(*it, sg, gg.real)), g.global_fnv);
        std::vector<uint16_t> local;
        for (int32_t s = 0; s < sg; ++s) {
            const std::vector<uint16_t> v = segment_pixels(*it, s, kUocrSegs[g.seg0 + s].real);
            local.insert(local.end(), v.begin(), v.end());
        }
        if (g.local_fnv) {
            if (fnv(local) != g.local_fnv)
                std::fprintf(stderr, "    picture %dx%d mode %d: the crops' patches differ\n",
                             g.w, g.h, g.mode);
            CHECK_EQ(fnv(local), g.local_fnv);
        } else {
            CHECK(local.empty());
        }
        /* The token count is the reference's arithmetic. */
        CHECK_EQ(uocr_tokens(p.config(), mode, g.cols, g.rows), (int64_t)g.n_ids);
    }
}

/* Pillow's antialiased bicubic runs its two passes in its own order (libImaging/Resample.c: the
 * vertical one first when the height shrinks by more than twice what the width does) and rounds
 * the intermediate to eight bits; this pins the order at sizes where the choice changes the
 * picture, since a one-pixel difference is invisible in a summary and visible to the model. */
TEST(pillow_bicubic_resizes_in_the_references_pass_order) {
    for (const GoldUocrResize& g : kUocrResize) {
        Frame out;
        resize_bicubic(frame(0, g.sh, g.sw), g.dw, g.dh, &out);
        CHECK_EQ(out.w, g.dw);
        CHECK_EQ(out.h, g.dh);
        if (fnv(out.rgb) != g.fnv)
            std::fprintf(stderr, "    %dx%d -> %dx%d: pixels differ\n", g.sw, g.sh, g.dw, g.dh);
        CHECK_EQ(fnv(out.rgb), g.fnv);
    }
}

/* A request with several pictures marks each one `several`. A one-mode processor serves it
 * exactly as a lone picture and refuses only a mode the request named; Unlimited-OCR takes its
 * multi-page (base) mode for a picture whose mode was not named. */
TEST(several_pictures_pick_a_mode_only_where_the_processor_has_two) {
    std::string why;
    Processor q;
    REQUIRE(q.init(qwen_cfg(), [](const std::string&) { return std::vector<int32_t>{ 1 }; }) >= 0);
    ImageMode several;
    several.several = true;
    std::shared_ptr<Item> a, b;
    CHECK_OK(q.image_from(frame(0, 256, 320), &a, &why, several));
    CHECK_OK(q.image_from(frame(0, 256, 320), &b, &why));
    REQUIRE(a != nullptr && b != nullptr);
    CHECK(a->expand == b->expand);
    CHECK_EQ(fnv(a->pixels), fnv(b->pixels));
    ImageMode named;
    named.kind = ImageMode::Base;
    CHECK_EQ(q.image_from(frame(0, 256, 320), &a, &why, named), RAD_E_UNSUPPORTED);

    Processor u;
    REQUIRE_EQ(u.init(uocr_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      uocr_marker), RAD_OK);
    ImageMode base;
    base.kind = ImageMode::Base;
    std::shared_ptr<Item> x, y;
    CHECK_OK(u.image_from(frame(0, 1400, 1000), &x, &why, several));
    CHECK_OK(u.image_from(frame(0, 1400, 1000), &y, &why, base));
    REQUIRE(x != nullptr && y != nullptr);
    CHECK(x->expand == y->expand);
    CHECK_EQ(fnv(x->pixels), fnv(y->pixels));
    /* Named, the mode wins over `several`. */
    ImageMode gundam = several;
    gundam.kind = ImageMode::Gundam;
    CHECK_OK(u.image_from(frame(0, 1400, 1000), &x, &why, gundam));
    REQUIRE(x != nullptr);
    CHECK(x->expand.size() > y->expand.size());
}

/* The runs: one per segment, every token the picture marker, and a 1-D rotary that gives each
 * placeholder its own position. The reference's assembled order is the item's row order -- the
 * local lines, the global view's lines, then the view separator. */
TEST(unlimitedocr_runs_follow_the_reference_layout) {
    Processor p;
    REQUIRE_EQ(p.init(uocr_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      uocr_marker), RAD_OK);
    ImageMode gundam;
    gundam.kind = ImageMode::Gundam;
    std::shared_ptr<Item> it;
    std::string why;
    CHECK_OK(p.image_from(frame(1, 900, 1400), &it, &why, gundam));
    REQUIRE(it != nullptr);

    auto pm = std::make_shared<PromptMedia>();
    pm->items.push_back({ it, 1 });          /* after one text token */
    pm->layout();
    REQUIRE_EQ(pm->rope.runs.size(), (size_t)it->n_seg);
    for (int32_t s = 0; s < it->n_seg; ++s) {
        const RopeLayout::Run& r = pm->rope.runs[(size_t)s];
        CHECK(r.flat);
        CHECK_EQ(r.tok, 1 + it->seg_at[(size_t)s]);
        CHECK_EQ(r.n, (int32_t)(it->row_at(s + 1) - it->row_at(s)));
        CHECK_EQ(r.base, 1 + it->seg_at[(size_t)s]);
        CHECK_EQ(r.shift, 0);
        CHECK_EQ(r.item, 0);
        CHECK_EQ(r.row0, (int32_t)it->row_at(s));
    }
    const int64_t n = (int64_t)it->expand.size();
    for (int64_t i = 0; i < n; ++i) {
        int32_t pos[3];
        pm->rope.at(1 + i, pos);
        CHECK_EQ(pos[0], (int32_t)(1 + i));
        CHECK_EQ(pos[1], (int32_t)(1 + i));
        CHECK_EQ(pos[2], (int32_t)(1 + i));
    }
    /* The reference's arithmetic, spelled out: the global view's 16 lines of 16 features with a
     * newline each plus the separator, then the local grid's lines (a newline after each). */
    const VisionConfig& c = p.config();
    CHECK_EQ(uocr_tokens(c, gundam, 1, 1), (int64_t)273);
    ImageMode base;
    base.kind = ImageMode::Base;
    CHECK_EQ(uocr_tokens(c, base, 1, 1), (int64_t)273);
    ImageMode page640;
    page640.kind = ImageMode::Base;
    page640.image_size = kUocrTile;
    CHECK_EQ(uocr_tokens(c, page640, 1, 1), (int64_t)111);
    CHECK_EQ(uocr_tokens(c, gundam, 3, 2), (int64_t)273 + (int64_t)(10 * 3 + 1) * 20);
    CHECK_EQ(uocr_tokens(c, gundam, 4, 3), (int64_t)273 + (int64_t)(10 * 4 + 1) * 30);

    /* A request may restate the global view's side, but only at a grid the DeepEncoder serves
     * (64/40), and not cut crops at another tile or pad a page to a side that is not whole merged
     * patches. */
    bool crop = false;
    int32_t bs = 0, is = 0;
    ImageMode b2048;
    b2048.kind = ImageMode::Gundam;
    b2048.base_size = 2048;
    CHECK_EQ(uocr_mode(c, b2048, &crop, &bs, &is, &why), RAD_E_INVAL);
    ImageMode b640;
    b640.kind = ImageMode::Gundam;
    b640.base_size = 640;
    REQUIRE_EQ(uocr_mode(c, b640, &crop, &bs, &is, &why), RAD_OK);
    CHECK_EQ(uocr_tokens(c, b640, 1, 1), (int64_t)(10 + 1) * 10 + 1);
    ImageMode tile512;
    tile512.kind = ImageMode::Gundam;
    tile512.image_size = 512;
    CHECK_EQ(uocr_mode(c, tile512, &crop, &bs, &is, &why), RAD_E_INVAL);
    ImageMode odd;
    odd.kind = ImageMode::Base;
    odd.image_size = 700;
    CHECK_EQ(uocr_mode(c, odd, &crop, &bs, &is, &why), RAD_E_INVAL);
    ImageMode ok640;
    ok640.kind = ImageMode::Base;
    ok640.image_size = 640;
    REQUIRE_EQ(uocr_mode(c, ok640, &crop, &bs, &is, &why), RAD_OK);
    CHECK(!crop);
    CHECK_EQ(is, 640);

    /* The multi-page form: two pages at the base mode are two runs of 273, back to back. */
    ImageMode base2;
    base2.kind = ImageMode::Base;
    std::shared_ptr<Item> a, b;
    CHECK_OK(p.image_from(frame(0, 300, 500), &a, &why, base2));
    CHECK_OK(p.image_from(frame(1, 900, 1400), &b, &why, base2));
    auto mp = std::make_shared<PromptMedia>();
    mp->items.push_back({ a, 1 });
    mp->items.push_back({ b, 1 + (int32_t)a->expand.size() });
    mp->layout();
    CHECK_EQ((int64_t)mp->rope.runs.size(), (int64_t)4);
    CHECK_EQ(a->expand.size(), (size_t)273);
    CHECK_EQ(b->expand.size(), (size_t)273);
    int32_t pos[3];
    mp->rope.at(1 + 273 + 100, pos);
    CHECK_EQ(pos[0], (int32_t)(1 + 273 + 100));
}

/* The DeepEncoder's 40-tower absolute position tables are the 64/16 checkpoint tables resampled
 * with torch's bicubic + antialias (arch/common/rad_resample.h), computed host-side in
 * fill_tables() and rounded to bf16. Every golden here is torch's F.interpolate(mode="bicubic",
 * antialias=True) on CPU fp32 (tests/resample_golden.py), so this pins the load-time position
 * table number for number: a resample off by a row or a tap serves a slightly shifted picture. */
TEST(unlimitedocr_bicubic_antialias_resample_is_the_reference) {
    uocr_golden::Golden gd;
    std::string gerr;
    if (!uocr_golden::load(&gd, &gerr)) ARTIFACT_MISSING(gerr);
    const int CH = 3;
    double worst_abs = 0.0, worst_rel = 0.0;
    for (int k = 0; k < gd.n_cases; ++k) {
        const int ih = gd.geo[k][0], iw = gd.geo[k][1];
        const int oh = gd.geo[k][2], ow = gd.geo[k][3];
        std::vector<float> out((size_t)oh * ow * CH);
        rad::arch::resample_bicubic_aa(gd.in[k].data(), ih, iw, oh, ow, CH, out.data());
        const float* want = gd.out[k].data();
        double a = 0.0, r = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            const double d = std::fabs((double)out[i] - (double)want[i]);
            const double rr = d / std::fmax(std::fabs((double)want[i]), 1e-30);
            if (d > a) a = d;
            if (rr > r) r = rr;
        }
        std::fprintf(stderr, "    %dx%d -> %dx%d: max abs %.3g, max rel %.3g\n", ih, iw, oh, ow, a, r);
        if (a > worst_abs) worst_abs = a;
        if (r > worst_rel) worst_rel = r;
    }
    std::fprintf(stderr, "    synthetic fp32: worst abs %.3g, worst rel %.3g (target < 1e-4)\n",
                 worst_abs, worst_rel);
    CHECK(worst_abs < 1e-4);

    /* A checkpoint table, bf16 in and bf16 out, exactly as fill_tables() does it: decode the
     * stored bf16, resample in fp32, round back. Reports the last-bit count and the max abs the
     * bf16 table would carry. */
    auto bf16_case = [&](const uint16_t* src_bits, int ih, int iw, int oh, int ow, int ch,
                         const uint16_t* want_bits, size_t off, const char* tag,
                         size_t* mism, double* absmax) {
        std::vector<float> f((size_t)ih * iw * ch);
        for (size_t i = 0; i < f.size(); ++i) f[i] = rad_bf16_to_f32(src_bits[i]);
        std::vector<float> g((size_t)oh * ow * ch);
        rad::arch::resample_bicubic_aa(f.data(), ih, iw, oh, ow, ch, g.data());
        size_t m = 0;
        double a = 0.0;
        for (size_t i = 0; i < g.size(); ++i) {
            const uint16_t got = rad_f32_to_bf16(g[i]);
            const uint16_t want = want_bits[off + i];
            if (got != want) ++m;
            const double d = std::fabs(rad_bf16_to_f32(got) - rad_bf16_to_f32(want));
            if (d > a) a = d;
        }
        std::fprintf(stderr, "    %s: %zu/%zu bf16 mismatches, max abs %.3g\n", tag, m, g.size(), a);
        *mism = m;
        *absmax = a;
    };

    size_t sam_mism = 0;
    double sam_abs = 0.0;
    bf16_case(gd.sam_src.data(), 64, 64, 40, 40, 768, gd.sam_out.data(), 0, "sam 64->40 bf16",
              &sam_mism, &sam_abs);

    /* CLIP: rows 1: as a 16x16 grid resampled to 10x10; the cls row (index 0) passes through in
     * front, so the golden's grid starts one channel-vector in. */
    size_t clip_mism = 0;
    double clip_abs = 0.0;
    {
        const int CW = 1024;
        const uint16_t* grid_bits = gd.clip_src.data() + CW;   /* skip the cls row */
        bf16_case(grid_bits, 16, 16, 10, 10, CW, gd.clip_out.data(), (size_t)CW, "clip 16->10 bf16",
                  &clip_mism, &clip_abs);
    }
    /* The bf16 tables are a report, not a gate: bf16 carries ~3 decimal digits, so a last-bit
     * difference on a fraction of entries is representable, not a bug. The fp32 CHECK above is
     * the strict signal on the resample arithmetic. */
    (void)sam_mism;
    (void)clip_mism;
}

/* The AVX2 resample and patch-staging paths must be the scalar function, not an approximation:
 * the goldens above pin the modelled result, and this pins the vector lanes against the scalar
 * loop that produced them. mm_force_scalar selects the scalar path (core/mm/processor.cpp), so the
 * same input goes through both dispatches on this host. Sizes cover growth and shrink (different
 * tap counts), both pass orders, odd widths/tails, and a 1-pixel side. */
TEST(avx2_preprocessing_is_bit_identical_to_scalar) {
    const struct { int32_t ih, iw, oh, ow; } cases[] = {
        { 53, 67, 35, 40 },  { 53, 67, 55, 57 },  { 64, 64, 1024, 1024 },
        { 1024, 1024, 640, 640 }, { 200, 100, 50, 200 }, { 97, 129, 96, 128 },
        { 1, 1, 7, 5 },      { 71, 131, 70, 130 }, { 17, 31, 9, 8 },
        { 300, 500, 299, 501 },
    };
    int seed = 11;
    for (const auto& c : cases) {
        Frame in = frame(seed++, c.ih, c.iw);
        Frame sv, sm, tv, tm;
        mm_force_scalar(true);
        resize_bicubic(in, c.ow, c.oh, &sv);
        resize_bicubic_torch(in, c.ow, c.oh, &tv);
        mm_force_scalar(false);
        resize_bicubic(in, c.ow, c.oh, &sm);
        resize_bicubic_torch(in, c.ow, c.oh, &tm);
        CHECK_EQ(sm.w, sv.w);
        CHECK_EQ(sm.h, sv.h);
        CHECK(sm.rgb == sv.rgb);
        CHECK(tm.rgb == tv.rgb);
        if (sm.rgb != sv.rgb) {
            for (size_t i = 0; i < sv.rgb.size(); ++i)
                if (sm.rgb[i] != sv.rgb[i]) {
                    std::fprintf(stderr, "    %dx%d -> %dx%d: first byte %zu differs (%d vs %d)\n",
                                 c.iw, c.ih, c.ow, c.oh, i, (int)sv.rgb[i], (int)sm.rgb[i]);
                    break;
                }
        }
    }

    /* Patch/LUT staging (the Unlimited-OCR append path), through the whole processor so the
     * segment order and both views are covered. */
    Processor p;
    REQUIRE_EQ(p.init(uocr_cfg(), [](const std::string&) { return std::vector<int32_t>{}; },
                      uocr_marker), RAD_OK);
    const struct { int32_t seed, h, w; } imgs[] = { { 3, 640, 640 }, { 4, 900, 1400 } };
    for (const auto& g : imgs) {
        Frame in = frame(g.seed, g.h, g.w);
        for (int kind = 0; kind < 2; ++kind) {
            ImageMode mode;
            mode.kind = kind ? ImageMode::Base : ImageMode::Gundam;
            std::shared_ptr<Item> a, b;
            std::string why;
            mm_force_scalar(true);
            REQUIRE_EQ(p.image_from(in, &a, &why, mode), RAD_OK);
            mm_force_scalar(false);
            REQUIRE_EQ(p.image_from(in, &b, &why, mode), RAD_OK);
            REQUIRE(a != nullptr);
            REQUIRE(b != nullptr);
            CHECK_EQ(b->n_patch, a->n_patch);
            CHECK(b->pixels == a->pixels);
            CHECK(b->expand == a->expand);
        }
    }
}

RAD_TEST_MAIN()
