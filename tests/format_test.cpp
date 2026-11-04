/* format_test.cpp -- the .rad container: write one, read it back, and check the invariants the
 * hot path relies on.
 *
 * The refusals matter most. The engine mmaps an untrusted file and addresses into it with
 * arithmetic and no bounds test, so every assumption must be checked at open and fail with
 * RAD_E_FORMAT and a message, not a later SIGSEGV.
 */
#include "rad_test.h"
#include "format/radfile.h"
#include "format/table_copy.h"
#include "format/table_share.h"
#include "format/tunecache.h"
#include "format/imatrix.h"
#include "format/safetensors.h"
#include "format/checkpoint.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <signal.h>
#include <string>
#include <sys/file.h>
#include <sys/wait.h>
#include <unordered_map>
#include <vector>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;

namespace {

/* Writes `bytes` to radtest::tmp_path(stem); the path, or "" when the file cannot be created. */
std::string write_tmp(const char* stem, const std::string& bytes) {
    const std::string p = radtest::tmp_path(stem);
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) return std::string();
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    return p;
}
std::string write_tmp(const char* stem, const std::vector<uint8_t>& bytes) {
    return write_tmp(stem, std::string(bytes.begin(), bytes.end()));
}

std::vector<uint8_t> ramp(int64_t n, uint8_t seed) {
    std::vector<uint8_t> v((size_t)n);
    for (int64_t i = 0; i < n; ++i) v[(size_t)i] = (uint8_t)((i * 31 + seed) & 0xFF);
    return v;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::vector<uint8_t> v;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        uint8_t buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) v.insert(v.end(), buf, buf + n);
        std::fclose(f);
    }
    return v;
}

bool exists(const std::string& path) { return ::access(path.c_str(), F_OK) == 0; }

/* The test container's two encodings: plain bf16, and int4 codes with an f16 scale per [1 x 32]
 * group. For the expert shape below the int4 codes plane ends off a 256-byte boundary, so the
 * scale plane's alignment is actually exercised. */
RadEncoding plain_bf16() { return rad_enc_plain(RAD_BF16); }
RadEncoding int4_g32() { return rad_enc_affine(RAD_I4, RAD_F16, 1, 32); }

RadWriter::Weight mkw(const char* name, const RadEncoding& enc, int64_t rows, int64_t cols,
                      int32_t layer, int32_t expert, int32_t slot) {
    RadWriter::Weight w;
    w.name     = name;
    w.enc      = enc;
    w.rank     = 2;
    w.shape[0] = rows;
    w.shape[1] = cols;
    w.layer    = layer;
    w.expert   = expert;
    w.slot     = slot;
    return w;
}

/* A weight's planes as add_weight(w, planes) takes them: plane k is a ramp of exactly
 * rad_enc_plane_bytes(), seeded per plane so no two planes of a weight match. */
struct Planes {
    std::vector<std::vector<uint8_t>> bytes;
    std::vector<const void*>          ptr;
};

Planes planes_of(const RadWriter::Weight& w, uint8_t seed) {
    Planes p;
    int64_t rows = 0, cols = 0;
    rad_enc_view(w.rank, w.shape, &rows, &cols);
    for (int k = 0; k < w.enc.n_planes; ++k)
        p.bytes.push_back(ramp(rad_enc_plane_bytes(&w.enc.plane[k], rows, cols),
                               (uint8_t)(seed + 16 * k)));
    for (const auto& b : p.bytes) p.ptr.push_back(b.data());
    return p;
}

/* The expert shape: [36, 64] int4 is 1152 bytes of codes, so the 144-byte scale plane starts at
 * 1280. Seeds are per weight kind and shared by every expert, so the stride test can compare two
 * experts' bytes. */
constexpr int64_t EXP_ROWS = 36, EXP_COLS = 64;
enum : uint8_t { SEED_EMB = 1, SEED_OUT = 2, SEED_QKV = 3, SEED_GATE = 4, SEED_UP = 5,
                 SEED_DOWN = 6 };
const char* const kRecipe = "blk.*.ffn_(gate|up).*=rtn:codes=i4,group=32,scale=f16\n";
const char* const kOptions = "codes=i4,group=32,scale=f16";

/* A small complete container: two model weights, and two layers each with a dense weight and four
 * experts of three slots, to exercise fixed-stride addressing. The gate is reserved and written a
 * plane at a time, scale first and codes in two halves, as a quantiser produces it; everything
 * else goes in whole. */
struct Built {
    std::string path;
};

Built write_model(const char* stem) {
    Built b;
    b.path = radtest::tmp_path(stem);

    RadWriter w;
    CHECK_OK(w.begin(b.path.c_str()));
    w.set_arch("radtest");
    w.set_model_name("format-test");
    w.set_quant("fp8_block");
    w.set_recipe(kRecipe);
    w.set_created_by("format_test");
    w.meta_i("n_layers", 2);
    w.meta_i("n_embd", 64);
    w.meta_f("rms_eps", 1e-6);
    w.meta_s("note", "written by format_test");

    auto whole = [&](const RadWriter::Weight& x, uint8_t seed) {
        const Planes p = planes_of(x, seed);
        CHECK(w.add_weight(x, p.ptr.data()) >= 0);
    };
    whole(mkw("token_embd.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_EMB);
    whole(mkw("output.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_OUT);

    /* Per layer: the dense weight first, so the expert run is contiguous in the directory as
     * index_experts() requires to derive the stride. */
    for (int32_t layer = 0; layer < 2; ++layer) {
        const std::string blk = "blk." + std::to_string(layer);
        whole(mkw((blk + ".attn_qkv.weight").c_str(), plain_bf16(), 64, 64, layer, -1, 0),
              SEED_QKV);
        for (int32_t e = 0; e < 4; ++e) {
            const std::string x = "." + std::to_string(e);
            RadWriter::Weight g = mkw((blk + ".ffn_gate" + x).c_str(), int4_g32(), EXP_ROWS,
                                      EXP_COLS, layer, e, 0);
            g.quantizer = "rtn";
            g.options = kOptions;
            const Planes gp = planes_of(g, SEED_GATE);
            const int64_t gi = w.add_weight(g);
            CHECK(gi >= 0);
            const int64_t half = (int64_t)gp.bytes[0].size() / 2;
            CHECK_OK(w.write_plane(gi, 1, 0, gp.bytes[1].data(), (int64_t)gp.bytes[1].size()));
            CHECK_OK(w.write_plane(gi, 0, half, gp.bytes[0].data() + half,
                                   (int64_t)gp.bytes[0].size() - half));
            CHECK_OK(w.write_plane(gi, 0, 0, gp.bytes[0].data(), half));

            RadWriter::Weight u = mkw((blk + ".ffn_up" + x).c_str(), int4_g32(), EXP_ROWS,
                                      EXP_COLS, layer, e, 1);
            u.quantizer = "rtn";
            u.options = kOptions;
            whole(u, SEED_UP);
            whole(mkw((blk + ".ffn_down" + x).c_str(), plain_bf16(), 64, 32, layer, e, 2),
                  SEED_DOWN);
        }
        w.add_profile(layer, 0, 0.4f);
        w.add_profile(layer, 1, 0.3f);
        w.add_profile(layer, 2, 0.2f);
        w.add_profile(layer, 3, 0.1f);
    }

    CHECK_OK(w.finish());
    return b;
}

/* Every plane of `e` holds what planes_of() made for `w` at `seed`. */
bool planes_match(const RadFile& f, const RadFileEntry& e, const RadWriter::Weight& w,
                  uint8_t seed) {
    const Planes want = planes_of(w, seed);
    if (e.n_planes != want.bytes.size()) return false;
    for (uint32_t k = 0; k < e.n_planes; ++k) {
        const RadFilePlane& p = f.plane(e, (int)k);
        if (p.bytes != want.bytes[k].size() ||
            std::memcmp(f.data(p), want.bytes[k].data(), want.bytes[k].size()) != 0)
            return false;
    }
    return true;
}

/* Writes a copy of `src` beside it with `edit` applied, and returns the copy's path. The edit
 * gets the header as written so it can locate the table to damage. */
using Edit = std::function<void(std::vector<uint8_t>&, const RadFileHeader&)>;

std::string patched(const std::string& src, const char* tag, const Edit& edit) {
    std::vector<uint8_t> all = slurp(src);
    RadFileHeader h{};
    std::memcpy(&h, all.data(), sizeof h);
    edit(all, h);
    const std::string dst = src + "." + tag;
    FILE* f = std::fopen(dst.c_str(), "wb");
    if (f) {
        std::fwrite(all.data(), 1, all.size(), f);
        std::fclose(f);
    }
    return dst;
}

template <typename T> T get(const std::vector<uint8_t>& f, uint64_t off) {
    T v;
    std::memcpy(&v, f.data() + off, sizeof v);
    return v;
}
template <typename T> void put(std::vector<uint8_t>& f, uint64_t off, const T& v) {
    std::memcpy(f.data() + off, &v, sizeof v);
}

}  /* namespace */

/* ------------------------------------------------------------------ round trip */
TEST(write_and_read_back) {
    Built b = write_model("roundtrip");

    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    CHECK_EQ(std::string(f.str(f.header().arch_id)), std::string("radtest"));
    CHECK_EQ(std::string(f.str(f.header().model_name)), std::string("format-test"));
    CHECK_EQ(std::string(f.str(f.header().quant)), std::string("fp8_block"));
    CHECK_EQ(std::string(f.str(f.header().recipe)), std::string(kRecipe));
    CHECK_EQ(std::string(f.str(f.header().created_by)), std::string("format_test"));
    CHECK_EQ((int)f.header().version, (int)RAD_FORMAT_VER);
    CHECK_EQ((int)f.header().magic, (int)RAD_MAGIC);

    int64_t n = 0;
    CHECK(f.get_i("n_layers", &n));
    CHECK_EQ(n, (int64_t)2);
    double eps = 0;
    CHECK(f.get_f("rms_eps", &eps));
    CHECK_NEAR(eps, 1e-6, 1e-12);
    CHECK_EQ(std::string(f.get_s("note", "")), std::string("written by format_test"));

    /* 2 model + 2 layers * (1 dense + 4 experts * 3 slots) = 2 + 26 = 28. */
    CHECK_EQ(f.entry_count(), (int64_t)28);
    CHECK_EQ(f.profile_count(), (int64_t)8);
    /* 28 entries, of which 16 are two-plane experts. */
    CHECK_EQ((int64_t)f.header().plane_count, (int64_t)(28 + 16));

    const RadFileEntry* e = f.find("token_embd.weight");
    REQUIRE(e != nullptr);
    const RadEncoding plain = plain_bf16();
    CHECK(rad_enc_equal(&f.encoding(*e), &plain));
    CHECK_EQ((int)e->rank, 2);
    CHECK_EQ(e->shape[0], (int64_t)128);
    CHECK_EQ(e->shape[1], (int64_t)64);
    CHECK_EQ((int)e->n_planes, 1);
    CHECK_EQ(std::string(f.str(e->quantizer)), std::string());
    CHECK_EQ((int64_t)f.plane(*e, 0).bytes, (int64_t)(128 * 64 * 2));
    /* The first plane starts the entry, and the bytes match, not just the directory row. */
    CHECK_EQ((const void*)f.data(f.plane(*e, 0)), (const void*)f.data(*e));
    CHECK(planes_match(f, *e, mkw("", plain, 128, 64, -1, -1, 0), SEED_EMB));

    /* An expert: two planes, plus the quantiser's provenance. */
    const RadFileEntry* g = f.find("blk.1.ffn_gate.3");
    REQUIRE(g != nullptr);
    const RadEncoding i4 = int4_g32();
    CHECK(rad_enc_equal(&f.encoding(*g), &i4));
    CHECK_EQ((int)g->n_planes, 2);
    CHECK_EQ(std::string(f.str(g->quantizer)), std::string("rtn"));
    CHECK_EQ(std::string(f.str(g->options)), std::string(kOptions));
    CHECK_EQ((int)g->layer, 1);
    CHECK_EQ((int)g->expert, 3);
    CHECK_EQ((int)g->slot, 0);
    /* Written in pieces out of order, read back whole. */
    CHECK(planes_match(f, *g, mkw("", i4, EXP_ROWS, EXP_COLS, 1, 3, 0), SEED_GATE));
    const RadFileEntry* u = f.find("blk.0.ffn_up.0");
    REQUIRE(u != nullptr);
    CHECK(planes_match(f, *u, mkw("", i4, EXP_ROWS, EXP_COLS, 0, 0, 1), SEED_UP));

    CHECK(f.find("no.such.weight") == nullptr);
    f.close();
}

/* ------------------------------------------------------------------ the encoding table */
TEST(an_encoding_is_stored_once_however_many_weights_share_it) {
    Built b = write_model("encs");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    /* 28 weights, two encodings: one table row per distinct encoding, each entry naming one.
     * Equality is byte-for-byte, so a kernel receives exactly what the quantiser described. */
    REQUIRE_EQ(f.encoding_count(), (int64_t)2);
    CHECK(!rad_enc_equal(&f.encoding_at(0), &f.encoding_at(1)));
    const RadEncoding plain = plain_bf16(), i4 = int4_g32();
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        CHECK(e.enc < 2);
        const bool expert_codes = e.expert >= 0 && e.slot < 2;
        CHECK(rad_enc_equal(&f.encoding(e), expert_codes ? &i4 : &plain));
        CHECK_EQ(e.n_planes, (uint32_t)f.encoding(e).n_planes);
    }
    f.close();
}

/* ------------------------------------------------------------------ alignment */
TEST(alignment_invariants) {
    Built b = write_model("align");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    /* Pool 2 MiB (huge-page mappable), entry 4 KiB (the movement unit), plane 256 B (spec §4.2). */
    CHECK_EQ((int64_t)(f.header().data_off % RAD_ALIGN_POOL), (int64_t)0);

    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        CHECK_EQ((int64_t)(e.offset % RAD_ALIGN_UNIT), (int64_t)0);
        /* Every entry lies inside the data blob, which makes data() safe. */
        CHECK(e.offset >= f.header().data_off);
        CHECK(e.offset + e.bytes <= f.header().data_off + f.header().data_bytes);

        /* Planes are in encoding order, exactly sized, 256-aligned and non-overlapping, and the
         * entry ends with its last plane: one span a mover can copy, each plane a slice by
         * arithmetic. */
        int64_t rows = 0, cols = 0;
        rad_enc_view(e.rank, e.shape, &rows, &cols);
        uint64_t end = e.offset;
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            CHECK_EQ((int64_t)p.bytes, rad_enc_plane_bytes(&f.encoding(e).plane[k], rows, cols));
            CHECK_EQ((int64_t)(p.offset % RAD_ALIGN_SUB), (int64_t)0);
            CHECK(p.offset >= end);
            end = p.offset + p.bytes;
        }
        CHECK_EQ(f.plane(e, 0).offset, e.offset);
        CHECK_EQ(e.offset + e.bytes, end);
    }

    /* The scale plane does not follow its codes directly: 1152 bytes of codes round up to 1280. */
    const RadFileEntry* g = f.find("blk.0.ffn_gate.0");
    REQUIRE(g != nullptr);
    CHECK_EQ((int64_t)f.plane(*g, 0).bytes, (int64_t)1152);
    CHECK_EQ((int64_t)(f.plane(*g, 1).offset - g->offset), (int64_t)1280);
    CHECK_EQ((int64_t)f.plane(*g, 1).bytes, (int64_t)(EXP_ROWS * 2 * 2));

    /* The string blob ends in NUL, so str() can return a bare pointer without scanning. */
    CHECK_EQ((int)((const char*)f.base())[f.header().str_off + f.header().str_bytes - 1], 0);

    f.close();
}

/* ------------------------------------------------------------------ fixed-stride experts */
TEST(fixed_stride_expert_addressing) {
    Built b = write_model("stride");
    RadFile f;
    REQUIRE(f.open(b.path.c_str()) >= 0);

    CHECK_EQ(f.group_count(), (int64_t)2);

    for (int32_t layer = 0; layer < 2; ++layer) {
        const RadFile::ExpertGroup* g = f.group_for_layer(layer);
        CHECK(g != nullptr);
        if (!g) continue;

        CHECK_EQ((int)g->n_expert, 4);
        CHECK_EQ((int)g->n_slot, 3);
        CHECK(g->stride > 0);
        CHECK(g->stride >= g->unit_bytes);
        CHECK_EQ((int64_t)(g->stride % RAD_ALIGN_UNIT), (int64_t)0);

        /* base + id*stride must equal the directory's offset for every expert and slot; the mover
         * and expert kernels rely on this without a bounds test. */
        for (int32_t e = 0; e < g->n_expert; ++e)
            for (int32_t s = 0; s < g->n_slot; ++s) {
                const RadFileEntry& row = f.expert_entry(*g, e, s);
                CHECK_EQ((int)row.layer, (int)layer);
                CHECK_EQ((int)row.expert, (int)e);
                CHECK_EQ((int)row.slot, (int)s);
                CHECK_EQ((const void*)f.expert_ptr(*g, e, s), (const void*)f.data(row));
                CHECK_EQ((int64_t)(row.offset - g->base - (uint64_t)e * g->stride),
                         (int64_t)g->slot_off[s]);
                /* A plane has the same displacement in every expert's slot, so a scale plane is
                 * addressed by the same arithmetic as its codes. */
                const RadFileEntry& row0 = f.expert_entry(*g, 0, s);
                for (uint32_t k = 0; k < row.n_planes; ++k)
                    CHECK_EQ(f.plane(row, (int)k).offset - row.offset,
                             f.plane(row0, (int)k).offset - row0.offset);
            }

        /* Every expert was written with the same payload, so a wrong stride also shows as
         * mismatched bytes. */
        const uint8_t* a = f.expert_ptr(*g, 0, 0);
        const uint8_t* c = f.expert_ptr(*g, 3, 0);
        CHECK(std::memcmp(a, c, (size_t)g->slot_bytes[0]) == 0);
    }

    f.close();
}

/* --ngram-placement ram: a table copied from the container into locked memory holds exactly the
 * mapping's bytes at an address no file backs, and a copy read from the wrong file offset fails the
 * check against the mapping. Small enough for a default RLIMIT_MEMLOCK, and an odd size to
 * exercise the copy's 2 MiB rounding. */
TEST(a_table_copied_into_locked_memory_is_the_mapping_byte_for_byte) {
    const std::string p = radtest::tmp_path("tcopy");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");
    const RadWriter::Weight x = mkw("ngram.weight", plain_bf16(), 1500, 999, -1, -1, 0);
    const Planes want = planes_of(x, 5);
    REQUIRE(w.add_weight(x, want.ptr.data()) >= 0);
    /* A trailing weight, so a read shifted past the table's end stays inside the file. */
    const RadWriter::Weight y = mkw("tail.weight", plain_bf16(), 64, 64, -1, -1, 0);
    const Planes ywant = planes_of(y, 6);
    REQUIRE(w.add_weight(y, ywant.ptr.data()) >= 0);
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const RadFileEntry* e = f.find("ngram.weight");
    REQUIRE(e != nullptr);
    const RadFilePlane& pl = f.plane(*e, 0);
    const uint8_t* src = f.data(pl);
    const int64_t n = (int64_t)pl.bytes;
    CHECK_EQ(n, (int64_t)1500 * 999 * 2);

    TableCopy c;
    std::string why;
    const int rc = table_copy_make(f, src, n, (int64_t)pl.offset, &c, &why);
    /* A locked-memory limit below the copy is the machine's setting, not a defect. */
    if (rc == RAD_E_NOMEM && why.find("mlock2") != std::string::npos) SKIP("%s", why.c_str());
    CHECK_OK(rc);
    CHECK(c.base != nullptr && c.base != (const void*)src);
    CHECK_EQ(c.bytes % ((size_t)2 << 20), (size_t)0);
    CHECK(c.bytes >= (size_t)n);
    if (c.base) CHECK(std::memcmp(c.base, src, (size_t)n) == 0);
    table_copy_free(&c);
    CHECK(c.base == nullptr);

    /* A 16-byte shift in the file offset reads real bytes from the wrong place. */
    TableCopy bad;
    std::string bwhy;
    CHECK_EQ(table_copy_make(f, src, n, (int64_t)pl.offset + 16, &bad, &bwhy), (int)RAD_E_IO);
    CHECK(bad.base == nullptr);
    CHECK(bwhy.find("does not match") != std::string::npos);
    f.close();
}

/* --ngram-placement disk=DIR (#130): the first open fills the table's own file from the container,
 * renamed into place whole; the mapping holds the container's bytes and is that file's (the disk
 * gather finds the file behind its mapping); a second open reuses it unfilled; a file that is not
 * this table is refused rather than served. */
TEST(a_table_file_on_another_disk_is_filled_once_and_mapped_in_place_of_the_container) {
    const std::string p = radtest::tmp_path("tfile");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");
    const RadWriter::Weight x = mkw("ngram.weight", plain_bf16(), 1500, 999, -1, -1, 0);
    const Planes want = planes_of(x, 7);
    REQUIRE(w.add_weight(x, want.ptr.data()) >= 0);
    REQUIRE(w.finish() >= 0);
    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const RadFilePlane& pl = f.plane(*f.find("ngram.weight"), 0);
    const uint8_t* src = f.data(pl);
    const int64_t n = (int64_t)pl.bytes;
    const std::string tab = radtest::tmp_path("tfile-table");
    ::unlink(tab.c_str());

    TableCopy a;
    bool filled = false;
    std::string why;
    REQUIRE_EQ(table_file_open(f, src, n, (int64_t)pl.offset, tab, &a, &filled, &why), (int)RAD_OK);
    CHECK(filled);
    CHECK_EQ(a.bytes, (size_t)n);
    CHECK(std::memcmp(a.base, src, (size_t)n) == 0);
    CHECK(::access((tab + ".part").c_str(), F_OK) != 0);
    /* The mapping is the file's, as /proc/self/maps names it. */
    {
        std::ifstream maps("/proc/self/maps");
        bool backed = false;
        for (std::string line; std::getline(maps, line);) {
            unsigned long long lo = 0, hi = 0;
            if (std::sscanf(line.c_str(), "%llx-%llx", &lo, &hi) != 2) continue;
            if ((uintptr_t)a.base >= lo && (uintptr_t)a.base < hi)
                backed = line.size() >= tab.size() && line.compare(line.size() - tab.size(), tab.size(), tab) == 0;
        }
        CHECK(backed);
    }
    ::munmap(a.base, a.bytes);

    TableCopy b;
    REQUIRE_EQ(table_file_open(f, src, n, (int64_t)pl.offset, tab, &b, &filled, &why), (int)RAD_OK);
    CHECK(!filled);
    CHECK(std::memcmp(b.base, src, (size_t)n) == 0);
    ::munmap(b.base, b.bytes);

    /* Same size, other bytes: refused, and left for the operator to remove. */
    {
        std::fstream t(tab, std::ios::in | std::ios::out | std::ios::binary);
        t.seekp(0);
        const char junk[16] = "not the table..";
        t.write(junk, sizeof junk);
    }
    TableCopy c;
    CHECK_EQ(table_file_open(f, src, n, (int64_t)pl.offset, tab, &c, &filled, &why), (int)RAD_E_IO);
    CHECK(c.base == nullptr);
    CHECK(why.find("does not match") != std::string::npos);
    ::unlink(tab.c_str());
    f.close();
}

/* --ngram-placement ram=DIR: the first holder to open the file fills it, a later one finds it
 * ready (a second descriptor in one process is a second holder, since flock locks belong to the
 * open file description), and the file goes when the last holder leaves. A file whose header is
 * not this table's is never overwritten, since a live engine may have it mapped. */
TEST(a_shared_table_file_is_filled_once_and_deleted_by_the_last_holder) {
    const std::string dir = radtest::tmp_path("share");
    REQUIRE(::mkdir(dir.c_str(), 0777) == 0 || errno == EEXIST);
    const std::string container = write_tmp("share-container", ramp(8192, 1));
    REQUIRE(!container.empty());
    const int64_t n = 64 << 10;
    const std::string id = table_share_id(container, 4096, n);
    CHECK_EQ(id.size(), (size_t)16);
    CHECK_EQ(table_share_id(container, 4096, n), id);
    CHECK(table_share_id(container, 8192, n) != id);
    CHECK(table_share_id(container, 4096, n + 1) != id);
    CHECK(table_share_id(radtest::tmp_path("share-absent"), 4096, n).empty());

    TableShare a, b, bad;
    std::string why;
    CHECK_OK(table_share_open(dir, id, n, &a, &why));
    CHECK(a.filler && !a.ready);
    CHECK(!table_share_is_ready(dir, id, n));
    const std::vector<uint8_t> table = ramp(n, 3);
    REQUIRE_EQ(::pwrite(a.fd, table.data(), (size_t)n, kTableShareHeader), (ssize_t)n);
    CHECK_OK(table_share_mark_ready(&a, n, id, &why));
    CHECK(a.ready);
    CHECK(table_share_is_ready(dir, id, n));
    CHECK(!table_share_is_ready(dir, id, n * 2));
    struct stat st{};
    REQUIRE_EQ(::stat(a.path.c_str(), &st), 0);
    CHECK_EQ((int64_t)st.st_size, table_share_file_bytes(n));
    CHECK_EQ(table_share_file_bytes(n), kTableShareHeader + ((int64_t)2 << 20));

    CHECK_OK(table_share_open(dir, id, n, &b, &why));
    CHECK(b.ready && !b.filler);
    CHECK_EQ(b.lock_fd, -1);
    std::vector<uint8_t> back((size_t)n);
    REQUIRE_EQ(::pread(b.fd, back.data(), (size_t)n, kTableShareHeader), (ssize_t)n);
    CHECK(back == table);

    /* The same name, asked for a table of another length. */
    CHECK_EQ(table_share_open(dir, id, n * 2, &bad, &why), (int)RAD_E_INVAL);
    CHECK(bad.fd == -1);
    CHECK(why.find("exists") != std::string::npos);

    /* A file at another table's name whose header names this one: the id in the header is checked. */
    const std::string other = table_share_id(container, 0, n);
    const std::string other_path = dir + "/ngram-" + other + ".tab";
    std::vector<uint8_t> head(4096);
    REQUIRE_EQ(::pread(a.fd, head.data(), head.size(), 0), (ssize_t)head.size());
    {
        const int fd = ::open(other_path.c_str(), O_RDWR | O_CREAT, 0644);
        REQUIRE(fd >= 0);
        REQUIRE_EQ(::pwrite(fd, head.data(), head.size(), 0), (ssize_t)head.size());
        REQUIRE_EQ(::ftruncate(fd, table_share_file_bytes(n)), 0);
        ::close(fd);
    }
    CHECK_EQ(table_share_open(dir, other, n, &bad, &why), (int)RAD_E_INVAL);
    CHECK_EQ(::access(other_path.c_str(), F_OK), 0);
    ::unlink(other_path.c_str());

    const std::string path = a.path, lock_path = a.lock_path;
    table_share_close(&a);
    CHECK(a.fd == -1);
    CHECK_EQ(::access(path.c_str(), F_OK), 0);   /* b still holds it, after a's wait for it */
    table_share_close(&b);
    CHECK(::access(path.c_str(), F_OK) != 0);
    CHECK(::access(lock_path.c_str(), F_OK) != 0);

    /* A filler that leaves without marking the table ready leaves nothing behind, and the next
     * holder fills. */
    TableShare c, d;
    CHECK_OK(table_share_open(dir, id, n, &c, &why));
    CHECK(c.filler);
    table_share_close(&c);
    CHECK(::access(path.c_str(), F_OK) != 0);
    CHECK_OK(table_share_open(dir, id, n, &d, &why));
    CHECK(d.filler && !d.ready);
    table_share_close(&d);
}

/* A shared file costs the header and the table's rounding to 2 MiB. From kTableShareMin on that is
 * under an eighth of the table, and kTableShareMin is the least such bound for every size above
 * it; the n-gram gather's two-byte scale would be a 4 MiB file, so it keeps a private copy. */
TEST(a_table_is_shared_only_where_the_file_costs_little_beside_it) {
    CHECK(2 < kTableShareMin);
    CHECK_EQ(table_share_file_bytes(2), (int64_t)4 << 20);
    const int64_t huge = (int64_t)2 << 20;
    for (int64_t n : { kTableShareMin, kTableShareMin + 1, kTableShareMin + huge - 1,
                       (int64_t)47 << 30 })
        CHECK((table_share_file_bytes(n) - n) * 8 < n);
    const int64_t below = kTableShareMin - huge + 1;
    CHECK((table_share_file_bytes(below) - below) * 8 >= below);
}

/* Engines stopped together leave together. A holder that leaves while another still holds the
 * file waits for it, so the file goes even when the other dies before its own close (killed at a
 * stop timeout; its locks go with it). A holder that gets the lock on a file already deleted, with
 * a new engine's file at the name, leaves the new one alone. */
TEST(a_shared_table_file_goes_when_its_holders_leave_together) {
    const std::string dir = radtest::tmp_path("share-leave");
    REQUIRE(::mkdir(dir.c_str(), 0777) == 0 || errno == EEXIST);
    const std::string container = write_tmp("share-leave-container", ramp(8192, 1));
    REQUIRE(!container.empty());
    const int64_t n = 64 << 10;
    const std::string id = table_share_id(container, 4096, n);
    REQUIRE(!id.empty());
    std::string why;

    TableShare a;
    CHECK_OK(table_share_open(dir, id, n, &a, &why));
    REQUIRE(a.filler);
    CHECK_OK(table_share_mark_ready(&a, n, id, &why));
    const std::string path = a.path, lock_path = a.lock_path;
    int ready[2];
    REQUIRE_EQ(::pipe(ready), 0);
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        TableShare b;
        std::string bwhy;
        char c = table_share_open(dir, id, n, &b, &bwhy) == RAD_OK && b.ready ? 'y' : 'n';
        (void)!::write(ready[1], &c, 1);
        ::usleep(300 * 1000);
        ::_exit(0);
    }
    char c = 0;
    CHECK_EQ(::read(ready[0], &c, 1), (ssize_t)1);
    CHECK_EQ(c, 'y');
    table_share_close(&a);
    CHECK(::access(path.c_str(), F_OK) != 0);
    CHECK(::access(lock_path.c_str(), F_OK) != 0);
    ::waitpid(child, nullptr, 0);
    ::close(ready[0]);
    ::close(ready[1]);

    TableShare p, q, r;
    CHECK_OK(table_share_open(dir, id, n, &p, &why));
    REQUIRE(p.filler);
    CHECK_OK(table_share_mark_ready(&p, n, id, &why));
    CHECK_OK(table_share_open(dir, id, n, &q, &why));
    CHECK(q.ready);
    REQUIRE_EQ(::flock(q.fd, LOCK_UN), 0);   /* q is on its way out too */
    table_share_close(&p);
    CHECK(::access(path.c_str(), F_OK) != 0);
    CHECK_OK(table_share_open(dir, id, n, &r, &why));
    CHECK(r.filler);
    table_share_close(&q);
    CHECK_EQ(::access(path.c_str(), F_OK), 0);
    CHECK_EQ(::access(lock_path.c_str(), F_OK), 0);
    table_share_close(&r);
    CHECK(::access(path.c_str(), F_OK) != 0);
}

/* ------------------------------------------------------------------ reserving and writing */
TEST(a_weight_is_written_in_pieces_in_any_order) {
    const std::string p = radtest::tmp_path("pieces");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");
    const RadWriter::Weight x = mkw("blk.0.ffn_up.weight", int4_g32(), EXP_ROWS, EXP_COLS, 0,
                                    -1, 0);
    const Planes want = planes_of(x, 9);
    const int64_t idx = w.add_weight(x);
    REQUIRE(idx >= 0);
    CHECK_EQ(w.plane_bytes(idx, 0), (int64_t)1152);
    CHECK_EQ(w.plane_bytes(idx, 1), (int64_t)144);
    CHECK(w.plane_bytes(idx, 2) < 0);
    CHECK(w.plane_bytes(idx + 1, 0) < 0);

    /* Refused without writing: past the plane, before it, a nonexistent plane, an unreserved
     * weight, and null data. */
    const uint8_t* c = want.bytes[0].data();
    CHECK_EQ(w.write_plane(idx, 0, 1100, c, 100), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 0, -1, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 2, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx + 1, 0, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(-1, 0, 0, c, 1), (int)RAD_E_INVAL);
    CHECK_EQ(w.write_plane(idx, 0, 0, nullptr, 1), (int)RAD_E_INVAL);

    /* Scale first, then codes a third at a time, last third first, as a producer emitting row
     * blocks of every plane at once would write them. */
    CHECK_OK(w.write_plane(idx, 1, 0, want.bytes[1].data(), 144));
    CHECK_OK(w.write_plane(idx, 0, 768, c + 768, 384));
    CHECK_OK(w.write_plane(idx, 0, 0, c, 384));
    CHECK_OK(w.write_plane(idx, 0, 384, c + 384, 384));
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const RadFileEntry* e = f.find("blk.0.ffn_up.weight");
    REQUIRE(e != nullptr);
    CHECK(planes_match(f, *e, x, 9));
    f.close();
}

/* A sealed writer produces the same file. seal() fixes the layout once every table is in; planes
 * then go straight to their final offsets in any order, with no spill or splice, and the result
 * must match the spilling writer byte for byte. A table changed after the seal would move the
 * blob, so finish() and add_weight() refuse it. */
TEST(a_sealed_writer_writes_the_same_file_with_no_spill) {
    struct One { RadWriter::Weight x; uint8_t seed; };
    std::vector<One> ws;
    ws.push_back({ mkw("token_embd.weight", plain_bf16(), 128, 64, -1, -1, 0), SEED_EMB });
    for (int32_t layer = 0; layer < 2; ++layer)
        for (int32_t e = 0; e < 4; ++e) {
            const std::string n = "blk." + std::to_string(layer) + ".ffn_up." + std::to_string(e);
            RadWriter::Weight u = mkw(n.c_str(), int4_g32(), EXP_ROWS, EXP_COLS, layer, e, 0);
            u.quantizer = "rtn";
            u.options = kOptions;
            ws.push_back({ u, (uint8_t)(SEED_UP + layer * 4 + e) });
        }

    auto build = [&](const std::string& p, bool sealed, bool late) -> int {
        RadWriter w;
        if (w.begin(p.c_str()) < 0) return RAD_E_IO;
        w.set_arch("radtest");
        w.set_model_name("seal-test");
        w.meta_i("n_layers", 2);
        std::vector<int64_t> idx;
        for (const One& o : ws) idx.push_back(w.add_weight(o.x));
        w.add_profile(0, 1, 0.5f);
        if (sealed) {
            CHECK_OK(w.seal());
            CHECK(::access((p + ".blob.tmp").c_str(), F_OK) != 0);
            CHECK_EQ(w.add_weight(ws[0].x), (int64_t)RAD_E_STATE);
        }
        for (size_t i = ws.size(); i-- > 0;) {
            const Planes pl = planes_of(ws[i].x, ws[i].seed);
            for (size_t k = 0; k < pl.bytes.size(); ++k)
                CHECK_OK(w.write_plane(idx[i], (int)k, 0, pl.bytes[k].data(),
                                       (int64_t)pl.bytes[k].size()));
        }
        if (late) w.meta_s("note", "after the seal");
        return w.finish();
    };

    const std::string a = radtest::tmp_path("seal-spilled"), b = radtest::tmp_path("seal-sealed");
    REQUIRE(build(a, false, false) >= 0);
    REQUIRE(build(b, true, false) >= 0);
    const std::vector<uint8_t> fa = slurp(a), fb = slurp(b);
    CHECK(!fa.empty());
    CHECK(fa == fb);

    RadFile f;
    REQUIRE(f.open(b.c_str()) >= 0);
    for (const One& o : ws) {
        const RadFileEntry* e = f.find(o.x.name.c_str());
        REQUIRE(e != nullptr);
        CHECK(planes_match(f, *e, o.x, o.seed));
    }
    f.close();

    const std::string c = radtest::tmp_path("seal-late");
    CHECK_EQ(build(c, true, true), (int)RAD_E_STATE);
    CHECK(::access(c.c_str(), F_OK) != 0);
}

TEST(a_plane_left_short_is_refused_at_finish) {
    /* An unwritten hole in a plane reads as zero codes (a quantiser that returned early, an
     * off-by-one block loop). The writer refuses the file and leaves neither target nor spill. */
    const RadWriter::Weight x = mkw("w", int4_g32(), EXP_ROWS, EXP_COLS, -1, -1, 0);
    const Planes want = planes_of(x, 3);

    for (int which = 0; which < 2; ++which) {
        const std::string p = radtest::tmp_path(which ? "short_never" : "short_scale");
        RadWriter w;
        REQUIRE(w.begin(p.c_str()) >= 0);
        w.set_arch("radtest");
        const RadWriter::Weight ok = mkw("whole", plain_bf16(), 4, 64, -1, -1, 0);
        const Planes okp = planes_of(ok, 1);
        CHECK(w.add_weight(ok, okp.ptr.data()) >= 0);
        const int64_t idx = w.add_weight(x);
        REQUIRE(idx >= 0);
        if (which == 0) {
            /* every plane begun, the last two bytes of the scale never written */
            CHECK_OK(w.write_plane(idx, 0, 0, want.bytes[0].data(), 1152));
            CHECK_OK(w.write_plane(idx, 1, 0, want.bytes[1].data(), 142));
        }
        /* which == 1: reserved and never written at all */
        CHECK_EQ(w.finish(), (int)RAD_E_STATE);
        CHECK(!exists(p));
        CHECK(!exists(p + ".blob.tmp"));
    }
}

TEST(a_weight_the_writer_cannot_place_is_refused_and_leaves_no_trace) {
    const std::string p = radtest::tmp_path("refuse");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");

    RadWriter::Weight no_enc = mkw("a", plain_bf16(), 4, 64, -1, -1, 0);
    no_enc.enc = RadEncoding{};                       /* no planes, no scheme */
    CHECK(w.add_weight(no_enc) < 0);
    RadWriter::Weight bad_dtype = mkw("b", plain_bf16(), 4, 64, -1, -1, 0);
    bad_dtype.enc.plane[0].dtype = RAD_DT_PLUGIN_BASE;   /* a dtype the core cannot size */
    CHECK(w.add_weight(bad_dtype) < 0);
    RadWriter::Weight no_rank = mkw("c", plain_bf16(), 4, 64, -1, -1, 0);
    no_rank.rank = 0;
    CHECK(w.add_weight(no_rank) < 0);
    CHECK(w.add_weight(mkw("d", plain_bf16(), 0, 64, -1, -1, 0)) < 0);
    CHECK(w.add_weight(mkw("", plain_bf16(), 4, 64, -1, -1, 0)) < 0);

    /* The refusals reserved nothing: the one placed weight is the only entry and plane, at the
     * start of the blob. */
    const RadWriter::Weight ok = mkw("ok", plain_bf16(), 4, 64, -1, -1, 0);
    const Planes okp = planes_of(ok, 2);
    CHECK_EQ(w.add_weight(ok, okp.ptr.data()), (int64_t)0);
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    CHECK_EQ(f.entry_count(), (int64_t)1);
    CHECK_EQ((int64_t)f.header().plane_count, (int64_t)1);
    CHECK_EQ(f.encoding_count(), (int64_t)1);
    CHECK_EQ(f.entry(0).offset, f.header().data_off);
    CHECK(planes_match(f, f.entry(0), ok, 2));
    f.close();
}

/* ------------------------------------------------------------------ the vocab section */
TEST(the_vocab_section_is_rebased_to_where_it_lands) {
    /* Serialised with section-relative offsets, since only the writer knows where the section
     * lands; finish() rebases them and the reader reads in place. */
    const std::string p = radtest::tmp_path("vocab");
    RadWriter w;
    REQUIRE(w.begin(p.c_str()) >= 0);
    w.set_arch("radtest");

    std::vector<uint8_t> blob(sizeof(RadVocabHeader) + 3 * sizeof(rad_stroff), 0);
    RadVocabHeader h{};
    h.kind = RAD_TOK_BPE;
    h.n_tokens = 3;
    h.tok_text_off = sizeof(RadVocabHeader);
    h.bos = h.eos = h.eot = h.pad_id = h.unk = h.sep = -1;
    h.chat_template = w.intern_string("{{ messages }}");
    const rad_stroff text[3] = { w.intern_string("a"), w.intern_string("bc"),
                                 w.intern_string("<eos>") };
    std::memcpy(blob.data(), &h, sizeof h);
    std::memcpy(blob.data() + sizeof h, text, sizeof text);

    /* Offsets outside the section are refused before anything is placed. */
    RadVocabHeader abs = h;
    abs.tok_text_off = 1u << 20;
    std::vector<uint8_t> bad = blob;
    std::memcpy(bad.data(), &abs, sizeof abs);
    CHECK_EQ(w.set_vocab_section(bad.data(), (int64_t)bad.size()), (int)RAD_E_INVAL);
    CHECK_EQ(w.set_vocab_section(blob.data(), 16), (int)RAD_E_INVAL);

    CHECK_OK(w.set_vocab_section(blob.data(), (int64_t)blob.size()));
    REQUIRE(w.finish() >= 0);

    RadFile f;
    REQUIRE(f.open(p.c_str()) >= 0);
    const VocabSection v = f.vocab();
    REQUIRE(bool(v));
    CHECK_EQ((int)v.h->n_tokens, 3);
    REQUIRE(v.tok_text != nullptr);
    CHECK_EQ((const void*)v.tok_text,
             (const void*)(f.base() + f.header().vocab_off + sizeof(RadVocabHeader)));
    CHECK_EQ(std::string(f.str(v.tok_text[1])), std::string("bc"));
    CHECK_EQ(std::string(f.str(v.tok_text[2])), std::string("<eos>"));
    CHECK_EQ(std::string(f.str(v.h->chat_template)), std::string("{{ messages }}"));
    CHECK(v.tok_score == nullptr);
    CHECK(v.merges == nullptr);
    f.close();
}

/* ------------------------------------------------------------------ append in place */
namespace {

RadWriter::Weight weight_of(const RadFile& f, const RadFileEntry& e) {
    RadWriter::Weight w;
    w.name      = f.str(e.name);
    w.quantizer = f.str(e.quantizer);
    w.options   = f.str(e.options);
    w.enc       = f.encoding(e);
    w.rank      = e.rank;
    for (uint32_t d = 0; d < RAD_MAX_RANK; ++d) w.shape[d] = e.shape[d];
    w.layer     = e.layer;
    w.expert    = e.expert;
    w.slot      = e.slot;
    return w;
}

/* Everything `old` holds, through add_existing, in its own order. */
void carry_over(RadWriter& w, const RadFile& old) {
    w.set_arch("radtest");
    w.set_model_name("format-test");
    w.set_quant("fp8_block");
    w.set_recipe(kRecipe);
    w.set_created_by("format_test, extended");
    w.meta_i("n_layers", 2);
    w.meta_s("note", "extended in place");
    for (int64_t i = 0; i < old.entry_count(); ++i)
        CHECK(w.add_existing(weight_of(old, old.entry(i)), old, old.entry(i)) >= 0);
    for (int64_t i = 0; i < old.profile_count(); ++i)
        w.add_profile(old.profile()[i].layer, old.profile()[i].expert, old.profile()[i].share);
}

}  /* namespace */

TEST(a_container_extended_in_place_keeps_every_old_weight_where_it_was) {
    Built b = write_model("append");
    const std::vector<uint8_t> before = slurp(b.path);
    REQUIRE(before.size() > sizeof(RadFileHeader));

    RadFile old;
    REQUIRE((old.open(b.path.c_str())) >= 0);
    const int64_t n_old = old.entry_count();
    std::vector<uint64_t> offs;
    std::vector<RadFilePlane> planes;
    for (int64_t i = 0; i < n_old; ++i) {
        offs.push_back(old.entry(i).offset);
        for (uint32_t k = 0; k < old.entry(i).n_planes; ++k)
            planes.push_back(old.plane(old.entry(i), (int)k));
    }

    RadWriter w;
    REQUIRE((w.begin_append(b.path.c_str(), old)) >= 0);
    carry_over(w, old);
    /* A carried-over weight's bytes belong to the old file and cannot be written. */
    const uint8_t one = 1;
    CHECK_EQ(w.write_plane(0, 0, 0, &one, 1), (int)RAD_E_INVAL);
    /* One weight in a new encoding, one in an existing one. */
    const RadWriter::Weight wv = mkw("v.patch.weight", rad_enc_affine(RAD_I8, RAD_BF16, 1, 64),
                                     48, 64, -1, -1, 0);
    const Planes pv = planes_of(wv, 9);
    CHECK(w.add_weight(wv, pv.ptr.data()) >= 0);
    RadWriter::Weight wb = mkw("v.patch.bias", plain_bf16(), 1, 64, -1, -1, 0);
    wb.rank = 1;
    wb.shape[0] = 64;
    wb.shape[1] = 0;
    const Planes pb = planes_of(wb, 10);
    CHECK(w.add_weight(wb, pb.ptr.data()) >= 0);
    w.meta_s("radiance.encoder", "vision");
    REQUIRE((w.finish()) >= 0);
    old.close();

    /* Nothing between the header and the old end moved: the old tables are dead but intact, and
     * every old weight's bytes stay in place. */
    const std::vector<uint8_t> after = slurp(b.path);
    REQUIRE(after.size() > before.size());
    CHECK(std::memcmp(after.data() + sizeof(RadFileHeader), before.data() + sizeof(RadFileHeader),
                      before.size() - sizeof(RadFileHeader)) == 0);

    RadFile f;
    REQUIRE((f.open(b.path.c_str())) >= 0);
    CHECK_EQ(f.entry_count(), n_old + 2);
    size_t pi = 0;
    for (int64_t i = 0; i < n_old; ++i) {
        CHECK_EQ(f.entry(i).offset, offs[(size_t)i]);
        for (uint32_t k = 0; k < f.entry(i).n_planes; ++k, ++pi) {
            REQUIRE(pi < planes.size());
            CHECK_EQ(f.plane(f.entry(i), (int)k).offset, planes[pi].offset);
            CHECK_EQ(f.plane(f.entry(i), (int)k).bytes, planes[pi].bytes);
        }
    }
    CHECK_EQ(f.group_count(), (int64_t)2);
    const RadFile::ExpertGroup* g = f.group_for_layer(1);
    REQUIRE(g != nullptr);
    CHECK_EQ((int)g->n_expert, 4);
    const RadFileEntry* e0 = f.find("blk.1.ffn_gate.2");
    REQUIRE(e0 != nullptr);
    CHECK_EQ(std::string(f.str(e0->quantizer)), std::string("rtn"));
    CHECK(planes_match(f, *e0, weight_of(f, *e0), SEED_GATE));

    /* The old encodings plus one new one; the bias shares the embedding's. */
    CHECK_EQ(f.encoding_count(), (int64_t)3);
    const RadFileEntry* ev = f.find("v.patch.weight");
    REQUIRE(ev != nullptr);
    CHECK((int64_t)ev->offset >= (int64_t)before.size());
    CHECK_EQ((int64_t)(ev->offset % RAD_ALIGN_UNIT), (int64_t)0);
    CHECK_EQ((int64_t)(f.plane(*ev, 1).offset % RAD_ALIGN_SUB), (int64_t)0);
    CHECK(planes_match(f, *ev, wv, 9));
    const RadFileEntry* eb = f.find("v.patch.bias");
    REQUIRE(eb != nullptr);
    CHECK_EQ(eb->enc, f.find("token_embd.weight")->enc);
    CHECK(planes_match(f, *eb, wb, 10));

    CHECK_EQ(std::string(f.get_s("note", "")), std::string("extended in place"));
    CHECK_EQ(std::string(f.get_s("radiance.encoder", "")), std::string("vision"));
    CHECK_EQ(std::string(f.str(f.header().created_by)), std::string("format_test, extended"));
    CHECK_EQ(std::string(f.str(f.header().recipe)), std::string(kRecipe));
    CHECK_EQ(f.profile_count(), (int64_t)8);
    CHECK_EQ((int64_t)(f.header().data_off % RAD_ALIGN_POOL), (int64_t)0);
    f.close();

    /* The saved header and length restore the original file byte for byte. */
    const std::vector<uint8_t> bak = slurp(b.path + ".pre-append");
    REQUIRE_EQ(bak.size(), sizeof(RadFileHeader) + sizeof(int64_t));
    int64_t len = 0;
    std::memcpy(&len, bak.data() + sizeof(RadFileHeader), sizeof len);
    CHECK_EQ(len, (int64_t)before.size());
    const int fd = ::open(b.path.c_str(), O_WRONLY);
    REQUIRE(fd >= 0);
    CHECK_EQ(::pwrite(fd, bak.data(), sizeof(RadFileHeader), 0), (ssize_t)sizeof(RadFileHeader));
    CHECK_EQ(::ftruncate(fd, (off_t)len), 0);
    ::close(fd);
    CHECK(slurp(b.path) == before);
}

TEST(an_in_place_extension_that_fails_leaves_the_container_as_it_was) {
    Built b = write_model("append_fail");
    const std::vector<uint8_t> before = slurp(b.path);

    /* A new expert cannot join its layer from past the end; the refusal truncates the file back. */
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        REQUIRE((w.begin_append(b.path.c_str(), old)) >= 0);
        carry_over(w, old);
        const RadWriter::Weight wv = mkw("v.patch.weight", plain_bf16(), 64, 64, -1, -1, 0);
        const Planes pv = planes_of(wv, 9);
        CHECK(w.add_weight(wv, pv.ptr.data()) >= 0);
        const RadWriter::Weight wg = mkw("blk.1.ffn_gate.4", int4_g32(), EXP_ROWS, EXP_COLS, 1,
                                         4, 0);
        const Planes pg = planes_of(wg, SEED_GATE);
        CHECK(w.add_weight(wg, pg.ptr.data()) < 0);
        w.abort();
    }
    CHECK(slurp(b.path) == before);

    /* Another process has the file open: refused before writing. */
    int ready[2];
    REQUIRE_EQ(::pipe(ready), 0);
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        const int fd = ::open(b.path.c_str(), O_RDONLY);
        char c = fd >= 0 ? 'y' : 'n';
        (void)!::write(ready[1], &c, 1);
        ::pause();
        ::_exit(0);
    }
    char c = 0;
    CHECK_EQ(::read(ready[0], &c, 1), (ssize_t)1);
    CHECK_EQ(c, 'y');
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        CHECK_EQ(w.begin_append(b.path.c_str(), old), (int)RAD_E_STATE);
    }
    ::kill(child, SIGKILL);
    ::waitpid(child, nullptr, 0);
    ::close(ready[0]);
    ::close(ready[1]);
    CHECK(slurp(b.path) == before);

    /* Once it exits, the same file extends. */
    {
        RadFile old;
        REQUIRE((old.open(b.path.c_str())) >= 0);
        RadWriter w;
        CHECK_OK(w.begin_append(b.path.c_str(), old));
        w.abort();
    }
    CHECK(slurp(b.path) == before);
}

/* ------------------------------------------------------------------ refusals */
TEST(truncated_file_is_refused_not_crashed) {
    Built b = write_model("trunc");

    /* Truncated copies: the header still claims the full length and table offsets point past the
     * end. Nothing may be dereferenced. */
    FILE* fp = std::fopen(b.path.c_str(), "rb");
    CHECK(fp != nullptr);
    std::fseek(fp, 0, SEEK_END);
    const long full = std::ftell(fp);
    std::fclose(fp);

    const std::string cut = b.path + ".cut";
    for (long keep : { (long)16, full / 3, full - 4096 }) {
        FILE* in = std::fopen(b.path.c_str(), "rb");
        FILE* outf = std::fopen(cut.c_str(), "wb");
        CHECK(in && outf);
        std::vector<uint8_t> buf((size_t)keep);
        CHECK_EQ((long)std::fread(buf.data(), 1, (size_t)keep, in), keep);
        std::fwrite(buf.data(), 1, (size_t)keep, outf);
        std::fclose(in);
        std::fclose(outf);

        RadFile f;
        const int s = f.open(cut.c_str());
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
    }

    /* Not a .rad at all. */
    const std::string junk = b.path + ".junk";
    FILE* j = std::fopen(junk.c_str(), "wb");
    CHECK(j != nullptr);
    const char noise[512] = { 'n', 'o', 't', 'r', 'a', 'd' };
    std::fwrite(noise, 1, sizeof noise, j);
    std::fclose(j);
    RadFile f;
    CHECK_EQ(f.open(junk.c_str()), (int)RAD_E_FORMAT);

    /* A missing path is RAD_E_IO, distinct from RAD_E_FORMAT. */
    CHECK_EQ(f.open((b.path + ".absent").c_str()), (int)RAD_E_IO);
}

TEST(a_container_of_another_version_is_refused_with_the_way_out) {
    /* Only one version is read. Any other is refused, and the message names rad-convert. */
    Built b = write_model("version");
    for (uint32_t v : { 1u, RAD_FORMAT_VER + 1 }) {
        const std::string bad = patched(b.path, "ver", [&](std::vector<uint8_t>& all,
                                                           const RadFileHeader& h) {
            RadFileHeader x = h;
            x.version = v;
            put(all, 0, x);
        });
        RadFile f;
        int s = RAD_OK;
        const std::string log = stderr_of([&] { s = f.open(bad.c_str()); });
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
        CHECK(log.find("rad-convert") != std::string::npos);
    }
}

TEST(a_table_count_that_overflows_its_span_is_refused) {
    /* A table count whose byte span overflows is refused. `count * sizeof(element)` is a uint64
     * product of a file-controlled count; every element size is even, so a count of 2^63 wraps to
     * zero and looks like an empty table, while later loops iterate the count and walk off the
     * mapping. Each table is corrupted in its own copy so a refusal is attributable. */
    Built b = write_model("ovf");
    for (int which = 0; which < 5; ++which) {
        const std::string bad = patched(b.path, "ovf", [&](std::vector<uint8_t>& all,
                                                           const RadFileHeader& h) {
            RadFileHeader x = h;
            const uint64_t wrap = 1ULL << 63;
            switch (which) {
                case 0: x.meta_count  = wrap; break;
                case 1: x.dir_count   = wrap; break;
                case 2: x.plane_count = wrap; break;
                case 3: x.enc_count   = wrap; break;
                default: x.prof_count = wrap; break;
            }
            put(all, 0, x);
        });
        RadFile f;
        CHECK_EQ(f.open(bad.c_str()), (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
    }
}

TEST(an_entry_whose_planes_are_not_its_encodings_is_refused) {
    /* A reader slices planes by arithmetic and hands them to a kernel's relayout as the encoding
     * describes. Every disagreement between directory, plane table and encoding table is refused
     * at open, each on its own copy of a good file. Entry 0 is the bf16 embedding (one plane);
     * entry 3 is layer 0's first expert gate (codes at +0, scale at +1280). */
    Built b = write_model("planes");
    {
        RadFile ok;
        REQUIRE(ok.open(b.path.c_str()) >= 0);
        REQUIRE(ok.encoding(ok.entry(3)).n_planes == 2);
        REQUIRE(ok.entry(3).expert == 0);
    }

    auto entry = [](const std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i) {
        return get<RadFileEntry>(all, h.dir_off + i * sizeof(RadFileEntry));
    };
    auto set_entry = [](std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i,
                        const RadFileEntry& e) {
        put(all, h.dir_off + i * sizeof(RadFileEntry), e);
    };
    auto plane_at = [](const RadFileHeader& h, uint64_t i) {
        return h.plane_off + i * sizeof(RadFilePlane);
    };

    struct Case { const char* what; Edit edit; };
    const Case cases[] = {
        { "an encoding index past the table", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.enc = (uint32_t)h.enc_count;
              set_entry(all, h, 0, e);
          } },
        { "a plane count that is not its encoding's", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.n_planes = 2;
              set_entry(all, h, 0, e);
          } },
        { "planes running off the plane table", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 3);
              e.plane_first = h.plane_count - 1;
              set_entry(all, h, 3, e);
          } },
        { "a plane index that wraps", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 0);
              e.plane_first = ~0ULL;
              set_entry(all, h, 0, e);
          } },
        { "a plane two bytes short of its encoding", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e = entry(all, h, 0);
              RadFilePlane p = get<RadFilePlane>(all, plane_at(h, e.plane_first));
              p.bytes -= 2;
              put(all, plane_at(h, e.plane_first), p);
          } },
        /* -16, not +16, so the scale plane still ends inside its entry after the codes and only
         * the alignment is wrong. */
        { "a plane off its 256-byte boundary", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e = entry(all, h, 3);
              RadFilePlane p = get<RadFilePlane>(all, plane_at(h, e.plane_first + 1));
              p.offset -= 16;
              put(all, plane_at(h, e.plane_first + 1), p);
          } },
        { "a plane before its own entry", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadFileEntry e0 = entry(all, h, 0), e1 = entry(all, h, 1);
              put(all, plane_at(h, e1.plane_first), get<RadFilePlane>(all, plane_at(h, e0.plane_first)));
          } },
        { "an encoding with no planes", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              x.n_planes = 0;
              put(all, h.enc_off, x);
          } },
        { "an encoding with a dtype the core cannot size", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              x.plane[0].dtype = RAD_DT_PLUGIN_BASE + 1;
              put(all, h.enc_off, x);
          } },
        /* An unknown scheme is a quantiser's own and rad_enc_valid() accepts it; only the
         * missing terminator is wrong. */
        { "an unterminated scheme", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadEncoding x = get<RadEncoding>(all, h.enc_off);
              std::memset(x.scheme, 'q', sizeof x.scheme);
              put(all, h.enc_off, x);
          } },
        { "an entry off its 4 KiB boundary", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = entry(all, h, 1);
              e.offset -= 256;
              set_entry(all, h, 1, e);
          } },
    };

    for (const Case& c : cases) {
        const std::string bad = patched(b.path, "planes", c.edit);
        RadFile f;
        const int s = f.open(bad.c_str());
        if (s != RAD_E_FORMAT) std::fprintf(stderr, "    not refused: %s\n", c.what);
        CHECK_EQ(s, (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
    }
}

TEST(a_layer_with_no_fixed_stride_is_refused) {
    /* Experts within a layer must share shape and format, or the hot path's unchecked
     * `base + id*stride` reads the wrong bytes. Both writer and reader refuse it. */
    const std::string p = radtest::tmp_path("nostride");
    const RadWriter::Weight small = mkw("e", plain_bf16(), 32, 64, 0, 0, 0);
    const Planes ps = planes_of(small, 1);

    /* (a) The writer refuses the offending weight by name, while the checkpoint is still at hand.
     * Expert 0 defines the unit shape and expert 1 does not match it slot for slot.
     *
     * The size check is not redundant with the offset check: an oversized last expert lands at the
     * right base and passes every offset test, yet base + id*stride runs off the end. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        CHECK(w.add_weight(mkw("e0", plain_bf16(), 32, 64, 0, 0, 0), ps.ptr.data()) >= 0);
        const RadWriter::Weight big = mkw("e1", plain_bf16(), 128, 64, 0, 1, 0);
        CHECK(w.add_weight(big, planes_of(big, 2).ptr.data()) < 0);
        w.abort();
    }

    /* (b) Out of order, which no stride can address; refused on the first weight. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        CHECK(w.add_weight(mkw("e1", plain_bf16(), 32, 64, 0, 1, 0), ps.ptr.data()) < 0);
        w.abort();
    }

    /* (c) The reader, on a container RadWriter did not produce: a .rad may come from anywhere,
     * so the reader re-derives the stride from the directory at open and refuses a group without
     * one.
     *
     * A well-formed file whose directory is then edited: expert 1 and its plane move one 4 KiB
     * unit onto expert 2. Every per-entry check still holds, but the experts sit at base,
     * base + 8K and base + 8K, which no single stride describes. */
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        for (int e = 0; e < 3; ++e)
            CHECK(w.add_weight(mkw(("e" + std::to_string(e)).c_str(), plain_bf16(), 32, 64, 0, e,
                                   0), ps.ptr.data()) >= 0);
        CHECK_OK(w.finish());

        /* It opens as written. */
        { RadFile f; CHECK_OK(f.open(p.c_str())); }

        const std::string bad = patched(p, "moved", [&](std::vector<uint8_t>& all,
                                                        const RadFileHeader& h) {
            const uint64_t at = h.dir_off + sizeof(RadFileEntry);
            RadFileEntry e = get<RadFileEntry>(all, at);
            e.offset += RAD_ALIGN_UNIT;
            put(all, at, e);
            const uint64_t pat = h.plane_off + e.plane_first * sizeof(RadFilePlane);
            RadFilePlane pl = get<RadFilePlane>(all, pat);
            pl.offset += RAD_ALIGN_UNIT;
            put(all, pat, pl);
        });
        RadFile f;
        CHECK_EQ(f.open(bad.c_str()), (int)RAD_E_FORMAT);
        CHECK(!f.is_open());
    }
}

/* ------------------------------------------------------------------ the remaining refusals
 *
 * Each check validate() and index_experts() make, on its own patched copy of a good file. The
 * refusal must be RAD_E_FORMAT with the container closed, and its message must be the check's
 * own, so a case is not passed by some earlier check the edit happens to trip. */
namespace {

struct Refusal {
    const char* what;
    const char* says;      /* a fragment of the intended check's message */
    Edit        edit;
};

/* Runs every case on `src` and reports the ones not refused, or refused by another check. */
void expect_refusals(const std::string& src, const std::vector<Refusal>& cases) {
    for (const Refusal& c : cases) {
        const std::string bad = patched(src, "refuse", c.edit);
        RadFile f;
        int st = RAD_OK;
        const std::string log = stderr_of([&] { st = f.open(bad.c_str()); });
        const bool ok = st == RAD_E_FORMAT && !f.is_open() && has(log, c.says);
        if (!ok)
            std::fprintf(stderr, "    %s: status %d, log: %s\n", c.what, st, log.c_str());
        CHECK(ok);
    }
}

RadFileEntry dir_entry(const std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i) {
    return get<RadFileEntry>(all, h.dir_off + i * sizeof(RadFileEntry));
}
void set_dir_entry(std::vector<uint8_t>& all, const RadFileHeader& h, uint64_t i,
                   const RadFileEntry& e) {
    put(all, h.dir_off + i * sizeof(RadFileEntry), e);
}

/* A container with a vocab section that has every array: three tokens with text and types, one
 * merge and one step, so each of the section's checks has something to refuse. The section is
 * laid out header, text, types (padded to 4), merges, then the step at the next multiple of 8. */
std::string write_vocab_model(const char* stem) {
    const std::string p = radtest::tmp_path(stem);
    RadWriter w;
    CHECK_OK(w.begin(p.c_str()));
    w.set_arch("radtest");
    const uint64_t text_off = sizeof(RadVocabHeader);
    const uint64_t type_off = text_off + 3 * sizeof(rad_stroff);
    const uint64_t merge_off = type_off + 4;
    const uint64_t step_off = (merge_off + 8 + 7) / 8 * 8;
    std::vector<uint8_t> blob(step_off + sizeof(RadVocabStep), 0);
    RadVocabHeader h{};
    h.kind = RAD_TOK_BPE;
    h.n_tokens = 3;
    h.n_merges = 1;
    h.n_steps = 1;
    h.tok_text_off = text_off;
    h.tok_type_off = type_off;
    h.merge_off = merge_off;
    h.step_off = step_off;
    h.bos = h.eos = h.eot = h.pad_id = h.unk = h.sep = -1;
    h.chat_template = w.intern_string("{{ messages }}");
    const rad_stroff text[3] = { w.intern_string("a"), w.intern_string("b"),
                                 w.intern_string("ab") };
    const uint8_t types[3] = { RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_NORMAL };
    const uint32_t merge[2] = { 0, 1 };
    RadVocabStep st{};
    st.kind = RAD_PRE_SPLIT;
    st.arg0 = w.intern_string("\\s+");
    std::memcpy(blob.data(), &h, sizeof h);
    std::memcpy(blob.data() + text_off, text, sizeof text);
    std::memcpy(blob.data() + type_off, types, sizeof types);
    std::memcpy(blob.data() + merge_off, merge, sizeof merge);
    std::memcpy(blob.data() + step_off, &st, sizeof st);
    CHECK_OK(w.set_vocab_section(blob.data(), (int64_t)blob.size()));
    w.meta_s("note", "a string value");
    w.add_weight(mkw("w", plain_bf16(), 4, 64, -1, -1, 0), planes_of(mkw("w", plain_bf16(), 4, 64,
                                                                         -1, -1, 0), 1).ptr.data());
    CHECK_OK(w.finish());
    return p;
}

RadVocabHeader vocab_header(const std::vector<uint8_t>& all, const RadFileHeader& h) {
    return get<RadVocabHeader>(all, h.vocab_off);
}

}  /* namespace */

/* The header, the string blob, the metadata and the vocab section: every offset into the string
 * blob or the section is checked, so str() and the vocab view are plain pointers afterwards. */
TEST(every_table_and_string_check_refuses_its_own_corruption) {
    const std::string p = write_vocab_model("tables");
    {
        RadFile ok;
        REQUIRE(ok.open(p.c_str()) == RAD_OK);
        REQUIRE(bool(ok.vocab()));
        REQUIRE(ok.vocab().merges != nullptr && ok.vocab().steps != nullptr);
        REQUIRE(ok.get_s("note") != nullptr);
    }
    auto set_header = [](std::vector<uint8_t>& all, const RadFileHeader& h) { put(all, 0, h); };
    const std::vector<Refusal> cases = {
        { "an empty string blob", "string blob is empty", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.str_bytes = 0; set_header(all, x); } },
        { "a string blob that does not end in NUL", "last byte is not NUL", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.str_bytes -= 1; set_header(all, x); } },
        { "a string blob past the end of the file", "the string blob lies outside", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.str_bytes = all.size(); set_header(all, x); } },
        { "an architecture name outside the blob", "header string offset", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.arch_id = h.str_bytes; set_header(all, x); } },
        { "a recipe outside the blob", "header string offset", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.recipe = ~0ull; set_header(all, x); } },
        { "provenance outside the blob", "header string offset", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.created_by = h.str_bytes + 7; set_header(all, x); } },
        { "a metadata key outside the blob", "metadata row 0", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileKV kv = get<RadFileKV>(all, h.meta_off); kv.key = h.str_bytes; put(all, h.meta_off, kv); } },
        { "a metadata string value outside the blob", "metadata row 0", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileKV kv = get<RadFileKV>(all, h.meta_off); kv.v.s = h.str_bytes; put(all, h.meta_off, kv); } },
        { "a data blob off its 2 MiB boundary", "not 2097152-aligned", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.data_off -= 4096; set_header(all, x); } },
        { "a vocab section past the end of the file", "the vocab section lies outside", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.vocab_bytes = all.size(); set_header(all, x); } },
        { "a vocab section shorter than its header", "shorter than its own header", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.vocab_bytes = sizeof(RadVocabHeader) - 8; set_header(all, x); } },
        { "a vocab section off its alignment", "the vocab section starts at", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileHeader x = h; x.vocab_off += 4; x.vocab_bytes -= 4; set_header(all, x); } },
        { "token text that runs off the section", "arrays do not fit", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadVocabHeader v = vocab_header(all, h); v.tok_text_off = h.vocab_off + h.vocab_bytes - 8;
              put(all, h.vocab_off, v); } },
        { "token types before the section", "arrays do not fit", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadVocabHeader v = vocab_header(all, h); v.tok_type_off = h.vocab_off - 8;
              put(all, h.vocab_off, v); } },
        { "merges off their element alignment", "arrays do not fit", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadVocabHeader v = vocab_header(all, h); v.merge_off += 2; put(all, h.vocab_off, v); } },
        { "a step count past the section", "arrays do not fit", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadVocabHeader v = vocab_header(all, h); v.n_steps = 1000; put(all, h.vocab_off, v); } },
        { "a merge naming a token past the vocabulary", "merge 0 names token 3", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadVocabHeader v = vocab_header(all, h); put(all, v.merge_off + 4, (uint32_t)3); } },
        { "token text outside the string blob", "vocab token 1 points outside", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadVocabHeader v = vocab_header(all, h);
              put(all, v.tok_text_off + sizeof(rad_stroff), (rad_stroff)h.str_bytes); } },
        { "a step string outside the string blob", "vocab step 0 points outside", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              const RadVocabHeader v = vocab_header(all, h);
              RadVocabStep st = get<RadVocabStep>(all, v.step_off); st.arg1 = h.str_bytes + 100;
              put(all, v.step_off, st); } },
        { "a chat template outside the string blob", "chat template offset", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadVocabHeader v = vocab_header(all, h); v.chat_template = h.str_bytes; put(all, h.vocab_off, v); } },
    };
    expect_refusals(p, cases);
}

/* A directory row: its rank, its extents and its span, each refused at open. */
TEST(every_directory_row_check_refuses_its_own_corruption) {
    Built b = write_model("rows");
    const std::vector<Refusal> cases = {
        { "rank 0", "has rank 0", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 0); e.rank = 0; set_dir_entry(all, h, 0, e); } },
        { "a rank past the ABI's", "the ABI allows", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 0); e.rank = RAD_MAX_RANK + 1; set_dir_entry(all, h, 0, e); } },
        { "an extent of zero", "extent 0 in dim 1", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 1); e.shape[1] = 0; set_dir_entry(all, h, 1, e); } },
        { "a negative extent", "extent -128 in dim 0", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 1); e.shape[0] = -128; set_dir_entry(all, h, 1, e); } },
        { "a span past the end of the file", "-- truncated", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 2); e.bytes = all.size(); set_dir_entry(all, h, 2, e); } },
        { "a name outside the string blob", "directory row 2 has a string offset", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 2); e.options = h.str_bytes; set_dir_entry(all, h, 2, e); } },
    };
    expect_refusals(b.path, cases);
}

/* The expert directory's layout rules, which `base + id * stride` relies on. In write_model's
 * directory, rows 3..14 are layer 0's four experts of three slots and rows 16..27 layer 1's. */
TEST(every_expert_layout_check_refuses_its_own_corruption) {
    Built b = write_model("experts");
    const std::vector<Refusal> cases = {
        { "expert 0's slots out of order", "expert 0's slots are 1 at position 0", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 3); e.slot = 1; set_dir_entry(all, h, 3, e); } },
        { "an expert 0 slot before the expert's base", "is before the expert's base", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 4); e.offset = dir_entry(all, h, 3).offset - RAD_ALIGN_UNIT;
              e.bytes += dir_entry(all, h, 4).offset - e.offset; set_dir_entry(all, h, 4, e); } },
        { "experts listed out of order", "listed out of order (expected 1, found 2)", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              for (uint64_t i = 6; i < 9; ++i) { RadFileEntry e = dir_entry(all, h, i); e.expert = 2; set_dir_entry(all, h, i, e); } } },
        { "a later expert without expert 0's slots", "does not have the same 3 slots", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 8); e.slot = 1; set_dir_entry(all, h, 8, e); } },
        { "a later expert's slot of another size", "must share a shape and format", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              RadFileEntry e = dir_entry(all, h, 6); e.bytes += RAD_ALIGN_SUB; set_dir_entry(all, h, 6, e); } },
        { "a layer's experts split by another weight", "not contiguous in the directory", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              for (uint64_t i = 16; i < 28; ++i) { RadFileEntry e = dir_entry(all, h, i); e.layer = 0; set_dir_entry(all, h, i, e); } } },
        { "more slots than the directory supports", "has 9 slots", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
              /* Expert 0 runs on through experts 1 and 2. */
              for (uint64_t i = 6; i < 12; ++i) { RadFileEntry e = dir_entry(all, h, i); e.expert = 0; set_dir_entry(all, h, i, e); } } },
    };
    expect_refusals(b.path, cases);

    /* The writer refuses a ninth slot while the weight is at hand, rather than write a container
     * the reader then refuses. */
    const std::string p = radtest::tmp_path("nineslots");
    RadWriter w;
    CHECK_OK(w.begin(p.c_str()));
    w.set_arch("radtest");
    for (int32_t s = 0; s < 9; ++s) {
        const RadWriter::Weight x = mkw(("e0." + std::to_string(s)).c_str(), plain_bf16(), 4, 64,
                                        0, 0, s);
        const int64_t idx = w.add_weight(x, planes_of(x, 1).ptr.data());
        CHECK(s < 8 ? idx >= 0 : idx == RAD_E_INVAL);
    }
    w.abort();
}

/* A weight is looked up by name, so a container holds each name once. The writer refuses a second
 * weight of a name it holds, from add_weight and from add_existing, and the reader refuses a
 * directory that lists one twice rather than serving the first and leaving the second dead. */
TEST(a_container_holds_each_name_once) {
    const std::string p = radtest::tmp_path("dupname");
    const RadWriter::Weight a = mkw("blk.0.attn_q.weight", plain_bf16(), 32, 64, 0, -1, 0);
    const Planes pa = planes_of(a, 1);
    {
        RadWriter w;
        CHECK_OK(w.begin(p.c_str()));
        w.set_arch("radtest");
        CHECK(w.add_weight(a, pa.ptr.data()) >= 0);
        CHECK_EQ(w.add_weight(a, pa.ptr.data()), (int64_t)RAD_E_INVAL);
        RadWriter::Weight b = a;
        b.name = "blk.0.attn_k.weight";
        CHECK(w.add_weight(b, pa.ptr.data()) >= 0);
        CHECK_OK(w.finish());
    }
    {
        RadFile f;
        REQUIRE(f.open(p.c_str()) == RAD_OK);
        CHECK_EQ(f.entry_count(), (int64_t)2);
        /* Extending in place takes the same name once too. */
        RadWriter w;
        REQUIRE(w.begin_append(p.c_str(), f) == RAD_OK);
        CHECK(w.add_existing(weight_of(f, f.entry(0)), f, f.entry(0)) >= 0);
        CHECK_EQ(w.add_existing(weight_of(f, f.entry(0)), f, f.entry(0)), (int64_t)RAD_E_INVAL);
        CHECK_EQ(w.add_weight(a, pa.ptr.data()), (int64_t)RAD_E_INVAL);
        w.abort();
    }
    /* Entry 1's name pointed at entry 0's: every other check holds. */
    const std::string bad = patched(p, "dup", [&](std::vector<uint8_t>& all, const RadFileHeader& h) {
        RadFileEntry e0 = get<RadFileEntry>(all, h.dir_off);
        RadFileEntry e1 = get<RadFileEntry>(all, h.dir_off + sizeof(RadFileEntry));
        e1.name = e0.name;
        put(all, h.dir_off + sizeof(RadFileEntry), e1);
    });
    RadFile f;
    int s = RAD_OK;
    const std::string log = stderr_of([&] { s = f.open(bad.c_str()); });
    CHECK_EQ(s, (int)RAD_E_FORMAT);
    CHECK(!f.is_open());
    CHECK(has(log, "blk.0.attn_q.weight"));
}

TEST(empty_container_still_opens) {
    /* A container with no weights (vocab- or metadata-only) is legal and needs no special case. */
    const std::string p = radtest::tmp_path("empty");
    RadWriter w;
    CHECK_OK(w.begin(p.c_str()));
    w.set_arch("radtest");
    w.meta_i("n_layers", 0);
    CHECK_OK(w.finish());

    RadFile f;
    CHECK_OK(f.open(p.c_str()));
    CHECK_EQ(f.entry_count(), (int64_t)0);
    CHECK_EQ(f.encoding_count(), (int64_t)0);
    CHECK_EQ(f.group_count(), (int64_t)0);
    CHECK(!f.vocab());
    CHECK_EQ(f.profile_count(), (int64_t)0);
    int64_t n = -1;
    CHECK(f.get_i("n_layers", &n));
    CHECK_EQ(n, (int64_t)0);
    f.close();
}

/* ------------------------------------------------------------------ the tuning cache */
TEST(tune_cache_round_trip) {
    const std::string home = radtest::tmp_path("home");
    ::mkdir(home.c_str(), 0755);

    TuneCache c;
    c.set_machine("gfx1201-x2-test");

    Geometry g;
    g.set_i("M", 64);
    g.set_i("N", 8192);
    g.set_i("K", 5120);
    g.set_s("dtype", "w4a8");
    const std::string key = TuneCache::canon_geometry(g);
    /* Sorted by key, so a shape is spelled the same whatever the plugin's parameter order. */
    CHECK_EQ(key, std::string("K=5120 M=64 N=8192 dtype=w4a8"));

    Geometry g2;
    g2.set_s("dtype", "w4a8");
    g2.set_i("K", 5120);
    g2.set_i("N", 8192);
    g2.set_i("M", 64);
    CHECK_EQ(TuneCache::canon_geometry(g2), key);

    TuneRow r;
    r.kernel = "gemm_w4a8_nt_m64";
    r.plugin = "libr4d";
    r.choice = "bm=128,bn=64,stages=3";
    r.geom = key;
    r.us = 41.203;
    r.measured = TuneCache::now_iso8601();
    c.put(r);

    TuneRow r2 = r;
    r2.choice = "";             /* a row with no declared axes */
    r2.geom = "K=1 M=1 N=1";
    r2.us = 1.5;
    c.put(r2);

    CHECK_OK(c.save(home));

    TuneCache back;
    CHECK_OK(back.load(home, "gfx1201-x2-test"));
    CHECK_EQ(back.rows().size(), (size_t)2);

    const TuneRow* got = back.find("gemm_w4a8_nt_m64", "libr4d", key);
    CHECK(got != nullptr);
    if (got) {
        CHECK_EQ(got->choice, std::string("bm=128,bn=64,stages=3"));
        CHECK_NEAR(got->us, 41.203, 1e-6);
        CHECK_EQ(got->measured, r.measured);
    }
    /* A missing row is absent, not a default. */
    CHECK(back.find("gemm_w4a8_nt_m64", "libr4d", "K=1") == nullptr);
    CHECK(back.find("nope", "libr4d", key) == nullptr);

    /* A missing cache is normal on a fresh install, not a failure. */
    TuneCache none;
    CHECK_OK(none.load(home, "no-such-machine"));
    CHECK(none.empty());
}

/* The cache may be hand-edited, so a whitespace-only line, before the magic line or among the
 * rows, is a blank line, not a malformed row. */
TEST(a_tune_cache_line_of_blanks_is_a_blank_line) {
    const std::string home = radtest::tmp_path("home_blank");
    ::mkdir(home.c_str(), 0755);
    ::mkdir((home + "/tune").c_str(), 0755);
    const std::string path = TuneCache::path_for(home, "blank-test");
    const char* bodies[] = {
        /* among the rows */
        "radtune 2\n"
        "   \n"
        "machine blank-test\n"
        "\t\r\n"
        "gemm_x libr4d mb=2 3.5 2026-01-01T00:00:00Z K=1 M=1 N=1\n"
        "  \n",
        /* ahead of the magic line */
        " \t \n"
        "radtune 2\n"
        "gemm_x libr4d mb=2 3.5 2026-01-01T00:00:00Z K=1 M=1 N=1\n",
    };
    for (const char* body : bodies) {
        FILE* f = std::fopen(path.c_str(), "wb");
        CHECK(f != nullptr);
        if (!f) return;
        std::fputs(body, f);
        std::fclose(f);

        TuneCache c;
        CHECK_OK(c.load(home, "blank-test"));
        CHECK_EQ(c.rows().size(), (size_t)1);
        CHECK(c.find("gemm_x", "libr4d", "K=1 M=1 N=1") != nullptr);
    }
}

/* A cache from another machine gives wrong answers, so the key names what changes them. On the
 * host backend the device is the CPU, named by its model: host kernels tune differently on an AVX2
 * part and an AVX-512 one, so the CPU must be in the key rather than a fixed "host". */
TEST(the_machine_key_names_the_card_or_the_cpu) {
    RadDeviceProps gpu{};
    std::snprintf(gpu.name, sizeof gpu.name, "AMD Radeon AI PRO R9700");
    std::snprintf(gpu.arch, sizeof gpu.arch, "gfx1201");
    CHECK_EQ(rad_machine_key_of(2, &gpu), std::string("gfx1201-x2-AMD_Radeon_AI_PRO_R9700"));

    RadDeviceProps cpu{};
    std::snprintf(cpu.name, sizeof cpu.name, "AMD Ryzen 9 7950X 16-Core Processor  ");
    std::snprintf(cpu.arch, sizeof cpu.arch, "host");
    cpu.is_host_backend = 1;
    CHECK_EQ(rad_machine_key_of(1, &cpu), std::string("host-x1-AMD_Ryzen_9_7950X_16_Core_Processor"));
    std::snprintf(cpu.name, sizeof cpu.name, "Intel(R) Xeon(R) Gold 6338");
    CHECK_EQ(rad_machine_key_of(1, &cpu), std::string("host-x1-Intel_R__Xeon_R__Gold_6338"));

    /* No device at all still names the CPU architecture. */
    const std::string none = rad_machine_key_of(0, nullptr);
    CHECK(none.rfind("host-", 0) == 0);
    CHECK(none != "host-");
}

/* The cache may be hand-edited and its geometries have no length bound, so a row is read whole
 * however long it is: a row split at a buffer's length would leave a truncated geometry that never
 * matches and a tail misread as a row of its own. A timing of nan or inf is a malformed row (every
 * comparison against it is false or wins), and a measurement date saved as "-" reads back empty,
 * as an empty choice does. */
TEST(a_tune_cache_row_is_read_whole_and_a_timing_must_be_finite) {
    const std::string home = radtest::tmp_path("home_long");
    ::mkdir(home.c_str(), 0755);
    ::mkdir((home + "/tune").c_str(), 0755);
    const std::string path = TuneCache::path_for(home, "long-test");
    std::string geom;
    for (int k = 0; geom.size() < 6000; ++k) geom += (k ? " " : "") + fmt("key_%04d=%d", k, k * 7);
    std::string body = "radtune 2\nmachine long-test\n";
    body += "gemm_long libr4d mb=2 3.5 2026-01-01T00:00:00Z " + geom + "\n";
    body += "gemm_nan libr4d - nan - K=1\n";
    body += "gemm_inf libr4d - inf - K=2\n";
    body += "gemm_neg libr4d - -inf - K=3\n";
    body += "gemm_ok libr4d - 2.25 - K=4\n";
    FILE* f = std::fopen(path.c_str(), "wb");
    REQUIRE(f != nullptr);
    std::fputs(body.c_str(), f);
    std::fclose(f);

    TuneCache c;
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = c.load(home, "long-test"); });
    CHECK_EQ(st, (int)RAD_OK);
    CHECK_EQ(c.rows().size(), (size_t)2);
    const TuneRow* lr = c.find("gemm_long", "libr4d", geom);
    REQUIRE(lr != nullptr);
    CHECK_EQ(lr->geom.size(), geom.size());
    CHECK(c.find("gemm_nan", "libr4d", "K=1") == nullptr);
    CHECK(c.find("gemm_inf", "libr4d", "K=2") == nullptr);
    CHECK(c.find("gemm_neg", "libr4d", "K=3") == nullptr);
    const TuneRow* ok = c.find("gemm_ok", "libr4d", "K=4");
    REQUIRE(ok != nullptr);
    CHECK_EQ(ok->measured, std::string());
    CHECK_EQ(ok->choice, std::string());
    CHECK(has(log, "3 malformed row(s)"));

    /* An empty date survives a save and a load as empty, not as "-". */
    CHECK_OK(c.save(home));
    TuneCache back;
    CHECK_OK(back.load(home, "long-test"));
    const TuneRow* again = back.find("gemm_ok", "libr4d", "K=4");
    REQUIRE(again != nullptr);
    CHECK_EQ(again->measured, std::string());
    REQUIRE(back.find("gemm_long", "libr4d", geom) != nullptr);
}

/* ------------------------------------------------------------------ the safetensors reader */

/* Tensor names are matched, so their decoding decides whether a weight is found. JSON writes a
 * codepoint above U+FFFF as a surrogate pair; encoding the halves separately yields CESU-8 (six
 * bytes) that never equals the standard four-byte UTF-8, and the tensor looks missing.
 *
 * The other cases are the boundaries: the last two-byte codepoint, the first three-byte one, and
 * an unpaired high surrogate, which is passed through rather than rejected since names are only
 * matched. */
static std::string st_write(const std::string& header_json) {
    const uint64_t n = header_json.size();
    std::string bytes((const char*)&n, 8);
    bytes += header_json;
    bytes.append(4, '\0');                         /* one f32 of data section */
    return write_tmp("st.safetensors", bytes);
}

TEST(safetensors_decodes_a_surrogate_pair_as_one_codepoint) {
    /* U+1F600 as a JSON surrogate pair. */
    const std::string path = st_write(
        "{\"a\\uD83D\\uDE00b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
    CHECK(!path.empty());
    if (path.empty()) return;

    SafeTensorsFile st;
    CHECK_OK(st.open(path));
    CHECK_EQ(st.tensors().size(), (size_t)1);
    if (st.tensors().size() == 1) {
        const std::string& got = st.tensors()[0].name;
        CHECK_EQ(got, std::string("a\xF0\x9F\x98\x80" "b"));
        CHECK_EQ(got.size(), (size_t)6);           /* 'a' + four bytes + 'b', not 'a' + six + 'b' */
        CHECK(st.find("a\xF0\x9F\x98\x80" "b") != nullptr);
    }
    st.close();
}

TEST(safetensors_decodes_the_escape_widths_either_side_of_it) {
    const std::string path = st_write(
        "{\"\\u007F\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\u07FF\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\u0800\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        " \"\\uD800x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}");
    CHECK(!path.empty());
    if (path.empty()) return;

    SafeTensorsFile st;
    CHECK_OK(st.open(path));
    CHECK(st.find("\x7F") != nullptr);                     /* one byte */
    CHECK(st.find("\xDF\xBF") != nullptr);                 /* two */
    CHECK(st.find("\xE0\xA0\x80") != nullptr);             /* three */
    CHECK(st.find("\xED\xA0\x80" "x") != nullptr);         /* lone surrogate, passed through */
    st.close();
}

/* Readers walk a tensor by shape and dtype, so a shape that disagrees with its data_offsets span
 * is refused at open. st_write's data section is one f32. */
TEST(safetensors_refuses_a_shape_that_is_not_its_span) {
    const char* bad[] = {
        /* two f32 claimed over four bytes */
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}",
        /* 2^62 x 4 elements, a product that wraps to zero */
        "{\"t\":{\"dtype\":\"U8\",\"shape\":[4611686018427387904,4],\"data_offsets\":[0,0]}}",
        /* a negative extent */
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[-1,-1],\"data_offsets\":[0,4]}}",
        /* fewer elements than the span */
        "{\"t\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,4]}}",
    };
    for (const char* h : bad) {
        const std::string path = st_write(h);
        CHECK(!path.empty());
        if (path.empty()) return;
        SafeTensorsFile st;
        CHECK_EQ(st.open(path), RAD_E_FORMAT);
    }

    const char* good[] = {
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}",        /* a scalar */
        "{\"t\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[0,4]}}",
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[0,7],\"data_offsets\":[4,4]}}",     /* empty */
        /* an unsized dtype is left to consumers, which refuse it by name */
        "{\"t\":{\"dtype\":\"F8_E8M0\",\"shape\":[9],\"data_offsets\":[0,4]}}",
    };
    for (const char* h : good) {
        const std::string path = st_write(h);
        CHECK(!path.empty());
        if (path.empty()) return;
        SafeTensorsFile st;
        CHECK_OK(st.open(path));
        CHECK_EQ(st.tensors().size(), (size_t)1);
    }
}

/* ================================================================== the shard index
 *
 * A sharded checkpoint's weight_map is the only record of which file holds each tensor. An index
 * parsed short makes rad-convert report a present tensor as missing, which looks like a broken
 * download.
 *
 * Absence is not an error (a single-file checkpoint has none and the caller falls back to
 * model.safetensors), so "not found" and "malformed" are distinct results.
 */
TEST(the_shard_index_reads_every_entry_of_a_real_weight_map) {
    /* As Hugging Face writes it: a metadata block, then the map. */
    const std::string p = write_tmp("index_ok", R"({
  "metadata": { "total_size": 123456789 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
    "model.layers.47.mlp.down_proj.weight": "model-00003-of-00003.safetensors",
    "lm_head.weight": "model-00003-of-00003.safetensors"
  }
})");
    CHECK(!p.empty());
    std::unordered_map<std::string, std::string> m;
    CHECK_OK(st_load_index(p, &m));
    CHECK_EQ(m.size(), 4u);
    CHECK_EQ(m["model.embed_tokens.weight"], std::string("model-00001-of-00003.safetensors"));
    CHECK_EQ(m["lm_head.weight"], std::string("model-00003-of-00003.safetensors"));
    CHECK_EQ(m.count("model.layers.47.mlp.down_proj.weight"), 1u);

    /* The out map is cleared first, so a failed second read leaves no stale shards behind. */
    const std::string q = write_tmp("index_one", R"({"weight_map":{"a":"s0.safetensors"}})");
    CHECK_OK(st_load_index(q, &m));
    CHECK_EQ(m.size(), 1u);
    CHECK_EQ(m.count("lm_head.weight"), 0u);

    /* An empty map is a valid index, not a parse failure. */
    const std::string e = write_tmp("index_empty", R"({"weight_map":{}})");
    CHECK_OK(st_load_index(e, &m));
    CHECK_EQ(m.size(), 0u);
}

TEST(a_missing_index_is_not_found_and_a_broken_one_is_a_format_error) {
    std::unordered_map<std::string, std::string> m;
    /* Not found is the ordinary single-file case; the caller falls back on it but refuses a
       corrupt index. */
    CHECK_EQ(st_load_index(radtest::tmp_path("does_not_exist"), &m), RAD_E_NOTFOUND);

    const std::string a = write_tmp("index_nomap", R"({"metadata":{"total_size":1}})");
    CHECK_EQ(st_load_index(a, &m), RAD_E_FORMAT);

    const std::string b = write_tmp("index_notobj", R"({"weight_map": "a string"})");
    CHECK_EQ(st_load_index(b, &m), RAD_E_FORMAT);

    const std::string c = write_tmp("index_trunc", R"({"weight_map":{"a":)");
    CHECK_EQ(st_load_index(c, &m), RAD_E_FORMAT);

    const std::string d = write_tmp("index_empty_file", "");
    CHECK_EQ(st_load_index(d, &m), RAD_E_FORMAT);

    /* A null out pointer is refused. */
    CHECK_EQ(st_load_index(radtest::tmp_path("whatever"), nullptr), RAD_E_INVAL);
}

/* Escapes in index names are decoded; a name with its backslashes intact would not match the
 * safetensors header and the tensor would look missing. */
TEST(the_shard_index_decodes_escaped_names) {
    const std::string p = write_tmp("index_esc",
        "{\"weight_map\":{\"a\\/b.weight\":\"s0.safetensors\","
        "\"q\\\"uote\":\"s1.safetensors\"}}");
    std::unordered_map<std::string, std::string> m;
    CHECK_OK(st_load_index(p, &m));
    CHECK_EQ(m.size(), 2u);
    CHECK_EQ(m.count("a/b.weight"), 1u);
    CHECK_EQ(m.count("q\"uote"), 1u);
}

/* ================================================================== the importance matrix
 *
 * An imatrix is an untrusted file feeding a long unattended quantisation. Both readers walk a
 * length-prefixed format out of a mapping, so a malformed file risks an out-of-bounds read, and an
 * in-bounds misparse silently degrades quantisation quality.
 *
 * llama.cpp has written two formats; only the newer has a per-expert dimension, so the join of
 * `<name>.in_sum2` and `<name>.counts` is what MoE quantisation depends on.
 */
namespace {

/* A little-endian byte builder for the length-prefixed formats, lengths included. */
struct Bin {
    std::vector<uint8_t> b;
    void raw(const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*)p;
        b.insert(b.end(), q, q + n);
    }
    void i32(int32_t v)  { raw(&v, 4); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v)    { raw(&v, 4); }
    void gstr(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void lstr(const std::string& s) { i32((int32_t)s.size()); raw(s.data(), s.size()); }
    void pad_to(size_t a) { while (b.size() % a) b.push_back(0); }
};

/* One legacy entry: name, ncall, and the raw sums. */
struct LegacyEntry { std::string name; int32_t ncall; std::vector<float> vals; };

std::vector<uint8_t> legacy_bytes(const std::vector<LegacyEntry>& es) {
    Bin w;
    w.i32((int32_t)es.size());
    for (const LegacyEntry& e : es) {
        w.lstr(e.name);
        w.i32(e.ncall);
        w.i32((int32_t)e.vals.size());
        for (float v : e.vals) w.f32(v);
    }
    return w.b;
}

/* One GGUF tensor: name, dims (dim 0 fastest) and payload. gguf_bytes computes offsets and the
 * data section. */
struct GgufTensor { std::string name; std::vector<uint64_t> dims; std::vector<float> data; };

struct GgufKv { std::string key; uint32_t type; uint32_t u; std::string s; };

std::vector<uint8_t> gguf_bytes(const std::vector<GgufTensor>& ts,
                                const std::vector<GgufKv>& kvs,
                                uint32_t version = 3, uint64_t alignment = 32) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(version);
    w.u64(ts.size());
    w.u64(kvs.size());
    for (const GgufKv& kv : kvs) {
        w.gstr(kv.key);
        w.u32(kv.type);
        if (kv.type == 4)      w.u32(kv.u);                        /* u32 */
        else if (kv.type == 8) w.gstr(kv.s);                       /* string */
        else if (kv.type == 9) { w.u32(8); w.u64(1); w.gstr(kv.s); }  /* array of one string */
        else if (kv.type == 12) { double d = (double)kv.u; w.raw(&d, 8); }  /* f64 */
    }
    uint64_t off = 0;
    for (const GgufTensor& t : ts) {
        w.gstr(t.name);
        w.u32((uint32_t)t.dims.size());
        for (uint64_t d : t.dims) w.u64(d);
        w.u32(0);                                 /* GGML_TYPE_F32 */
        w.u64(off);
        off += t.data.size() * sizeof(float);
    }
    w.pad_to((size_t)alignment);
    for (const GgufTensor& t : ts) for (float v : t.data) w.f32(v);
    return w.b;
}

/* One float of an entry's payload, read by memcpy since a GGUF payload may be unaligned in the
 * mapping. */
float f32_at(const float* p, size_t i) {
    float v = 0.0f;
    std::memcpy(&v, (const char*)p + i * sizeof(float), sizeof v);
    return v;
}

double mean_of(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return v.empty() ? 0.0 : s / (double)v.size();
}

}  /* namespace */

TEST(a_legacy_imatrix_reads_every_entry_it_declares) {
    const std::string path = write_tmp("legacy.imatrix", legacy_bytes({
        { "blk.0.attn_q.weight",    10, { 1.0f, 2.0f, 3.0f, 4.0f } },
        { "blk.0.ffn_down.weight",   4, { 8.0f, 8.0f } },
    }));
    CHECK(!path.empty());

    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(!im.is_gguf());
    CHECK_EQ(im.entries().size(), (size_t)2);
    CHECK_EQ(im.path(), path);

    const ImatrixEntry* e = im.find("blk.0.attn_q.weight");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->cols, (int64_t)4);
        CHECK_EQ(e->ncall, (int64_t)10);
        /* The legacy format has no per-expert dimension, so every tensor is dense and all
         * experts in a layer share the layer's aggregate. */
        CHECK_EQ(e->n_expert, (int64_t)1);
        CHECK(e->counts == nullptr);
        CHECK(e->sum2 != nullptr);
        if (e->sum2) CHECK_EQ(f32_at(e->sum2, 2), 3.0f);
    }
    CHECK(im.find("blk.0.ffn_up.weight") == nullptr);
}

/* Importance is normalised to mean 1, so a weighted quantisation error is comparable with an
 * unweighted one. */
TEST(importance_is_normalised_to_mean_one) {
    const std::string path = write_tmp("norm.imatrix", legacy_bytes({
        { "t", 7, { 1.0f, 3.0f, 4.0f, 8.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(im.importance("t", 0, &w));
    CHECK_EQ(w.size(), (size_t)4);
    CHECK_NEAR(mean_of(w), 1.0, 1e-5);
    /* The row's shape survives: column 3 is eight times column 0. */
    if (w.size() == 4) CHECK_NEAR(w[3] / w[0], 8.0, 1e-4);
}

/* A uniform scale of the row does not change the result, since mean-1 divides it out. The
 * per-expert division relies on this, and it makes the weights independent of calibration-set
 * size. */
TEST(importance_is_invariant_to_a_uniform_scale_of_the_row) {
    const std::string a = write_tmp("scale_a.imatrix", legacy_bytes({
        { "t", 1, { 1.0f, 2.0f, 5.0f, 8.0f } },
    }));
    const std::string b = write_tmp("scale_b.imatrix", legacy_bytes({
        { "t", 1000, { 1000.0f, 2000.0f, 5000.0f, 8000.0f } },
    }));
    ImatrixFile ia, ib;
    CHECK_OK(ia.open(a));
    CHECK_OK(ib.open(b));
    std::vector<float> wa, wb;
    CHECK(ia.importance("t", 0, &wa));
    CHECK(ib.importance("t", 0, &wb));
    CHECK_EQ(wa.size(), wb.size());
    for (size_t i = 0; i < wa.size() && i < wb.size(); ++i) CHECK_NEAR(wa[i], wb[i], 1e-4);
}

/* A tensor the calibration set never activated has an all-zero row, which cannot be normalised,
 * so it falls back to uniform weights (as if quantised without an imatrix). */
TEST(a_degenerate_row_falls_back_to_uniform_weights) {
    const std::string path = write_tmp("zero.imatrix", legacy_bytes({
        { "t", 3, { 0.0f, 0.0f, 0.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(im.importance("t", 0, &w));
    CHECK_EQ(w.size(), (size_t)3);
    for (float x : w) CHECK_EQ(x, 1.0f);
}

TEST(importance_declines_a_tensor_or_an_expert_it_does_not_have) {
    const std::string path = write_tmp("absent.imatrix", legacy_bytes({
        { "t", 1, { 1.0f, 2.0f } },
    }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    std::vector<float> w;
    CHECK(!im.importance("not_here", 0, &w));
    CHECK(!im.importance("t", 1, &w));      /* the legacy format has one expert */
    CHECK(!im.importance("t", -1, &w));
    /* With no output buffer, the query reports whether importance is available. */
    CHECK(im.importance("t", 0, nullptr));
}

/* ------------------------------------------------------------------ gguf */

TEST(a_gguf_imatrix_joins_the_sums_with_the_counts) {
    const std::string path = write_tmp("moe.imatrix", gguf_bytes(
        {
            /* Two experts of four columns, expert-major. */
            { "blk.0.ffn_down.in_sum2", { 4, 2 },
              { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f } },
            { "blk.0.ffn_down.counts",  { 1, 2 }, { 100.0f, 7.0f } },
            { "blk.0.attn_q.in_sum2",   { 3 },    { 2.0f, 4.0f, 6.0f } },
        },
        { { "imatrix.chunk_count", 4, 128, "" },
          { "imatrix.chunk_size",  4, 512, "" },
          { "imatrix.datasets",    9, 0,   "wiki.train.raw" } }));
    CHECK(!path.empty());

    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(im.is_gguf());
    CHECK_EQ(im.chunk_count(), (int64_t)128);
    CHECK_EQ(im.chunk_size(), (int64_t)512);
    CHECK_EQ(im.dataset(), std::string("wiki.train.raw"));
    CHECK_EQ(im.entries().size(), (size_t)2);

    const ImatrixEntry* e = im.find("blk.0.ffn_down");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->cols, (int64_t)4);
        CHECK_EQ(e->n_expert, (int64_t)2);
        CHECK(e->counts != nullptr);
    }
    /* A dense tensor in the same file is one expert; the suffix is stripped from both names. */
    const ImatrixEntry* q = im.find("blk.0.attn_q");
    CHECK(q != nullptr);
    if (q) {
        CHECK_EQ(q->cols, (int64_t)3);
        CHECK_EQ(q->n_expert, (int64_t)1);
        CHECK(q->counts == nullptr);
    }
}

/* Each expert is weighted by its own row, not expert 0's and not a layer aggregate as in the
 * legacy format. */
TEST(a_per_expert_importance_comes_from_that_experts_own_row) {
    const std::string path = write_tmp("expert.imatrix", gguf_bytes(
        {
            { "t.in_sum2", { 4, 2 }, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f } },
            { "t.counts",  { 1, 2 }, { 100.0f, 7.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));

    std::vector<float> w0, w1;
    CHECK(im.importance("t", 0, &w0));
    CHECK(im.importance("t", 1, &w1));
    CHECK_EQ(w0.size(), (size_t)4);
    CHECK_EQ(w1.size(), (size_t)4);
    /* Expert 0 has a flat row, so every column weighs the same. */
    for (float x : w0) CHECK_NEAR(x, 1.0, 1e-5);
    /* Expert 1's ratios are its own. */
    CHECK_NEAR(mean_of(w1), 1.0, 1e-5);
    if (w1.size() == 4) {
        CHECK_NEAR(w1[3] / w1[0], 4.0, 1e-4);
        CHECK(w1[0] != w1[3]);
    }
    CHECK(!im.importance("t", 2, &w1));
}

/* The converter names weights as container entries; importance_for maps them onto the imatrix's
 * GGUF names. A wrong mapping hands a quantiser another tensor's importance, or none, with nothing
 * reported, so each spelling is pinned: a dense weight, an expert slice of a stacked entry, a width
 * that disagrees, and an expert the calibration never routed to (all-zero sums), which takes
 * uniform weights rather than zeros that leave a weighted search degenerate. */
TEST(a_converter_weight_name_finds_its_imatrix_row) {
    const std::string path = write_tmp("names.imatrix", gguf_bytes(
        {
            { "blk.0.attn_q.in_sum2", { 4 }, { 1.0f, 2.0f, 3.0f, 6.0f } },
            /* Three experts of four columns; expert 1 never routed. */
            { "blk.5.ffn_down_exps.in_sum2", { 4, 3 },
              { 1.0f, 1.0f, 1.0f, 1.0f,  0.0f, 0.0f, 0.0f, 0.0f,  8.0f, 4.0f, 2.0f, 2.0f } },
            { "blk.5.ffn_down_exps.counts", { 1, 3 }, { 10.0f, 0.0f, 3.0f } },
        },
        {}));
    ImatrixFile im;
    REQUIRE(im.open(path) == RAD_OK);
    std::vector<float> w;

    REQUIRE(im.importance_for("blk.0.attn_q.weight", 4, &w));
    REQUIRE_EQ(w.size(), (size_t)4);
    CHECK_NEAR(w[0], 1.0 / 3.0, 1e-6);             /* mean of the row is 3 */
    CHECK_NEAR(w[3], 2.0, 1e-6);

    REQUIRE(im.importance_for("blk.5.ffn_down_exps.2.weight", 4, &w));
    REQUIRE_EQ(w.size(), (size_t)4);
    CHECK_NEAR(w[0], 2.0, 1e-6);                   /* expert 2's own row, mean 4 */
    CHECK_NEAR(w[2], 0.5, 1e-6);

    REQUIRE(im.importance_for("blk.5.ffn_down_exps.1.weight", 4, &w));
    for (float x : w) CHECK_EQ(x, 1.0f);

    CHECK(!im.importance_for("blk.0.attn_q.weight", 5, &w));          /* another width */
    CHECK(!im.importance_for("blk.5.ffn_down_exps.3.weight", 4, &w)); /* no expert 3 */
    CHECK(!im.importance_for("blk.5.ffn_down_exps.-1.weight", 4, &w));
    CHECK(!im.importance_for("blk.5.ffn_down_exps.x.weight", 4, &w));
    CHECK(!im.importance_for("blk.0.attn_k.weight", 4, &w));
}

/* llama.cpp's imatrix (tools/imatrix, "%s.in_sum2" of the tensor's name) keeps `.weight` in every
 * entry's name; a converter weight finds that spelling too, dense or an expert of a stack. */
TEST(a_converter_weight_name_finds_llama_cpps_spelling) {
    const std::string path = write_tmp("names-llama.imatrix", gguf_bytes(
        {
            { "blk.0.attn_q.weight.in_sum2", { 4 }, { 1.0f, 2.0f, 3.0f, 6.0f } },
            { "blk.5.ffn_down_exps.weight.in_sum2", { 4, 2 },
              { 1.0f, 1.0f, 1.0f, 1.0f,  8.0f, 4.0f, 2.0f, 2.0f } },
            { "blk.5.ffn_down_exps.weight.counts", { 1, 2 }, { 10.0f, 3.0f } },
        },
        {}));
    ImatrixFile im;
    REQUIRE(im.open(path) == RAD_OK);
    std::vector<float> w;
    REQUIRE(im.importance_for("blk.0.attn_q.weight", 4, &w));
    CHECK_NEAR(w[0], 1.0 / 3.0, 1e-6);
    CHECK_NEAR(w[3], 2.0, 1e-6);
    REQUIRE(im.importance_for("blk.5.ffn_down_exps.1.weight", 4, &w));
    CHECK_NEAR(w[0], 2.0, 1e-6);                   /* expert 1's own row, mean 4 */
    CHECK_NEAR(w[2], 0.5, 1e-6);
    CHECK(!im.importance_for("blk.5.ffn_down_exps.2.weight", 4, &w)); /* no expert 2 */
    CHECK(!im.importance_for("blk.0.attn_q.weight", 5, &w));
}

/* With general.alignment 1 every float of a payload may sit at an odd address; the importance is
 * read by memcpy, so its values are those the file holds. */
TEST(an_importance_at_an_odd_offset_reads_the_values_the_file_holds) {
    const std::string path = write_tmp("odd.imatrix", gguf_bytes(
        { { "a.in_sum2", { 3 }, { 3.0f, 3.0f, 3.0f } },
          { "b.in_sum2", { 2 }, { 1.0f, 3.0f } } },
        { { "general.alignment", 4, 1, "" } }, 3, 1));
    ImatrixFile im;
    REQUIRE(im.open(path) == RAD_OK);
    const ImatrixEntry* e = im.find("b");
    REQUIRE(e != nullptr);
    CHECK(((uintptr_t)e->sum2 & 3) != 0);           /* the case under test: misaligned */
    std::vector<float> w;
    REQUIRE(im.importance_for("b.weight", 2, &w));
    REQUIRE_EQ(w.size(), (size_t)2);
    CHECK_NEAR(w[0], 0.5, 1e-6);
    CHECK_NEAR(w[1], 1.5, 1e-6);
}

/* The activation count stays raw: the placement planner warm-starts the expert profile from how
 * often each expert ran, and normalising would make every expert equally popular. */
TEST(the_activation_count_is_reported_unnormalised) {
    const std::string path = write_tmp("count.imatrix", gguf_bytes(
        {
            { "t.in_sum2", { 2, 3 }, { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f } },
            { "t.counts",  { 1, 3 }, { 900.0f, 30.0f, 1.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_NEAR(im.count("t", 0), 900.0, 1e-6);
    CHECK_NEAR(im.count("t", 1), 30.0, 1e-6);
    CHECK_NEAR(im.count("t", 2), 1.0, 1e-6);
    CHECK_EQ(im.count("t", 3), 0.0);
    CHECK_EQ(im.count("absent", 0), 0.0);
}

TEST(a_gguf_entry_with_counts_and_no_sums_is_dropped) {
    const std::string path = write_tmp("orphan.imatrix", gguf_bytes(
        {
            { "good.in_sum2", { 2 }, { 1.0f, 3.0f } },
            { "orphan.counts", { 1 }, { 5.0f } },
        },
        {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_EQ(im.entries().size(), (size_t)1);
    CHECK(im.find("good") != nullptr);
    CHECK(im.find("orphan") == nullptr);
}

/* general.alignment sets where the data section starts; reading from the wrong place would give
 * plausible wrong numbers, not an error. */
TEST(the_declared_alignment_places_the_data_section) {
    for (uint64_t a : { (uint64_t)32, (uint64_t)64, (uint64_t)1 }) {
        const std::string path = write_tmp("align.imatrix", gguf_bytes(
            { { "t.in_sum2", { 3 }, { 2.0f, 4.0f, 6.0f } } },
            { { "general.alignment", 4, (uint32_t)a, "" } }, 3, a));
        ImatrixFile im;
        CHECK_OK(im.open(path));
        const ImatrixEntry* e = im.find("t");
        CHECK(e != nullptr);
        if (e && e->sum2) {
            CHECK_EQ(f32_at(e->sum2, 0), 2.0f);
            CHECK_EQ(f32_at(e->sum2, 2), 6.0f);
        }
    }
}

/* An unrelated KV must be stepped over correctly, or every following tensor info is misread. */
TEST(an_unrelated_kv_is_stepped_over_rather_than_read) {
    const std::string path = write_tmp("kv.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2 }, { 1.0f, 3.0f } } },
        { { "general.architecture", 8, 0, "llama" },
          { "some.f64",             12, 42, "" },
          { "imatrix.chunk_count",  4, 9, "" } }));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK_EQ(im.chunk_count(), (int64_t)9);
    const ImatrixEntry* e = im.find("t");
    CHECK(e != nullptr);
    if (e && e->sum2) CHECK_EQ(f32_at(e->sum2, 1), 3.0f);
}

/* ------------------------------------------------------------------ refusals
 *
 * The reader walks lengths out of a mapping of an untrusted file, so malformed input must be
 * refused rather than read past the end. */

TEST(a_file_that_is_not_there_is_an_io_error) {
    ImatrixFile im;
    CHECK_EQ(im.open("/nonexistent/radiance/imatrix.dat"), RAD_E_IO);
}

TEST(a_file_too_short_to_be_an_imatrix_is_refused) {
    const std::string path = write_tmp("short.imatrix", std::vector<uint8_t>{ 1, 2, 3, 4 });
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

TEST(a_legacy_entry_that_runs_past_the_end_is_refused) {
    std::vector<uint8_t> bytes = legacy_bytes({ { "t", 1, { 1.0f, 2.0f, 3.0f } } });
    bytes.resize(bytes.size() - 6);          /* cuts into the last two floats */
    const std::string path = write_tmp("trunc.imatrix", bytes);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

TEST(a_legacy_entry_count_that_is_not_one_is_refused) {
    Bin w;
    w.i32(0);
    w.i32(0);
    const std::string path = write_tmp("zeroentries.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

/* A GGUF's rotary scaling becomes the config.json keys the architecture reads, including the
 * type, never a bare factor in RadModelMeta::rope_scale. A YaRN modifier with no config.json
 * spelling is refused. */
TEST(a_gguf_rotary_scaling_becomes_the_config_keys_with_its_type) {
    auto meta = [](const std::vector<GgufKv>& extra, const char* leaf, MetaOwn* out) -> int {
        std::vector<GgufKv> kvs = { { "general.architecture", 8, 0, "llama" },
                                    { "llama.block_count", 4, 2, "" } };
        kvs.insert(kvs.end(), extra.begin(), extra.end());
        GgufFile g;
        if (g.open(write_tmp(leaf, gguf_bytes({}, kvs)).c_str()) != RAD_OK) return RAD_E_IO;
        return meta_from_gguf(g, out);
    };
    {
        MetaOwn m;
        CHECK_OK(meta({ { "llama.rope.scaling.type", 8, 0, "yarn" },
                        { "llama.rope.scaling.factor", 12, 4, "" },
                        { "llama.rope.scaling.original_context_length", 4, 32768, "" } },
                      "yarn.gguf", &m));
        CHECK_EQ(m.m.rope_scale, 1.0f);
        REQUIRE(m.get("rope_scaling.type"));
        CHECK_EQ(std::string(m.get("rope_scaling.type")), std::string("yarn"));
        CHECK_EQ(std::string(m.get("rope_scaling.factor")), std::string("4"));
        CHECK_EQ(std::string(m.get("rope_scaling.original_max_position_embeddings")),
                 std::string("32768"));
    }
    {   /* llama.cpp's default type is linear; the older key names its factor. */
        MetaOwn m;
        CHECK_OK(meta({ { "llama.rope.scale_linear", 12, 2, "" } }, "linear.gguf", &m));
        CHECK_EQ(std::string(m.get("rope_scaling.type")), std::string("linear"));
        CHECK_EQ(std::string(m.get("rope_scaling.factor")), std::string("2"));
        CHECK_EQ(m.m.rope_scale, 1.0f);
    }
    {   /* An unimplemented type is carried by name for the architecture to refuse. */
        MetaOwn m;
        CHECK_OK(meta({ { "llama.rope.scaling.type", 8, 0, "longrope" },
                        { "llama.rope.scaling.factor", 12, 4, "" } }, "longrope.gguf", &m));
        CHECK_EQ(std::string(m.get("rope_scaling.type")), std::string("longrope"));
    }
    {
        MetaOwn m;
        CHECK_OK(meta({ { "llama.rope.scaling.type", 8, 0, "none" },
                        { "llama.rope.scaling.factor", 12, 4, "" } }, "none.gguf", &m));
        CHECK(!m.get("rope_scaling.type"));
        MetaOwn n;
        CHECK_OK(meta({}, "plain.gguf", &n));
        CHECK(!n.get("rope_scaling.type"));
    }
    {
        MetaOwn m;
        CHECK_EQ(meta({ { "llama.rope.scaling.type", 8, 0, "yarn" },
                        { "llama.rope.scaling.factor", 12, 4, "" },
                        { "llama.rope.scaling.attn_factor", 12, 2, "" } }, "attn.gguf", &m),
                 RAD_E_UNSUPPORTED);
    }
}

TEST(a_gguf_version_this_reader_does_not_know_is_refused) {
    const std::string path = write_tmp("v2.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2 }, { 1.0f, 2.0f } } }, {}, /*version=*/2));
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

TEST(a_gguf_tensor_that_runs_past_the_end_is_refused) {
    std::vector<uint8_t> bytes =
        gguf_bytes({ { "t.in_sum2", { 64 }, { 1.0f, 2.0f } } }, {});
    const std::string path = write_tmp("over.imatrix", bytes);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

/* A second entry under one tensor name could never be looked up, so the file is refused. */
TEST(a_gguf_that_lists_a_tensor_twice_is_refused) {
    const std::string path = write_tmp("twice.gguf", gguf_bytes(
        { { "t.weight", { 2 }, { 1.0f, 2.0f } }, { "t.weight", { 2 }, { 3.0f, 4.0f } } }, {}));
    GgufFile g;
    CHECK(g.open(path.c_str()) != RAD_OK);
}

/* 2^62 x 4 floats wraps to zero elements, which passes any bounds test while the entry claims
 * 2^62 columns. */
TEST(a_gguf_tensor_whose_dims_wrap_is_refused) {
    const std::string path = write_tmp("wrap.imatrix", gguf_bytes(
        { { "t.in_sum2", { (uint64_t)1 << 62, 4 }, { 1.0f, 2.0f } } }, {}));
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

/* A tensor count sizes an allocation, so one the file cannot hold is refused. */
TEST(a_gguf_tensor_count_the_file_cannot_hold_is_refused) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64((uint64_t)1 << 60);   /* tensors */
    w.u64(0);                   /* kvs */
    w.u64(0);
    const std::string path = write_tmp("manytensors.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

/* count() reads one value per expert of the .in_sum2 dims, so a shorter .counts is ignored: the
 * importance stands and the activation count reads as absent. */
TEST(a_gguf_counts_shorter_than_the_expert_count_is_ignored) {
    const std::string path = write_tmp("shortcounts.imatrix", gguf_bytes(
        { { "t.in_sum2", { 2, 3 }, { 1.f, 1.f, 1.f, 1.f, 2.f, 6.f } },
          { "t.counts",  { 1, 2 }, { 900.0f, 30.0f } } }, {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    const ImatrixEntry* e = im.find("t");
    CHECK(e != nullptr);
    if (e) {
        CHECK_EQ(e->n_expert, (int64_t)3);
        CHECK(e->counts == nullptr);
    }
    CHECK_EQ(im.count("t", 2), 0.0);
    std::vector<float> imp;
    CHECK(im.importance("t", 2, &imp));
    CHECK_EQ(imp.size(), (size_t)2);
    if (imp.size() == 2) CHECK_NEAR(imp[1] / imp[0], 3.0f, 1e-6);
}

/* An .in_sum2 whose trailing dims multiply to less than cols x n_expert (e.g. a zero third dim)
 * is dropped, since every query indexes by cols x n_expert. */
TEST(a_gguf_sums_smaller_than_their_claimed_extent_are_dropped) {
    const std::string path = write_tmp("emptysums.imatrix", gguf_bytes(
        { { "t.in_sum2", { 4, 2, 0 }, {} },
          { "u.in_sum2", { 2 },       { 1.0f, 3.0f } } }, {}));
    ImatrixFile im;
    CHECK_OK(im.open(path));
    CHECK(im.find("t") == nullptr);
    CHECK(im.find("u") != nullptr);
}

TEST(an_alignment_that_is_not_a_power_of_two_is_refused) {
    Bin w;
    w.raw("GGUF", 4);
    w.u32(3);
    w.u64(0);           /* no tensors */
    w.u64(1);
    w.gstr("general.alignment");
    w.u32(4);
    w.u32(24);          /* not a power of two */
    const std::string path = write_tmp("badalign.imatrix", w.b);
    ImatrixFile im;
    CHECK_EQ(im.open(path), RAD_E_FORMAT);
}

TEST(reopening_replaces_everything_the_previous_file_left) {
    const std::string a = write_tmp("first.imatrix", legacy_bytes({
        { "only_in_a", 1, { 1.0f, 2.0f } },
    }));
    const std::string b = write_tmp("second.imatrix", gguf_bytes(
        { { "only_in_b.in_sum2", { 2 }, { 3.0f, 4.0f } } },
        { { "imatrix.chunk_count", 4, 5, "" } }));
    ImatrixFile im;
    CHECK_OK(im.open(a));
    CHECK(im.find("only_in_a") != nullptr);
    CHECK_EQ(im.chunk_count(), (int64_t)0);

    CHECK_OK(im.open(b));
    CHECK(im.find("only_in_a") == nullptr);
    CHECK(im.find("only_in_b") != nullptr);
    CHECK(im.is_gguf());
    CHECK_EQ(im.chunk_count(), (int64_t)5);
    CHECK_EQ(im.entries().size(), (size_t)1);
}

RAD_TEST_MAIN()
