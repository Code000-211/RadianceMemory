/* dispatch.cpp -- picks a backend once, then forwards.
 *
 * The only definition of every rad_dev_* / rad_stream_* / rad_event_* symbol. host.cpp and hip.cpp
 * export vtables only, so there is no #ifdef above this file and both backends share one contract.
 *
 * A build with HIP still runs the host backend when no device is visible (e.g. a CI container
 * with ROCm headers and no /dev/kfd); the chosen backend is logged at Info.
 */
#include "device.h"
#include "../rad_internal.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>
#include <string>
#include <utility>
#include <cstdint>

namespace rad {

/* ------------------------------------------------------------------ the error slot */
/* Thread-local and cleared by reading, per the ABI. Each rank and the mover run on their own
 * thread (spec §1), so an error must stay on the thread that raised it. */
static thread_local std::string t_last_error;

void device_set_error(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    t_last_error.assign(buf);
    /* Debug, not Error: callers often recover (e.g. a pool retrying smaller), and fatal ones are
     * reported by the caller with more context. */
    RAD_DEBUG("device: %s", buf);
}

/* ------------------------------------------------------------------ selection */

/* The startup line's list of every visible device by index and arch. A machine may report more
 * devices than accelerators: an APU shows up beside discrete cards and claims system RAM as
 * "VRAM". The arch tells them apart. Filtering is the engine's decision (via rad_dev_props). */
static std::string device_inventory(const DeviceBackend* b) {
    std::string s;
    const int n = b->count();
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (b->props(i, &p) < 0) continue;
        s += fmt("%s%d:%s", i ? " " : "", i, p.arch);
    }
    return s;
}

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
/* With amdgpu.cwsr_enable=0 (compute wave save/restore off) a queue eviction hangs the GPU. The
 * driver evicts a process's queues when card-mapped host memory is moved or reclaimed, and a serve
 * maps gigabytes of it. Without CWSR the driver waits for the queue to drain, a busy queue never
 * does, MES fails REMOVE_QUEUE and the card resets. Warned here as the backend comes up, not only
 * in the startup report: a hang during load kills the process first, and rad-kbench and rad-tune
 * print no report. */
static void warn_cwsr_off() {
    FILE* f = std::fopen("/sys/module/amdgpu/parameters/cwsr_enable", "r");
    if (!f) return;
    int v = -1;
    const int got = std::fscanf(f, "%d", &v);
    std::fclose(f);
    if (got == 1 && v == 0)
        RAD_WARN("device: amdgpu cwsr_enable is 0, so the driver cannot preempt a running queue: "
                 "any queue eviction (host memory the card maps, moved under memory pressure) waits "
                 "for a queue a serve never lets drain, and ends in a GPU hang and reset. Set "
                 "amdgpu.cwsr_enable=1 (module option, then rebuild the initramfs and reboot)");
}

/* The host backend in a HIP build still counts plugin GPU code. This build's fat-binary
 * registration hooks see every dlopen'ed kernel library, card or no card, so the plugin loader can
 * tell which library brought device code. A serving engine drops those device rows from selection
 * (Registry::set_no_device_code); a selection-only tool keeps them. */
static DeviceBackend g_host_beside_hip;
#endif

static const DeviceBackend* select_backend() {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    if (const DeviceBackend* b = aql_backend_if_present()) {
        RAD_INFO("device: AQL backend, %d device(s) [%s]", b->count(),
                 device_inventory(b).c_str());
        warn_cwsr_off();
        return b;
    }
    RAD_INFO("device: built with HIP but no device is visible -- using the host backend");
    g_host_beside_hip = *host_backend();
    g_host_beside_hip.code_mark = aql_code_registered;
    const DeviceBackend* b = &g_host_beside_hip;
#else
    const DeviceBackend* b = host_backend();
#endif
    RAD_INFO("device: host backend, %d device(s) [%s]", b->count(), device_inventory(b).c_str());
    return b;
}

/* Selected exactly once, under std::call_once; read from every rank thread. */
static const DeviceBackend* g_backend = nullptr;
static std::once_flag       g_once;

static inline const DeviceBackend* dev() {
    std::call_once(g_once, [] { g_backend = select_backend(); });
    return g_backend;
}

bool        device_is_host()      { return std::strcmp(dev()->name, "host") == 0; }
const char* device_backend_name() { return dev()->name; }

}  /* namespace rad */

/* ------------------------------------------------------------------ the ledger
 *
 * Counts every device allocation the engine makes. The VRAM budget (core/mem/vram_budget.cpp)
 * spends the card down to --gpu-headroom-mib, so whatever lands outside the two pools and is not
 * derived from the buffer plan or run configuration must be counted here.
 *
 * Not seen: plugins that call hipMalloc directly (libr4d's split-tail workspaces and all-reduce
 * peer scratch). Engine::start reads the card back after allocation for that reason.
 *
 * RADIANCE_TRACE_ALLOC=N logs each allocation of at least N bytes.
 *
 * RADIANCE_DEBUG_POISON_ALLOC=B fills every fresh DEVICE allocation with byte B (0..255), so a
 * read-before-write shows up as garbage instead of lucky zeros (as after a driver reset, "VRAM is
 * lost"). A correct engine serves the same output with it set. RADIANCE_DEBUG_POISON_RANGE=lo:hi
 * limits the fill to device allocations with ordinal in [lo, hi), counted from 1 per allocating
 * thread (the trace's `#n on this thread`), for bisecting. The fill is synchronous on one stream
 * per thread and device, kept until exit: a stream is a hardware queue, and per-allocation queue
 * churn is slow and faults in ROCr's Trim path. Bad values abort at the first allocation. */
namespace rad {
namespace {

struct Ledger {
    std::atomic<int64_t> bytes{0};
    std::atomic<int64_t> count{0};
};
Ledger g_ledger[4];        /* indexed by RAD_MEM_*; rad_dev_alloc bounds the kind */
/* Each live allocation's size, so rad_dev_free (which has none) takes it back off the ledger's
 * bytes: pinned ranges are freed and taken again during bring-up, and the pin-cap check reads the
 * ledger as what the process holds. Allocations are rare, never per step. */
struct LiveAlloc { int64_t bytes; int kind; };
std::mutex                           g_ledger_mu;
std::unordered_map<void*, LiveAlloc> g_ledger_live;

void ledger_add(void* p, int64_t bytes, int kind) {
    g_ledger[kind].bytes.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_ledger_mu);
    g_ledger_live[p] = LiveAlloc{ bytes, kind };
}
void ledger_drop(void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_ledger_mu);
    const auto it = g_ledger_live.find(p);
    if (it == g_ledger_live.end()) return;
    g_ledger[it->second.kind].bytes.fetch_sub(it->second.bytes, std::memory_order_relaxed);
    g_ledger_live.erase(it);
}

/* Live event handles, counted here so the number is backend-independent; the only place an
 * event leaked at teardown shows. */
std::atomic<int64_t> g_live_events{0};
/* Host blocks through the ABI (rad_event_sync, rad_stream_sync), counted here so a test can
 * assert that a path never stops the host, whichever backend runs it. */
std::atomic<int64_t> g_host_blocks{0};

int64_t trace_floor() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_TRACE_ALLOC");
        if (!e || !*e) return (int64_t)-1;
        const long long x = std::atoll(e);
        return x >= 0 ? (int64_t)x : (int64_t)0;
    }();
    return v;
}

/* -1 when off; otherwise the byte. */
int poison_byte() {
    static const int v = [] {
        const char* e = std::getenv("RADIANCE_DEBUG_POISON_ALLOC");
        if (!e || !*e) return -1;
        char* end = nullptr;
        const long x = std::strtol(e, &end, 0);
        if (*end || x < 0 || x > 255) {
            RAD_ERR("RADIANCE_DEBUG_POISON_ALLOC=%s is not a byte (0..255)", e);
            std::abort();
        }
        return (int)x;
    }();
    return v;
}

/* [lo, hi) over device-allocation ordinals; the whole run when unset. */
bool poison_in_range(int64_t ordinal) {
    static const std::pair<int64_t, int64_t> r = [] {
        const char* e = std::getenv("RADIANCE_DEBUG_POISON_RANGE");
        long long lo = 0, hi = INT64_MAX;
        if (e && *e && (std::sscanf(e, "%lld:%lld", &lo, &hi) != 2 || lo < 0 || hi <= lo)) {
            RAD_ERR("RADIANCE_DEBUG_POISON_RANGE=%s is not lo:hi with 0 <= lo < hi", e);
            std::abort();
        }
        return std::make_pair((int64_t)lo, (int64_t)hi);
    }();
    return ordinal >= r.first && ordinal < r.second;
}

const char* kind_name(int kind) {
    switch (kind) {
        case RAD_MEM_DEVICE:      return "device";
        case RAD_MEM_HOST:        return "host";
        case RAD_MEM_HOST_PINNED: return "host-pinned";
        case RAD_MEM_HOST_MAPPED: return "host-mapped";
        default:                  return "?";
    }
}

}  /* namespace */

void rad_dev_alloc_totals(int kind, int64_t* bytes, int64_t* count) {
    if (kind < 0 || kind >= (int)std::size(g_ledger)) {
        if (bytes) *bytes = 0;
        if (count) *count = 0;
        return;
    }
    if (bytes) *bytes = g_ledger[kind].bytes.load(std::memory_order_relaxed);
    if (count) *count = g_ledger[kind].count.load(std::memory_order_relaxed);
}

int64_t rad_dev_live_events(void) { return g_live_events.load(std::memory_order_relaxed); }
int64_t rad_dev_host_blocks(void) { return g_host_blocks.load(std::memory_order_relaxed); }


/* ------------------------------------------------------------------ elastic memory */
/* See device.h. Interval arithmetic and rounding here; the backends do the mapping. */

namespace {
std::atomic<int64_t> g_vmem_committed{0};
}  /* namespace */

int64_t rad_dev_vmem_committed(void) { return g_vmem_committed.load(std::memory_order_relaxed); }

/* Both checks: a backend may have the entries on a driver with no virtual memory manager
 * (granularity 0). Checking only the pointers would reserve a range that can never be backed. */
bool    VMem::supported()   { return dev()->vmem_reserve != nullptr && VMem::granularity() > 0; }
int64_t VMem::granularity() {
    if (!dev()->vmem_granularity) return 0;
    const int64_t g = dev()->vmem_granularity();
    return g > 0 ? g : 0;
}

int VMem::reserve(int64_t bytes, int64_t align) {
    release();
    if (bytes <= 0) return RAD_E_INVAL;
    if (!supported()) {
        device_set_error("elastic memory: the %s backend has none", device_backend_name());
        return RAD_E_UNSUPPORTED;
    }
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    if (align <= 0) align = g;
    /* Round up to a granule: a partial last granule could never be committed. */
    const int64_t want = align_up(bytes, g);
    void* p = dev()->vmem_reserve(want, align);
    if (!p) return RAD_E_NOMEM;
    base_ = p;
    size_ = want;
    committed_ = 0;
    ext_.clear();
    return RAD_OK;
}

void VMem::release() {
    if (!base_) { size_ = 0; committed_ = 0; ext_.clear(); return; }
    /* Decommit before release: releasing a range with pages still mapped leaks them, and some
     * drivers refuse the free. One granule at a time, as each was mapped. */
    const int64_t g = granularity();
    for (const auto& e : ext_) {
        if (dev()->vmem_decommit && g > 0)
            for (int64_t o = e.first; o < e.first + e.second; o += g)
                (void)dev()->vmem_decommit((char*)base_ + o, g);
        g_vmem_committed.fetch_sub(e.second, std::memory_order_relaxed);
    }
    if (dev()->vmem_release) dev()->vmem_release(base_, size_);
    base_ = nullptr;
    size_ = 0;
    committed_ = 0;
    ext_.clear();
}

bool VMem::is_committed(int64_t off, int64_t bytes) const {
    if (bytes <= 0) return true;
    if (off < 0 || off + bytes > size_) return false;
    for (const auto& e : ext_)
        if (off >= e.first && off + bytes <= e.first + e.second) return true;
    return false;                       /* extents are merged, so one of them covers it or none does */
}
/* The granules to map are gathered before mapping any: add_extent merges, so walking ext_ while
 * mapping would skip entries and map the tail twice (the driver refuses a double map).
 *
 * Driver rules:
 *   - A mapping can only be unbacked exactly as it was backed, so each granule is mapped
 *     separately; that lets the pool hand back the top of its range.
 *   - Access cannot be granted per granule: a granule adjacent to an accessible one is refused on
 *     its own. Access is granted once per merged extent the new granules belong to; re-granting
 *     access to a range is free. */
int VMem::commit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* Round outward: every requested byte must be committed. */
    const int64_t lo = (off < 0 ? 0 : off / g * g);
    int64_t hi = align_up(off + bytes, g);
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    std::vector<int64_t> want;
    want.reserve((size_t)((hi - lo) / g));
    for (int64_t o = lo; o < hi; o += g)
        if (!is_committed(o, g)) want.push_back(o);
    if (want.empty()) return RAD_OK;

    /* On a failure part way, what already mapped stays mapped and recorded: a partial grow is
     * usable, untracked mappings are not. */
    int rc = RAD_OK;
    size_t took = 0;
    for (; took < want.size(); ++took) {
        rc = dev()->vmem_commit((char*)base_ + want[took], g);
        if (rc < 0) break;
        add_extent(want[took], g);
    }

    /* Grant access even if the grow stopped short, so the granules that did map are usable. */
    int prc = RAD_OK;
    if (took > 0 && dev()->vmem_protect) {
        for (const auto& e : ext_) {
            if (e.first + e.second <= lo || e.first >= hi) continue;
            prc = dev()->vmem_protect((char*)base_ + e.first, e.second);
            if (prc < 0) break;
        }
    }
    if (prc < 0) {
        /* Mapped but inaccessible memory faults the first kernel that touches it. Undo exactly
         * what this call mapped; earlier extents were already accessible. */
        for (size_t i = 0; i < took; ++i) {
            if (dev()->vmem_decommit) (void)dev()->vmem_decommit((char*)base_ + want[i], g);
            drop_extent(want[i], g);
        }
        return prc;
    }
    return rc;
}

int VMem::decommit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* Round inward: a granule holding any byte outside the range stays. */
    const int64_t lo = align_up(off < 0 ? 0 : off, g);
    int64_t hi = (off + bytes) / g * g;
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    /* Gathered first and unmapped per granule, as in commit. */
    std::vector<int64_t> drop;
    for (int64_t o = lo; o < hi; o += g)
        if (is_committed(o, g)) drop.push_back(o);

    for (const int64_t o : drop) {
        const int rc = dev()->vmem_decommit((char*)base_ + o, g);
        if (rc < 0) return rc;
        drop_extent(o, g);
    }
    return RAD_OK;
}

void VMem::add_extent(int64_t off, int64_t bytes) {
    committed_ += bytes;
    g_vmem_committed.fetch_add(bytes, std::memory_order_relaxed);
    /* Insert in order, then merge touching neighbours, so is_committed() stays a short scan
     * although pools grow one granule at a time. */
    size_t i = 0;
    while (i < ext_.size() && ext_[i].first < off) ++i;
    ext_.insert(ext_.begin() + (long)i, { off, bytes });
    for (size_t j = 0; j + 1 < ext_.size();) {
        if (ext_[j].first + ext_[j].second >= ext_[j + 1].first) {
            const int64_t end = std::max(ext_[j].first + ext_[j].second,
                                         ext_[j + 1].first + ext_[j + 1].second);
            ext_[j].second = end - ext_[j].first;
            ext_.erase(ext_.begin() + (long)j + 1);
        } else {
            ++j;
        }
    }
}

/* Precondition: the piece lies wholly inside one extent (decommit only drops committed granules).
 * What remains of that extent is nothing, a head, a tail, or both. */
void VMem::drop_extent(int64_t off, int64_t bytes) {
    committed_ -= bytes;
    g_vmem_committed.fetch_sub(bytes, std::memory_order_relaxed);
    for (size_t i = 0; i < ext_.size(); ++i) {
        const int64_t es = ext_[i].first, ee = es + ext_[i].second;
        if (off < es || off + bytes > ee) continue;
        ext_.erase(ext_.begin() + (long)i);
        if (off + bytes < ee) ext_.insert(ext_.begin() + (long)i, { off + bytes, ee - off - bytes });
        if (es < off)         ext_.insert(ext_.begin() + (long)i, { es, off - es });
        return;
    }
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the ABI surface */

using rad::dev;

extern "C" {

int  rad_dev_count(void)                          { return dev()->count(); }
/* The device this thread last selected, for the poison fill's stream (rad_dev_alloc). */
static thread_local int t_device = 0;
static thread_local std::map<int, RadStream> t_poison_stream;
int  rad_dev_set(int device) {
    const int st = dev()->set(device);
    if (st == RAD_OK) t_device = device;
    return st;
}
int  rad_dev_props(int device, RadDeviceProps* o)  { return dev()->props(device, o); }
int  rad_dev_enable_peer(int self, int peer)      { return dev()->enable_peer(self, peer); }

void* rad_dev_alloc(int64_t bytes, int kind) {
    /* Poison ordinals are per thread: each rank allocates in a fixed order on its own thread, but
     * threads interleave, so a process-wide ordinal would not be repeatable. */
    thread_local int64_t t_device_ordinal = 0;
    void* p = dev()->alloc(bytes, kind);
    if (p && kind >= 0 && kind < (int)std::size(rad::g_ledger)) {
        rad::ledger_add(p, bytes, kind);
        const int64_t total = rad::g_ledger[kind].count.fetch_add(1, std::memory_order_relaxed) + 1;
        const int64_t ordinal = kind == RAD_MEM_DEVICE ? ++t_device_ordinal : total;
        const int64_t floor_ = rad::trace_floor();
        if (floor_ >= 0 && bytes >= floor_)
            RAD_INFO("alloc %s %s (#%lld on this thread; %lld live in %lld allocations)",
                     rad::kind_name(kind), rad::humanb(bytes).c_str(), (long long)ordinal,
                     (long long)rad::g_ledger[kind].bytes.load(std::memory_order_relaxed),
                     (long long)total);
        /* Poison fill (see the ledger header): synchronous, so it lands before p is returned. */
        const int poison = rad::poison_byte();
        if (poison >= 0 && kind == RAD_MEM_DEVICE && rad::poison_in_range(ordinal)) {
            RadStream& s = t_poison_stream[t_device];
            if (!s && dev()->stream_create(&s, 0) != RAD_OK) s = nullptr;
            if (!s || dev()->memset_async(p, poison, bytes, s) != RAD_OK ||
                dev()->stream_sync(s) != RAD_OK)
                RAD_WARN("poison: device allocation #%lld (%s) was not filled",
                         (long long)ordinal, rad::humanb(bytes).c_str());
            else if (floor_ >= 0 && bytes >= floor_)
                RAD_INFO("poison: device allocation #%lld filled with 0x%02x", (long long)ordinal,
                         poison);
        }
    }
    return p;
}
void  rad_dev_free(void* p, int kind) {
    rad::ledger_drop(p);
    dev()->free(p, kind);
}
void* rad_dev_host_ptr(void* p)                   { return dev()->host_ptr(p); }
void* rad_dev_device_ptr(void* p)                 { return dev()->device_ptr(p); }

int  rad_stream_create(RadStream* out, int hi)    { return dev()->stream_create(out, hi); }
void rad_stream_destroy(RadStream s)              { dev()->stream_destroy(s); }
int  rad_stream_sync(RadStream s) {
    rad::g_host_blocks.fetch_add(1, std::memory_order_relaxed);
    return dev()->stream_sync(s);
}

}  /* extern "C" */

namespace rad {

bool    tape_supported()       { return dev()->tape_begin != nullptr; }
int     dev_vram_used(int device, int64_t* out) {
    return dev()->vram_used ? dev()->vram_used(device, out) : RAD_E_UNSUPPORTED;
}
int     tape_begin()           { return dev()->tape_begin ? dev()->tape_begin() : RAD_E_UNSUPPORTED; }
int     tape_mark()            { return dev()->tape_mark ? dev()->tape_mark() : RAD_E_UNSUPPORTED; }
void    tape_pause(bool on)    { if (dev()->tape_pause) dev()->tape_pause(on ? 1 : 0); }
RadTape tape_end(bool keep)    { return dev()->tape_end ? dev()->tape_end(keep ? 1 : 0) : nullptr; }
int     tape_len(RadTape t)    { return dev()->tape_len ? dev()->tape_len(t) : RAD_E_UNSUPPORTED; }
int     tape_play(RadTape t, int from, int to) {
    return dev()->tape_play ? dev()->tape_play(t, from, to) : RAD_E_UNSUPPORTED;
}
int     tape_diff(RadTape a, RadTape b, std::string* why) {
    if (!dev()->tape_diff) return RAD_E_UNSUPPORTED;
    char buf[512];
    buf[0] = 0;
    const int at = dev()->tape_diff(a, b, buf, (int)sizeof buf);
    if (why) *why = buf;
    return at;
}
void    tape_destroy(RadTape t) { if (dev()->tape_destroy) dev()->tape_destroy(t); }
int     tape_record(RadTape t, int i, int* kind, uint32_t* queue) {
    return dev()->tape_record ? dev()->tape_record(t, i, kind, queue) : RAD_E_UNSUPPORTED;
}
int     tape_set_overlap(RadTape t, int i, bool on) {
    return dev()->tape_set_overlap ? dev()->tape_set_overlap(t, i, on ? 1 : 0) : RAD_E_UNSUPPORTED;
}
bool    dev_integrated(int device) { return dev()->integrated && dev()->integrated(device) > 0; }
/* These getters answer "" on failure and restore the error slot, so the next
 * rad_dev_last_error reader does not see an unrelated error. `get` fills a buffer of `n` bytes. */
template <class Get> static std::string dev_string(int n, Get get) {
    std::string buf((size_t)n, '\0');
    const std::string before = t_last_error;
    if (get(buf.data(), n) < 0) {
        t_last_error = before;
        return std::string();
    }
    buf.resize(std::strlen(buf.c_str()));
    return buf;
}
std::string dev_pci_address(int device) {
    if (!dev()->pci_address) return std::string();
    return dev_string(64, [&](char* b, int n) { return dev()->pci_address(device, b, n); });
}
int64_t dev_stuck_ms() {
    const auto f = dev()->stuck_ms;
    return f ? f() : 0;
}
std::string dev_dma_lane(int device) {
    if (!dev()->dma_lane) return std::string();
    return dev_string(256, [&](char* b, int n) { return dev()->dma_lane(device, b, n); });
}
std::string dev_runtime_version() {
    if (!dev()->runtime_version) return std::string();
    return dev_string(128, [&](char* b, int n) { return dev()->runtime_version(b, n); });
}
int     dev_peer_access(int self, int peer) {
    if (self == peer) return 1;
    return dev()->peer_access ? dev()->peer_access(self, peer) : 1;
}
size_t  dev_code_mark() { return dev()->code_mark ? dev()->code_mark() : 0; }
int     dev_code_runs(size_t from, size_t to, int device, std::string* why) {
    if (!dev()->code_runs || from >= to) return 1;
    char buf[1024];
    buf[0] = 0;
    const int r = dev()->code_runs(from, to, device, buf, (int)sizeof buf);
    if (why) *why = buf;
    return r;
}

}  /* namespace rad */

extern "C" {

int  rad_dev_peer_access(int self, int peer)      { return rad::dev_peer_access(self, peer); }

int  rad_event_create_as(RadEvent* out, unsigned flags) {
    const int s = dev()->event_create(out, flags);
    if (s >= 0 && out && *out) rad::g_live_events.fetch_add(1, std::memory_order_relaxed);
    return s;
}
int  rad_event_create(RadEvent* out)              { return rad_event_create_as(out, RAD_EVENT_HOST_WAIT); }
int  rad_event_create_local(RadEvent* out)        { return rad_event_create_as(out, RAD_EVENT_LOCAL); }
void rad_event_destroy(RadEvent e) {
    if (!e) return;
    rad::g_live_events.fetch_sub(1, std::memory_order_relaxed);
    dev()->event_destroy(e);
}
int  rad_event_record(RadEvent e, RadStream s)    { return dev()->event_record(e, s); }
int  rad_event_dma(RadEvent e, RadStream s, uint32_t** w) {
    if (const auto f = dev()->event_dma) return f(e, s, w);
    rad::device_set_error("rad_event_dma: the device backend has no DMA queue to complete an event");
    return RAD_E_UNSUPPORTED;
}
int  rad_event_wait(RadStream s, RadEvent e)      { return dev()->event_wait(s, e); }
int  rad_event_query(RadEvent e)                  { return dev()->event_query(e); }
int  rad_event_sync(RadEvent e) {
    rad::g_host_blocks.fetch_add(1, std::memory_order_relaxed);
    return dev()->event_sync(e);
}
void rad_dev_host_wrote(void)                     { dev()->host_wrote(); }

/* Whether launches chain, from the engine (rad_dev_chain_infer), and the answer once taken. Only
 * where the backend chains: the AQL backend writes the record a chained form reads, while a HIP
 * launch carries only the kernel's own arguments. */
static std::atomic<int> g_chain_pays{0};
static std::atomic<int> g_chain{-1};
void rad_dev_chain_infer(int pays) {
    g_chain_pays.store(pays ? 1 : 0, std::memory_order_relaxed);
    g_chain.store(-1, std::memory_order_relaxed);
}
int rad_dev_chain_enabled(void) {
    int on = g_chain.load(std::memory_order_relaxed);
    if (on < 0) {
        on = g_chain_pays.load(std::memory_order_relaxed) != 0 &&
                     std::strcmp(rad::device_backend_name(), "aql") == 0
                 ? 1
                 : 0;
        g_chain.store(on, std::memory_order_relaxed);
    }
    return on;
}

int  rad_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    return dev()->event_elapsed_ms(a, b, out);
}

int rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    /* One address space (unified memory, or a host view of a device allocation): a copy
     * onto itself is already done. */
    if (dst == src) return RAD_OK;
    return dev()->memcpy_async(dst, src, bytes, s);
}
int rad_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                        int64_t width, int64_t height, RadStream s) {
    if (dst == src && dpitch == spitch) return RAD_OK;
    return dev()->memcpy_2d_async(dst, dpitch, src, spitch, width, height, s);
}
int rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    return dev()->memset_async(dst, value, bytes, s);
}
int rad_memcpy_ranges_async(void* dst, const void* src, int64_t prefix,
                            const RadCopyRange* ranges, int n, RadStream s) {
    return dev()->memcpy_ranges_async(dst, src, prefix, ranges, n, s);
}
int rad_memcpy_list_async(const RadCopySeg* segs, int n, unsigned flags, RadStream s) {
    if (const auto f = dev()->memcpy_list_async) return f(segs, n, flags, s);
    for (int i = 0; i < n; ++i) RAD_TRY(dev()->memcpy_async(segs[i].dst, segs[i].src, segs[i].bytes, s));
    return RAD_OK;
}
int rad_link_hold(RadStream s, int on) {
    if (const auto f = dev()->link_hold) return f(s, on);
    return RAD_OK;
}
void* rad_link_gate(RadStream s) {
    if (const auto f = dev()->link_gate) return f(s);
    return nullptr;
}
int rad_xp_supported(int device) {
    const auto* d = dev();
    return d->xp_supported && d->xp_stream_create && d->ride && d->ride_wait && d->ride_release && d->xp_list &&
           d->xp_behind && d->alloc_uncached && d->free_uncached && d->xp_supported(device);
}
int rad_xp_stream_create(RadStream* out) {
    if (const auto f = dev()->xp_stream_create) return f(out);
    return RAD_E_UNSUPPORTED;
}
int rad_ride(RadStream s, void** token) {
    if (const auto f = dev()->ride) return f(s, token);
    return RAD_E_UNSUPPORTED;
}
int rad_ride_wait(RadStream s, void* token) {
    if (const auto f = dev()->ride_wait) return f(s, token);
    return RAD_E_UNSUPPORTED;
}
void rad_ride_release(void* token, RadStream after, int64_t packets) {
    if (const auto f = dev()->ride_release) f(token, after, packets);
}
int rad_xp_list(int device, void* mark, int n, const uint64_t* src, const uint64_t* dst, const uint64_t* bytes,
                uint32_t* done, uint32_t done_val, unsigned flags) {
    if (const auto f = dev()->xp_list) return f(device, mark, n, src, dst, bytes, done, done_val, flags);
    return RAD_E_UNSUPPORTED;
}
int64_t rad_xp_behind(int device) {
    if (const auto f = dev()->xp_behind) return f(device);
    return 0;
}
void* rad_dev_alloc_uncached(int64_t bytes) {
    if (const auto f = dev()->alloc_uncached) return f(bytes);
    return nullptr;
}
void* rad_dev_map_file(int fd, int64_t foff, int64_t bytes) {
    const auto f = dev()->map_file;
    if (!f) {
        rad::device_set_error("rad_dev_map_file: the %s backend has no pinned host memory to map a "
                              "file into", dev()->name);
        return nullptr;
    }
    void* p = f(fd, foff, bytes);
    if (p) {
        rad::ledger_add(p, bytes, RAD_MEM_HOST_MAPPED);
        rad::g_ledger[RAD_MEM_HOST_MAPPED].count.fetch_add(1, std::memory_order_relaxed);
    }
    return p;
}
void rad_dev_free_uncached(void* p) {
    if (const auto f = dev()->free_uncached) f(p);
}
int rad_stream_gate(RadStream s, void** gate) {
    if (gate) *gate = nullptr;
    if (const auto f = dev()->stream_gate) return f(s, gate);
    rad::device_set_error("rad_stream_gate: the %s backend has no gates", dev()->name);
    return RAD_E_UNSUPPORTED;
}
int rad_gate_open(void* gate) {
    if (const auto f = dev()->gate_open) return f(gate);
    return RAD_E_UNSUPPORTED;
}
int64_t rad_gate_room(RadStream s) {
    if (const auto f = dev()->gate_room) return f(s);
    return 0;
}

/* Cleared by reading. Never null; "" when nothing failed. */
const char* rad_dev_last_error(void) {
    static thread_local std::string drained;
    drained.swap(rad::t_last_error);
    rad::t_last_error.clear();
    return drained.c_str();
}

}  /* extern "C" */
