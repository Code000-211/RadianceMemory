# libhalo

libhalo is radiance's kernel library for Strix Halo (gfx1151, RDNA 3.5, the Radeon 8060S in the
Ryzen AI Max+ 395): specialised HIP kernels behind the same plain C ABI libr4d exposes, with
`halo_` in place of `r4d_` ([`halo.h`](halo.h)), and the kernel-plugin rows the engine selects from
(`halo_rows.cpp`).

## libhalo and libr3d

libr3d serves the whole RDNA3 family (`^gfx11`), gfx1151 included; libhalo serves gfx1151 only.
Both are built when `RAD_GPU_TARGETS` names gfx1151, and on a Strix Halo libhalo is the one used:

- with no `--kernels` list the loader takes libraries in file-name order, so `libhalo.so` comes
  ahead of `libr3d.so`, and the first library with a row for an op supplies it (docs/PLUGIN.md).
  An op libhalo has no row for falls through to libr3d;
- `rad_plugin_open` (`halo_rows.cpp`) declines every machine whose KFD topology lists no gfx1151,
  so on another RDNA3 card libhalo is not loaded at all and declares no schema: libr3d serves it
  alone. The loader would leave libhalo's device rows out there anyway (its code objects are
  gfx1151's); declining also keeps its schemas out, and keeps rad-convert, which resolves kernels
  without that check, from laying weights out for it. A process without `/dev/kfd` (a container
  given no card, a machine with no amdgpu) sees no gfx1151 and declines it too.

An explicit `--kernels` list orders the libraries as given; name `libhalo` before `libr3d`.

## What is shared with libr4d, and what is not

Shared, by design, so one container serves both cards:

- the op vocabulary and schemas (the loader refuses a library whose schema for an op disagrees
  with the first declaration, so these cannot drift silently);
- row names, so `halo_selftest` (derived from `r4d_selftest`, whose cases address rows by name)
  and `rad-kbench` check libhalo's kernels against the same cases and recordings as libr4d's;
- stored weight layouts whose bytes do not encode a WMMA fragment order, tags included (`r4d.*`
  layout tags name arrangements, not libraries).

Not shared: anything measured. Priorities, constraint bands, tunable defaults, launch geometries
and the kernel bodies themselves are libhalo's own, derived from libr4d's sources (each file says
so) and re-measured on gfx1151. Fragment-ordered weights are stored in gfx1151's fragment order
under `halo.*` tags: a fragment order is a statement about one architecture's WMMA lanes
(see `halo_wmma.h`), and storing gfx1201 order would make every halo GEMM pay an in-register
shuffle to read it. The canonical planes a container holds, the dequantised values and the
reference outputs are identical either way (spec §4.3).

Two properties of the part shape everything here and are not restated per kernel:

- unified memory: device memory is host memory (a 120 GiB GTT pool inside the machine's RAM).
  There is no link, so "stream the weight" and "stage the weight" differ only in which copy of
  the bytes a kernel reads, and the core's unified-memory path keeps one ledger for both pools;
- ~256 GB/s of LPDDR5X shared with the CPU, 40 CUs, wave32. Decode is a pure weight stream at
  a quarter of the R9700's bandwidth; launch geometry that fills 64 CUs idles here.

## Porting state

Kernel units land family by family in `HALO_UNITS` (`CMakeLists.txt`). Until a unit lands, its
rows resolve to generated stubs in `halo_stubs.cpp` that return `RAD_E_UNSUPPORTED`, so the
library loads, selection works, and the gap is reported by name at first use. The stub count is
the parity scoreboard; done means zero. Chained launch forms (`halo_chain.h`) follow the plain
forms per family; `rad_kernel_chain` names a row only once all of its kernels have one.

## Adding a kernel

As in libr4d: write the `.hip` named for its entry point and geometry, declare it in `halo.h`,
add or adjust the `RadKernelInfo` row in `halo_rows.cpp`, and add a case to `halo_selftest.cpp`.
`HALO_UNITS` and the stubs are generated: run `python3 libhalo/gen_stubs.py` after adding the
file instead of editing either by hand (`libhalo/PORTING.md`). Stubs are weak, so there is no
stub line to delete when a unit lands. Check numerics with `halo_selftest <path to libhalo.so>
--case <row>` and `rad-kbench --kernels libhalo,libref`; check speed on the card, in the same
run, against the shape's roofline. `libhalo/PORTING.md` is the porting guide: the rename
transform, the gfx1151 fragment rules, the wrapper-file pattern and the gotchas already paid for.
