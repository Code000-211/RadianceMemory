/* cfg_stub.cpp -- the tuned-configuration readers the M2XFP4 GEMM units and the paged attention
 * call (r4d_cfg_mxfp4_decode, r4d_cfg_mxfp4_tiled, r4d_cfg_split), which libr4d defines on its
 * declaration side (r4d_rows.cpp) beside the row tables. That side is not in the emu build, and
 * every r4d_emu_* binary links every unit, so the readers are defined once here. Every axis default
 * of tGemmMxfp4Decode, tGemmMxfp4Tiled and tAttnSplit is 0, the kernels' own rule, so reading the
 * key or 0 is what r4d_rows.cpp's readers return for a caller with no tuned point.
 *
 * And r4d_p2p_copy2d (r4d_ar_*.hip, the copy engines), which r4d_model_bf16.hip's cast calls for a
 * device-to-device copy and no emulated test reaches: it reports a failed launch. */
#include "r4d.h"
#include "r4d_args.h"

extern "C" void r4d_cfg_mxfp4_decode(const RadArgs* a, R4dMxDecCfg* out) {
    out->ks = (int)rad_args_geti_or(a, "ks", 0);
    out->bk = (int)rad_args_geti_or(a, "bk", 0);
    out->nt = (int)rad_args_geti_or(a, "nt", 0);
}

extern "C" void r4d_cfg_mxfp4_tiled(const RadArgs* a, R4dMxTileCfg* out) {
    out->tn = (int)rad_args_geti_or(a, "tn", 0);
}

extern "C" void r4d_cfg_split(const RadArgs* a, R4dSplitCfg* out) {
    out->splits = (int)rad_args_geti_or(a, "splits", 0);
}

extern "C" int r4d_p2p_copy2d(long, long, long, long, long, long, long) { return -100 - 1; }
