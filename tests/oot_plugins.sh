#!/bin/sh
# oot_plugins.sh BUILD SOURCE WORK [HIP_COMPILER HIP_TARGETS [VOPD_REPEATABLE [CXX_COMPILER]]]
#
# THE PLUGINS RADIANCE SHIPS, BUILT THE WAY A THIRD PARTY BUILDS THEIRS. The tree's build is
# installed into a prefix; then every plugin directory -- the kernel libraries, the quantisers,
# every architecture, the plugins under plugin/ -- is configured ON ITS OWN against that prefix with find_package(radiance),
# built, and installed into a home of its own. Nothing in the tree's build is reused: what is
# tested is the installed package a stranger would have.
#
# What has to hold afterwards:
#   1. the out-of-tree home holds the same plugins, file for file, as the tree's own build put in
#      its home -- no plugin needs the tree to be built;
#   2. every kernel library declares exactly the vocabulary its in-tree twin declares;
#   3. the search path registers the same plugins, in the same order, with the same declarations
#      (rad-info --plugins), from the out-of-tree home alone;
#   4. the installed rad-kbench replays part of the recorded fixture through the host libraries
#      from that home, and every check passes. The kbench test replays all of it in tree; this
#      replay proves the installed tools serve from the out-of-tree home, so three cheap ops that
#      every host library implements (a norm, the fp8 activation quantiser, a gated activation)
#      are enough;
#   5. with HIP_COMPILER given, the device library the tree built (libr4d) is built out of tree
#      too, for HIP_TARGETS and at the tree's RAD_HIP_VOPD_REPEATABLE (VOPD_REPEATABLE, default
#      OFF; on LLVM 24 it decides whether VOPD pairs, so it changes the code), and compiles to the
#      tree's device code. A replay of a device library needs a card, and this test must not touch
#      one a server is using; the same instructions mean the tree's own GPU suite (ctest -L gpu)
#      has already checked what the out-of-tree build produced.
set -eu
B=$1; S=$2; W=$3
HIPC=${4:-}; HIPT=${5:-}; VOPD=${6:-OFF}; CXXC=${7:-}
# The tree's own C++ compiler, so the out-of-tree build does not depend on the host's default
# (`c++` may be a different GCC than the one the installed ABI headers require).
CCXX=""
[ -n "$CXXC" ] && CCXX="-DCMAKE_CXX_COMPILER=$CXXC"
GEN=${CMAKE_GENERATOR:-Ninja}
J=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || echo 4)}

rm -rf "$W"
mkdir -p "$W/b" "$W/logs"
P="$W/prefix"
H="$W/home"

say() { printf '%s\n' "$*"; }
fail() { say "FAIL: $*"; exit 1; }
run() {   # run LOG CMD... -- quiet unless it fails
  log=$1; shift
  if ! "$@" > "$log" 2>&1; then tail -40 "$log"; fail "$* (log: $log)"; fi
}

say "installing $B into $P"
run "$W/logs/install.log" cmake --install "$B" --prefix "$P"

dirs="libref libquant"
[ -d "$S/libavx" ] && [ -f "$B/radiance_home/kernels/libavx.so" ] && dirs="$dirs libavx"
for d in "$S"/arch/*/ "$S"/plugin/*/; do
  [ -f "$d/CMakeLists.txt" ] && dirs="$dirs $(basename "$(dirname "$d")")/$(basename "$d")"
done
# The device libraries: built out of tree only with the tree's HIP compiler, and left out of every
# comparison below without it. Each declares the family it serves, so a tree built for one family
# produces one of them: libr3d returns without a target when RAD_GPU_TARGETS holds no gfx11, and the
# checks below compare the two builds of the same list, not a fixed list.
devlibs=""
for d in libr4d libr3d; do
  [ -f "$S/$d/CMakeLists.txt" ] && [ -f "$B/radiance_home/kernels/$d.so" ] && devlibs="$devlibs $d"
done
[ -n "$HIPC" ] && dirs="$dirs $devlibs"
is_dev() { case " $devlibs " in *" $1 "*) return 0 ;; esac; return 1; }

# Each plugin installs its plugin file under $H and anything else it ships (libr4d's r4d_selftest)
# under the package's prefix, as a third party installing into that radiance would.
for d in $dirs; do
  n=$(basename "$d")
  say "building $d against the installed package"
  extra="$CCXX"
  is_dev "$n" && extra="$extra -DCMAKE_HIP_COMPILER=$HIPC -DRAD_GPU_TARGETS=$HIPT
                        -DRAD_HIP_VOPD_REPEATABLE=$VOPD"
  # A bundled plugin under plugin/ may carry device rows beside its host ones; it gets the tree's
  # HIP compiler as a third party would pass theirs.
  case "$d" in plugin/*) [ -n "$HIPC" ] && extra="-DCMAKE_HIP_COMPILER=$HIPC" ;; esac
  # shellcheck disable=SC2086
  run "$W/logs/$n.configure.log" cmake -S "$S/$d" -B "$W/b/$n" -G "$GEN" \
      -DCMAKE_PREFIX_PATH="$P" -DCMAKE_INSTALL_PREFIX="$P" -DRAD_PLUGIN_INSTALL_HOME="$H" $extra
  run "$W/logs/$n.build.log" cmake --build "$W/b/$n" -j "$J"
  # --prefix for the same reason as the tree install above: a plugin's own CMakeLists may install
  # companions (libr4d ships r4d_selftest) to relative destinations under CMAKE_INSTALL_PREFIX, and
  # the test must not need root. The plugins themselves ignore the prefix -- rad_add_plugin installs
  # them under the absolute RAD_PLUGIN_INSTALL_HOME given at configure time -- so the checks below,
  # which look in $H, are unaffected.
  run "$W/logs/$n.install.log" cmake --install "$W/b/$n" --prefix "$W/i"
done

# 1. The same plugins, file for file.
sets() {
  for k in kernels architectures quantizers; do
    [ -d "$1/$k" ] && (cd "$1/$k" && ls *.so 2>/dev/null | sed "s|^|$k/|")
  done | sort
}
want=$(sets "$B/radiance_home")
if [ -z "$HIPC" ]; then
  for d in $devlibs; do want=$(printf '%s\n' "$want" | grep -v "^kernels/$d\.so\$" || true); done
fi
have=$(sets "$H")
if [ "$want" != "$have" ]; then
  say "in-tree:"; say "$want"; say "out of tree:"; say "$have"
  fail "the out-of-tree home does not hold the plugins the tree's build does"
fi
say "1. same plugins: $(printf '%s\n' "$have" | wc -l | tr -d ' ') files"

# 2. The same vocabulary, op for op.
for so in "$H"/kernels/*.so; do
  n=$(basename "$so")
  "$B/bin/rad-schemas" "$B/radiance_home/kernels/$n" > "$W/schemas.tree.$n.txt"
  "$B/bin/rad-schemas" "$so" > "$W/schemas.oot.$n.txt"
  cmp -s "$W/schemas.tree.$n.txt" "$W/schemas.oot.$n.txt" ||
    fail "$n declares a different vocabulary out of tree (diff $W/schemas.*.$n.txt)"
done
say "2. same schemas"

# 3. The same registration, from each home alone. No card: the test must not depend on one, and
# must not touch one a server is using.
listing() {
  HIP_VISIBLE_DEVICES=-1 "$P/bin/rad-info" --plugins --home "$1" 2>/dev/null |
    grep -v '^plugins on' | grep -v '^     /'
}
tree_list=$(listing "$B/radiance_home")
if [ -z "$HIPC" ]; then
  for d in $devlibs; do tree_list=$(printf '%s\n' "$tree_list" | grep -v " $d " || true); done
fi
oot_list=$(listing "$H")
# The in-tree list numbers a device library's position when it is left out above; compare without the order
# column and check the order separately by name.
strip() { printf '%s\n' "$1" | sed 's/^ *[0-9]* //'; }
if [ "$(strip "$tree_list")" != "$(strip "$oot_list")" ]; then
  say "in-tree:"; say "$tree_list"; say "out of tree:"; say "$oot_list"
  fail "the plugins register differently out of tree"
fi
say "3. same registration:"
say "$oot_list"

# 4. The installed tools serve from the out-of-tree home. The replay runs every library its home
# holds, so it is given the out-of-tree home less its device libraries, which have no card here.
HH="$W/hosthome"
mkdir -p "$HH/kernels"
for k in architectures quantizers; do [ -d "$H/$k" ] && ln -s "$H/$k" "$HH/$k"; done
for so in "$H"/kernels/*.so; do
  is_dev "$(basename "$so" .so)" || ln -s "$so" "$HH/kernels/"
done
# A fixed --max-bytes, so which cases run does not depend on the memory free at the start.
run "$W/logs/kbench.log" env HIP_VISIBLE_DEVICES=-1 RADIANCE_HOME="$HH" \
    "$P/bin/rad-kbench" --fixture "$P/share/radiance/kernels.rkb" --max-cases 1 \
    --max-bytes 2048 --op rmsnorm --op quant_act_fp8 --op silu_mul
grep -E "passed|failed" "$W/logs/kbench.log" | tail -3
# rad-kbench fails only on a case that ran and failed, so a replay that skipped every case would
# pass unseen: its summary must name at least one passed case and no failed one.
sum=$(grep -E '^[0-9]+ checked, [0-9]+ passed, [0-9]+ failed' "$W/logs/kbench.log" | tail -1)
n_pass=$(printf '%s\n' "$sum" | sed -nE 's/^[0-9]+ checked, ([0-9]+) passed, ([0-9]+) failed.*/\1/p')
n_fail=$(printf '%s\n' "$sum" | sed -nE 's/^[0-9]+ checked, ([0-9]+) passed, ([0-9]+) failed.*/\2/p')
[ -n "$n_pass" ] && [ "$n_pass" -gt 0 ] && [ "$n_fail" -eq 0 ] ||
  fail "the installed rad-kbench replay checked nothing or failed: '$sum' (log: $W/logs/kbench.log)"
say "4. the installed rad-kbench replays fixture ops from the out-of-tree home: $(ls "$HH/kernels" | tr '\n' ' ')"

# 5. The device library's machine code, against the tree's, object by object. A code object also
# carries its compilation unit's ID, which rad_add_plugin names from the plugin and the source's
# path inside it rather than leaving clang to hash the build directory into it
# (cmake/RadAddPlugin.cmake) -- so the instructions (.text), the kernel descriptors and the rest of
# the read-only data agree between the two builds wherever each was made, unless the flags that
# reach the device compiler differ.
#
# THE DESCRIPTORS ARE COMPARED BY NAME, NOT AS .rodata's BYTES. The compiler lays them out in an
# order that changes from one run of the same command to the next -- ROCm 7's clang does it to the
# eight instances of the template in r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip -- and each holds
# its kernel's entry as an offset from itself, so the section's bytes differ while every kernel and
# every descriptor is the same. Each descriptor is read by its symbol, its entry resolved to the
# address in .text it names, and the rest of .rodata is compared with the descriptors blanked.
if [ -n "$HIPC" ]; then
  L=$(dirname "$HIPC")
  for t in llvm-objcopy llvm-readelf clang-offload-bundler; do
    [ -x "$L/$t" ] || fail "no $t beside $HIPC"
  done
  mkdir -p "$W/co"
  fatbin() {   # fatbin OBJ -- its fat binary into $W/co/fb; false for a source with no kernels
    "$L/llvm-objcopy" --dump-section .hip_fatbin="$W/co/fb" "$1" /dev/null 2>/dev/null
  }
  descriptors() {   # descriptors CO RODATA -- each descriptor by name; blanks it in RODATA
    set -- "$1" "$2" $("$L/llvm-readelf" -SW "$1" |
                       awk '{ for (i = 1; i < NF; ++i) if ($i == ".rodata") print $(i + 2), $(i + 3) }')
    [ $# -eq 4 ] || return 0
    base=$((0x$3)); foff=$((0x$4))
    "$L/llvm-readelf" -sW "$1" | awk '$4 == "OBJECT" && $8 ~ /\.kd$/ { print $2, $3, $8 }' |
      sort -u -k3 |
      while read -r val size name; do
        a=$((0x$val)); o=$((foff + a - base))
        e=$(od -An -t d8 -j $((o + 16)) -N 8 "$1" | tr -d ' ')
        printf '%s %s entry=%x %s %s\n' "$name" "$size" $((a + e)) \
          "$(od -An -v -t x1 -j "$o" -N 16 "$1" | tr -d ' \n')" \
          "$(od -An -v -t x1 -j $((o + 24)) -N $((size - 24)) "$1" | tr -d ' \n')"
        dd if=/dev/zero of="$2" bs=1 seek=$((a - base)) count="$size" conv=notrunc 2>/dev/null
      done
  }
  code() {     # code OBJ TARGET -- checksums of that target's instructions, its descriptors by
               # name, and the rest of its read-only data
    fatbin "$1"
    "$L/clang-offload-bundler" --type=o --input="$W/co/fb" --unbundle --targets="$2" \
        --output="$W/co/co"
    for sec in .text .rodata; do
      "$L/llvm-objcopy" --dump-section "$sec=$W/co/$sec" "$W/co/co" /dev/null 2>/dev/null ||
        : > "$W/co/$sec"
    done
    cksum < "$W/co/.text"
    descriptors "$W/co/co" "$W/co/.rodata" | cksum
    cksum < "$W/co/.rodata"
  }
  for lib in $devlibs; do
    td="$B/$lib/CMakeFiles/$lib.dir"
    od="$W/b/$lib/CMakeFiles/$lib.dir"
    [ "$(ls "$td"/*.hip.o | wc -l)" = "$(ls "$od"/*.hip.o | wc -l)" ] ||
      fail "$lib compiles a different set of HIP sources out of tree"
    n=0
    for o in "$td"/*.hip.o; do
      f=$(basename "$o")
      [ -f "$od/$f" ] || fail "$lib's $f has no out-of-tree twin"
      if ! fatbin "$o"; then
        if fatbin "$od/$f"; then fail "$lib's $f has device code out of tree only"; fi
        continue
      fi
      for t in $("$L/clang-offload-bundler" --type=o --input="$W/co/fb" --list | grep -v '^host-'); do
        [ "$(code "$o" "$t")" = "$(code "$od/$f" "$t")" ] ||
          fail "$lib's $f compiles to different device code for $t out of tree"
        n=$((n + 1))
      done
    done
    say "5. $lib's device code is the tree's: $n code objects, the same instructions and descriptors"
  done
fi
say "PASS"
