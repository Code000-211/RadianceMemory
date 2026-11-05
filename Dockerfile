# syntax=docker/dockerfile:1
#
# radiance as a container image: the engine, plugins and tools built in the ROCm toolchain image,
# the host test suite run against that build, and a FROM scratch runtime image holding the install
# tree and exactly the shared libraries it loads (no distribution, no package manager, only a
# static busybox).
#
#   docker/build.sh                                       # the usual way (BuildKit / buildx)
#   docker buildx build -t radiance .                     # the same, by hand
#   docker buildx build --build-arg GPU_TARGETS="gfx1200;gfx1201" -t radiance .
#
# Stages:
#   ffmpeg   FFmpeg with only the demuxers core/mm/media.cpp admits, the decoders their streams
#            need (pictures, video, audio with swresample) and the three encoders the images API
#            writes (png, mjpeg, libwebp); the distribution's build pulls ~200 unused packages
#   llvm     ROCm's libLLVM rebuilt from ROCm's llvm-project at the commit the image's clang
#            reports, with docker/llvm-vopd-pairing.patch, which makes LLVM 24's device code
#            repeatable across builds (cmake/RadAddPlugin.cmake, rad_hip_repeatable)
#   build    ROCm dev image + cmake/ninja/g++-14; configures, builds and installs to /stage, with
#            device code compiled by ROCm's clang from a shadow ROCm tree using the llvm stage's
#            libLLVM
#   test     docker/llvm-check.sh, then the host-only ctest suites
#   rootfs   the stripped install tree plus every library its binaries and plugins load (resolved
#            by ldd), copied into one directory and checked from inside a chroot
#   runtime  FROM scratch + rootfs; the shipped image
#
# Device code is not built with --offload-compress: the AQL backend reads the fat binaries itself
# and refuses a compressed bundle.
#
# GPU_TARGETS must be named: a build container has no card to detect, unlike a bare-metal
# ./build.sh. They are the targets of the shipped kernel libraries, each of which keeps only its own
# family: libr4d takes gfx1201 (RDNA4), libr3d gfx11-generic (every RDNA3 chip) and gfx1151, and
# libhalo gfx1151 (Strix Halo), where the loader prefers it over libr3d. On any one card the other
# libraries decline or are left out, so the default image serves all three. The engine's own
# kernels carry the generic family targets and run on every AMD GPU family, so a kernel library for
# another card is added by mounting it on the search path:
#   docker run ... -v /my/plugins:/plugins -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance
#
# The image's ROCm userspace must be supported by the host's amdgpu kernel driver; it is pinned to
# the serving host's version, so move both together. From ROCm 10 on, the dev image is published
# only as the -full tag.

ARG ROCM_VERSION=10.1.0-full
ARG UBUNTU_VERSION=24.04
ARG FFMPEG_VERSION=8.1.3
ARG FFMPEG_SHA256=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3

# ----------------------------------------------------------------------------------------- ffmpeg
FROM ubuntu:${UBUNTU_VERSION} AS ffmpeg
ARG FFMPEG_VERSION
ARG FFMPEG_SHA256
ARG JOBS=
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential nasm pkg-config ca-certificates curl xz-utils zlib1g-dev libdav1d-dev \
      libwebp-dev libopenh264-dev \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp
# Demuxers: media.cpp's kFormats whitelist. Decoders: what those containers carry -- still
# images, H.264 and HEVC (MP4, MOV, HEIF), VP8 and VP9 (WebM, WebP), AV1 through dav1d (WebM, MP4,
# AVIF), MPEG-1/2/4, Theora and the FLV/H.263 family. JPEG XL is recognised and refused by name
# (its only decoder is libjxl's). Encoders: the image output formats of media.cpp encode_image
# (png, jpeg as mjpeg, webp through libwebp), and a generated video's MP4 (core/mm/mp4.cpp): H.264
# through libopenh264 (BSD-licensed, so the build stays LGPL) and FFmpeg's own AAC, in the mp4
# muxer. No protocols: media.cpp and mp4.cpp read and write memory through their own I/O
# contexts.
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz \
 && echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - \
 && tar -xJf ffmpeg.tar.xz && cd "ffmpeg-${FFMPEG_VERSION}" \
 && ./configure --prefix=/opt/ffmpeg --enable-shared --disable-static --disable-debug \
      --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything \
      --disable-avdevice --disable-avfilter --enable-swresample \
      --enable-zlib --enable-libdav1d --enable-libwebp \
      --enable-demuxer=mov,matroska,avi,mpegts,flv,ogg,gif,apng,image2pipe,image_png_pipe,image_jpeg_pipe,image_jpegls_pipe,image_jpegxl_pipe,image_webp_pipe,image_bmp_pipe,image_tiff_pipe,image_qoi_pipe,image_pgm_pipe,image_ppm_pipe,image_pbm_pipe,image_pam_pipe,image_gif_pipe,wav,w64,mp3,flac,aac,aiff \
      --enable-decoder=h264,hevc,vp8,vp9,libdav1d,mpeg4,mpeg1video,mpeg2video,mjpeg,jpegls,png,apng,gif,webp,bmp,tiff,qoi,pgm,ppm,pbm,pam,pgmyuv,theora,vp3,flv,h263,pcm_u8,pcm_s16le,pcm_s16be,pcm_s24le,pcm_s24be,pcm_s32le,pcm_s32be,pcm_f32le,pcm_f32be,pcm_f64le,pcm_alaw,pcm_mulaw,mp3float,mp2float,flac,vorbis,opus,aac \
      --enable-encoder=png,mjpeg,libwebp,libopenh264,aac --enable-muxer=mp4 --enable-libopenh264 \
      --enable-parser=h264,hevc,vp8,vp9,av1,mpeg4video,mpegvideo,mjpeg,png,gif,bmp,webp,vp3,qoi,pnm,h263,jpegxl,mpegaudio,flac,aac,vorbis,opus \
 && make -j"${JOBS:-$(nproc)}" \
 && make install \
 && mkdir -p /opt/ffmpeg/share/doc/ffmpeg \
 && cp LICENSE.md COPYING.LGPLv2.1 /opt/ffmpeg/share/doc/ffmpeg/ \
 && sed -n 's/^#define FFMPEG_CONFIGURATION "\(.*\)"$/\1/p' config.h \
      > /opt/ffmpeg/share/doc/ffmpeg/CONFIGURATION \
 && echo "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz, sha256 ${FFMPEG_SHA256}" \
      > /opt/ffmpeg/share/doc/ffmpeg/SOURCE

# ------------------------------------------------------------------------------------------- llvm
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION} AS llvm
ARG JOBS=
# clang, lld and the offload tools all load lib/llvm/lib/libLLVM.so, so this stage rebuilds only
# that file: ROCm's llvm-project at exactly the commit `clang --version` names (the stage stops if
# clang names none or the patch does not apply), configured as ROCm's
# lib/cmake/llvm/LLVMConfig.cmake records -- Release without assertions, AMDGPU;SPIRV;X86, every
# component in the dylib, no RTTI or exceptions, zlib and zstd from rocm_sysdeps -- and compiled
# by ROCm's own clang and lld. With VOPD pairing off it produces
# ROCm's code byte for byte; docker/llvm-check.sh holds the build stage to that. The fetch is
# sparse and depth one: llvm/ plus the cmake/, libc/ and third-party/ directories configure reads,
# without tests or documentation.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git cmake ninja-build ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY docker/llvm-vopd-pairing.patch /tmp/
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    set -e; L=/opt/rocm/lib/llvm; S=/opt/rocm/lib/rocm_sysdeps; \
    rev=$($L/bin/clang --version | sed -n 's|.*ROCm/llvm-project.git \([0-9a-f]\{40\}\).*|\1|p'); \
    [ -n "$rev" ] || { echo "llvm: no ROCm/llvm-project commit in: $($L/bin/clang --version | head -1)" >&2; exit 1; }; \
    git init -q /llvm-project; cd /llvm-project; \
    git remote add origin https://github.com/ROCm/llvm-project.git; \
    git sparse-checkout set --no-cone '/*' '!/*/' /cmake/ /libc/ /llvm/ /third-party/ \
      '!/llvm/test/' '!/llvm/unittests/' '!/llvm/docs/' '!/llvm/benchmarks/'; \
    git fetch -q --depth 1 --filter=blob:none origin "$rev"; \
    git checkout -q FETCH_HEAD; \
    git apply /tmp/llvm-vopd-pairing.patch; \
    cmake -G Ninja -S llvm -B /build-llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_TARGETS_TO_BUILD="AMDGPU;SPIRV;X86" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
      -DLLVM_DYLIB_COMPONENTS=all -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_FFI=OFF \
      -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_CURL=OFF \
      -DLLVM_ENABLE_HTTPLIB=OFF -DLLVM_ENABLE_TELEMETRY=ON -DLLVM_ENABLE_PLUGINS=ON \
      -DLLVM_ENABLE_LIBPFM=OFF \
      -DLLVM_ENABLE_ZLIB=FORCE_ON -DZLIB_INCLUDE_DIR=$S/include -DZLIB_LIBRARY=$S/lib/librocm_sysdeps_z.so.1 \
      -DLLVM_ENABLE_ZSTD=FORCE_ON -Dzstd_INCLUDE_DIR=$S/include -Dzstd_LIBRARY=$S/lib/librocm_sysdeps_zstd.so.1 \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
      -DLLVM_INCLUDE_DOCS=OFF -DLLVM_BUILD_TOOLS=OFF -DLLVM_USE_LINKER=lld -DLLVM_PARALLEL_LINK_JOBS=2 \
      -DCMAKE_C_COMPILER=$L/bin/clang -DCMAKE_CXX_COMPILER=$L/bin/clang++ > /tmp/llvm-configure.log 2>&1 \
      || { tail -40 /tmp/llvm-configure.log >&2; exit 1; }; \
    ninja -C /build-llvm ${JOBS:+-j${JOBS}} LLVM > /tmp/llvm-build.log 2>&1 \
      || { grep -m20 -B2 -A8 'error' /tmp/llvm-build.log >&2; exit 1; }; \
    so=$(basename $(readlink -f $L/lib/libLLVM.so)); \
    mkdir -p /out; cp /build-llvm/lib/$so /out/; \
    rm -rf /build-llvm /llvm-project

# ------------------------------------------------------------------------------------------ build
FROM rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION} AS build
ARG GPU_TARGETS="gfx1201;gfx11-generic;gfx1151"
# Parallel compile jobs; empty means one per core. A device-kernel translation unit takes 2-3 GB
# of RAM under hipcc, so lower it on a host that is also serving.
ARG JOBS=

# g++-14 rather than the default 13: hipcc compiles device-side translation units against its
# libstdc++, which is also the runtime image's. gawk: tools/ops_doc_check.sh uses gensub, which
# mawk lacks. libdav1d7, libwebp7 and libopenh264-7 are the runtime libraries of the ffmpeg stage's
# AV1 decoder, WebP encoder and H.264 encoder; busybox-static goes into the runtime image.
# libopenblas-dev: libquant's GPTQ factorises its Hessians through LAPACK where it finds OpenBLAS
# (libquant/CMakeLists.txt); without it rad-convert's gptq rule runs single-threaded loops.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake ninja-build pkg-config g++-14 gawk libdav1d7 libwebp7 libopenh264-7 busybox-static \
      libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ffmpeg /opt/ffmpeg /opt/ffmpeg
# Register ROCm's library directory beside FFmpeg's: the ROCm 10 images register none, and the
# rootfs stage resolves the installed binaries' libraries with ldd.
RUN echo /opt/ffmpeg/lib > /etc/ld.so.conf.d/ffmpeg.conf \
 && echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm CC=gcc-14 CXX=g++-14 \
    PKG_CONFIG_PATH=/opt/ffmpeg/lib/pkgconfig

# The HIP compiler: ROCm's clang loading the llvm stage's libLLVM. ROCm's tools find libraries
# through RPATHs relative to themselves ($ORIGIN/../lib for libLLVM, which LD_LIBRARY_PATH cannot
# override, plus $ORIGIN/../../rocm_sysdeps/lib and $ORIGIN/../../../lib), so the compiler runs
# from /opt/rocm-repeatable, a shadow of the whole ROCm tree at the same depth. Every entry links
# back into /opt/rocm except lib/llvm/bin (real copies of the executables) and the rebuilt
# libLLVM. Only compiling uses it. The build passes --rocm-path=/opt/rocm so headers and device
# libraries are ROCm's and linked plugins record /opt/rocm/lib. Without it clang treats its own
# tree as ROCm, and the HIP runtime would reach comgr through the rebuilt libLLVM, which lacks the
# libstdc++ symbols ROCm's GCC-built one re-exports for comgr. The rootfs stage refuses any
# library resolved through the shadow tree.
COPY --from=llvm /out/ /tmp/llvm-out/
RUN set -e; R=/opt/rocm; D=/opt/rocm-repeatable; L=lib/llvm; \
    so=$(basename $(readlink -f $R/$L/lib/libLLVM.so)); \
    [ -f /tmp/llvm-out/$so ] || { echo "the llvm stage built no $so" >&2; exit 1; }; \
    mkdir -p $D/$L/lib; \
    for e in $R/*; do n=${e##*/}; [ $n = lib ] || ln -s $e $D/$n; done; \
    for e in $R/lib/*; do n=${e##*/}; [ $n = llvm ] || ln -s $e $D/lib/$n; done; \
    for e in $R/$L/*; do n=${e##*/}; [ $n = bin ] || [ $n = lib ] || ln -s $e $D/$L/$n; done; \
    cp -a $R/$L/bin $D/$L/; \
    for e in $R/$L/lib/*; do n=${e##*/}; \
      if [ $n = $so ]; then mv /tmp/llvm-out/$so $D/$L/lib/; \
      elif [ -L $e ] && [ "$(readlink $e)" = $so ]; then ln -s $so $D/$L/lib/$n; \
      else ln -s $e $D/$L/lib/$n; fi; \
    done; \
    rm -rf /tmp/llvm-out

WORKDIR /src
COPY . .

# The prefix is fixed at configure time because the default $RADIANCE_HOME is compiled in as
# <prefix>/share/radiance; DESTDIR stages the tree without moving that default.
# oom_score_adj 1000: on a host that is also serving, the OOM killer takes the compiler first.
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    cmake -S . -B /build -G Ninja \
      -DCMAKE_INSTALL_PREFIX=/opt/radiance \
      -DRAD_WITH_HIP=ON -DRAD_WITH_FFMPEG=ON \
      "-DRAD_GPU_TARGETS=${GPU_TARGETS}" \
      -DCMAKE_HIP_COMPILER=/opt/rocm-repeatable/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
      -DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm \
      -DRAD_HIP_VOPD_REPEATABLE=ON \
 && cmake --build /build ${JOBS:+-j${JOBS}} \
 && DESTDIR=/stage cmake --install /build \
 && mkdir -p /stage/opt/radiance/share/radiance/recipes \
 && cp data/recipes/*.recipe /stage/opt/radiance/share/radiance/recipes/

# ------------------------------------------------------------------------------------------- test
FROM build AS test
ARG JOBS=
# Tests at a time when it should differ from JOBS (below).
ARG TEST_JOBS=
# A factor on every test's CPU budget (RAD_TEST_CPU_BUDGET_SCALE, tests/rad_test.h), for a host
# slower per thread than the one the budgets were set on: a VM on an EPYC 7302 spends 4.6x the
# quoted CPU time on tokenizer_test's reference pass.
ARG TEST_CPU_BUDGET_SCALE=
# A ctest -E regex of entries to leave out, for a host where a timing assertion does not hold:
# tokenizer_perf_test's one-word ratio reads 10-12 against its 9 on a Zen 2 VM with 512 KiB of L2,
# where the sandbox leaves it CPU time rather than instructions to count.
ARG TEST_EXCLUDE=
# Host-only: a build container has no /dev/kfd. The `gpu`-labelled tests (the *_card entries,
# r4d_selftest, kbench) run against a card from this stage:
#   docker buildx build --target build -t radiance-build . && docker run --rm \
#       --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined radiance-build \
#       ctest --test-dir /build -L gpu --output-on-failure
# Tests run TEST_JOBS at a time, else JOBS (default one per two hardware threads, about one per
# core on an SMT machine: the per-binary CPU budgets in tests/CMakeLists.txt hold for tests that do
# not share a core); the `perf` entries, which assert wall-clock costs, run alone (RUN_SERIAL). The
# build sandbox forbids io_uring, so avx_check skips its file-backed gather case, and
# perf_event_open, so the tests that bound work count CPU time rather than instructions
# (tests/work_counter.h). The `oot` suite (every shipped plugin rebuilt against the installed
# package) repeats this stage's whole compile, so it runs from the development tree instead
# (tests/oot_plugins.sh). The `parity` entry (halo_parity: libhalo's tables against libr4d's) is
# left out until libhalo has ported the ops libr4d added after it (the test names each one).
RUN echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true; \
    bash /src/docker/llvm-check.sh /build \
 && RAD_TEST_CPU_BUDGET_SCALE="${TEST_CPU_BUDGET_SCALE}" \
    ctest --test-dir /build -LE 'gpu|oot|parity' ${TEST_EXCLUDE:+-E "${TEST_EXCLUDE}"} \
      -j"${TEST_JOBS:-${JOBS:-$(( ($(nproc) + 1) / 2 ))}}" --output-on-failure \
 && touch /build/host-tests-passed

# ----------------------------------------------------------------------------------------- rootfs
FROM build AS rootfs
# hfget.sh (fetches the Hugging Face repo a --model id names) needs curl, jq and a CA bundle.
# Installed here rather than in `build` so changing them does not recompile the engine.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      curl jq ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# The marker makes this stage, and so the shipped image, depend on the test stage.
COPY --from=test /build/host-tests-passed /rootfs/opt/radiance/share/radiance/.host-tests-passed
# Every library the binaries and plugins load, plus comgr (opened by the HIP runtime at run time
# and required to initialise a device), goes into one directory the loader searches by default:
# no ld.so.cache, no LD_LIBRARY_PATH. ROCm's libdrm_amdgpu has the card name table compiled in, so
# no amdgpu.ids is copied. A static busybox provides the health check's wget and a shell for
# `docker exec`; curl, jq and the CA bundle are for hfget.sh (the bundle is ca-certificates'
# Mozilla set, named by that directory in the license list). docker/licenses.sh records each
# file's package, license and source in /usr/share/doc/THIRD-PARTY. Finally every ELF file is
# resolved from inside the tree, so none depends on a library the image lacks.
RUN set -e; R=/rootfs; L=$R/usr/lib/x86_64-linux-gnu; \
    mkdir -p $L $R/lib64 $R/usr/bin $R/bin $R/etc/ssl/certs $R/data $R/tmp; \
    chmod 1777 $R/tmp; \
    cp -a /stage/opt/radiance $R/opt/; \
    elves=$(find $R/opt/radiance -type f -exec sh -c 'head -c4 "$1" | grep -q ELF' _ {} \; -print); \
    strip --strip-unneeded $elves; \
    libs=$(for f in $elves /usr/bin/curl /usr/bin/jq /opt/rocm/lib/libamd_comgr.so.3; do ldd "$f"; done \
           | awk '/=> \//{print $3} /^\t\/lib64\//{print $1}' | sort -u); \
    if echo "$libs" | grep /opt/rocm-repeatable/; then \
      echo "these resolve through the compiler's shadow ROCm tree, not /opt/rocm" >&2; exit 1; fi; \
    for lib in $libs; do \
      case "$lib" in */ld-linux-x86-64.so.2) cp -L "$lib" $R/lib64/ ;; *) cp -L "$lib" $L/ ;; esac; \
    done; \
    cp -L /opt/rocm/lib/libamd_comgr.so.3 $L/; \
    cp /bin/busybox $R/usr/bin/busybox; ln -s /usr/bin/busybox $R/bin/sh; \
    cp /usr/bin/curl /usr/bin/jq $R/usr/bin/; \
    cp /etc/ssl/certs/ca-certificates.crt $R/etc/ssl/certs/; \
    sh /src/docker/licenses.sh $R $libs /opt/rocm/lib/libamd_comgr.so.3 /bin/busybox \
      /usr/bin/curl /usr/bin/jq /usr/share/ca-certificates/mozilla; \
    for f in $(cd $R && find opt/radiance usr/lib -type f -exec sh -c 'head -c4 "$1" | grep -q ELF' _ {} \; -print) \
             usr/bin/curl usr/bin/jq; do \
      out=$(chroot $R /lib64/ld-linux-x86-64.so.2 --list "/$f" 2>&1) || { echo "$f: $out" >&2; exit 1; }; \
      if echo "$out" | grep -q 'not found'; then echo "$f: $out" >&2; exit 1; fi; \
    done; \
    chroot $R /opt/radiance/bin/radiance --help > /dev/null; \
    chroot $R /usr/bin/curl --version > /dev/null; chroot $R /usr/bin/jq --version > /dev/null; \
    du -sh $R $L $R/opt/radiance

# ---------------------------------------------------------------------------------------- runtime
FROM scratch AS runtime
ARG GPU_TARGETS="gfx1201;gfx11-generic;gfx1151"
ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.title="radiance" \
      org.opencontainers.image.description="LLM inference engine for AMD GPUs" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      radiance.gpu-targets="${GPU_TARGETS}"

COPY --from=rootfs /rootfs/ /

# Let one allocation take the whole card and the HIP heap grow to all of VRAM; the runtime's
# defaults cap both below what the placement plan hands out.
ENV PATH=/opt/radiance/bin:/usr/bin:/bin \
    GPU_MAX_ALLOC_PERCENT=100 \
    GPU_MAX_HEAP_SIZE=100

WORKDIR /data
EXPOSE 8000
ENTRYPOINT ["radiance"]
CMD ["--help"]
