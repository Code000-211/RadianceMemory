#!/bin/sh
# docker/build.sh -- build the radiance image from a commit with BuildKit (docker buildx) and tag it.
#
# The build context is `git archive` of the commit, so the image holds only committed files (no
# build trees, uncommitted edits or model files). With -s the archive is streamed over ssh to
# another machine's `docker buildx`, which builds and keeps the image; that machine needs only
# Docker with buildx.
#
#   docker/build.sh                       HEAD, here
#   docker/build.sh -s HOST               HEAD, built on and kept by HOST
#   docker/build.sh -r COMMIT             another commit
#   docker/build.sh -t 'gfx1200;gfx1201'  other GPU architectures (default
#                                         'gfx1201;gfx11-generic;gfx1151')
#   docker/build.sh -j 8                  fewer compile jobs, for a host that is also serving
#   docker/build.sh -J 4                  fewer tests at a time (default: the compile jobs)
#   docker/build.sh -B 5                  multiply every test's CPU budget, for a host slower per
#                                         thread than the one the budgets were set on
#   docker/build.sh -E tokenizer_perf_test  leave tests matching a regex out of the test stage
#   docker/build.sh --target build        stop at a stage (tagged radiance-<stage>), e.g. for the
#                                         GPU test suites
#
# Tags: radiance:<project version>, radiance:<commit>, radiance:latest.
set -e
cd "$(dirname "$0")/.."

REV=HEAD
HOST=
TARGETS='gfx1201;gfx11-generic;gfx1151'
JOBS=
TEST_JOBS=
BUDGET_SCALE=
TEST_EXCLUDE=
STAGE=
while [ $# -gt 0 ]; do
    case "$1" in
        -s) shift; HOST="$1" ;;
        -r) shift; REV="$1" ;;
        -t) shift; TARGETS="$1" ;;
        -j) shift; JOBS="$1" ;;
        -J) shift; TEST_JOBS="$1" ;;
        -B) shift; BUDGET_SCALE="$1" ;;
        -E) shift; TEST_EXCLUDE="$1" ;;
        --target) shift; STAGE="$1" ;;
        *)  echo "docker/build.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

SHA=$(git rev-parse --short=12 "$REV^{commit}")
VERSION=$(git show "$SHA:CMakeLists.txt" | sed -n 's/^project(radiance VERSION \([0-9.]*\).*/\1/p')
[ -n "$VERSION" ] || { echo "docker/build.sh: no project version in CMakeLists.txt" >&2; exit 1; }

if [ -n "$STAGE" ]; then
    TAGS="-t radiance-$STAGE --target $STAGE"
else
    TAGS="-t radiance:$VERSION -t radiance:$SHA -t radiance:latest"
fi
CMD="docker buildx build --load --build-arg GPU_TARGETS='$TARGETS' --build-arg JOBS='$JOBS'"
CMD="$CMD --build-arg TEST_JOBS='$TEST_JOBS' --build-arg TEST_CPU_BUDGET_SCALE='$BUDGET_SCALE'"
CMD="$CMD --build-arg TEST_EXCLUDE='$TEST_EXCLUDE'"
CMD="$CMD --build-arg VERSION='$VERSION' --build-arg REVISION='$SHA' $TAGS -"

echo "radiance $VERSION ($SHA) for $TARGETS${HOST:+ on $HOST}${STAGE:+, stage $STAGE}"
if [ -n "$HOST" ]; then
    git archive --format=tar "$SHA" | ssh "$HOST" "$CMD"
else
    git archive --format=tar "$SHA" | sh -c "$CMD"
fi
