#!/bin/sh
# ops_doc_check.sh -- hold docs/OPS.md to the schemas the plugins actually declare.
#
# OPS.md is the authority for the ops it defines, and architecture plugins are written against it,
# so its drift is caught here: a dropped optional operand, a missing `:inout` marker, a parameter
# that moved into an operand, a parameter no plugin declares.
#
#     tools/ops_doc_check.sh docs/OPS.md build/bin/rad-schemas build/radiance_home/kernels/*.so
#
# A row with a fourth column (a proposal) is exempt and its marker is printed. Exit 1 on any
# disagreement; exit 77 (ctest's SKIP_RETURN_CODE) when a plugin will not dump. A return code, not
# a "  SKIP " line: ctest evaluates SKIP_REGULAR_EXPRESSION before the return code, so a skip line
# would mask a disagreement found later in the same run.
set -e
[ $# -ge 3 ] || { echo "usage: $0 <docs/OPS.md> <rad-schemas> <plugin.so>..." >&2; exit 2; }
doc=$1; schemas=$2; shift 2
[ -r "$doc" ] || { echo "  SKIP no $doc"; exit 77; }
[ -x "$schemas" ] || { echo "  SKIP no rad-schemas at $schemas"; exit 77; }

dump=$(mktemp); trap 'rm -f "$dump"' EXIT
for so in "$@"; do
    [ -r "$so" ] || { echo "  SKIP no plugin at $so"; exit 77; }
    "$schemas" "$so" >> "$dump" 2>/dev/null || { echo "  SKIP $so would not dump"; exit 77; }
done

awk -v doc="$doc" '
# ---- tool notation: name:type[/optional|/derived], name:role[?] ----------------------------
function t_param(t,   f,o) {
    if (sub(/\/optional$/, "", t)) o = "/opt"; else if (sub(/\/derived$/, "", t)) o = "/der"
    f = (t ~ /:f64$/) ? ":f64" : ""; sub(/:(int|str|f64)$/, "", t); return t f o }
function t_opd(t,   q,w,io) {
    if (sub(/\?$/, "", t)) q = "?"
    # A weight table is a weight operand whose issue names a run of handles, annotated apart:
    # `w`wt in the document. See docs/OPS.md and RAD_WTAB in abi/rad_runtime.h.
    if (sub(/:wtab$/, "", t)) w = "wt"
    if (sub(/:weight$/, "", t)) w = "w"
    if (sub(/:inout$/, "", t)) io = ":inout"
    sub(/:(in|out)$/, "", t); return t w io q }
# ---- doc notation: `name` followed by its annotations -------------------------------------
# POSIX awk only: mawk, Debian's and Ubuntu's default awk, has no gensub.
function trim(t) { sub(/^ +/, "", t); sub(/ +$/, "", t); return t }
function d_name(t,   a) { a = index(substr(t, 2), "`"); return substr(t, 2, a - 1) }
function d_rest(t,   a) { a = index(substr(t, 2), "`"); return substr(t, a + 2) }
function d_param(t,   n,r,f,o) {
    if (substr(t, 1, 1) != "`") return ""
    n = d_name(t); r = d_rest(t); sub(/\(range\)/, "", r)
    f = (r ~ /:f64/) ? ":f64" : ""
    if (r ~ /\/opt/) o = "/opt"; else if (r ~ /\/der/) o = "/der"
    return n f o }
function d_opd(t,   n,r,w,io,q) {
    if (substr(t, 1, 1) != "`") return ""
    n = d_name(t); r = d_rest(t)
    if (substr(r, 1, 2) == "wt") w = "wt"; else if (substr(r, 1, 1) == "w") w = "w"
    if (r ~ /:inout/) io = ":inout"
    if (r ~ /\?/) q = "?"
    return n w io q }
# ---- the dump: the first declarer fixes the schema, as in the loader (spec 2.3) -------------
FILENAME != doc {
    if ($0 ~ /^  [a-z]/)         { op = $1; seen[op] = 1; next }
    if ($0 ~ /^      params /)   { s = ""; for (i = 2; i <= NF; i++) s = s (i > 2 ? " " : "") t_param($i)
                                   if (!(op in P)) P[op] = s; next }
    if ($0 ~ /^      operands /) { s = ""; for (i = 2; i <= NF; i++) s = s (i > 2 ? " " : "") t_opd($i)
                                   if (!(op in O)) O[op] = s; next }
    next }
# ---- the doc rows --------------------------------------------------------------------------
/^\| `/ {
    n = split($0, c, "|")
    op = d_name(trim(c[2])); documented[op] = 1
    if (n > 5) { exempt++; printf "exempt  %-24s line %d -- %s\n", op, FNR,
                        trim(c[5]); next }
    pc = c[3]; oc = c[4]; gsub(/[,;]/, " ", pc); gsub(/[,;]|→/, " ", oc)
    ps = ""; np = split(pc, pa, /[ \t]+/)
    for (i = 1; i <= np; i++) { v = d_param(pa[i]); if (v != "") ps = ps (ps == "" ? "" : " ") v }
    os = ""; no = split(oc, oa, /[ \t]+/)
    for (i = 1; i <= no; i++) { v = d_opd(oa[i]);   if (v != "") os = os (os == "" ? "" : " ") v }
    rows++
    # An op no loaded plugin declares is a row this configuration cannot check, not a
    # disagreement: a host-only build (spec 18) loads no device library. It is counted and named.
    if (!(op in P)) { unchecked++; u_list = u_list (u_list == "" ? "" : " ") op; next }
    ok = 1
    if (ps != P[op]) { ok = 0; printf "PARAMS  %-24s line %d\n          doc  %s\n          real %s\n", op, FNR, ps, P[op] }
    if (os != O[op]) { ok = 0; printf "OPERAND %-24s line %d\n          doc  %s\n          real %s\n", op, FNR, os, O[op] }
    if (!ok) bad++ }
END {
    for (op in seen) if (!(op in documented)) {
        bad++; printf "UNDOCUMENTED %-20s declared by a plugin, in no table\n", op }
    if (unchecked) printf "\n%d op(s) no loaded plugin declares, so unchecked here: %s\n",
                          unchecked, u_list
    printf "\n%s: %d rows, %d exempt, %d unchecked, %d disagree\n", doc, rows, exempt, unchecked, bad
    exit bad ? 1 : 0 }
' "$dump" "$doc"
