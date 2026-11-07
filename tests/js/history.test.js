/* The history summary line: how it groups, and how it wraps.
 *
 * The line packs several categories into one strip -- a count, tok/s, prefill, ttft -- and each
 * category carries its own figures. Two rules the page must keep:
 *
 *   a category is ATOMIC. If it does not fit on the line it moves whole to the next line. It
 *     must never break mid-figure: "810" at the end of one line and "ms" on the next is worse
 *     than the wrap it replaced.
 *   the category break is a DOT and the break between figures inside one category is a PIPE.
 *
 * Both rules are structural, not string-level: HTML collapses runs of spaces, so the width of a
 * category break cannot come from spaces in the joined string. It comes from a padded element.
 * That is why these checks inspect the DOM the shipped code builds rather than the string it
 * used to build -- a string assertion here would have passed while the page rendered wrong. */
const fs = require("fs");
const path = require("path");
const src = require("./page.js");           // the page's <script> block
/* The CSS lives in the <style> block, which page.js does not return (the JS tests have no
   reason to read it). Read it from the same shipped file rather than widen page.js's export and
   break the two suites that treat its result as a plain string. */
const cpp = fs.readFileSync(path.join(__dirname, "..", "..", "core", "server", "dashboard.cpp"), "utf8");
const css = (cpp.match(/<style>([\s\S]*?)<\/style>/) || [, ""])[1];
const L = src.split("\n");
let fails = 0, checks = 0;
const ok = (c, m) => { checks++; if (!c) { fails++; console.log("  FAIL " + m); } };
const eq = (a, b, m) => { checks++; if (a !== b) { fails++; console.log("  FAIL " + m + ": got " + JSON.stringify(a) + " want " + JSON.stringify(b)); } };
function lines(a, b) {
  const i = L.findIndex((x) => x.startsWith(a));
  if (i < 0) throw new Error("no start line: " + a);
  let j = -1;
  for (let k = i; k < L.length; k++) if (L[k].startsWith(b)) { j = k; break; }
  if (j < 0) throw new Error("no end line: " + b);
  return L.slice(i, j + 1).join("\n");
}

console.log("== the CSS makes a category atomic and the break actually wide ==");
const grpCss = css.match(/#hist \.hgrp \{([^}]*)\}/);
ok(css.length > 200, "read the page <style> block (" + css.length + " chars)");
ok(grpCss, "#hist .hgrp rule exists");
ok(/white-space:\s*nowrap/.test(grpCss ? grpCss[1] : ""), "a category cannot break mid-figure");
const sepCss = css.match(/#hist \.hsep \{([^}]*)\}/);
ok(sepCss, "#hist .hsep rule exists");
const padM = sepCss ? sepCss[1].match(/padding:\s*0\s+(\d+)px/) : null;
ok(padM, "the category break gets width from padding, not spaces");
const pad = padM ? +padM[1] : 0;
ok(pad >= 4, "the padding is wide enough to read as a break (" + pad + "px, want >= 4)");

console.log("== the shipped summary assembly builds one atomic group per category ==");
/* Slice ONLY the summary-line assembly (mean/group -> the forEach that fills #hsum), not the
   whole renderer: the rest of histRender draws the table and needs a dozen module globals that
   have nothing to do with how the line groups or wraps. */
const si = L.findIndex((x) => x.startsWith("  const mean = (a) =>"));
if (si < 0) throw new Error("no summary start");
const ai = L.findIndex((x, k) => k > si && x.includes("hsum.appendChild(g);"));
if (ai < 0) throw new Error("no assembly anchor");
let ei = -1;
for (let k = ai; k < L.length; k++) if (L[k] === "  });") { ei = k; break; }
if (ei < 0) throw new Error("no assembly end");
const BLOCK = L.slice(si, ei + 1).join("\n");
ok(BLOCK.includes("hgrp") && BLOCK.includes("hsep"), "slice captured the grouping code (" + BLOCK.length + " chars)");

function mk(tag, cls, txt) {
  const o = { tag, cls, _t: txt || "", kids: [],
    appendChild(c) { this.kids.push(c); return c; },
    set textContent(v) { this._t = v; if (v === "") this.kids = []; },
    get textContent() { return this._t || ""; } };
  Object.defineProperty(o, "txt", { get() { return o._t || ""; } });
  return o;
}
const hsum = mk("span", "hsum");
const el = (t, c, x) => mk(t, c, x);
const doc = { createTextNode: (t) => ({ tag: "#text", cls: null, txt: t }) };
/* The formatters come from the SHIPPED source, not re-typed here: a guard that asserts against a
   stub's own " ms" tests the stub. Slice DASH..dur (self-contained: ladder, sig3, cnt, pctn all
   inside) and drive the real tps/cnt3/dur. */
/* lines() is INCLUSIVE of the end anchor, which would leave a dangling `const stepTime =` and
   throw a SyntaxError inside new Function. Slice to the line BEFORE it instead. */
const fi = L.findIndex((x) => x.startsWith("const DASH = "));
const fj = L.findIndex((x) => x.startsWith("const stepTime = "));
if (fi < 0 || fj < 0 || fj <= fi) throw new Error("formatter block not located");
const FMT = L.slice(fi, fj).join("\n");
const { tps: tpsF, cnt3: cnt3F, dur: durF } = new Function(FMT + "\nreturn { tps, cnt3, dur };")();
/* Fixture with DISTINCT means: the old cycling values made mean(all) == mean(last 20) by
   coincidence, so a window mutation could not show up in the figures. */
const mkRows = (n) => Array.from({ length: n }, (_, i) => ({ tps: 100 + i, pf: 600 + 2 * i, ttft: 0.5 + 0.1 * i, age: i * 3 }));
const rows = mkRows(24);
new Function("el", "document", "$", "hRows", "tps", "cnt3", "dur", BLOCK)(
  el, doc, () => hsum, rows, tpsF, cnt3F, durF);

/* #hsum now holds the four category groups AND the zero-width-space break points between them,
   so select by class rather than assuming every child is a group. */
const groups = hsum.kids.filter((k) => k.cls === "hgrp");
const zwsps = hsum.kids.filter((k) => k.tag === "#text" && k.txt === "\u200b");
eq(groups.length, 4, "four categories: count, tok/s, prefill, ttft");
console.log("== the line has a break point, so a long summary wraps instead of clipping ==");
/* The groups are nowrap; without a break opportunity between them the whole line is one
 * unbreakable run that overflows the row and clips whatever follows it. */
eq(zwsps.length, 3, "a zero-width break point between each pair of categories");
ok(hsum.kids.every((k) => k.cls === "hgrp" || (k.tag === "#text" && k.txt === "\u200b")),
   "#hsum holds only groups and break points, nothing else");
/* The break point must sit BETWEEN groups, not inside one: a ZWSP inside a nowrap span cannot
 * break, so it would buy nothing. */
for (let i = 1; i < groups.length; i++) {
  const first = hsum.kids.indexOf(groups[i]);
  eq(hsum.kids[first - 1].txt, "\u200b", "break point immediately precedes category " + i);
}
ok(!JSON.stringify(groups.map((g) => g.kids)).includes("\u200b"),
   "no break point buried inside a nowrap group, where it could not act");
console.log("== the dot introduces its category and travels with it ==");
eq(groups[0].kids.filter((k) => k.cls === "hsep").length, 0, "the first category carries no leading dot");
[1, 2, 3].forEach((i) => {
  const seps = groups[i].kids.filter((k) => k.cls === "hsep");
  eq(seps.length, 1, "category " + i + " carries exactly one dot");
  eq(groups[i].kids[0].cls, "hsep", "the dot is the FIRST child of its own group, not a sibling of it");
  eq(groups[i].kids[0].txt, "\u00b7", "and it is a middle dot, not a pipe");
});
console.log("== within a category the fields are separated by pipes ==");
const textOf = (g) => g.kids.map((k) => (k.txt || "")).join("");
/* The category's own text, excluding the leading dot that introduces it (that dot is the
   category BREAK and is SUPPOSED to be a dot; "no dot inside" is about the figures. */
const ownText = (g) => g.kids.filter((k) => k.cls !== "hsep").map((k) => k.txt || "").join("");
const tpsGroup = textOf(groups[1]);
const tpsOwn = ownText(groups[1]);
ok(tpsGroup.includes("|"), "tok/s separates mean / last 20 / last 5 with pipes: " + JSON.stringify(tpsGroup));
ok(!tpsOwn.includes("\u00b7"), "no dot among a category's own figures (the dot means category change): " + JSON.stringify(tpsOwn));
ok(/last 20:.*\|.*last 5:/.test(tpsGroup), "both windows present and pipe-separated");
console.log("== each figure is the mean of the rows its label names ==");
/* Computed independently of the shipped group(): if the shipped slice or threshold drifts
   (slice(-20) -> slice(-100) survived the old suite), rendered and expected diverge. */
const mean = (a) => a.reduce((p, c) => p + c, 0) / a.length;
const tv = rows.map((r) => r.tps), pv = rows.map((r) => r.pf);
const wantTps = "tok/s mean " + tpsF(mean(tv)) + " | last 20: " + tpsF(mean(tv.slice(-20))) + " | last 5: " + tpsF(mean(tv.slice(-5)));
ok(tpsOwn === wantTps, "tok/s figures are the means of all / last 20 / last 5: " + JSON.stringify(tpsOwn) + " want " + JSON.stringify(wantTps));
const pfOwn = ownText(groups[2]);
const wantPf = "prefill mean " + cnt3F(mean(pv)) + " | last 20: " + cnt3F(mean(pv.slice(-20))) + " | last 5: " + cnt3F(mean(pv.slice(-5)));
ok(pfOwn === wantPf, "prefill figures are the means of all / last 20 / last 5: " + JSON.stringify(pfOwn) + " want " + JSON.stringify(wantPf));
console.log("== nothing can orphan a unit ==");
/* The failure this exists to prevent: "ttft median 810" ending a line and "ms" starting the next.
 * That happens when the value and its unit are separate wrap opportunities. */
const ttft = textOf(groups[3]);
const ttftGroup = groups[3];
const textNodes = ttftGroup.kids.filter((k) => k.cls !== "hsep");
eq(textNodes.length, 1, "the ttft category is a single text node, so value and unit cannot split");
const tfsSorted = rows.map((r) => r.ttft).sort((a, b) => a - b);
const wantTtft = "ttft median " + durF(tfsSorted[Math.floor(tfsSorted.length / 2)]);
ok(textNodes[0].txt === wantTtft,
   "value and unit are one string, rendered by the shipped dur(): " + JSON.stringify(textNodes[0].txt) + " want " + JSON.stringify(wantTtft));
console.log("== the window thresholds: last 20 needs 20 rows, last 5 needs 5 ==");
/* Every count sits ADJACENT to a threshold, because that is the only place a threshold can move
   unnoticed: 19/20 pins the 20 boundary (>=19 fails at 19), 4/5 pins the 5 boundary (>=6 or >=7
   fails at 5). Counts spaced further apart leave the off-by-one free to move. */
for (const [n, want20, want5] of [[4, false, false], [5, false, true], [6, false, true],
                                  [7, false, true], [12, false, true], [19, false, true],
                                  [20, true, true], [24, true, true]]) {
  const box = mk("span", "hsum");
  new Function("el", "document", "$", "hRows", "tps", "cnt3", "dur", BLOCK)(
    el, doc, () => box, mkRows(n), tpsF, cnt3F, durF);
  const t = box.kids.filter((k) => k.cls === "hgrp").map((g) => g.kids.map((k) => k.txt || "").join("")).join(" ");
  eq(t.includes("last 20:"), want20, n + " rows: last 20 " + (want20 ? "shows" : "stays hidden (it would repeat the overall mean)"));
  eq(t.includes("last 5:"), want5, n + " rows: last 5 " + (want5 ? "shows" : "stays hidden"));
}
console.log("== the shipped formatters render these exact strings ==");
/* The window guards above call tpsF/cnt3F/durF on BOTH sides, so a formatter bug cancels out --
   replacing tps with a constant passed the whole suite. These literals are the only place the
   formatters are pinned to something that is not themselves. Values derived from the shipped
   source; DASH is U+2013, not a hyphen. */
eq(tpsF(58), "58.0", "tps keeps one decimal through the middle band");
eq(tpsF(54), "54.0", "tps: the GUIDE's F1 example figure");
eq(tpsF(9.995), "10.0", "tps lower band boundary is inclusive");
eq(tpsF(9.994), "9.99", "tps just under the lower boundary keeps two decimals (kills sig3 9.995->9.994)");
eq(tpsF(99.949), "99.9", "tps just under the upper boundary keeps a decimal");
eq(tpsF(99.95), "100", "tps upper band boundary is exclusive");
eq(tpsF(120), "120", "tps above the band is an integer");
eq(tpsF(0), "0", "tps of zero is 0, not 0.00");
eq(tpsF(NaN), "\u2013", "tps of a non-finite is the en-dash");
eq(cnt3F(691), "691", "cnt3 below the kilo step is a plain integer");
eq(cnt3F(1234567), "1.23M", "cnt3 steps to M with three significant figures");
eq(durF(0.813), "813 ms", "dur under a second is whole milliseconds");
eq(durF(0), "0 ms", "dur of zero is 0 ms");
eq(durF(1.7), "1.70 s", "dur just over a second carries two decimals");
eq(durF(59.97), "1m 00s", "dur rounds to the shown resolution BEFORE splitting (not 60.0 s)");

console.log("== the median is falsifiable: an ODD row count ==");
/* At n=24 floor(12) and ceil(12) are the same row, so floor->ceil in the shipped median passed
   every earlier fixture. Only an odd count makes the two candidates differ. */
const box25 = mk("span", "hsum");
const r25 = mkRows(25);
const s25 = r25.map((r) => r.ttft).sort((a, b) => a - b);
if (s25[Math.floor(25 / 2)] === s25[Math.ceil(25 / 2)]) throw new Error(
  "fixture broken: n=25 must make floor and ceil pick different rows");
new Function("el", "document", "$", "hRows", "tps", "cnt3", "dur", BLOCK)(
  el, doc, () => box25, r25, tpsF, cnt3F, durF);
const ttft25 = box25.kids.filter((k) => k.cls === "hgrp")[3].kids
  .filter((k) => k.cls !== "hsep").map((k) => k.txt || "").join("");
eq(ttft25, "ttft median 1.70 s", "the shipped median takes the floor row at n=25");

console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped history line");
process.exit(fails ? 1 : 0);
