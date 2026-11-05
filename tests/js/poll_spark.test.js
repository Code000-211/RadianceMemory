/* Poll cadence and the time-proportional sparkline. Executes the shipped clampMs,
   pollCadence, pollSummary, spark and setMetric, sliced by line range out of the page.
   The window expectations DERIVE from the shipped HIST_S rather than hardcoding 60, so
   changing the constant moves the expectation instead of silently passing. */
const src = require("./page.js");
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
const HIST_S_SRC = lines("const HIST_S =", "const HIST_S =");
const HS = parseInt(HIST_S_SRC.match(/HIST_S\s*=\s*([0-9]+)/)[1], 10);
const SPARK = lines("function spark(key, pts) {", "}");
const CADENCE = lines("const POLL_K", "const pollCadence =");
const SUMMARY = lines("function pollSummary(", "}");
const SETMETRIC = lines("function setMetric(key, value, text) {", "}");

console.log("== the window is a constant, and the expectations read it ==");
eq(HS, 60, "HIST_S is 60 as documented");

function cadence(raw) {
  const els = {};
  const lsGet = () => raw;
  const $ = (id) => (els[id] || (els[id] = { textContent: "" }));
  return new Function("lsGet", "$", CADENCE + "\n" + SUMMARY +
    "\npollSummary();\nreturn {clampMs, pollCadence, idle:()=>pollIdleMs, busy:()=>pollBusyMs, sum:()=>$('advsum').textContent};")(lsGet, $);
}
console.log("== clamp: the field cannot be talked out of a sane rate ==");
let c = cadence(null);
eq(c.clampMs(50, 500), 100, "below floor clamps to the floor");
eq(c.clampMs(9999, 500), 5000, "above ceiling clamps to the ceiling");
eq(c.clampMs("", 500), 500, "blank field falls back to its default");
eq(c.clampMs("abc", 300), 300, "garbage falls back to its default");
eq(c.clampMs(1500, 500), 1500, "in-range passes through");
console.log("== defaults, and the in-flight trigger ==");
eq(c.idle(), 500, "shipped idle default"); eq(c.busy(), 300, "shipped busy default");
eq(c.pollCadence(0), 500, "idle when nothing in flight");
eq(c.pollCadence(3), 300, "busy when requests are in flight");
console.log("== stored values, and the old key migrating to a flat rate ==");
let m = cadence(JSON.stringify({ idle: 700, busy: 200 }));
eq(m.idle(), 700, "stored idle honored"); eq(m.busy(), 200, "stored busy honored");
m = cadence(JSON.stringify({ ms: 800, adaptive: true }));
eq(m.idle(), 800, "old {ms} migrates to idle"); eq(m.busy(), 800, "old {ms} migrates to a FLAT rate");
m = cadence("{ not json");
eq(m.idle(), 500, "corrupt storage falls back to the default");
eq(m.busy(), 300, "corrupt storage keeps the busy default");
eq(cadence(JSON.stringify({ idle: 50 })).idle(), 100, "a stored sub-floor value is clamped on read");
console.log("== the summary label never lies about what is set ==");
eq(c.sum(), "500 ms idle | 300 ms busy", "two values shown when they differ");
eq(cadence(JSON.stringify({ idle: 500, busy: 500 })).sum(), "500 ms", "one value shown when equal");

let lastD = null;
const M = {}, hist = {};
let nowS = 0;
eval(HIST_S_SRC + "\n" + SPARK);
const xs = (d) => (d.match(/[ML]([0-9.]+)/g) || []).map((s) => parseFloat(s.slice(1)));
console.log("== the sparkline's x axis is time, not sample index ==");
M.k = { line: { setAttribute: (n, v) => { if (n === "d") lastD = v; } } };
let pts = []; for (let i = 0; i < 60; i++) pts.push({ t: i, v: 50 + (i % 7) });
spark("k", pts); let X = xs(lastD);
eq(X.length, 60, "all 60 points drawn");
ok(Math.abs(X[59] - 200) < 0.06, "newest sample sits on the right edge (got " + X[59] + ")");
ok(Math.abs(X[0] - 200 * (1 - 59 / HS)) < 0.06, "oldest of a 59s span sits at its time position (got " + X[0] + ")");
pts = []; for (let i = 0; i < 10; i++) pts.push({ t: i * 0.1, v: 1 });
for (let i = 1; i <= 10; i++) pts.push({ t: 0.9 + i, v: 1 });
spark("k", pts); X = xs(lastD);
const ratio = (X[9] - X[0]) / (X[19] - X[0]);
const want = 0.9 / 10.9;
ok(Math.abs(ratio - want) < 0.01, "the busy stretch gets its 0.9s of the 10.9s span (got " + ratio.toFixed(4) + " want " + want.toFixed(4) + ")");
ok(Math.abs(ratio - 0.5) > 0.2, "and NOT the 10/20 index-spacing answer 0.5 (got " + ratio.toFixed(4) + ")");
console.log("== startup fills in from the right; a sample gap draws as a line gap ==");
spark("k", [{ t: 100, v: 1 }, { t: 101, v: 2 }, { t: 102, v: 3 }]); X = xs(lastD);
ok(X[0] > 190, "3 samples spanning 2s all sit near the right edge (leftmost x=" + X[0] + ")");
spark("k", [{ t: 0, v: 1 }, { t: 1, v: 2 }, { t: 50, v: 3 }]); X = xs(lastD);
ok(X[2] - X[1] > 10 * (X[1] - X[0]), "a 49s gap draws far wider than a 1s step");
spark("k", [{ t: 1, v: 1 }, { t: 2, v: 2 }]); eq(lastD, "", "2 points draw nothing");
console.log("== the prune is the SHIPPED setMetric, actually called ==");
function shippedPrune(seed, nowSVal, value) {
  const Mm = { m: { n: { firstChild: { nodeValue: "" } }, line: { setAttribute: () => {} } } };
  const hh = { m: seed.slice() };
  /* v is passed as a parameter, not interpolated into the source: interpolating the string
     "null" and the literal null produce the same text, so the null case would not be null. */
  new Function("M", "hist", "nowS", "spark", "HIST_S", "v", SETMETRIC + "\nsetMetric('m', v, 'x');\nreturn hist;")(Mm, hh, nowSVal, () => {}, HS, value);
  return hh.m;
}
let fast = []; for (let i = 0; i < 600; i++) fast.push({ t: i * 0.1, v: 1 });
let r = shippedPrune(fast, 60, 1);
eq(r.length, 601, HS + "s of 100ms samples: nothing pruned, plus the sample just recorded");
eq(r[r.length - 1].t, 60, "the new sample is stamped with nowS");
let slow = []; for (let i = 0; i < 600; i++) slow.push({ t: i * 5, v: 1 });
r = shippedPrune(slow, 3000, 1);
eq(r.length, Math.ceil(HS / 5) + 1, HS + "s of 5s samples keeps " + Math.ceil(HS / 5) + " plus the new one, not 601");
eq(r[0].t, 3000 - HS, "the oldest survivor is the first sample inside the window");
let edge = []; for (let i = 0; i < 5; i++) edge.push({ t: 1000 + i, v: 1 });
eq(shippedPrune(edge, 1000 + HS, 1).length, 6, "a sample exactly at nowS-HIST_S survives (exclusive <)");
eq(shippedPrune(edge, 1001 + HS, 1).length, 5, "one second later that same sample is pruned");
r = shippedPrune([{ t: 10, v: 1 }], 1005, null);
eq(r.length, 1, "a null reading records nothing and prunes nothing");
eq(r[0].t, 10, "the earlier sample is untouched");
console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped page JS");
process.exit(fails ? 1 : 0);
