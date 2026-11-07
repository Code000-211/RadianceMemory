/* Session extremes: the age clause, and the gap/restart split. Executes the shipped
   noteSample / triple / acceptTriple / resetWindows / newEngineRun / poll, sliced by line
   range out of the page and CALLED -- defining a sliced function proves nothing.
   pctn/pct are the SHIPPED formatters rather than a re-typed stand-in: a copy of a formatter
   drifts from the real one, and an assertion on its output then silently means nothing.
   nowS is declared here because in the page it is module-level, so it is invisible inside
   new Function. */
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
const COMMON = [
  "let nowS = 0;",
  'const DASH = "\\u2013";',
  "const finite = (v) => v != null && isFinite(v);",
  /* resetWindows() clears these; they are not under test but it READS them. */
  "const hist = {}, moveHist = { up: [], down: [] };",
  "let prev = null, prevAt = 0, accPrev = null;",
  lines("function pctn(", "}"),
  lines("const pct =", "const pct ="),
  lines("function dur(", "}"),
  lines("const RECENT_S =", "const RECENT_PREFILL_S"),
  lines("const extAt =", "let lastActiveC"),
  lines("function noteSample(", "}"),
  lines("function triple(", "}"),
  lines("function acceptTriple(", "}"),
  lines("const accPool =", "const accPool ="),
  lines("function resetWindows", "}"),
  lines("function newEngineRun", "}"),
].join("\n");

function fresh() {
  const box = {};
  new Function("box", COMMON + "\nbox.api={noteSample,triple,acceptTriple,resetWindows,newEngineRun,extAt,winAt,extAtS,accPool,setNow:(v)=>{nowS=v;}};")(box);
  return box.api;
}
const F = (v) => v + " t/s";

async function main() {
console.log("== the age clause appears only when the extreme stands alone ==");
let b = fresh();
b.setNow(100); b.noteSample("dec", 1, 500, 15, true);
let out = b.triple("dec", 1, "best", F);
ok(!out.includes("ago"), "window full: no age clause (got " + JSON.stringify(out) + ")");
ok(out.includes("recent"), "and the window figures still render");
b.resetWindows(); b.setNow(150);
out = b.triple("dec", 1, "best", F);
ok(/best 500 t\/s \| 50\.0 s ago$/.test(out), "extreme alone: age names the span (got " + JSON.stringify(out) + ")");
b.setNow(100.4);
ok(!b.triple("dec", 1, "best", F).includes("ago"), "an extreme set 0.4s ago is current: no clause");

console.log("== a stale extreme is distinguishable from a fresh one ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, false); b.setNow(1000);
out = b.triple("dec", 1, "best", F);
ok(out.includes("15m 00s ago"), "15-minute-old extreme says so (got " + JSON.stringify(out) + ")");
ok(out.includes("best 500"), "and still shows the figure itself");

console.log("== the stamp marks when the EXTREME was set, not when anything was sampled ==");
/* Re-sampling a key with a WORSE value must not refresh the stamp: the age belongs to the
   figure on the left. Refreshing it on every sample makes every age read ~0 -- wrong output,
   not missing output, which is the hardest kind of defect for a label to survive. */
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, false);
b.setNow(200); b.noteSample("dec", 1, 400, 15, false);   /* not a new extreme */
b.setNow(260);
out = b.triple("dec", 1, "best", F);
ok(out.includes("best 500"), "the extreme is still the earlier, better figure (got " + JSON.stringify(out) + ")");
ok(out.includes("2m 40s ago"), "age is measured from the extreme, not the last sample (got " + JSON.stringify(out) + ")");
b.setNow(300); b.noteSample("dec", 1, 900, 15, false);   /* a NEW extreme */
b.setNow(330);
out = b.triple("dec", 1, "best", F);
ok(out.includes("best 900") && out.includes("30.0 s ago"), "a new extreme restamps (got " + JSON.stringify(out) + ")");

console.log("== acceptTriple's age clause is exercised on a LIVE extreme ==");
/* The restart case below clears extAt first, so acceptTriple returns at its guard and never
   reaches the clause. Nothing else drove it, so the clause could be deleted unnoticed. */
b = fresh(); b.setNow(100); b.noteSample("accept", 1, 0.75, 15, false);
b.setNow(100.5);
ok(!b.acceptTriple(1).includes("ago"), "an acceptance extreme set 0.5s ago is current: no clause (got " + JSON.stringify(b.acceptTriple(1)) + ")");
b.setNow(400);
out = b.acceptTriple(1);
ok(/^highest 75% \| 5m 00s ago$/.test(out), "once it stands alone it names its age (got " + JSON.stringify(out) + ")");
b.accPool[1] = [{ dr: 10, da: 8 }, { dr: 10, da: 6 }];
out = b.acceptTriple(1);
ok(out.includes("recent") && !out.includes("ago"), "with a ring: window figures, no age clause (got " + JSON.stringify(out) + ")");

console.log("== a GAP keeps the extreme, and says how old it is ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, true); b.noteSample("accept", 1, 0.9, 15, false);
b.setNow(400); b.resetWindows();
out = b.triple("dec", 1, "best", F);
ok(out.includes("best 500"), "the session extreme survives a gap (got " + JSON.stringify(out) + ")");
ok(out.includes("5m 00s ago"), "and names its age (got " + JSON.stringify(out) + ")");
eq(Object.keys(b.extAt).length, 2, "extAt untouched by a gap");

console.log("== a RESTART drops the extremes: a new process inherits nothing ==");
b = fresh(); b.setNow(100); b.noteSample("dec", 1, 500, 15, true); b.noteSample("accept", 1, 0.9, 15, false);
b.setNow(400); b.newEngineRun();
eq(Object.keys(b.extAt).length, 0, "extAt cleared by newEngineRun");
eq(Object.keys(b.extAtS).length, 0, "extAtS cleared by newEngineRun");
eq(b.triple("dec", 1, "best", F), "", "the previous run's extreme no longer renders");
eq(b.acceptTriple(1), "", "the previous run's acceptance extreme no longer renders");
b.setNow(200); b.noteSample("dec", 1, 700, 15, false);
out = b.triple("dec", 1, "best", F);
ok(!out.includes("ago"), "a fresh extreme carries no age clause (got " + JSON.stringify(out) + ")");
ok(/best 700 t\/s$/.test(out), "and reads as the new run's own figure");

/* ------------------------------------------------------------------ poll() itself
 * The restart DETECTION lives in poll(), not in newEngineRun(). Calling newEngineRun() directly
 * proves only that the function clears things -- deleting the detection line in poll() would
 * leave every such assertion green. So drive poll() with a scripted /stats sequence and assert
 * what it decides. resetWindows / newEngineRun / update are replaced with counters here on
 * purpose: this is testing the decision, not the clearing (covered above). */
const POLL = lines("async function poll() {", "}");
function makePoller(seq, clock) {
  const body = COMMON + "\n" + [
    "let lastPollAt = 0;",
    "const GAP_S = 60; let pollIdleMs = 500;",
    "const calls = { run: 0, gap: 0, update: 0 };",
    "function resetWindows() { calls.gap++; }",
    "function newEngineRun() { calls.run++; }",
    "function update() { calls.update++; }",
    "function schedulePoll() {}",
    "function pollCadence() { return 500; }",
    "const els = {};",
    "function $(id) { return els[id] || (els[id] = { hidden: false, className: \"\", classList: { remove() {} } }); }",
    "let i = 0;",
    "async function api() { const s = seq[Math.min(i++, seq.length - 1)]; return { ok: true, json: async () => JSON.parse(JSON.stringify(s)) }; }",
    POLL,
    "return { poll, calls };",
  ].join("\n");
  return new Function("document", "performance", "Date", "seq", "clock", body)(
    { hidden: false }, { now: () => 0 }, { now: () => clock.wall }, seq, clock);
}
const fixed = { wall: 1000 };   /* a fixed clock keeps the GAP path out of these assertions */
console.log("== poll() detects a new engine process ==");
let p = makePoller([
  { started_at: 100, uptime_s: 10 },  /* first poll: baseline, clears nothing */
  { started_at: 100, uptime_s: 20 },  /* same process, uptime rising: nothing happens */
  { started_at: 200, uptime_s: 5 },   /* started_at moved: a new process */
], fixed);
await p.poll(); eq(p.calls.run, 0, "the first poll is a baseline, not a restart");
await p.poll(); eq(p.calls.run, 0, "same started_at with a rising uptime is the same process");
await p.poll(); eq(p.calls.run, 1, "a changed started_at clears the extremes");

console.log("== uptime_s is derived, so it is not a restart signal ==");
/* uptime_s is unix_now() - started_at_, both whole seconds, so started_at + uptime_s is exactly
   the wall clock at that poll. A later poll of the SAME process therefore always reports a
   larger uptime; a smaller one means the wall clock moved backwards, which is an NTP adjustment
   and not a new process. Clearing on it would throw away the session's extremes on a clock
   correction, so nothing here treats it as a restart. */
p = makePoller([
  { started_at: 100, uptime_s: 40 },
  { started_at: 100, uptime_s: 2 },   /* same started_at, uptime went backwards */
], fixed);
await p.poll(); await p.poll();
eq(p.calls.run, 0, "a backward uptime with an unchanged started_at is a clock step, not a restart");

console.log("== poll() takes the GAP path, not the restart path, when only the clock moved ==");
const slow = { wall: 1000 };
p = makePoller([{ started_at: 100, uptime_s: 10 }, { started_at: 100, uptime_s: 130 }], slow);
await p.poll();
slow.wall = 1000 + 90 * 1000;         /* 90s with nothing polled: a gap */
await p.poll();
eq(p.calls.gap, 1, "a long quiet span clears the windows");
eq(p.calls.run, 0, "and is NOT read as a restart: the extremes survive it");
}
main().then(() => {
  console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped extreme code");
  process.exit(fails ? 1 : 0);
});
