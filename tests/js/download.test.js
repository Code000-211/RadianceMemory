/* The Requests-header JSON export. It used to be a "json" text button in the history row that
 * copied to the clipboard; it is now a download icon that saves a file. A button that looks
 * clickable and silently does nothing is the failure this guards: so drive the SHIPPED handler
 * against stubs and assert it actually builds a file, names it, clicks it, and cleans up. */
const fs = require("fs");
const path = require("path");
const src = require("./page.js");           // the <script> block: the handler lives here
/* The MARKUP lives in the page body, which page.js does not return (the JS suites have no
   reason to read it). Read the whole shipped page for markup assertions. */
const cpp = fs.readFileSync(path.join(__dirname, "..", "..", "core", "server", "dashboard.cpp"), "utf8");
const page = (cpp.match(/R"HTML\(([\s\S]*)\)HTML"/) || [, ""])[1];
const css = (cpp.match(/<style>([\s\S]*?)<\/style>/) || [, ""])[1];
const L = src.split("\n");
let fails = 0, checks = 0;
const ok = (c, m) => { checks++; if (!c) { fails++; console.log("  FAIL " + m); } };
const eq = (a, b, m) => { checks++; if (a !== b) { fails++; console.log("  FAIL " + m + ": got " + JSON.stringify(a) + " want " + JSON.stringify(b)); } };

console.log("== the export lives in the Requests header, not the history row ==");
ok(/id="reqjson"/.test(page), "a #reqjson control exists");
ok(!/id="hjson"/.test(page), "the old #hjson button in the history row is gone");
/* The button spans several source lines, so match the whole element, not one line. */
const btn = (page.match(/<button[^>]*id="reqjson"[\s\S]*?<\/button>/) || [""])[0];
ok(btn.length > 0, "found the reqjson button element");
ok(/title="Download JSON file"/.test(btn), "the tooltip says 'Download JSON file'");
ok(/aria-label="Download JSON file"/.test(btn), "and is labelled for assistive tech");
ok(/<svg[\s\S]*<\/svg>/.test(btn), "it holds an icon, not a text label");
ok(!/>\s*json\s*</.test(btn), "no leftover 'json' text label inside the button");
/* Two things put the tray bar level with the bottom of the caption's letters, and BOTH are
   load-bearing -- measured in headless Chrome, not reasoned out:

   1. the glyph fills its box: the bar's STROKE bottom lands on the viewBox bottom, so there is
      no dead space inside the SVG. An inset bar (the original sat at 13.5 of 16) shifts the
      bar up and the nudge below would then be the wrong size.
   2. the glyph is nudged down 2px: align-items:baseline puts the SVG's BOX on the text
      baseline, but the baseline sits ABOVE the box bottom, so even a box-filling glyph floated
      ~2px high. The nudge is what actually closes the gap.

   The first version of this guard checked only (1) and asserted it "sits on the baseline" --
   that claim was false, and the passing test is why the float survived a commit. */
const vb = +(btn.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/) || [, , 0])[2];
const sw = +(btn.match(/stroke-width="([\d.]+)"/) || [, 0])[1];
const bar = +(btn.match(/M[\d.]+ ([\d.]+)h/) || [, 0])[1];
ok(vb > 0 && sw > 0 && bar > 0, "parsed the glyph geometry (viewBox " + vb + ", stroke " + sw + ", bar " + bar + ")");
eq(Math.round((bar + sw / 2) * 1000) / 1000, vb,
   "the glyph fills its box: the tray bar's stroke bottom is flush with the viewBox bottom");
/* The nudge that actually delivers the alignment. Pinned because it is a measured value, not a
   taste value: at 2px the bar's bottom and the caption's bottom land on the same device row. */
const nudge = css.match(/#reqjson svg \{[^}]*transform:\s*translateY\(([\d.]+)px\)/);
ok(nudge, "the download glyph carries an explicit vertical nudge");
eq(nudge ? +nudge[1] : NaN, 2, "the nudge is the measured 2px (removing it floats the bar ~2px high)");
/* It must sit in the Requests header, in the LEFT group, before the right-hand aside. Index
   ordering is sturdier than a regex across the whole block. */
const iLeft = page.indexOf('class="pleft"');
const iBtn  = page.indexOf('id="reqjson"');
const iAside= page.indexOf('id="reqaside"');
ok(iLeft >= 0 && iBtn > iLeft, "reqjson comes after the phead left-group opens");
ok(iAside > iBtn, "reqjson comes before the right-hand aside (so it sits on the LEFT)");
/* Exactly one: a duplicate id is invalid HTML and getElementById binds only the first, so a
   second button would render and do nothing -- the silent-dead-control failure itself. */
const nBtn = (page.match(/id="reqjson"/g) || []).length;
eq(nBtn, 1, "exactly one reqjson control (a duplicate id would leave one dead)");
/* And it is inside the Requests panel, not some other phead: the phead that opens with
   class="pleft" is the Requests one (only that header gained the group). */
const iCap = page.indexOf('<span class="cap">Requests</span>');
ok(iCap >= 0, "the Requests caption exists (indexOf -1 would make the order check vacuous)");
ok(iCap >= 0 && iCap < iBtn, "the button is in the Requests header, after its title");

console.log("== the shipped handler builds a real download ==");
const si = L.findIndex((x) => x.startsWith('$("reqjson").addEventListener'));
if (si < 0) throw new Error("no reqjson handler");
let ei = -1;
for (let k = si; k < L.length; k++) if (L[k] === "});") { ei = k; break; }
if (ei < 0) throw new Error("no handler end");
const HANDLER = L.slice(si, ei + 1).join("\n");

/* Stubs: capture the click handler, then invoke it and watch what it builds. */
let clickHandler = null;
const seq = [];   /* one ordered log: the ORDER of DOM ops is the bug class, not the counts */
const fakeEl = (tag) => ({ tag, href: "", download: "", clicked: 0, removed: 0,
  click() { this.clicked++; seq.push("click"); }, remove() { this.removed++; seq.push("remove"); } });
const anchors = [];
const el = (t) => { const a = fakeEl(t); anchors.push(a); return a; };
const body = { appended: [], appendChild(n) { this.appended.push(n); seq.push("append"); } };
const created = [];
const URL = { createObjectURL(b) { const u = "blob:fake/" + created.length; created.push({ b, u }); return u; },
              revoked: [], revokeObjectURL(u) { this.revoked.push(u); seq.push("revoke"); } };
class Blob { constructor(parts, opts) { this.parts = parts; this.type = opts && opts.type; } }
let timeoutFn = null;
const setTimeout = (fn, ms) => { timeoutFn = fn; return 1; };
const hRows = [{ tps: 208, pf: 651, ttft: 796 }, { tps: 190 }];

const $ = (id) => ({ addEventListener(ev, fn) { if (ev === "click") clickHandler = fn; } });
new Function("$", "el", "document", "URL", "Blob", "setTimeout", "hRows", HANDLER)(
  $, el, { body }, URL, Blob, setTimeout, hRows);

ok(clickHandler !== null, "a click handler was registered on #reqjson");
clickHandler();

eq(anchors.length, 1, "exactly one anchor built");
const a = anchors[0];
eq(a.tag, "a", "it is an anchor");
eq(a.download, "radiance-history.json", "it names a downloadable file");
ok(a.href.startsWith("blob:"), "href is a blob URL: " + a.href);
eq(a.clicked, 1, "the anchor was clicked (that is what triggers the save)");
eq(body.appended.length, 1, "the anchor was attached to the document (Firefox needs it in the DOM)");
eq(a.removed, 1, "the anchor was removed afterwards");
/* Order, not just occurrence: Firefox ignores a click on an anchor that is not yet in the DOM, so
   appendChild MUST precede click, and remove must follow it. Counts alone let a reordered handler
   pass while the button is dead in the browser. */
const iApp = seq.indexOf("append"), iClick = seq.indexOf("click"), iRem = seq.indexOf("remove");
ok(iApp >= 0 && iClick >= 0 && iApp < iClick,
   "the anchor is in the DOM BEFORE it is clicked (Firefox needs it): " + JSON.stringify(seq));
ok(iClick >= 0 && iRem > iClick, "the anchor is removed only AFTER the click: " + JSON.stringify(seq));
eq(created[0].b.type, "application/json", "the blob is typed application/json");
const payload = JSON.parse(created[0].b.parts[0]);
ok(Array.isArray(payload) && payload.length === 2, "the file holds the stored rows as JSON");
eq(payload[0].tps, 208, "the row data survived serialization");
ok(created[0].b.parts[0].includes("\n"), "pretty-printed (indented), not one dense line");

console.log("== cleanup: the blob URL is revoked, not leaked ==");
eq(URL.revoked.length, 0, "not revoked synchronously (the click must finish first)");
ok(typeof timeoutFn === "function", "a revoke was scheduled");
timeoutFn();
eq(URL.revoked.length, 1, "the revoke ran and released the blob URL");
eq(URL.revoked[0], a.href, "and revoked exactly the URL it handed the anchor");

console.log(fails ? "\n" + fails + " FAILURE(S) of " + checks : "\nALL " + checks + " CHECKS PASS on the shipped download");
process.exit(fails ? 1 : 0);
