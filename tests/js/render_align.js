/* MANUAL render check -- not part of the suite, and not wired into CMake.
 *
 * It loads the dashboard in headless Chrome, screenshots the Requests header at 4x, and finds
 * the lowest inked pixel row of the caption's letters and of the download glyph's tray bar.
 * That is the only way to answer "is the icon level with the text": the CSS says where the
 * BOX goes, and the box is not the glyph.
 *
 * Needs puppeteer + a Chrome binary, which are NOT repo dependencies:
 *   npm i puppeteer && npx puppeteer browsers install chrome
 * Then, with the page served (the viewer on :8900, or production on :8800):
 *   node tests/js/render_align.js [url]
 * Exits nonzero if the two bottoms are not on the same device pixel row.
 *
 * Why this exists: the separator and glyph work went through three commits that each asserted
 * an alignment from the geometry -- "13px per boundary", "flush with the viewBox bottom, so it
 * sits on the baseline" -- and both claims were false. The first collapsed to zero width in
 * HTML; the second aligned the SVG box to the baseline while the baseline sits above the box
 * bottom, leaving the bar 2px high. A unit test cannot see either, because neither is about the
 * string the code builds. Measure the pixels.
 */
const puppeteer = require("puppeteer");
const URL0 = process.argv[2] || "http://127.0.0.1:8900/";
(async () => {
  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--font-render-hinting=none"],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900, deviceScaleFactor: 4 });
  await page.goto(URL0, { waitUntil: "networkidle2", timeout: 30000 });
  await page.waitForSelector("#reqjson", { timeout: 10000 });

  // the Requests caption and the download glyph, as laid out on the real page
  const boxes = await page.evaluate(() => {
    const btn = document.getElementById("reqjson");
    const cap = btn.closest(".pleft").querySelector(".cap");
    const svg = btn.querySelector("svg");
    const cs = getComputedStyle(cap);
    const r = (e) => { const b = e.getBoundingClientRect(); return {x:b.x,y:b.y,w:b.width,h:b.height,bottom:b.bottom}; };
    return { cap: r(cap), svg: r(svg), btn: r(btn), font: cs.fontFamily, size: cs.fontSize,
             align: getComputedStyle(btn.closest(".pleft")).alignItems };
  });
  console.log("cap  ", JSON.stringify(boxes.cap));
  console.log("svg  ", JSON.stringify(boxes.svg));
  console.log("align-items:", boxes.align, " font:", boxes.size, boxes.font.slice(0, 40));

  // screenshot the strip containing both, then decode IN THE BROWSER and find ink rows
  const x0 = Math.floor(boxes.cap.x) - 2, x1 = Math.ceil(boxes.svg.x + boxes.svg.w) + 2;
  const y0 = Math.floor(Math.min(boxes.cap.y, boxes.svg.y)) - 4;
  const y1 = Math.ceil(Math.max(boxes.cap.bottom, boxes.svg.bottom)) + 6;
  const shot = await page.screenshot({ encoding: "base64",
    clip: { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } });

  const res = await page.evaluate(async (b64, capX, svgX, x0, y0, scale) => {
    const img = new Image();
    img.src = "data:image/png;base64," + b64;
    await img.decode();
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    // ink = pixel meaningfully brighter than the near-black background
    const rowInk = (xFrom, xTo) => {
      const xa = Math.max(0, Math.round((xFrom - x0) * scale));
      const xb = Math.min(c.width,  Math.round((xTo   - x0) * scale));
      const rows = [];
      for (let y = 0; y < c.height; y++) {
        let mx = 0;
        for (let x = xa; x < xb; x++) {
          const i = (y * c.width + x) * 4;
          mx = Math.max(mx, d[i], d[i + 1], d[i + 2]);
        }
        rows.push(mx);
      }
      return rows;
    };
    const lastInk = (rows, thr) => { for (let y = rows.length - 1; y >= 0; y--) if (rows[y] > thr) return y; return -1; };
    const firstInk = (rows, thr) => { for (let y = 0; y < rows.length; y++) if (rows[y] > thr) return y; return -1; };
    const capRows = rowInk(capX.x, capX.x + capX.w);
    const svgRows = rowInk(svgX.x, svgX.x + svgX.w);
    const T = 60;   // background is ~#0a0a0a; text/icon are #7b7b7b / #4d4d4d
    return {
      imgH: c.height, scale,
      capFirst: firstInk(capRows, T), capLast: lastInk(capRows, T),
      svgFirst: firstInk(svgRows, T), svgLast: lastInk(svgRows, T),
      peakCap: Math.max(...capRows), peakSvg: Math.max(...svgRows),
    };
  }, shot, boxes.cap, boxes.svg, x0, y0, 4);

  console.log("url:", URL0);
  console.log("decoded:", JSON.stringify(res));
  if (res.capLast < 0 || res.svgLast < 0) { console.log("MEASUREMENT FAILED: no ink found in a region"); }
  else {
    const gapDevice = res.svgLast - res.capLast;
    const gapCss = gapDevice / 4;
    console.log("\n  bottom of REQUESTS ink  : device row " + res.capLast);
    console.log("  bottom of tray-bar ink  : device row " + res.svgLast);
    console.log("  GAP = " + gapDevice + " device px = " + gapCss.toFixed(2) + " CSS px");
    console.log("  " + (gapCss === 0 ? "PERFECTLY FLUSH" :
      (gapCss > 0 ? "icon sits " + gapCss.toFixed(2) + "px BELOW the letters"
                  : "icon sits " + Math.abs(gapCss).toFixed(2) + "px ABOVE the letters")));
  }
  const flush = res.capLast >= 0 && res.svgLast >= 0 && res.svgLast === res.capLast;
  console.log(flush ? "\nPASS: tray bar and caption bottoms share a device row"
                    : "\nFAIL: not level (see GAP above)");
  await browser.close();
  process.exit(flush ? 0 : 1);
})().catch((e) => { console.log("ERROR: " + e.message); process.exit(1); });
