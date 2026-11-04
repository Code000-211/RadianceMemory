# Dashboard page tests

The dashboard is one C++ string: `core/server/dashboard.cpp` embeds the whole page,
so its JavaScript has no file of its own to test. These harnesses extract the page
from the translation unit (`page.js`) and **execute the shipped functions**, sliced
by line range, against mocked DOM. A retyped copy would test the copy, not the bytes
that ship, and each harness calls the functions it slices: one that is only defined proves
nothing.

    node tests/js/poll_spark.test.js
    node tests/js/extremes.test.js
    node tests/js/history.test.js
    node tests/js/download.test.js

## What each one covers

| file | the shipped code it drives |
| --- | --- |
| `poll_spark.test.js` | the cadence fields, the sparkline |
| `extremes.test.js` | `noteSample`, `triple`, `acceptTriple`, `resetWindows`, `newEngineRun` |
| `history.test.js` | the history summary line: how it groups, and how it wraps |
| `download.test.js` | the Requests-header JSON export: its markup and its handler |

Each asserts structure, not just text. `history.test.js` inspects the DOM the page builds and the
CSS it ships, because the rules it guards -- a category wraps whole, the category break is a dot
and the break within one is a pipe -- are not properties of any string the code builds.
`download.test.js` invokes the click handler against stubs and checks it really builds a file,
names it, clicks it and releases the blob URL: a button that looks clickable and does nothing is
the failure it exists to catch.

## The render check (manual, needs Chrome)

`render_align.js` is not a unit test and cannot be: it answers whether the download glyph is
level with the caption by loading the page in headless Chrome, screenshotting the header at 4x,
and comparing the lowest inked pixel row of each. It needs puppeteer and a Chrome binary, which
are deliberately not repo dependencies:

    npm i puppeteer && npx puppeteer browsers install chrome
    node tests/js/render_align.js http://127.0.0.1:8800/

Exits nonzero when the two bottoms land on different device rows. Run it after touching glyph or
header geometry: alignment cannot be read off the CSS and the SVG viewBox (collapsed whitespace can
make a separator zero pixels wide, and an SVG flush with the baseline still floats, because the
baseline sits above the box bottom), so measure the rendered page.

## Why these are not in ctest

They need node, which is not a build dependency of this project and will not be
present on a contributor's box or in the ROCm build image. Wiring them into ctest
would turn a missing optional tool into a red build, so they stay opt-in and are run
by hand when the page changes. `dashboard_test` covers the C++ side only.
