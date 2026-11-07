/* Extract the embedded dashboard page's <script> from core/server/dashboard.cpp.
   The page lives inside a C++ raw string, so a JS test has to pull it out of the
   translation unit it ships in. Reading the real file is the point: a copy of the
   page would test the copy. */
const fs = require("fs");
const path = require("path");
const cpp = fs.readFileSync(
  path.join(__dirname, "..", "..", "core", "server", "dashboard.cpp"),
  "utf8",
);
const page = cpp.match(/R"HTML\(([\s\S]*)\)HTML"/);
if (!page) throw new Error("no R\"HTML(...)HTML\" page block in dashboard.cpp");
const script = page[1].match(/<script>([\s\S]*)<\/script>/);
if (!script) throw new Error("no <script> block in the dashboard page");
module.exports = script[1];
