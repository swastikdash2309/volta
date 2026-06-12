/*
 verify_site.js
 --------------
 Headless zero-error check for the dashboard. Loads volta_dashboard.html into a
 real DOM (jsdom), stubs the canvas 2D context and fetch, runs the page script,
 then drives every interaction (mode toggle, privacy toggle, scrub, play, hover)
 and asserts that nothing throws. Run:  node verify_site.js
*/
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const html = fs.readFileSync(path.join(__dirname, "volta_dashboard.html"), "utf8");
const errors = [];

const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;
const document = window.document;

// ---- stubs that a browser provides but jsdom does not ----
window.devicePixelRatio = 1;
window.fetch = () => Promise.reject(new Error("offline"));   // forces static mode path
const fakeCtx = new Proxy({}, {
  get(_t, prop) {
    if (prop === "createLinearGradient") return () => ({ addColorStop() {} });
    if (prop === "measureText") return () => ({ width: 0 });
    if (prop === "canvas") return { width: 700, height: 240 };
    return () => {};
  },
  set() { return true; },
});
window.HTMLCanvasElement.prototype.getContext = () => fakeCtx;
window.HTMLCanvasElement.prototype.getBoundingClientRect =
  () => ({ left: 0, top: 0, width: 700, height: 240, right: 700, bottom: 240 });
// bound rAF so the render loop runs a handful of frames then stops
let frames = 0;
window.requestAnimationFrame = (cb) => { if (frames++ < 8) setTimeout(() => cb(Date.now()), 0); return frames; };

window.addEventListener("error", (e) => errors.push("window error: " + (e.error && e.error.stack || e.message)));

// ---- run the page script ----
const script = html.split("<script>")[1].split("</script>")[0];
try { window.eval(script); } catch (e) { errors.push("script load: " + e.stack); }

function fire(desc, fn) { try { fn(); } catch (e) { errors.push(desc + ": " + (e.stack || e)); } }
const $ = (id) => document.getElementById(id);
const click = (el) => el && el.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));

// ---- drive every interaction ----
setTimeout(() => {
  fire("toggle VOLTA->Baseline", () => click(document.querySelector('.vseg button[data-m="0"]')));
  fire("toggle Baseline->VOLTA", () => click(document.querySelector('.vseg button[data-m="1"]')));
  fire("privacy Standard->Private", () => click(document.querySelector('.ptog button[data-p="1"]')));
  fire("privacy Private->Standard", () => click(document.querySelector('.ptog button[data-p="0"]')));
  fire("play", () => click($("play")));
  fire("scrub slider", () => { const s = $("sl"); s.value = "24"; s.dispatchEvent(new window.Event("input", { bubbles: true })); });
  fire("hover chart", () => $("cv").dispatchEvent(new window.MouseEvent("mousemove", { clientX: 300, clientY: 60, bubbles: true })));
  fire("leave chart", () => $("cv").dispatchEvent(new window.MouseEvent("mouseleave", { bubbles: true })));
  fire("resize", () => window.dispatchEvent(new window.Event("resize")));

  // structural sanity
  const need = ["seg", "kpis", "cv", "play", "sl", "cars", "ptog"];
  need.forEach((id) => { if (!$(id)) errors.push("missing element #" + id); });
  if (document.querySelectorAll(".kpi").length !== 4) errors.push("expected 4 KPI cards");
  if (document.querySelectorAll(".vbar").length !== 12) errors.push("expected 12 vehicle bars");

  setTimeout(() => {
    if (errors.length) { console.log("FAIL: " + errors.length + " issue(s)\n" + errors.join("\n")); process.exit(1); }
    console.log("PASS: dashboard ran with zero runtime errors across all interactions.");
    console.log("  verified: mode toggle, privacy toggle, play, scrub, hover, leave, resize, structure.");
    process.exit(0);
  }, 80);
}, 60);
