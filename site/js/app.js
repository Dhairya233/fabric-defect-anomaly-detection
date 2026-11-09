(() => {
"use strict";

const NS = "http://www.w3.org/2000/svg";
const METHODS = [
  ["contrast", "Raw contrast", "Baseline"],
  ["brightness", "Brightness deviation", "Baseline"],
  ["rf", "Random forest", "Supervised"],
  ["svm", "SVM (RBF)", "Supervised"],
  ["maha_global", "Mahalanobis, global", "Anomaly"],
  ["maha_fabric", "Mahalanobis, per fabric", "Anomaly"],
  ["iforest", "Isolation forest", "Anomaly"],
  ["ocsvm", "One-class SVM", "Anomaly"],
];
const LABEL = Object.fromEntries(METHODS.map(([k, n]) => [k, n]));
const SPLITS = [
  ["grouped", "Grouped by strip", "circle"],
  ["lofo", "Whole fabric held out", "diamond"],
  ["leaky", "Leaky patch split", "square"],
];
const METRICS = {
  image_macro_fabric_auroc: {
    name: "Strip AUROC, per fabric", chance: 0.5, domain: [0.2, 1], ticks: [0.2, 0.4, 0.6, 0.8, 1],
    help: "Can the method tell defective strips from clean ones of the same fabric? AUROC is the chance that a random defective strip scores higher than a random clean one: 0.5 is guessing, 1 is perfect. It is computed for each fabric separately, then averaged over the six fabrics with at least five defective strips.",
  },
  image_auroc: {
    name: "Strip AUROC, pooled", chance: 0.5, domain: [0.2, 1], ticks: [0.2, 0.4, 0.6, 0.8, 1],
    help: "The same AUROC with all 245 strips in one pool. This partly rewards recognising which fabric a strip is, because most defective strips come from two fabrics, so prefer the per-fabric version.",
  },
  patch_ap: {
    name: "Patch precision", chance: null, domain: [0, 0.8], ticks: [0, 0.2, 0.4, 0.6, 0.8],
    help: "Average precision over individual 256-pixel patches: when the method ranks patches by suspicion, how many of the top ones really contain a defect? The dashed line is what random ranking scores, which equals the share of defective patches.",
  },
};
const TABLE_METRICS = [
  ["image_macro_fabric_auroc", "Strip AUROC, per fabric"],
  ["image_auroc", "Strip AUROC, pooled"],
  ["image_ap", "Strip average precision"],
  ["image_f1", "Strip F1 at chosen threshold"],
  ["patch_ap", "Patch average precision"],
  ["pixel_auroc", "Pixel AUROC"],
  ["pixel_iou", "Pixel overlap (IoU)"],
];
const TABLE_HELP = {
  image_f1: "F1 of exactly 0.600 is what flagging every strip scores (105 of 245 are defective), so a method at 0.60 is doing nothing useful at its threshold.",
  pixel_auroc: "Capped by design: scores exist per patch, so no method can locate a defect more finely than patch resolution.",
  pixel_iou: "Below 0.05 for every method. This reflects patch resolution, not model quality.",
  image_macro_fabric_auroc: METRICS.image_macro_fabric_auroc.help,
  image_auroc: METRICS.image_auroc.help,
  patch_ap: METRICS.patch_ap.help,
  image_ap: "Average precision when ranking strips by their highest patch score.",
};
const READ_AS = {
  contrast: "Tracks fabric and margins, not defects.",
  brightness: "Flags nearly every strip in some fabrics.",
  rf: "Strong in-fabric, chance on a new fabric.",
  svm: "Same pattern as the forest, lower in-fabric.",
  maha_global: "Fabric differences look like defects.",
  maha_fabric: "Needs the fabric known in advance.",
  iforest: "Below chance on unseen fabrics.",
  ocsvm: "Near chance on the cleanest fabrics.",
};
const CIVIDIS = ["#00224E", "#123570", "#3B496C", "#575C6D", "#707173", "#8A8779", "#A69D75", "#C4B56C", "#E3CB5C", "#FEE838"];

const $ = (s, r = document) => r.querySelector(s);
const el = (tag, attrs = {}, kids = []) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v; else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) n.append(c);
  return n;
};
const sv = (tag, attrs = {}, kids = []) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  for (const c of [].concat(kids)) n.append(c);
  return n;
};
const f2 = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(2));
const f3 = (x) => (x === null || x === undefined ? "n/a" : x.toFixed(3));
const signed = (x) => (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(2);
const ci = (c) => (c ? `${f2(c[0])} [${f2(c[1])}, ${f2(c[2])}]` : null);
const colour = (v) => {
  const t = Math.min(1, Math.max(0, (v + 100) / 200)) * (CIVIDIS.length - 1);
  const i = Math.min(CIVIDIS.length - 2, Math.floor(t)), u = t - i;
  const a = CIVIDIS[i], b = CIVIDIS[i + 1];
  const mix = (o) => Math.round(parseInt(a.slice(o, o + 2), 16) * (1 - u) + parseInt(b.slice(o, o + 2), 16) * u);
  return `rgb(${mix(1)},${mix(3)},${mix(5)})`;
};

const state = { results: null, strips: null, run: "masked", metric: "image_macro_fabric_auroc", method: "rf", shown: 20, table: false };

/* ---------- loading ---------- */
async function load() {
  const targets = ["#leak-chart", "#strip-list", "#tbl-out"];
  targets.forEach((t) => { const n = $(t); n.replaceChildren(el("div", { class: "state", role: "status" }, [el("p", { text: "Loading results\u2026" })])); });
  try {
    const [r, s] = await Promise.all(["data/results.json", "data/strips.json"].map(async (u) => {
      const resp = await fetch(u);
      if (!resp.ok) throw new Error(`${u}: ${resp.status}`);
      return resp.json();
    }));
    state.results = r; state.strips = s;
  } catch (err) {
    const msg = location.protocol === "file:"
      ? "The page needs to be served over HTTP. From the repository root run: python -m http.server -d site 8000, then open http://localhost:8000."
      : "The results could not be loaded.";
    targets.forEach((t) => {
      const btn = el("button", { type: "button", text: "Try again" });
      btn.addEventListener("click", load);
      $(t).replaceChildren(el("div", { class: "state", role: "alert" }, [el("p", { text: msg }), el("p", { class: "muted", text: String(err.message) }), btn]));
    });
    return;
  }
  init();
}

/* ---------- leakage chart ---------- */
function marker(shape, x, y, r, c, filled) {
  const common = { fill: filled ? c : "#FBF8F1", stroke: c, "stroke-width": 2 };
  if (shape === "circle") return sv("circle", { cx: x, cy: y, r, ...common });
  if (shape === "diamond") return sv("path", { d: `M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z`, ...common });
  return sv("rect", { x: x - r + 0.5, y: y - r + 0.5, width: 2 * r - 1, height: 2 * r - 1, ...common });
}
const SPLIT_STYLE = { grouped: ["#24507A", true], lofo: ["#A23B2A", true], leaky: ["#4A5560", false] };

function drawLegend() {
  const ul = $("#leak-legend");
  ul.replaceChildren(...SPLITS.map(([k, name, shape]) => {
    const s = sv("svg", { width: 18, height: 18, "aria-hidden": "true" }, [marker(shape, 9, 9, 5, SPLIT_STYLE[k][0], SPLIT_STYLE[k][1])]);
    return el("li", {}, [s, el("span", { text: name })]);
  }));
}

function metricRow(m, split) {
  const run = state.results.runs[state.run];
  const row = run.table[m] && run.table[m][split];
  return row ? row[state.metric] : null;
}

function drawChart() {
  const host = $("#leak-chart");
  const W = Math.max(300, host.clientWidth || 800);
  const narrow = W < 620;
  const cfg = METRICS[state.metric];
  const LW = narrow ? 12 : 196, RW = 16, top = 30;
  const RH = narrow ? 84 : 76;
  const H = top + RH * METHODS.length + 8;
  const x0 = LW + 8, x1 = W - RW;
  const X = (v) => x0 + ((v - cfg.domain[0]) / (cfg.domain[1] - cfg.domain[0])) * (x1 - x0);
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `${cfg.name} for each method under three splits. A table version is available with the Show as table button.` });

  const axis = sv("g", { class: "axis" });
  cfg.ticks.forEach((t) => {
    axis.append(sv("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 6, stroke: "#CFC6B3", "stroke-width": 1 }));
    axis.append(sv("text", { x: X(t), y: top - 12, "text-anchor": "middle", text: String(t) }));
  });
  svg.append(axis);
  let chance = cfg.chance;
  if (chance === null) { const c = state.results.runs[state.run].counts; chance = c.defective_patches / c.labelled_patches; }
  svg.append(sv("line", { x1: X(chance), x2: X(chance), y1: top - 6, y2: H - 6, stroke: "#6B6458", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  svg.append(sv("text", { x: X(chance) + 4, y: H - 10, fill: "#4A5560", style: "font:12px 'IBM Plex Sans',sans-serif;fill:#4A5560", text: "chance" }));

  METHODS.forEach(([key, name], i) => {
    const y = top + i * RH;
    const g = sv("g", { class: "row", tabindex: 0, role: "button", "aria-pressed": String(key === state.method), "aria-label": `${name}. Select to compare splits.` });
    g.append(sv("rect", { class: "rowbg", x: 0, y: y, width: W, height: RH - 2 }));
    const sel = key === state.method;
    g.append(sv("text", { x: 12, y: narrow ? y + 18 : sel ? y + 14 : y + RH / 2 + 5, "font-size": 14, "font-weight": 600, text: name }));
    const lanes = narrow ? [y + 38, y + 54, y + 70] : [y + 26, y + 44, y + 62];
    if (sel && narrow) {
      const vals = SPLITS.map(([s2]) => { const c2 = metricRow(key, s2); return c2 && c2[0] !== null ? f2(c2[0]) : "n/a"; });
      g.append(sv("text", { class: "val", x: W - 12, y: y + 18, "text-anchor": "end", text: vals.join(" \u00b7 ") }));
    }
    SPLITS.forEach(([split, sname, shape], j) => {
      const c = metricRow(key, split);
      const yy = lanes[j];
      const [col, filled] = SPLIT_STYLE[split];
      if (!c || c[0] === null) {
        g.append(sv("text", { class: "na", x: x0 + 4, y: yy + 4, text: split === "lofo" && key === "maha_fabric" ? "n/a: needs a model for each fabric" : "n/a" }));
        if (sel && !narrow) { g.append(marker(shape, 18, yy, 3.5, col, filled)); g.append(sv("text", { class: "val", x: 30, y: yy + 4, text: "n/a" })); }
        return;
      }
      const [v, lo, hi] = c;
      if (lo !== null) {
        g.append(sv("line", { x1: X(lo), x2: X(hi), y1: yy, y2: yy, stroke: col, "stroke-width": 2 }));
        g.append(sv("line", { x1: X(lo), x2: X(lo), y1: yy - 4, y2: yy + 4, stroke: col, "stroke-width": 2 }));
        g.append(sv("line", { x1: X(hi), x2: X(hi), y1: yy - 4, y2: yy + 4, stroke: col, "stroke-width": 2 }));
      }
      const mk = marker(shape, X(v), yy, 5, col, filled);
      mk.append(sv("title", { text: `${LABEL[key]}, ${sname}: ${f2(v)} (95% interval ${f2(lo)} to ${f2(hi)})` }));
      g.append(mk);
      if (sel && !narrow) { g.append(marker(shape, 18, yy, 3.5, col, filled)); g.append(sv("text", { class: "val", x: 30, y: yy + 4, text: ci(c) })); }
    });
    const select = () => { state.method = key; drawChart(); drawGap(); drawLeakTable(); $("#leak-chart .row[aria-pressed=true]")?.focus(); };
    g.addEventListener("click", select);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); } });
    svg.append(g);
  });
  host.replaceChildren(svg);
}

function drawGap() {
  const g = state.results.runs[state.run].gaps[state.method] || {};
  const box = $("#leak-gap");
  const block = (title, cmp, explain) => {
    const d = g[cmp];
    const kids = [el("h3", { text: title, style: "margin-top:0" })];
    if (!d) { kids.push(el("p", { class: "muted", text: "Not applicable for this method." })); return el("div", {}, kids); }
    const m = state.metric in d ? state.metric : "patch_ap";
    const [v, lo, hi] = d[m];
    kids.push(el("p", {}, [el("span", { class: "big mono", text: signed(v) }), document.createTextNode(`  ${METRICS[m].name}`)]));
    kids.push(el("p", { class: "mono", text: `95% interval ${signed(lo)} to ${signed(hi)}` }));
    kids.push(el("p", { class: "note", text: explain }));
    return el("div", {}, kids);
  };
  box.replaceChildren(
    block(`${LABEL[state.method]}: what leakage adds`, "leaky minus grouped", "Difference between the leaky patch split and the honest split by strip, paired over the same resampled strips. A positive number is inflation."),
    block("What an unseen fabric costs", "grouped minus lofo", "Difference between the split by strip and holding a whole fabric out. A positive number is performance lost on a new fabric."),
  );
}

function drawLeakTable() {
  const host = $("#leak-table");
  if (!state.table) { host.hidden = true; return; }
  host.hidden = false;
  const thead = el("thead", {}, [el("tr", {}, [el("th", { scope: "col", text: "Method" }), ...SPLITS.map(([, n]) => el("th", { scope: "col", text: n }))])]);
  const body = el("tbody");
  METHODS.forEach(([key, name]) => {
    body.append(el("tr", {}, [el("th", { scope: "row", text: name }), ...SPLITS.map(([s]) => {
      const c = ci(metricRow(key, s));
      return el("td", { class: c ? "num" : "na", text: c || "n/a" });
    })]));
  });
  host.replaceChildren(el("div", { class: "tablewrap" }, [el("table", {}, [el("caption", { text: `${METRICS[state.metric].name}: value [95% interval]` }), thead, body])]));
}

function redrawLeak() {
  $("#metric-help").textContent = METRICS[state.metric].help;
  drawChart(); drawGap(); drawLeakTable();
}

/* ---------- strip explorer ---------- */
function outcomeOf(strip, method) {
  const flag = strip.scores[method].flag;
  if (strip.defective) return flag ? "detected" : "missed";
  return flag ? "false" : "clear";
}
const OUTCOME_TEXT = { detected: "Detected", missed: "Missed", false: "False alarm", clear: "Clear" };
function outcomeBadge(o) {
  const bad = o === "missed" || o === "false";
  const shapes = { detected: marker("circle", 8, 8, 4, "#1B2430", true), clear: marker("circle", 8, 8, 4, "#1B2430", false),
    missed: sv("path", { d: "M8 2L14 14H2Z", fill: "#A23B2A", stroke: "#A23B2A", "stroke-width": 1 }), false: marker("diamond", 8, 8, 4, "#A23B2A", true) };
  return el("span", { class: "badge" + (bad ? " bad" : "") }, [sv("svg", { width: 16, height: 16, "aria-hidden": "true" }, [shapes[o]]), el("span", { text: OUTCOME_TEXT[o] })]);
}

function stripSvg(strip, method, W) {
  const n = state.strips.bins;
  const cell = W / n;
  const gtH = 14, scH = 22, gap = 4, ruler = 14;
  const H = gtH + gap + scH + ruler;
  const sc = strip.scores[method];
  const peak = sc.v.reduce((a, v, i) => (v !== null && (a === null || v > sc.v[a]) ? i : a), null);
  const gtText = strip.gt.length ? `true defect at ${strip.gt.map(([a, b]) => `${Math.round(a * strip.width)} to ${Math.round(b * strip.width)}`).join(", ")} pixels` : "no defect";
  const label = `Strip ${strip.id}, ${gtText}. Highest score ${peak === null ? "none" : `at pixels ${Math.round(peak / n * strip.width)} to ${Math.round((peak + 1) / n * strip.width)}, ${sc.v[peak] >= 0 ? "above" : "below"} the threshold`}. ${OUTCOME_TEXT[outcomeOf(strip, method)]}.`;
  const svg = sv("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": label });
  svg.append(sv("rect", { x: 0.5, y: 0.5, width: W - 1, height: gtH - 1, fill: "#FBF8F1", stroke: "#CFC6B3" }));
  strip.gt.forEach(([a, b]) => {
    const w = Math.max(5, (b - a) * W);
    svg.append(sv("rect", { x: a * W - (w - (b - a) * W) / 2, y: 0.5, width: w, height: gtH - 1, fill: "url(#hatch-gt)", stroke: "#1B2430" }));
  });
  const y0 = gtH + gap;
  sc.v.forEach((v, i) => {
    const x = i * cell;
    if (v === null) {
      svg.append(sv("rect", { x, y: y0, width: cell + 0.5, height: scH, fill: "url(#hatch-mask)" }));
      return;
    }
    const r = sv("rect", { x, y: y0, width: cell + 0.5, height: scH, fill: colour(v) }, [sv("title", { text: `pixels ${Math.round(i / n * strip.width)} to ${Math.round((i + 1) / n * strip.width)}: relative score ${v > 0 ? "+" : ""}${v} (0 is the threshold)` })]);
    svg.append(r);
    if (v >= 0) svg.append(sv("rect", { x, y: y0, width: cell + 0.5, height: 4, fill: "#1B2430" }));
  });
  svg.append(sv("rect", { x: 0.5, y: y0 + 0.5, width: W - 1, height: scH - 1, fill: "none", stroke: "#1B2430" }));
  const ry = y0 + scH;
  [0, 1024, 2048, 3072, 4096].forEach((px) => {
    const x = (px / strip.width) * W;
    if (x > W + 1) return;
    svg.append(sv("line", { x1: x, x2: x, y1: ry, y2: ry + 4, stroke: "#4A5560" }));
    svg.append(sv("text", { x: Math.min(Math.max(x, 8), W - 12), y: ry + 13, "text-anchor": px === 0 ? "start" : px >= 4096 ? "end" : "middle", text: String(px) }));
  });
  return svg;
}

function filteredStrips() {
  const f = { fabric: $("#f-fabric").value, defect: $("#f-defect").value, outcome: $("#f-outcome").value, sort: $("#f-sort").value };
  const m = state.method;
  let list = state.strips.strips.filter((s) => {
    if (f.fabric !== "all" && s.fabric !== f.fabric) return false;
    if (f.defect === "none" && s.defective) return false;
    if (f.defect !== "all" && f.defect !== "none" && String(s.code) !== f.defect) return false;
    const o = outcomeOf(s, m);
    if (f.outcome === "correct" && (o === "missed" || o === "false")) return false;
    if (f.outcome === "missed" && o !== "missed") return false;
    if (f.outcome === "false" && o !== "false") return false;
    return true;
  });
  const rank = { missed: 0, false: 1, detected: 2, clear: 3 };
  list.sort((a, b) => (f.sort === "attention" ? rank[outcomeOf(a, m)] - rank[outcomeOf(b, m)] || a.id.localeCompare(b.id) : a.id.localeCompare(b.id)));
  return list;
}

function drawStrips() {
  const host = $("#strip-list");
  const list = filteredStrips();
  const m = state.method;
  const count = $("#strip-count");
  if (!list.length) {
    count.textContent = "No strips match these filters.";
    const reset = el("button", { type: "button", text: "Clear filters" });
    reset.addEventListener("click", () => { ["#f-fabric", "#f-defect", "#f-outcome"].forEach((s) => { $(s).value = s === "#f-outcome" ? "all" : "all"; }); state.shown = 20; drawStrips(); });
    host.replaceChildren(el("div", { class: "state" }, [el("p", { text: "Try a different combination, for example a fabric that has that defect type." }), reset]));
    return;
  }
  const errors = list.filter((s) => ["missed", "false"].includes(outcomeOf(s, m))).length;
  count.textContent = `Showing ${Math.min(state.shown, list.length)} of ${list.length} strips for ${LABEL[m]} (${errors} errors among them).`;
  const W = Math.max(240, (host.clientWidth || 800) - (window.innerWidth <= 760 ? 0 : 230 + 24));
  const ol = el("ol", { class: "strips" });
  list.slice(0, state.shown).forEach((s) => {
    const o = outcomeOf(s, m);
    const left = el("div", {}, [
      el("h4", { text: s.id }),
      el("p", { class: "meta", text: `Fabric ${s.fabric} \u00b7 ${s.defective ? s.defect : "clean"}` }),
      outcomeBadge(o),
    ]);
    ol.append(el("li", { class: "strip" }, [left, el("div", { class: "track" }, [stripSvg(s, m, W)])]));
  });
  const kids = [ol];
  if (list.length > state.shown) {
    const more = el("button", { type: "button", class: "more", text: `Show ${Math.min(20, list.length - state.shown)} more` });
    more.addEventListener("click", () => { state.shown += 20; drawStrips(); });
    kids.push(more);
  }
  host.replaceChildren(...kids);
}

/* ---------- results table ---------- */
function drawTable() {
  const run = state.results.runs[$("input[name=trun]:checked").value];
  const metric = $("#tbl-metric").value;
  $("#tbl-help").textContent = TABLE_HELP[metric] || "";
  const thead = el("thead", {}, [el("tr", {}, [el("th", { scope: "col", text: "Method" }), ...SPLITS.map(([, n]) => el("th", { scope: "col", text: n })), el("th", { scope: "col", text: "Read it like this" })])]);
  const body = el("tbody");
  let family = "";
  METHODS.forEach(([key, name, fam]) => {
    if (fam !== family) { family = fam; body.append(el("tr", { class: "fam" }, [el("th", { scope: "colgroup", colspan: 5, text: fam })])); }
    const cells = SPLITS.map(([s]) => {
      const c = run.table[key] && run.table[key][s] && run.table[key][s][metric];
      const t = c && c[0] !== null ? (metric === "pixel_iou" ? `${f3(c[0])} [${f3(c[1])}, ${f3(c[2])}]` : ci(c)) : null;
      return el("td", { class: t ? "num" : "na", text: t || "n/a" });
    });
    body.append(el("tr", {}, [el("th", { scope: "row", text: name }), ...cells, el("td", { text: READ_AS[key] })]));
  });
  $("#tbl-out").replaceChildren(el("div", { class: "tablewrap" }, [el("table", {}, [el("caption", { text: `${TABLE_METRICS.find(([k]) => k === metric)[1]}: value [95% interval over strips]` }), thead, body])]));
  const c = run.counts;
  $("#tbl-counts").textContent = `Tested on ${c.images} strips (${c.defective_images} defective). Patch metrics use ${c.labelled_patches.toLocaleString("en")} labelled patches, ${c.defective_patches} of them defective. Baselines have no training, so they score the same under every split. ${run.not_applicable.length ? "n/a: per-fabric Mahalanobis needs a model for the fabric, so it cannot be run on a fabric held out of training." : ""}`;
}

/* ---------- data section ---------- */
const num = (x, d = 1) => (x === null || x === undefined ? "n/a" : x.toFixed(d));

function drawData() {
  const rows = state.results.artefact_check;
  const thead = el("thead", {}, [el("tr", {}, ["Fabric", "Clean strips", "Defective strips", "Mean brightness, clean", "Mean brightness, defective", "AUROC from brightness alone"].map((t) => el("th", { scope: "col", text: t })))]);
  const body = el("tbody", {}, rows.map((r) => el("tr", {}, [
    el("th", { scope: "row", text: String(r.fabric).padStart(2, "0") }),
    el("td", { class: "num", text: String(r.n_clean) }),
    el("td", { class: "num", text: String(r.n_defective) }),
    el("td", { class: "num", text: `${num(r.clean_mean_brightness)} \u00b1 ${num(r.clean_sd_across_strips)}` }),
    el("td", { class: "num", text: r.defective_sd_across_strips === null ? `${num(r.defective_mean_brightness)} (1 strip)` : `${num(r.defective_mean_brightness)} \u00b1 ${num(r.defective_sd_across_strips)}` }),
    el("td", { class: "num", text: r.auroc_brightness === null ? "n/a" : r.auroc_brightness.toFixed(2) + (r.n_defective < 5 ? " (too few)" : "") }),
  ])));
  $("#artefact-out").replaceChildren(el("div", { class: "tablewrap" }, [el("table", {}, [el("caption", { text: "Brightness is the strip mean over patches away from white margins (grey levels, mean \u00b1 spread across strips). 0.5 would mean brightness tells nothing." }), thead, body])]));

  const sub = state.results.artefact_free_subset;
  const hd = el("thead", {}, [el("tr", {}, [el("th", { scope: "col", text: "Method" }), ...SPLITS.map(([, n]) => el("th", { scope: "col", text: n }))])]);
  const b2 = el("tbody", {}, METHODS.map(([k, n]) => el("tr", {}, [el("th", { scope: "row", text: n }), ...SPLITS.map(([s]) => {
    const t = ci(sub[`${s}:${k}`]); return el("td", { class: t ? "num" : "na", text: t || "n/a" });
  })])));
  $("#subset-out").replaceChildren(el("div", { class: "tablewrap" }, [el("table", {}, [el("caption", { text: "Mean of per-fabric strip AUROC, value [95% interval]. 0.5 is chance." }), hd, b2])]));
}

/* ---------- wiring ---------- */
function fillSelect(sel, opts) { sel.replaceChildren(...opts.map(([v, t]) => el("option", { value: v, text: t }))); }

function init() {
  drawLegend();
  document.querySelectorAll("input[name=run]").forEach((i) => i.addEventListener("change", () => { state.run = i.value; redrawLeak(); }));
  document.querySelectorAll("input[name=metric]").forEach((i) => i.addEventListener("change", () => { state.metric = i.value; redrawLeak(); }));
  const tb = $("#leak-astable");
  tb.addEventListener("click", () => { state.table = !state.table; tb.setAttribute("aria-pressed", String(state.table)); tb.textContent = state.table ? "Show as chart" : "Show as table"; $("#leak-chart").hidden = state.table; drawLeakTable(); });
  redrawLeak();

  fillSelect($("#f-method"), METHODS.map(([k, n]) => [k, n]));
  $("#f-method").value = state.method;
  fillSelect($("#f-fabric"), [["all", "All fabrics"], ...["00", "01", "02", "03", "04", "05", "06"].map((f) => [f, `Fabric ${f}`])]);
  const types = {};
  state.strips.strips.forEach((s) => { if (s.defective) types[s.code] = [s.defect, (types[s.code] ? types[s.code][1] : 0) + 1]; });
  fillSelect($("#f-defect"), [["all", "All strips"], ["none", "Clean strips only"], ...Object.entries(types).sort((a, b) => b[1][1] - a[1][1]).map(([c, [n, k]]) => [c, `${n} (${k})`])]);
  ["#f-method", "#f-fabric", "#f-defect", "#f-outcome", "#f-sort"].forEach((s) => $(s).addEventListener("change", () => {
    if (s === "#f-method") state.method = $(s).value;
    state.shown = 20; drawStrips();
  }));
  drawStrips();

  fillSelect($("#tbl-metric"), TABLE_METRICS);
  $("#tbl-metric").addEventListener("change", drawTable);
  document.querySelectorAll("input[name=trun]").forEach((i) => i.addEventListener("change", drawTable));
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => { if (!state.table) drawChart(); drawStrips(); }, 120); });
  drawTable();
  drawData();

}

load();
})();
