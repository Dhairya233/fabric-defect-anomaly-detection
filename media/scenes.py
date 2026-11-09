"""HTML for the title cards. Timings (seconds from scene start) come from the voice track."""

from __future__ import annotations

from html import escape

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/site/css/style.css"><link rel="stylesheet" href="/media/media.css">
</head><body class="{orient}"><div class="stage"><div class="selvage"></div>"""
TAIL = """<div class="selvage bottom"></div></div><script src="/media/director.js"></script></body></html>"""


def _page(orient: str, body: str) -> str:
    return HEAD.format(orient=orient) + body + TAIL


def hook(f: dict, T: dict, orient: str) -> str:
    a, b = f["_rf_ap_grouped_raw"], f["_rf_ap_leaky_raw"]
    s1, s2 = T["H1"] + 0.9, T["H1"] + 3.4
    return _page(
        orient,
        f"""
<p class="kicker" data-start="0.3">Same model &middot; same data</p>
<h1 class="big" data-start="0.5" data-anim="rise">One random forest, two scores</h1>
<div class="bars">
  <div class="bar" data-start="{s1}" data-dur="1.1" data-anim="grow" style="--v:{a}">
    <span class="lab">Split by whole strip</span><div class="trk"><i></i></div><b>{f["rf_ap_grouped"]}</b></div>
  <div class="bar leaky" data-start="{s2}" data-dur="1.1" data-anim="grow" style="--v:{b}">
    <span class="lab">Leaky patch split</span><div class="trk"><i></i></div><b>{f["rf_ap_leaky"]}</b></div>
</div>
<p class="ratio" data-start="{T.get("H2", s2 + 2)}" data-anim="rise">{f["ratio"]}&times; higher, from leakage</p>
<p class="note" data-start="{s1}">Patch-level average precision, margins left out. Precomputed on held-out strips.</p>""",
    )


def problem(f: dict, T: dict, orient: str) -> str:
    s = T["P2"]
    patches = []
    # (colour, row) for the leaky split: neighbours alternate sides
    pattern = ["tr", "tr", "te", "tr", "te", "te", "tr", "tr", "te"]
    for i, side in enumerate(pattern):
        y = 70 if i % 2 == 0 else 112
        col = "#24507A" if side == "tr" else "#A23B2A"
        patches.append(
            f'<rect x="{30 + i * 100}" y="{y}" width="200" height="38" fill="{col}" fill-opacity=".92" stroke="#1B2430" stroke-width="2" '
            f'data-start="{s + 1.2 + i * 0.12}" data-dur=".4"/>'
        )
    strip = (
        f'<rect x="30" y="64" width="1000" height="92" fill="#E9E2D2" stroke="#1B2430" stroke-width="2" data-start="{s}" data-dur=".6"/>'
        f'<text x="30" y="40" font-size="26" font-weight="600" fill="#1B2430" data-start="{s}">One strip, cut into overlapping patches</text>'
    )
    band = (
        f'<rect x="230" y="64" width="100" height="92" fill="url(#ov)" stroke="#1B2430" stroke-width="3" '
        f'stroke-dasharray="8 5" data-start="{s + 3.0}"/>'
    )
    shared = (
        f'<rect x="230" y="64" width="0" height="0"/><path d="M230 160 v18 M330 160 v18 M230 169 H330" stroke="#1B2430" stroke-width="3" fill="none" data-start="{s + 3.2}"/>'
        f'<text x="280" y="204" text-anchor="middle" font-size="24" font-weight="600" fill="#1B2430" data-start="{s + 3.2}">shared pixels</text>'
        f'<text x="530" y="204" font-size="24" fill="#A23B2A" font-weight="600" data-start="{s + 3.6}">Train and test patches overlap: leakage</text>'
    )
    row_b = (
        f'<text x="30" y="272" font-size="26" font-weight="600" fill="#1B2430" data-start="{T.get("P3", T["P2"] + 3.2)}">Split by whole strip</text>'
        f'<rect x="30" y="288" width="480" height="40" fill="#24507A" stroke="#1B2430" stroke-width="2" data-start="{T.get("P3", T["P2"] + 3.2) + 0.4}"/>'
        f'<rect x="550" y="288" width="480" height="40" fill="#A23B2A" stroke="#1B2430" stroke-width="2" data-start="{T.get("P3", T["P2"] + 3.2) + 0.9}"/>'
        f'<text x="30" y="360" font-size="22" fill="#4A5560" data-start="{T.get("P3", T["P2"] + 3.2) + 0.9}">training strip: all patches stay together</text>'
        f'<text x="550" y="360" font-size="22" fill="#4A5560" data-start="{T.get("P3", T["P2"] + 3.2) + 0.9}">test strip: never seen in training</text>'
    )
    return _page(
        orient,
        f"""
<p class="kicker" data-start="0.3">The problem</p>
<h2 class="mid" data-start="0.5" data-anim="rise">Patches of one strip share pixels</h2>
<div class="legend2" data-start="{s + 1.2}"><span class="tr">train</span><span class="te">test</span></div>
<svg class="diagram" viewBox="0 0 1060 380" role="img" aria-label="Diagram of overlapping patches">
<defs><pattern id="ov" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
<rect width="12" height="12" fill="#F3EEE3"/><rect width="5" height="12" fill="#1B2430"/></pattern></defs>
{strip}{"".join(patches)}{band}{shared}{row_b}</svg>""",
    )


def methods(f: dict, T: dict, orient: str) -> str:
    s = T["M1"]
    return _page(
        orient,
        f"""
<p class="kicker" data-start="0.3">The study</p>
<h2 class="mid" data-start="0.5" data-anim="rise">Eight methods, three ways of splitting</h2>
<div class="flow">
  <div class="step" data-start="{s + 0.6}" data-anim="rise"><span class="tag">Input</span><h3>Strips</h3><p>{f["n_strips"]} strips of {f["n_fabrics"]} fabrics, 4096 &times; 256 px</p></div>
  <span class="arrow" data-start="{s + 1.2}">&rarr;</span>
  <div class="step" data-start="{s + 1.4}" data-anim="rise"><span class="tag">Features</span><h3>90 per patch</h3><p>Co-occurrence, local patterns, Gabor, Fourier, moments</p></div>
  <span class="arrow" data-start="{s + 2.2}">&rarr;</span>
  <div class="step" data-start="{s + 2.4}" data-anim="rise"><span class="tag">Methods</span><h3>Learn or compare</h3><p>Baselines, supervised models, anomaly detectors trained on clean strips only</p></div>
</div>
<div class="splits" data-start="{T.get("M2", T["M1"] + 8) + 0.3}" data-anim="rise">
  <span>By strip</span><span>Whole fabric held out</span><span class="ctl">Leaky patch split (control)</span>
</div>""",
    )


def limits(f: dict, T: dict, orient: str) -> str:
    s = T["L1"]
    l2 = T.get("L2", s + 5.2)
    rows = [
        (
            f"One small public dataset: <b>{f['n_strips']} strips</b>, <b>{f['n_fabrics']} fabrics</b>",
            s + 0.6,
        ),
        (
            "<b>No evidence of transfer</b> to a new fabric; a few fabrics cannot rule out a modest amount",
            s + 2.0,
        ),
        ("<b>Precomputed</b> results, not a live model. Not tested on a factory line", s + 4.2),
        (
            f"<b>{f['margin_strips']} of {f['n_strips']} strips</b> have blank margins that flatter models",
            l2 + 0.4,
        ),
    ]
    items = "".join(
        f'<li data-start="{t}" data-anim="rise"><span>{txt}</span></li>' for txt, t in rows
    )
    return _page(
        orient,
        f"""
<p class="kicker" data-start="0.3">Honest limits</p>
<h2 class="mid" data-start="0.5" data-anim="rise">What this does not show</h2>
<ul class="lim">{items}</ul>""",
    )


def end(f: dict, T: dict, orient: str, live_url: str) -> str:
    return _page(
        orient,
        f"""
<p class="kicker" data-start="0.3">Fabric defect lab</p>
<h2 class="mid" data-start="0.5" data-anim="rise">Results explorer, code and tables</h2>
<div class="end" data-start="1.0" data-anim="rise">
  <span class="lab">Results explorer</span>
  <span class="url">{escape(live_url.replace("https://", "")).replace("/", "/<wbr>")}</span>
  <span class="lab">Code and tables</span>
  <span class="url">github.com/<wbr>Dhairya233/<wbr>fabric-defect-anomaly-detection</span>
</div>
<p class="credit" data-start="1.8">Data: AITEX Fabric Image Database (AFID). Silvestre-Blanes, Albero-Albero, Miralles,
P&eacute;rez-Llorens, Moreno. AUTEX Research Journal, No. 4, 2019. No dataset images are shown.</p>""",
    )


BUILDERS = {"hook": hook, "problem": problem, "methods": methods, "limits": limits}
