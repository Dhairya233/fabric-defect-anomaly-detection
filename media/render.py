"""Frame-by-frame rendering of the scenes with Playwright, piped into ffmpeg.

Every frame is a pure function of time (see director.js), so a render is repeatable.
Frames whose page state is identical to the previous one reuse the previous screenshot.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

import scenes as card_scenes
from audio import ffmpeg_exe

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
CHROME_ARGS = ["--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--font-render-hinting=none"]
VIEWPORTS = {"h": ((1280, 720), 1.5), "v": ((540, 960), 2.0)}  # CSS size, scale -> 1080p frames


def explorer_program(rel: dict[str, float], orient: str) -> list[dict]:
    """What the cursor does on the real results page, timed to the narration."""

    def at(line: str, off: float, **kw) -> dict:
        return {"t": rel[line] + off, **kw}

    if orient == "v":
        t0 = rel["E3"]
        return [
            at("E3", 0.0, do="chip", text="Three fabrics where strips look alike"),
            at("E3", 0.3, do="scrollTo", sel="#subset-out", dur=1.0, margin=70),
            at("E3", 1.6, do="highlight", sel="#subset-out tbody tr:nth-child(3)", dur=t0 + 5 - t0),
        ]
    prog = [
        at("E1", 0.0, do="chip", text="Results explorer \u00b7 precomputed"),
        at("E1", 1.2, do="scrollTo", sel="#leak-legend", dur=1.0, margin=64),
        at("E2", 0.2, do="moveTo", sel="#ctl-metric label:nth-child(3)", dur=0.8),
        at("E2", 1.1, do="click", sel="#ctl-metric input[value=patch_ap]"),
        at("E2", 1.2, do="chip", text="Leakage chart \u00b7 patch precision"),
        at("E2", 2.2, do="moveTo", sel='#leak-chart g.row[aria-label^="Random forest"]', dur=0.9),
        at(
            "E2",
            2.3,
            do="highlight",
            sel='#leak-chart g.row[aria-label^="Random forest"]',
            dur=4.5,
            pad=2,
        ),
        at("E2", 7.6, do="scrollTo", sel="#leak-gap", dur=1.0, margin=150),
        at("E2", 8.8, do="highlight", sel="#leak-gap .big", dur=4.5, pad=8),
        at("E3", 0.0, do="chip", text="Three fabrics where strips look alike"),
        at("E3", 0.2, do="scrollTo", sel="#subset-out", dur=1.2, margin=90),
        at("E3", 2.2, do="highlight", sel="#subset-out tbody tr:nth-child(3)", dur=5.5, pad=3),
        at("E4", 0.0, do="chip", text="Strip explorer \u00b7 random forest misses"),
        at("E4", 0.2, do="scrollTo", sel="#strips", dur=1.0, margin=56),
        at("E4", 1.2, do="moveTo", sel="#f-outcome", dur=0.7),
        at("E4", 2.0, do="click", sel="#f-outcome", value="missed"),
        at("E4", 2.3, do="scrollTo", sel="#strip-list", dur=0.9, margin=150),
        at(
            "E4", 3.4, do="highlight", sel="#strip-list li.strip:first-child .track", dur=3.0, pad=4
        ),
        at("E5", 0.0, do="chip", text="About the data \u00b7 capture artefacts"),
        at("E5", 0.2, do="scrollTo", sel="#artefact-out", dur=1.2, margin=90),
        at("E5", 2.2, do="highlight", sel="#artefact-out tbody tr:nth-child(2)", dur=6.0, pad=3),
    ]
    for a in prog:
        if a["do"] in ("moveTo", "click", "highlight"):
            a.setdefault("pad", 6)
    return prog


def card_html(kind: str, facts: dict, rel: dict, orient: str, live_url: str) -> str:
    if kind == "end":
        return card_scenes.end(facts, rel, orient, live_url)
    return card_scenes.BUILDERS[kind](facts, rel, orient)


def render(spec_scenes, facts, orient, out_mp4, base, chromium, live_url, work: Path, only=None):
    work.mkdir(parents=True, exist_ok=True)
    (w, h), scale = VIEWPORTS[orient]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        [
            ffmpeg_exe(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            str(FPS),
            "-c:v",
            "png",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "no-info=1",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )
    shots = 0
    frames = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium, args=CHROME_ARGS)
        for sc in spec_scenes:
            if only and sc["id"] not in only:
                continue
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=scale)
            page = ctx.new_page()
            dur = sc["end"] - sc["start"]
            if sc["kind"] == "explorer":
                page.goto(f"{base}/site/index.html", wait_until="load")
                page.wait_for_selector("#leak-chart svg")
                page.wait_for_selector("#subset-out table")
                page.add_script_tag(path=str(ROOT / "media/director.js"))
                program = explorer_program(sc["rel"], orient)
                page.evaluate(
                    "p => Director.load(p)",
                    {
                        "program": program,
                        "duration": dur,
                        "cursor": True,
                        "fadeIn": 0.35,
                        "fadeOut": 0.35,
                    },
                )
            else:
                kind = (
                    sc["id"] if sc["id"] in card_scenes.BUILDERS or sc["id"] == "end" else sc["id"]
                )
                html = card_html(kind, facts, sc["rel"], orient, live_url)
                f = work / f"card_{orient}_{sc['id']}.html"
                f.write_text(html)
                page.goto(f"{base}/{f.relative_to(ROOT).as_posix()}", wait_until="load")
                page.evaluate(
                    "p => Director.load(p)",
                    {"program": [], "duration": dur, "fadeIn": 0.35, "fadeOut": 0.35},
                )
            page.evaluate("document.fonts.ready")
            n = round(dur * FPS)
            last_sig, png = None, b""
            for k in range(n):
                sig = page.evaluate("t => Director.frame(t)", k / FPS)
                if sig != last_sig:
                    png = page.screenshot(type="png")
                    shots += 1
                    last_sig = sig
                ff.stdin.write(png)
                frames += 1
            ctx.close()
            print(f"  scene {sc['id']:9s} {n:4d} frames", flush=True)
        browser.close()
    ff.stdin.close()
    ff.wait()
    print(f"  {frames} frames, {shots} screenshots -> {out_mp4.name}")
