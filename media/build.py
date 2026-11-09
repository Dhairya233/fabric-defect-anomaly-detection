"""Build the demo media: video, captions, GIF, screenshots and the vertical short.

    python media/build.py all            # everything (about 25 minutes on one CPU core)
    python media/build.py voice          # script, timings, captions, audio only
    python media/build.py main|short|gif|shots

Outputs go to docs/media/. Needs: pip install -r requirements-media.txt, a Chromium for
Playwright (playwright install chromium, or --chromium PATH) and the repository's results.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audio  # noqa: E402
import facts as facts_mod  # noqa: E402
import fetch_models  # noqa: E402
import render  # noqa: E402
import serve  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "docs/media"
WORK = ROOT / "data/cache/media"
LIVE_URL = "https://dhairya233.github.io/fabric-defect-anomaly-detection/"
META = [
    "-map_metadata",
    "-1",
    "-metadata",
    "title=Fabric defect lab: results explorer",
    "-metadata",
    "artist=Dhairya Shah",
]
STRIP = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact"]


def ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def chunks(text: str, limit: int = 72) -> list[str]:
    """Split a line into balanced caption pieces: sentence ends first, then even word splits."""
    out = []
    for sent in (p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()):
        words = sent.split()
        k = -(-len(sent) // limit)  # pieces needed
        per = -(-len(words) // k)
        out += [" ".join(words[i : i + per]) for i in range(0, len(words), per)]
    return out


def write_srt(lines: list[dict], path: Path) -> None:
    n, rows = 1, []
    for ln in lines:
        pieces = chunks(ln["text"])
        total = sum(len(p) for p in pieces)
        t = ln["start"]
        for p in pieces:
            d = (ln["end"] - ln["start"]) * len(p) / total
            rows.append(f"{n}\n{ts(t)} --> {ts(t + d)}\n{p}\n")
            n += 1
            t += d
    path.write_text("\n".join(rows))


def write_script_md(name: str, scenes: list[dict], lines: list[dict], path: Path) -> None:
    out = [
        f"# Voiceover script: {name}\n",
        "Timings come from the synthesised voice. Every number is read from "
        "`site/data/results.json` and `results/margin_summary.csv` when the video is built.\n",
        "| Time | Scene | Line |",
        "|---|---|---|",
    ]
    for ln in lines:
        out.append(f"| {ln['start']:6.1f}s | {ln['scene']} | {ln['text']} |")
    out.append(f"\nTotal length: {scenes[-1]['end']:.1f} s.\n")
    path.write_text("\n".join(out))


def mux(video: Path, wav: Path, out: Path, srt: Path | None = None, orient: str = "h") -> None:
    ff = audio.ffmpeg_exe()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video), "-i", str(wav)]
    if srt:
        fonts = WORK / "fonts"
        fonts.mkdir(parents=True, exist_ok=True)
        from fontTools.ttLib import TTFont

        src = ROOT / "site/fonts/ibm-plex-sans-latin-600-normal.woff2"
        f = TTFont(src)
        f.flavor = None
        ttf = fonts / "IBMPlexSans-SemiBold.ttf"
        f.save(ttf)
        family = f["name"].getDebugName(1)
        # libass sizes SRT fonts on a 288-line scale: 11 is about 41 px on 1080 lines.
        size, outline, margin = (11, 4, 17) if orient == "h" else (8, 3, 46)
        style = (
            f"FontName={family},FontSize={size},Bold=0,BorderStyle=3,Outline={outline},Shadow=0,"
            f"PrimaryColour=&H00E3EEF3,OutlineColour=&H2030241B,BackColour=&H2030241B,"
            f"MarginV={margin},Alignment=2"
        )
        vf = f"subtitles={srt.as_posix()}:fontsdir={fonts.as_posix()}:force_style='{style}'"
        cmd += [
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-x264-params",
            "no-info=1",
        ]
    else:
        cmd += ["-c:v", "copy"]
    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        *META,
        *STRIP,
        str(out),
    ]
    subprocess.run(cmd, check=True)


def make_web(src: Path, out: Path, limit_mb: float = 9.0) -> None:
    """A smaller captioned copy for GitHub's README upload (limit about 10 MB on free accounts)."""
    ff = audio.ffmpeg_exe()
    for crf in (24, 26, 28, 30, 32):
        subprocess.run(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=1600:-2",
                "-c:a",
                "aac",
                "-b:a",
                "112k",
                "-movflags",
                "+faststart",
                *META,
                *STRIP,
                str(out),
            ],
            check=True,
        )
        size = out.stat().st_size / 1e6
        print(f"  web copy crf {crf}: {size:.1f} MB")
        if size < limit_mb:
            return
    raise SystemExit("error: could not get the web copy under the size limit")


def make_gif(video: Path, start: float, out: Path, seconds: float = 12.0) -> None:
    ff = audio.ffmpeg_exe()
    for fps, width in ((12, 960), (10, 800), (8, 720)):
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4"
        )
        subprocess.run(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.2f}",
                "-t",
                str(seconds),
                "-i",
                str(video),
                "-vf",
                vf,
                "-loop",
                "0",
                *META,
                str(out),
            ],
            check=True,
        )
        size = out.stat().st_size / 1e6
        print(f"  gif {width}px {fps}fps: {size:.1f} MB")
        if size < 8:
            return
    raise SystemExit("error: could not get the GIF under 8 MB")


def screenshots(base: str, chromium: str | None, out: Path) -> None:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)

    def settle(pg) -> None:
        pg.add_style_tag(content="*{transition:none!important} html{scrollbar-width:none}")
        pg.evaluate("document.fonts.ready")

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium, args=render.CHROME_ARGS)
        pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        pg.goto(f"{base}/site/index.html", wait_until="load")
        pg.wait_for_selector("#subset-out table")
        settle(pg)
        pg.screenshot(path=str(out / "01-hero.png"))
        pg.click("#ctl-metric label:nth-child(3)")
        pg.evaluate("document.querySelector('#leak-legend').scrollIntoView()")
        pg.evaluate("window.scrollBy(0,-64)")
        pg.screenshot(path=str(out / "02-leakage-chart.png"))
        pg.select_option("#f-outcome", "missed")
        pg.evaluate("document.querySelector('#strip-count').scrollIntoView()")
        pg.evaluate("window.scrollBy(0,-120)")
        pg.screenshot(path=str(out / "03-failure-gallery-misses.png"))
        pg.select_option("#f-outcome", "false")
        pg.screenshot(path=str(out / "04-failure-gallery-false-alarms.png"))
        pg.select_option("#f-outcome", "all")
        pg.evaluate("document.querySelector('#table').scrollIntoView()")
        pg.evaluate("window.scrollBy(0,-56)")
        pg.screenshot(path=str(out / "05-results-table.png"))
        pg.close()
        m = b.new_page(
            viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True
        )
        m.goto(f"{base}/site/index.html", wait_until="load")
        m.wait_for_selector("#leak-chart svg")
        settle(m)
        m.evaluate("document.querySelector('#leak-chart').scrollIntoView()")
        m.evaluate("window.scrollBy(0,-60)")
        m.screenshot(path=str(out / "06-mobile-chart.png"))
        b.close()
    print(f"  screenshots -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["all", "voice", "main", "short", "gif", "shots"])
    ap.add_argument(
        "--chromium", default=None, help="path to a Chromium/Chrome binary (default: Playwright's)"
    )
    ap.add_argument("--model-dir", type=Path, default=WORK / "models")
    ap.add_argument(
        "--music", type=Path, default=None, help="a music file to use instead of the generated bed"
    )
    ap.add_argument("--live-url", default=LIVE_URL)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    facts = facts_mod.load()
    cfg = json.loads((HERE / "script.json").read_text())
    fetch_models.fetch(args.model_dir)

    plans = {}
    for name in ("main", "short"):
        scenes, lines, voice = audio.plan(cfg[name], facts, cfg, WORK, args.model_dir)
        plans[name] = (scenes, lines, voice)
        write_srt(lines, OUT / ("demo.srt" if name == "main" else "short_vertical.srt"))
        write_script_md(
            name,
            scenes,
            lines,
            OUT / ("voiceover_script.md" if name == "main" else "voiceover_script_short.md"),
        )
        dur = scenes[-1]["end"]
        wav = WORK / f"audio_{name}.wav"
        audio.mix(voice, args.music, wav, dur, WORK / f"mix_{name}")
        print(f"{name}: {dur:.1f} s, audio mixed")
    if args.step == "voice":
        return 0

    server, base = serve.start(ROOT)
    try:
        if args.step in ("all", "main", "gif"):
            scenes = plans["main"][0]
            silent = WORK / "main_silent.mp4"
            if args.step != "gif" or not silent.exists():
                print("rendering main video ...")
                render.render(scenes, facts, "h", silent, base, args.chromium, args.live_url, WORK)
            mux(silent, WORK / "audio_main.wav", OUT / "demo.mp4")
            mux(silent, WORK / "audio_main.wav", OUT / "demo_captioned.mp4", OUT / "demo.srt", "h")
            ex = next(s for s in scenes if s["id"] == "explorer")
            make_web(OUT / "demo_captioned.mp4", OUT / "demo_web.mp4")
            make_gif(OUT / "demo.mp4", ex["start"] + ex["rel"]["E2"] - 0.3, OUT / "hero.gif")
        if args.step in ("all", "short"):
            scenes = plans["short"][0]
            silent = WORK / "short_silent.mp4"
            print("rendering vertical short ...")
            render.render(scenes, facts, "v", silent, base, args.chromium, args.live_url, WORK)
            mux(silent, WORK / "audio_short.wav", OUT / "short_vertical.mp4")
            mux(
                silent,
                WORK / "audio_short.wav",
                OUT / "short_vertical_captioned.mp4",
                OUT / "short_vertical.srt",
                "v",
            )
        if args.step in ("all", "shots"):
            screenshots(base, args.chromium, OUT / "screenshots")
    finally:
        server.shutdown()

    report = {}
    for f in ("demo.mp4", "short_vertical.mp4"):
        if (OUT / f).exists():
            report[f] = audio.measure(OUT / f)
    if report:
        lines = [
            "Loudness of the final audio (EBU R128, measured with ffmpeg ebur128)."
            " Targets: integrated about -16 LUFS, true peak at most -1 dBTP.\n"
        ]
        for f, m in report.items():
            lines.append(
                f"{f}: integrated {m['integrated_lufs']:.1f} LUFS, loudness range "
                f"{m['loudness_range_lu']:.1f} LU, true peak {m['true_peak_dbtp']:.1f} dBTP"
            )
        (OUT / "loudness.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
