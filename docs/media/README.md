# Demo media

Everything here is built from the repository's result tables by `media/build.py`; no number
in the video is typed by hand. No dataset images appear anywhere.

| File | What it is |
|---|---|
| `demo_web.mp4` | The walkthrough (about 99 s, 1600x900, about 6 MB) with burned-in captions |
| `short_vertical_captioned.mp4` | 1080x1920 version (about 32 s) with burned-in captions, for phones |
| `demo.srt`, `short_vertical.srt` | Captions as text files |
| `hero.gif` | 12-second loop of the chart interaction, under 8 MB |
| `screenshots/` | Six full-resolution page captures |
| `voiceover_script.md`, `voiceover_script_short.md` | The narration with timings |
| `loudness.txt` | Measured loudness of the final audio |

`python media/build.py all` also writes full-quality 1080p files (`demo.mp4`,
`demo_captioned.mp4`, `short_vertical.mp4`); those are git-ignored because of their size.

Narration is generated with a text-to-speech model; the cursor is drawn over the real page.
Credits and licences are in [media/CREDITS.md](../../media/CREDITS.md); the music options are
in [media/MUSIC.md](../../media/MUSIC.md).

## Rebuild

```bash
pip install -r requirements-media.txt
playwright install chromium            # or pass --chromium PATH
python media/build.py all              # about 25 minutes on one CPU core
python media/build.py voice            # only the script, captions and audio
python media/build.py all --music my_track.mp3 --live-url https://example.org/
```

The first run downloads the text-to-speech model (about 350 MB, checksum-verified) into
`data/cache/media/models/`. The address on the end card is the standard GitHub Pages
address for this repository; it only works once Pages is switched on, so check it before
posting, or rebuild with `--live-url`.
