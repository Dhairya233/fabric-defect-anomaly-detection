"""Voice, music bed, mixing and loudness.

The voice is synthesised locally with Kokoro (Apache-2.0 model, MIT wrapper). The default
music bed is generated here from scratch, so it has no licence attached; a track you
choose can replace it (see media/MUSIC.md).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR_VOICE = 24000
SR_OUT = 48000
TARGET_LUFS = -16.0
LIMIT = 0.80  # linear ceiling (about -1.9 dBFS) leaves room for inter-sample peaks


def ffmpeg_exe() -> str:
    import shutil

    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def fill(text: str, facts: dict) -> str:
    out = re.sub(r"\{(\w+)\}", lambda m: facts[m.group(1)], text)
    if "{" in out:
        raise ValueError(f"unfilled placeholder in: {out}")
    return out


_KOKORO = None


def synth(text: str, voice: str, speed: float, cache: Path, model_dir: Path) -> np.ndarray:
    key = hashlib.sha256(f"{voice}|{speed}|{text}".encode()).hexdigest()[:16]
    path = cache / f"voice_{key}.wav"
    if not path.exists():
        global _KOKORO
        if _KOKORO is None:
            from kokoro_onnx import Kokoro

            _KOKORO = Kokoro(
                str(model_dir / "kokoro-v1.0.onnx"), str(model_dir / "voices-v1.0.bin")
            )
        audio, sr = _KOKORO.create(text, voice=voice, speed=speed, lang="en-us")
        assert sr == SR_VOICE
        cache.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, sr)
    return sf.read(path, dtype="float32")[0]


def plan(spec: dict, facts: dict, cfg: dict, cache: Path, model_dir: Path, lead=0.6, tail=0.9):
    """Synthesise every line and lay the scenes out on one timeline.

    Returns (scenes, lines, voice) where every scene/line has absolute start and end times.
    """
    gap = cfg["line_gap"]
    t = 0.0
    scenes, lines, chunks = [], [], []
    for sc in spec["scenes"]:
        s0 = t
        t += lead
        rel = {}
        for ln in sc["lines"]:
            text = fill(ln["text"], facts)
            wav = synth(text, cfg["voice"], cfg["speed"], cache, model_dir)
            dur = len(wav) / SR_VOICE
            rel[ln["id"]] = t - s0
            lines.append(
                {"id": ln["id"], "scene": sc["id"], "text": text, "start": t, "end": t + dur}
            )
            chunks.append((t, wav))
            t += dur + gap
        t += tail - gap
        scenes.append({**sc, "start": s0, "end": t, "rel": rel})
    voice = np.zeros(int((t + 1) * SR_VOICE), dtype=np.float32)
    for start, wav in chunks:
        i = int(start * SR_VOICE)
        voice[i : i + len(wav)] += wav
    return scenes, lines, voice[: int(t * SR_VOICE)]


def music_bed(duration: float) -> np.ndarray:
    """A quiet original pad (Am, F, C, G) built from soft sine partials. Mono, 48 kHz."""
    n = int(duration * SR_OUT)
    tt = np.arange(n) / SR_OUT
    chords = [
        (220.0, 261.63, 329.63),
        (174.61, 220.0, 261.63),
        (261.63, 329.63, 392.0),
        (196.0, 246.94, 293.66),
    ]
    seg = 8.0
    out = np.zeros(n)
    for k in range(int(duration / seg) + 2):
        f = chords[k % 4]
        start, end = k * seg - 1.5, (k + 1) * seg + 1.5
        env = np.clip(np.minimum((tt - start) / 2.0, (end - tt) / 2.0), 0, 1) ** 1.5
        for note in f:
            for h, a in ((1, 1.0), (2, 0.25), (3, 0.08)):
                out += env * a * np.sin(2 * np.pi * note * h * tt + 0.3 * h)
        out += env * 0.7 * np.sin(2 * np.pi * (f[0] / 2) * tt)
    out /= np.abs(out).max() + 1e-9
    fade = np.clip(np.minimum(tt / 2.0, (duration - tt) / 3.0), 0, 1)
    return (out * fade * 0.5).astype(np.float32)


def mix(voice: np.ndarray, music_file: Path | None, out_wav: Path, duration: float, work: Path):
    """Voice over a ducked music bed, loudness-normalised to -16 LUFS with a -1.9 dBFS limiter."""
    ff = ffmpeg_exe()
    work.mkdir(parents=True, exist_ok=True)
    sf.write(work / "voice.wav", voice, SR_VOICE)
    if music_file and music_file.exists():
        music_in = ["-stream_loop", "-1", "-i", str(music_file)]
    else:
        sf.write(work / "bed.wav", music_bed(duration + 1), SR_OUT)
        music_in = ["-i", str(work / "bed.wav")]
    graph = (
        "[0:a]aresample=48000,asplit=2[v][sc];"
        "[1:a]aresample=48000,lowpass=f=2500,volume=0.55,"
        f"afade=t=in:st=0:d=1.5,afade=t=out:st={duration - 3:.2f}:d=3[m];"
        "[m][sc]sidechaincompress=threshold=0.02:ratio=9:attack=30:release=500:makeup=1[duck];"
        "[v][duck]amix=inputs=2:duration=first:normalize=0[mix]"
    )
    base = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(work / "voice.wav"),
        *music_in,
    ]
    raw = work / "mix_raw.wav"
    subprocess.run(
        [
            *base,
            "-filter_complex",
            graph.replace("[mix]", "[o]"),
            "-map",
            "[o]",
            "-t",
            f"{duration:.2f}",
            "-ar",
            str(SR_OUT),
            "-ac",
            "2",
            str(raw),
        ],
        check=True,
    )
    # Apply the measured gain, limit the peaks, re-measure and correct until on target.
    gain = TARGET_LUFS - measure(raw)["integrated_lufs"]
    for _ in range(5):
        subprocess.run(
            [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw),
                "-af",
                f"volume={gain:.2f}dB,alimiter=limit={LIMIT}:attack=3:release=60:level=disabled",
                "-ar",
                str(SR_OUT),
                str(out_wav),
            ],
            check=True,
        )
        m = measure(out_wav)
        if abs(m["integrated_lufs"] - TARGET_LUFS) <= 0.4:
            return
        gain += TARGET_LUFS - m["integrated_lufs"]
    raise RuntimeError(f"could not reach {TARGET_LUFS} LUFS; last measurement {m}")


def measure(path: Path) -> dict:
    """Integrated loudness and true peak of a file (EBU R128)."""
    res = subprocess.run(
        [
            ffmpeg_exe(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    tail = res.stderr[res.stderr.rindex("Summary:") :]
    return {
        "integrated_lufs": float(re.search(r"I:\s+(-?[\d.]+) LUFS", tail).group(1)),
        "loudness_range_lu": float(re.search(r"LRA:\s+([\d.]+) LU", tail).group(1)),
        "true_peak_dbtp": float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail).group(1)),
    }
