# Music

## What the video uses now

An original, quiet pad generated in code (`music_bed` in `media/audio.py`): four soft chords
looping every 8 seconds, low-passed, faded in over 1.5 s and out over 3 s, and ducked
under the voice. It has no licence to manage. I could not audition it, so listen before
publishing; if it is not to your taste, use one of the options below.

## Use your own track

```
python media/build.py all --music path/to/track.mp3
```

The track loops if it is shorter than the video, is ducked under the voice automatically and
the final mix is normalised to about -16 LUFS. Keep it instrumental, slow and low in
the mid range so it does not compete with speech.

## Where to find a licensed track

I could not open any of these sites from my environment, so I cannot name a specific track
with its licence confirmed. Check the licence on the track's own page at download time.

1. **Kevin MacLeod (incompetech.com)**: large catalogue under Creative Commons Attribution 4.0.
   Requires visible credit. Search the "Calm" or "Ambient" categories for an instrumental of
   90 to 120 seconds. Titles I recall from his catalogue, unverified: "Inspired", "Wholesome".
2. **Pixabay Music**: free under the Pixabay Content Licence (no credit required, but read
   its limits on redistributing the file by itself).
3. **Free Music Archive**: licence varies per track. Only pick tracks marked CC0 or CC BY.
   Avoid anything marked NonCommercial or NoDerivatives for a public portfolio video.

Credit string format for Creative Commons Attribution tracks (fill the brackets from the
track page): `Music: "[title]" by [artist] ([source URL]), licensed under [licence name and link]`.

## Cue sheet

| Time (main video) | What the music does |
|---|---|
| 0:00 to 0:02 | Fades in |
| While the voice speaks | Sits about 12 to 18 dB under the voice (automatic ducking) |
| Between lines | Rises slightly in the gaps, returns when the voice starts |
| Last 3 seconds | Fades out over the end card |

Keep the music at or below the voice level at all times, and never let it mask the numbers.
