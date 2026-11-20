# Divya TechVerse studio pipeline

Converts a raw talking-head recording into the Divya TechVerse virtual studio look: background replaced, colour corrected, audio cleaned, exported as 1080p H.264 MP4.

## Install (one time, Linux/macOS/WSL)
```bash
sudo apt-get install -y ffmpeg libegl1 libgles2   # macOS: brew install ffmpeg
pip install opencv-python-headless mediapipe numpy pillow
```
`multiclass.tflite` (MediaPipe selfie_multiclass segmentation model) is included.

## Run
```bash
python3 remove_pauses.py my_raw_video.mp4 tight.mp4
```
```bash
python3 studio_render.py tight.mp4 my_studio_video.mp4
```
`remove_pauses.py` cuts silences longer than 0.4 s, keeping 0.12 s of breathing room on each side (`--min-pause`, `--pad`). Skip that step to keep every pause.
Options: `--subject-x 1120` (where you sit horizontally), `--crf 17` (quality; lower = bigger file), `--bg studio_bg.png`.

The script processes frames as a stream, so memory use stays flat for 15+ minute videos. On a 4-core CPU it runs at about 1.7 fps, so a 15-minute video takes roughly 4–5 hours.

To change the studio set, edit `make_studio_bg.py` and run `python3 make_studio_bg.py studio_bg.png`. Every video then uses the same plate, so the look stays consistent.

## Recording tips (these matter more than the software)
- Sit 1–1.5 m in front of a plain, evenly lit wall, and avoid hair-coloured walls.
- Put a window or soft light in front of you at about 45°. This also cuts the glare on your glasses: tilt the lenses slightly down or raise the light.
- Use a clip-on mic. The audio chain only cleans the sound up; it can't replace a good recording.

## Keep your real background (studio relight only)
```bash
python3 remove_pauses.py raw.mp4 tight.mp4
```
```bash
python3 studio_relight.py tight.mp4 final.mp4
```
This keeps your room, removes the yellow cast and adds a soft key light on your face, faint violet/blue edge lights, a slightly dimmed and cooled wall, and cleaned audio. `--bg-dim 0.9` sets how much the wall is dimmed (1.0 = unchanged).
