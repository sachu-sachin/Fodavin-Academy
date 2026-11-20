"""Cut silent pauses (jump cuts) from a recording, keeping audio/video in sync.

    python3 remove_pauses.py raw.mp4 tight.mp4 [--min-pause 0.4] [--pad 0.12]
The silence threshold adapts to the recording loudness (peak - 19 dB).
"""
import argparse, re, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--min-pause", type=float, default=0.4, help="pauses longer than this are cut (s)")
ap.add_argument("--pad", type=float, default=0.12, help="breathing room kept on each side (s)")
ap.add_argument("--below-peak", type=float, default=19.0)
a = ap.parse_args()

def ff(*args):
    return subprocess.run(["ffmpeg", "-hide_banner", *args], capture_output=True, text=True).stderr

peak = float(re.search(r"max_volume: (-?[\d.]+)", ff("-i", a.inp, "-af", "highpass=f=100,volumedetect", "-f", "null", "-")).group(1))
thr = peak - a.below_peak
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.inp]))
log = ff("-i", a.inp, "-af", f"highpass=f=100,silencedetect=n={thr}dB:d={a.min_pause}", "-f", "null", "-")
starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
if len(ends) < len(starts):
    ends.append(dur)

keep, t = [], 0.0
for s, e in zip(starts, ends):
    s2, e2 = max(0, s) + a.pad, e - a.pad
    if s2 > t:
        keep.append((t, s2))
    t = max(t, e2)
if t < dur:
    keep.append((t, dur))
keep = [(s, e) for s, e in keep if e - s > 0.15]
total = sum(e - s for s, e in keep)
print(f"threshold {thr:.1f} dB, {len(keep)} segments, {dur:.1f}s -> {total:.1f}s")

parts, labels = [], []
for i, (s, e) in enumerate(keep):
    fd = min(0.02, (e - s) / 4)
    parts.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS[v{i}];"
                 f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d={fd:.3f},afade=t=out:st={e - s - fd:.3f}:d={fd:.3f}[a{i}];")
    labels.append(f"[v{i}][a{i}]")
fc = "".join(parts) + "".join(labels) + f"concat=n={len(keep)}:v=1:a=1[v][a]"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.inp, "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-c:a", "pcm_s16le" if a.out.endswith(".mov") else "aac",
                "-b:a", "256k", a.out], check=True)
