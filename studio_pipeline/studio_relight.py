"""Keep the real background, but relight the video like a studio shoot.

    python3 studio_relight.py INPUT.mp4 OUTPUT.mp4

- Global: removes yellow cast, restores blacks/contrast (phone footage is often washed out).
- Subject (MediaPipe segmentation): soft key light on the face, natural skin tone, clarity,
  subtle violet/blue edge lights.
- Background: kept as-is, only slightly dimmed and cooled toward navy/purple with a
  soft vignette so you stand out.
- Audio: hum/rumble cut, noise reduction, voice presence, loudness -16 LUFS.
Output: 1920x1080, H.264 high profile, 30 fps, CRF 16.
"""
import argparse, os, subprocess
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from studio_render import guided

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1920, 1080


def global_grade(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    lo, hi = np.percentile(L, 0.5), np.percentile(L, 99.7)   # restore black/white points
    L = 0.5 * L + 0.5 * np.clip((L - lo) / max(hi - lo, 0.2), 0, 1)
    L = L * L * (3 - 2 * L) * 0.2 + L * 0.8                  # S-curve contrast
    lab[..., 0] = L * 255
    lab[..., 1] = (lab[..., 1] - 128) * 1.0 + 128
    lab[..., 2] -= (lab[..., 2].mean() - 136) * 0.75           # neutralise yellow cast
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", default=os.path.join(HERE, "multiclass.tflite"))
    ap.add_argument("--bg-dim", type=float, default=0.9, help="background brightness (1 = unchanged)")
    ap.add_argument("--crf", type=int, default=16)
    a = ap.parse_args()

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.inp, "-vf",
        f"fps=30,scale={OW}:{OH}:force_original_aspect_ratio=increase:flags=lanczos,crop={OW}:{OH},hqdn3d=1.5:1.5:3:3",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", "30", "-i", "-",
        "-i", a.inp, "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-profile:v", "high", "-tune", "film",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-af", "highpass=f=85,lowpass=f=14000,afftdn=nf=-30:tn=1,anlmdn=s=4,"
               "equalizer=f=250:t=q:w=1:g=-2,equalizer=f=3200:t=q:w=1.2:g=3,"
               "acompressor=threshold=-22dB:ratio=3:attack=10:release=150,loudnorm=I=-16:TP=-1.5:LRA=9",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)

    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    vig = np.clip(1 - 0.30 * (((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2), 0.6, 1)[..., None]
    bgtint = np.array([0.96, 0.97, 1.05])          # cool navy/violet wash on the wall
    nb = OW * OH * 3
    prev, prevf, i = None, None, 0
    while True:
        buf = dec.stdout.read(nb)
        if len(buf) < nb:
            break
        f = np.frombuffer(buf, np.uint8).reshape(OH, OW, 3)
        small = cv2.resize(f, (OW // 2, OH // 2), interpolation=cv2.INTER_AREA)
        r = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)),
                                  int(i * 1000 / 30))
        cm = [c.numpy_view().astype(np.float32) for c in r.confidence_masks]
        m = cv2.resize(1 - cm[0], (OW // 2, OH // 2))
        g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        m = np.clip(guided(g, m, 8, 1e-2), 0, 1)
        skin = cv2.resize(cm[2] + cm[3], (OW // 8, OH // 8))
        prev = m if prev is None else 0.6 * m + 0.4 * prev
        prevf = skin if prevf is None else 0.6 * skin + 0.4 * prevf
        m = cv2.GaussianBlur(cv2.resize(prev, (OW, OH)), (0, 0), 4)[..., None]
        fp = cv2.GaussianBlur(prevf, (0, 0), 5)
        fp = cv2.resize(fp / (fp.max() + 1e-6), (OW, OH))[..., None]

        img = global_grade(f)
        # subject: key light pool on face + local clarity
        subj = img * (1 + 0.18 * fp)
        subj = subj + fp * (255 - subj) * 0.10 * (1 - subj / 255)  # lift face shadows
        lab = cv2.cvtColor(subj.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        subj = subj * 0.65 + cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) * 0.35
        # edge lights (violet left, blue right)
        mm = m[..., 0]
        rim = np.clip(mm - cv2.GaussianBlur(mm, (0, 0), 18), 0, 1)[..., None]
        side = (xx / OW)[..., None]
        rimcol = (1 - side) * np.array([170, 100, 255]) + side * np.array([70, 150, 255])
        subj = subj + rim * rimcol * 0.18 * np.clip(1.1 - yy / OH, 0, 1)[..., None]
        # background: same room, just a bit darker and cooler
        bg = img * bgtint * a.bg_dim * vig
        out = bg * (1 - m) + subj * m
        enc.stdin.write(out.clip(0, 255).astype(np.uint8).tobytes())
        i += 1
    enc.stdin.close(); enc.wait(); dec.wait()
    print(f"done: {i} frames -> {a.out}")


if __name__ == "__main__":
    main()
