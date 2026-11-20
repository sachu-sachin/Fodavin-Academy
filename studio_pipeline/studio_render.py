"""Divya TechVerse studio render: raw talking-head video -> studio composite MP4.

Usage:
    python3 studio_render.py INPUT.mp4 OUTPUT.mp4 [--bg studio_bg.png] [--subject-x 1120]

Streams frames (constant memory) so 15+ minute recordings work.
Steps: auto-crop black bars -> denoise -> person/hair segmentation (MediaPipe
selfie_multiclass) -> guided-filter edge refinement + temporal smoothing ->
colour correction (remove yellow cast, soft key light) -> subtle brand rim
light -> composite on fixed studio plate -> H.264 + cleaned original audio.
"""
import argparse, re, subprocess, sys, os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

HERE = os.path.dirname(os.path.abspath(__file__))
OW, OH = 1920, 1080


def probe(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path]).decode()
    w, h, fr = out.strip().split(",")
    return int(w), int(h), fr


def detect_crop(path):
    log = subprocess.run(["ffmpeg", "-t", "5", "-i", path, "-vf", "cropdetect=24:2:0",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", log)
    return tuple(map(int, m[-1])) if m else None


def box(x, r):
    return cv2.blur(x, (2 * r + 1, 2 * r + 1))


def guided(I, p, r=8, eps=1e-3):
    mI, mp_ = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp_) / (box(I * I, r) - mI * mI + eps)
    b = mp_ - a * mI
    return box(a, r) * I + box(b, r)


def grade(rgb):
    """Neutralise yellow cast, lift exposure gently, add soft contrast."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    B -= 6.0           # less yellow
    A += 1.5           # keep skin warm-natural, not grey
    L = L / 255.0
    L = L + 0.10 * np.sin(np.pi * L) * (1 - L)       # lift mids/shadows
    L = 0.5 + (L - 0.5) * 1.06                        # gentle contrast
    lab[..., 0] = np.clip(L * 255, 0, 255)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    # mild saturation trim
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.03
    return cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--bg", default=os.path.join(HERE, "studio_bg.png"))
    ap.add_argument("--model", default=os.path.join(HERE, "multiclass.tflite"))
    ap.add_argument("--subject-x", type=int, default=1120, help="subject centre x in output")
    ap.add_argument("--crf", type=int, default=17)
    a = ap.parse_args()

    w, h, fr = probe(a.inp)
    crop = detect_crop(a.inp) or (w, h, 0, 0)
    cw, ch, cx, cy = crop
    scale = OH / ch
    sw, sh = int(round(cw * scale / 2) * 2), OH
    print(f"source {w}x{h}@{fr}, crop {crop}, subject {sw}x{sh}", flush=True)

    bg = cv2.cvtColor(cv2.imread(a.bg), cv2.COLOR_BGR2RGB).astype(np.float32)
    bg = cv2.resize(bg, (OW, OH))
    x0 = a.subject_x - sw // 2
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    # blurred studio plate behind the subject, used for light wrap
    bgb = cv2.GaussianBlur(bg, (0, 0), 30)
    bgblur_sub = np.zeros((OH, sw, 3), np.float32)
    lo, hi = max(0, x0), min(OW, x0 + sw)
    bgblur_sub[:, lo - x0:hi - x0] = bgb[:, lo:hi]

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.inp, "-vf",
        f"crop={cw}:{ch}:{cx}:{cy},hqdn3d=2:1.5:4:3,scale={sw}:{sh}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", fr, "-i", "-",
        "-i", a.inp, "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-af", "highpass=f=80,afftdn=nf=-28,equalizer=f=3000:t=q:w=1.2:g=2,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)

    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))

    fps = eval(fr)
    nbytes = sw * sh * 3
    prev = None
    i = 0
    # Precompute key-light gradient (brighter from upper-left like a softbox)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    key = (1.0 + 0.06 * (1 - xx / sw) - 0.05 * (yy / sh))[..., None]
    while True:
        buf = dec.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        fr_rgb = np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
        small = cv2.resize(fr_rgb, (sw // 2, sh // 2), interpolation=cv2.INTER_AREA)
        res = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(small)), int(i * 1000 / fps))
        m = 1.0 - res.confidence_masks[0].numpy_view().astype(np.float32)
        hair = res.confidence_masks[1].numpy_view().astype(np.float32)
        face = res.confidence_masks[3].numpy_view().astype(np.float32)
        skin = face + res.confidence_masks[2].numpy_view().astype(np.float32)
        m = cv2.resize(m, (sw // 2, sh // 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        m = np.clip(guided(gray, m, r=6, eps=4e-3), 0, 1)
        # temporal smoothing to avoid edge flicker
        prev = m if prev is None else 0.55 * m + 0.45 * prev
        m = prev
        m = np.clip((m - 0.35) / 0.45, 0, 1)            # tighten, remove halo
        m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)

        subj = grade(fr_rgb).astype(np.float32) * key
        # hair: remove yellow wall bounce shining through strands -> natural dark brown/black
        hair = cv2.GaussianBlur(cv2.resize(hair, (sw, sh)), (0, 0), 3)[..., None]
        hl = subj.mean(axis=2, keepdims=True)
        dark = np.minimum(subj, hl) * np.array([1.0, 0.92, 0.88]) * 0.62
        subj = subj * (1 - 0.8 * hair) + dark * 0.8 * hair
        # decontaminate: pull yellow wall spill out of semi-transparent edge pixels
        edge = ((m > 0.02) & (m < 0.6))[..., None]
        lum = subj.mean(axis=2, keepdims=True)
        subj = np.where(edge, subj * 0.8 + lum * 0.2, subj)

        # ---- studio lighting ----
        # key light: soft pool of light on the face/skin from front-left (softbox)
        fpool = cv2.GaussianBlur(cv2.resize(skin, (sw // 8, sh // 8)), (0, 0), 6)
        fpool = cv2.resize(fpool / (fpool.max() + 1e-6), (sw, sh))[..., None]
        keydir = np.clip(1.15 - xx / sw * 0.5, 0.75, 1.1)[..., None]
        subj = subj * (1 + 0.22 * fpool * keydir)
        # face clarity: gentle local contrast (like a sharper lens), no smoothing
        lab = cv2.cvtColor(subj.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        cl = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        subj = subj * 0.6 + cl * 0.4
        # overall subject fall-off (lower body slightly darker -> face draws the eye)
        subj *= (1.0 - 0.18 * np.clip((yy / sh - 0.55) / 0.45, 0, 1))[..., None]

        # rim / hair lights: violet from left, electric blue from right
        rim = np.clip(m - cv2.GaussianBlur(m, (0, 0), 14), 0, 1) ** 0.8
        side = (xx / sw)[..., None]
        rimcol = (1 - side) * np.array([170, 100, 255]) + side * np.array([70, 150, 255])
        topw = np.clip(1.2 - yy / sh, 0.3, 1)[..., None]
        subj = subj + rim[..., None] * rimcol * 0.7 * topw
        # light wrap: studio ambient bleeding onto edges so subject sits in the room
        wrap = np.clip(1 - cv2.GaussianBlur(m, (0, 0), 25), 0, 1)[..., None] * m[..., None]
        subj = subj * (1 - 0.35 * wrap) + bgblur_sub * 0.35 * wrap * 1.6
        subj = np.clip(subj, 0, 255)

        out = bg.copy()
        xs0, xs1 = max(0, x0), min(OW, x0 + sw)
        region = out[:, xs0:xs1]
        sm = m[:, xs0 - x0:xs1 - x0, None]
        region[:] = region * (1 - sm) + subj[:, xs0 - x0:xs1 - x0] * sm
        enc.stdin.write(out.clip(0, 255).astype(np.uint8).tobytes())
        i += 1
        if i % 100 == 0:
            print(f"frame {i}", flush=True)
    enc.stdin.close(); enc.wait(); dec.wait()
    print(f"done: {i} frames -> {a.out}")


if __name__ == "__main__":
    main()
