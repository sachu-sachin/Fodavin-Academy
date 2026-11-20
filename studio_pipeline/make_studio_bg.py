"""Generate the Divya TechVerse studio background plate (1920x1080).

Run once; the same PNG is reused for every video so the set stays identical.
    python3 make_studio_bg.py studio_bg.png
"""
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

NAVY = np.array([14, 18, 46])
PURPLE = np.array([92, 52, 170])
EBLUE = np.array([40, 120, 255])
VIOLET = np.array([150, 90, 230])


def glow(canvas, cx, cy, r, color, strength):
    yy, xx = np.mgrid[0:H, 0:W]
    d = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    canvas += np.exp(-d)[..., None] * color * strength


def main(out):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Wall: deep navy with a vertical gradient
    t = (yy / H)[..., None]
    img = NAVY * (1.15 - 0.45 * t) + np.zeros((H, W, 3))
    # Ambient LED washes
    glow(img, 380, 180, 520, PURPLE, 0.55)
    glow(img, 1650, 260, 560, EBLUE, 0.35)
    glow(img, 1000, 1100, 700, VIOLET, 0.18)

    # Vertical acoustic slats on the right
    slat = ((xx > 1380) & ((xx.astype(int) // 26) % 2 == 0)).astype(np.float32)
    img *= (1 - 0.18 * slat)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil, "RGBA")

    # LED strip along ceiling line
    d.rectangle([0, 92, W, 98], fill=(170, 120, 255, 200))

    # Floating shelves (left) with small objects
    for sy in (330, 520):
        d.rectangle([90, sy, 560, sy + 14], fill=(30, 26, 60, 255))
        d.rectangle([90, sy + 14, 560, sy + 18], fill=(150, 110, 255, 160))  # under-glow
    d.rectangle([120, 250, 150, 330], fill=(60, 50, 110, 255))
    d.rectangle([155, 262, 180, 330], fill=(80, 60, 140, 255))
    d.rectangle([185, 240, 210, 330], fill=(45, 70, 140, 255))
    d.ellipse([420, 270, 480, 330], fill=(230, 200, 255, 230))  # orb lamp
    d.polygon([(300, 520), (330, 450), (360, 520)], fill=(40, 90, 70, 255))  # plant
    d.rectangle([310, 500, 350, 520], fill=(70, 60, 110, 255))
    d.rounded_rectangle([430, 455, 520, 520], 8, fill=(36, 40, 90, 255))

    # Desk + monitor (left/back)
    d.rectangle([0, 860, 820, 890], fill=(28, 24, 52, 255))
    d.rectangle([0, 890, 820, H], fill=(16, 14, 34, 255))
    d.rectangle([300, 780, 330, 860], fill=(20, 20, 30, 255))
    d.rounded_rectangle([60, 470, 640, 790], 10, fill=(10, 10, 16, 255))
    d.rectangle([76, 486, 624, 774], fill=(18, 20, 36, 255))
    f = ImageFont.truetype(MONO, 20)
    code = [
        ("def ", (200, 120, 255)), ("greet", (110, 180, 255)), ("(name):", (220, 220, 235)),
    ]
    lines = [
        [("import ", (200, 120, 255)), ("python", (220, 220, 235))],
        [],
        code,
        [("    return ", (200, 120, 255)), ('f"Hello, {name}"', (140, 230, 160))],
        [],
        [("for ", (200, 120, 255)), ("i ", (220, 220, 235)), ("in ", (200, 120, 255)), ("range(3):", (110, 180, 255))],
        [("    print", (110, 180, 255)), ("(greet(", (220, 220, 235)), ('"TechVerse"', (140, 230, 160)), ("))", (220, 220, 235))],
    ]
    y = 506
    for ln in lines:
        x = 96
        for txt, col in ln:
            d.text((x, y), txt, font=f, fill=col)
            x += f.getlength(txt)
        y += 34

    # Python-ish logo tile on the wall (abstract, original)
    d.rounded_rectangle([1460, 300, 1600, 440], 24, outline=(120, 160, 255, 140), width=4)
    d.text((1484, 330), "</>", font=ImageFont.truetype(SANS_B, 52), fill=(150, 180, 255, 170))

    # Depth of field
    pil = pil.filter(ImageFilter.GaussianBlur(9))

    # Bokeh lights
    bok = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bok)
    rng = np.random.default_rng(7)
    for _ in range(28):
        x, y = rng.integers(0, W), rng.integers(120, 700)
        r = int(rng.integers(10, 30))
        c = tuple(int(v) for v in (VIOLET if rng.random() < .5 else EBLUE)) + (int(rng.integers(30, 70)),)
        bd.ellipse([x - r, y - r, x + r, y + r], fill=c)
    bok = bok.filter(ImageFilter.GaussianBlur(4))
    pil = Image.alpha_composite(pil.convert("RGBA"), bok)

    # Brand sign (soft neon, less blurred so it reads)
    sign = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sign)
    sf = ImageFont.truetype(SANS_B, 58)
    txt = "Divya TechVerse"
    tw = sf.getlength(txt)
    sx, sy = 100, 140
    sd.text((sx, sy), txt, font=sf, fill=(190, 150, 255, 255))
    halo = sign.filter(ImageFilter.GaussianBlur(14))
    pil = Image.alpha_composite(pil, halo)
    pil = Image.alpha_composite(pil, sign.filter(ImageFilter.GaussianBlur(2.2)))

    # Vignette
    a = np.asarray(pil.convert("RGB")).astype(np.float32)
    v = 1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    a = np.clip(a * np.clip(v, 0.55, 1)[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(a).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "studio_bg.png")
