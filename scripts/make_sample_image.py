# -*- coding: utf-8 -*-
"""Membuat citra uji sintetis 512x512 (BUKAN Mandrill) agar aplikasi punya contoh bawaan."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(7)
size = 512
y, x = np.mgrid[0:size, 0:size] / size

base = np.stack([0.5 + 0.5 * np.sin(6 * x + 2 * y),
                 0.5 + 0.5 * np.cos(5 * y - 3 * x),
                 0.5 + 0.5 * np.sin(4 * (x + y))], axis=-1)
noise = np.zeros((size, size, 3))
for scale, amp in ((64, 0.30), (16, 0.20), (4, 0.12)):
    small = rng.random((size // scale, size // scale, 3))
    big = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
    noise += amp * (big - 0.5)
img = np.clip(base * 0.8 + noise + 0.1, 0, 1)
pil = Image.fromarray((img * 255).astype(np.uint8))
d = ImageDraw.Draw(pil)
d.ellipse((90, 90, 280, 280), outline=(255, 255, 255), width=6)
d.rectangle((260, 250, 440, 430), outline=(20, 20, 20), width=6)
d.line((40, 460, 470, 60), fill=(255, 220, 0), width=5)
pil = pil.filter(ImageFilter.GaussianBlur(0.6))
out = ROOT / "sample_images" / "uji_sintetis_512.png"
pil.save(out)
print("Disimpan:", out)
