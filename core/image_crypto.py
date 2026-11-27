# -*- coding: utf-8 -*-
"""
image_crypto.py
===============
Enkripsi dan dekripsi citra memakai S-Box (dapat dibalik 100%).

Alur ENKRIPSI (tiap piksel/kanal diperlakukan sebagai byte 0..255):

  1. SUBSTITUSI  : tiap byte diganti lewat S-Box.            B = S[P]
  2. PERMUTASI   : posisi byte diacak dengan permutasi yang
                   diturunkan dari kunci.                     Q = B[perm]
  3. DIFUSI maju : A[i] = S[(Q[i] + K1[i] + A[i-1]) mod 256]
  4. DIFUSI mundur: C[i] = S[(A[i] + K2[i] + C[i+1]) mod 256]

Difusi dua arah membuat perubahan satu byte pada citra asli menyebar ke
seluruh citra terenkripsi (diuji lewat NPCR/UACI).

Alur DEKRIPSI adalah kebalikannya, memakai S-Box invers:

  - Balik difusi mundur dan maju (cukup rumus aljabar, tanpa pencarian).
  - Balik permutasi, lalu balik substitusi dengan S-Box invers.

Kunci berupa teks (passphrase). Dari kunci diturunkan: permutasi, dua aliran
kunci K1 dan K2, serta dua nilai awal (IV), memakai SHAKE-256 (hashlib bawaan
Python) sehingga hasilnya sama di semua komputer.

CATATAN: Ini adalah rancangan untuk keperluan belajar/tugas kuliah. Skema ini
belum melalui analisis kriptografi formal dan tidak dimaksudkan untuk
melindungi data sensitif di dunia nyata. Enkripsi bersifat deterministik
(kunci dan citra yang sama menghasilkan citra sandi yang sama).
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image

from .sbox_io import inverse_sbox
from .sbox_metrics import as_sbox_array


# --------------------------------------------------------------------------
# Turunan kunci
# --------------------------------------------------------------------------
def _shake(seed: bytes, label: bytes, nbytes: int) -> bytes:
    return hashlib.shake_256(seed + b"|" + label).digest(nbytes)


def derive_key_material(key: str, n: int):
    """Turunkan (perm, k1, k2, iv1, iv2) dari kunci teks untuk n byte data."""
    if key is None or len(key) == 0:
        raise ValueError("Kunci tidak boleh kosong.")
    seed = hashlib.sha256(key.encode("utf-8")).digest()
    rand = np.frombuffer(_shake(seed, b"perm", 8 * n), dtype=">u8").astype(np.uint64)
    perm = np.argsort(rand, kind="stable")
    k1 = np.frombuffer(_shake(seed, b"k1", n), dtype=np.uint8).astype(np.int64)
    k2 = np.frombuffer(_shake(seed, b"k2", n), dtype=np.uint8).astype(np.int64)
    iv = _shake(seed, b"iv", 2)
    return perm, k1, k2, int(iv[0]), int(iv[1])


# --------------------------------------------------------------------------
# Utilitas citra
# --------------------------------------------------------------------------
def pil_to_array(img: Image.Image, max_side: int | None = None) -> np.ndarray:
    """PIL -> array uint8. Mode abu-abu tetap abu-abu, lainnya menjadi RGB."""
    if img.mode in ("L", "1", "LA"):
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Array uint8 -> PNG (lossless). Wajib PNG agar dekripsi dari berkas tetap sempurna."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _check_image(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        raise ValueError("Citra harus bertipe uint8 (0..255).")
    if a.ndim not in (2, 3):
        raise ValueError("Citra harus berdimensi 2 (abu-abu) atau 3 (berwarna).")
    return a


# --------------------------------------------------------------------------
# Difusi (inti berantai)
# --------------------------------------------------------------------------
def _diffuse_forward(x: np.ndarray, k: np.ndarray, s: np.ndarray, iv: int) -> np.ndarray:
    xl, kl, sl = x.tolist(), k.tolist(), s.tolist()
    out = [0] * len(xl)
    prev = iv
    for i in range(len(xl)):
        prev = sl[(xl[i] + kl[i] + prev) & 255]
        out[i] = prev
    return np.array(out, dtype=np.int64)


def _diffuse_backward(x: np.ndarray, k: np.ndarray, s: np.ndarray, iv: int) -> np.ndarray:
    xl, kl, sl = x.tolist(), k.tolist(), s.tolist()
    n = len(xl)
    out = [0] * n
    prev = iv
    for i in range(n - 1, -1, -1):
        prev = sl[(xl[i] + kl[i] + prev) & 255]
        out[i] = prev
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------
# Enkripsi / dekripsi
# --------------------------------------------------------------------------
def encrypt_image(img, sbox, key: str, return_stages: bool = False):
    """Enkripsi citra uint8 (H,W) atau (H,W,3).

    Jika return_stages=True, mengembalikan dict berisi hasil tiap tahap.
    """
    arr = _check_image(img)
    s = as_sbox_array(sbox)
    inverse_sbox(s)  # memastikan bijektif (melempar ValueError bila tidak)
    shape = arr.shape
    flat = arr.reshape(-1).astype(np.int64)
    n = flat.size

    perm, k1, k2, iv1, iv2 = derive_key_material(key, n)

    substituted = s[flat]
    permuted = substituted[perm]
    forward = _diffuse_forward(permuted, k1, s, iv1)
    cipher = _diffuse_backward(forward, k2, s, iv2)

    def to_img(v):
        return v.astype(np.uint8).reshape(shape)

    if return_stages:
        return {
            "substitution": to_img(substituted),
            "permutation": to_img(permuted),
            "diffusion_forward": to_img(forward),
            "cipher": to_img(cipher),
        }
    return to_img(cipher)


def decrypt_image(cipher, sbox, key: str) -> np.ndarray:
    """Dekripsi citra sandi (uint8) dengan S-Box dan kunci yang sama."""
    arr = _check_image(cipher)
    s = as_sbox_array(sbox)
    inv = inverse_sbox(s)
    shape = arr.shape
    c = arr.reshape(-1).astype(np.int64)
    n = c.size

    perm, k1, k2, iv1, iv2 = derive_key_material(key, n)

    # balik difusi mundur: A[i] = inv[C[i]] - K2[i] - C[i+1]
    c_next = np.empty_like(c)
    c_next[:-1] = c[1:]
    c_next[-1] = iv2
    a = (inv[c] - k2 - c_next) & 255

    # balik difusi maju: Q[i] = inv[A[i]] - K1[i] - A[i-1]
    a_prev = np.empty_like(a)
    a_prev[1:] = a[:-1]
    a_prev[0] = iv1
    permuted = (inv[a] - k1 - a_prev) & 255

    # balik permutasi
    substituted = np.empty_like(permuted)
    substituted[perm] = permuted

    # balik substitusi
    plain = inv[substituted]
    return plain.astype(np.uint8).reshape(shape)
