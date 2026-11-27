# -*- coding: utf-8 -*-
"""
image_metrics.py
================
Evaluasi kualitas enkripsi citra: entropy, histogram, NPCR, UACI, korelasi
piksel bertetangga, serta pemeriksaan dekripsi (MSE/PSNR/identik).

Tentang istilah pada catatan tugas:
  * "MBCR"      : belum ada definisi pasti -> TIDAK dihitung, TIDAK dikarang.
                  Harus dikonfirmasi ke dosen.
  * "distogram" : diasumsikan salah ketik dari "histogram". Perlu konfirmasi.
"""
from __future__ import annotations

import math

import numpy as np

from .image_crypto import encrypt_image

# Nilai harapan teoretis untuk citra 8-bit acak ideal (rujukan umum di literatur)
NPCR_EXPECTED = (1 - 2 ** -8) * 100          # 99.6094 %
UACI_EXPECTED = 33.4635                      # %
ENTROPY_IDEAL = 8.0


def shannon_entropy(arr) -> float:
    """Entropy Shannon (bit/piksel) seluruh nilai byte pada citra."""
    a = np.asarray(arr, dtype=np.uint8).reshape(-1)
    counts = np.bincount(a, minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def channel_entropies(arr) -> dict:
    a = np.asarray(arr, dtype=np.uint8)
    if a.ndim == 2:
        return {"Gray": shannon_entropy(a)}
    return {name: shannon_entropy(a[..., i]) for i, name in enumerate(("R", "G", "B"))}


def histograms(arr) -> dict:
    """Histogram 256-bin per kanal. Kunci: 'Gray' atau 'R','G','B'."""
    a = np.asarray(arr, dtype=np.uint8)
    if a.ndim == 2:
        return {"Gray": np.bincount(a.reshape(-1), minlength=256)}
    return {name: np.bincount(a[..., i].reshape(-1), minlength=256)
            for i, name in enumerate(("R", "G", "B"))}


def npcr(c1, c2) -> float:
    """Number of Pixels Change Rate (%)."""
    a, b = np.asarray(c1), np.asarray(c2)
    if a.shape != b.shape:
        raise ValueError("Ukuran citra harus sama.")
    return float((a != b).mean() * 100)


def uaci(c1, c2) -> float:
    """Unified Average Changing Intensity (%)."""
    a = np.asarray(c1).astype(np.int64)
    b = np.asarray(c2).astype(np.int64)
    if a.shape != b.shape:
        raise ValueError("Ukuran citra harus sama.")
    return float((np.abs(a - b) / 255.0).mean() * 100)


def mse(a, b) -> float:
    d = np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64)
    return float((d ** 2).mean())


def psnr(a, b) -> float:
    m = mse(a, b)
    return float("inf") if m == 0 else float(10 * math.log10(255.0 ** 2 / m))


def adjacent_correlation(arr, n_pairs: int = 5000, seed: int = 0) -> dict:
    """Koefisien korelasi piksel bertetangga (horizontal, vertikal, diagonal)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    rng = np.random.default_rng(seed)
    out = {}
    for name, (dy, dx) in {"Horizontal": (0, 1), "Vertikal": (1, 0), "Diagonal": (1, 1)}.items():
        if h <= dy or w <= dx:
            out[name] = None
            continue
        ys = rng.integers(0, h - dy, size=n_pairs)
        xs = rng.integers(0, w - dx, size=n_pairs)
        u = a[ys, xs]
        v = a[ys + dy, xs + dx]
        if u.std() == 0 or v.std() == 0:
            out[name] = None
        else:
            out[name] = float(np.corrcoef(u, v)[0, 1])
    return out


def plaintext_sensitivity(img, sbox, key: str, cipher_ref=None, n_trials: int = 5,
                          seed: int = 0, progress_cb=None) -> dict:
    """Uji sensitivitas terhadap perubahan citra asli (standar NPCR/UACI).

    Satu byte pada citra asli diubah 1 bit (LSB dibalik), lalu kedua citra
    dienkripsi dan dibandingkan. Diulang n_trials kali pada posisi acak.
    """
    arr = np.asarray(img, dtype=np.uint8)
    if cipher_ref is None:
        cipher_ref = encrypt_image(arr, sbox, key)
    rng = np.random.default_rng(seed)
    n = arr.size
    npcr_list, uaci_list, positions = [], [], []
    for t in range(n_trials):
        pos = int(rng.integers(0, n))
        mod = arr.copy().reshape(-1)
        mod[pos] ^= 1
        c2 = encrypt_image(mod.reshape(arr.shape), sbox, key)
        npcr_list.append(npcr(cipher_ref, c2))
        uaci_list.append(uaci(cipher_ref, c2))
        positions.append(pos)
        if progress_cb:
            progress_cb(t + 1, n_trials)
    return {
        "n_trials": int(n_trials),
        "npcr_mean": float(np.mean(npcr_list)), "npcr_min": float(np.min(npcr_list)),
        "npcr_max": float(np.max(npcr_list)),
        "uaci_mean": float(np.mean(uaci_list)), "uaci_min": float(np.min(uaci_list)),
        "uaci_max": float(np.max(uaci_list)),
        "npcr_each": [float(v) for v in npcr_list],
        "uaci_each": [float(v) for v in uaci_list],
        "positions": positions,
        "npcr_expected": NPCR_EXPECTED, "uaci_expected": UACI_EXPECTED,
    }


def build_image_report(original, cipher, decrypted, sensitivity: dict | None = None) -> dict:
    """Kumpulkan seluruh hasil analisis citra menjadi satu dict (aman untuk JSON)."""
    rep = {
        "ukuran_citra": list(np.asarray(original).shape),
        "entropy": {
            "asli": shannon_entropy(original),
            "terenkripsi": shannon_entropy(cipher),
            "hasil_dekripsi": shannon_entropy(decrypted),
            "ideal": ENTROPY_IDEAL,
            "per_kanal_asli": channel_entropies(original),
            "per_kanal_terenkripsi": channel_entropies(cipher),
        },
        "verifikasi_dekripsi": {
            "identik_dengan_asli": bool(np.array_equal(original, decrypted)),
            "mse": mse(original, decrypted),
            "psnr_db": None if psnr(original, decrypted) == float("inf") else psnr(original, decrypted),
        },
        "asli_vs_terenkripsi": {
            "npcr_persen": npcr(original, cipher),
            "uaci_persen": uaci(original, cipher),
            "catatan": "Perbandingan langsung citra asli dan sandi (informasi tambahan, bukan uji NPCR/UACI standar).",
        },
        "korelasi_piksel_bertetangga": {
            "asli": adjacent_correlation(original),
            "terenkripsi": adjacent_correlation(cipher),
        },
        "mbcr": "Tidak dihitung: definisi MBCR belum jelas. Perlu konfirmasi dosen.",
    }
    if sensitivity is not None:
        rep["sensitivitas_citra_asli"] = sensitivity
    return rep


def report_rows(rep: dict) -> list[dict]:
    """Baris tabel ringkas dari build_image_report (untuk tampilan dan CSV)."""
    e = rep["entropy"]
    v = rep["verifikasi_dekripsi"]
    a = rep["asli_vs_terenkripsi"]
    rows = [
        {"Metrik": "Entropy citra asli (bit)", "Nilai": round(e["asli"], 5), "Acuan": "-"},
        {"Metrik": "Entropy citra terenkripsi (bit)", "Nilai": round(e["terenkripsi"], 5), "Acuan": "mendekati 8"},
        {"Metrik": "Entropy hasil dekripsi (bit)", "Nilai": round(e["hasil_dekripsi"], 5), "Acuan": "sama dengan asli"},
        {"Metrik": "Dekripsi identik dengan asli", "Nilai": "Ya" if v["identik_dengan_asli"] else "Tidak", "Acuan": "Ya"},
        {"Metrik": "MSE asli vs dekripsi", "Nilai": round(v["mse"], 6), "Acuan": "0"},
        {"Metrik": "PSNR asli vs dekripsi (dB)", "Nilai": "tak hingga" if v["psnr_db"] is None else round(v["psnr_db"], 3), "Acuan": "tak hingga"},
        {"Metrik": "NPCR asli vs sandi (%) [tambahan]", "Nilai": round(a["npcr_persen"], 4), "Acuan": "-"},
        {"Metrik": "UACI asli vs sandi (%) [tambahan]", "Nilai": round(a["uaci_persen"], 4), "Acuan": "-"},
    ]
    s = rep.get("sensitivitas_citra_asli")
    if s:
        rows += [
            {"Metrik": f"NPCR sensitivitas, rata-rata {s['n_trials']} uji (%)", "Nilai": round(s["npcr_mean"], 4), "Acuan": f"{NPCR_EXPECTED:.4f}"},
            {"Metrik": f"UACI sensitivitas, rata-rata {s['n_trials']} uji (%)", "Nilai": round(s["uaci_mean"], 4), "Acuan": f"{UACI_EXPECTED:.4f}"},
        ]
    for tag, d in rep["korelasi_piksel_bertetangga"].items():
        for arah, val in d.items():
            rows.append({"Metrik": f"Korelasi {arah.lower()} ({tag})",
                         "Nilai": "n/a" if val is None else round(val, 5),
                         "Acuan": "mendekati 0 untuk citra sandi"})
    return rows
