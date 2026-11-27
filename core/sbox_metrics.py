# -*- coding: utf-8 -*-
"""
sbox_metrics.py
===============
Modul pengujian S-Box 8-bit (256 nilai, tiap nilai 0..255).

Semua nilai dihitung langsung dari tabel S-Box yang diberikan (tidak ada nilai
yang ditulis manual). Pengujian yang tersedia:

  * Bijectivity       : apakah S-Box berupa permutasi 0..255
  * NL  (Nonlinearity): min nonlinearity seluruh fungsi komponen (mask 1..255)
  * SAC (Strict Avalanche Criterion): matriks 8x8, ideal 0.5
  * BIC-NL & BIC-SAC  : Bit Independence Criterion (pasangan bit keluaran)
  * DAP (Differential Approximation Probability)
  * LAP (Linear Approximation Probability)

Catatan istilah:
  - Fungsi komponen untuk mask b adalah f_b(x) = b . S(x)  (parity bit-AND).
  - Spektrum Walsh: W(a,b) = sum_x (-1)^( a.x  XOR  b.S(x) ).
  - NL = 128 - max_{a, b!=0} |W(a,b)| / 2.
  - LAP = max_{a!=0, b!=0} |W(a,b)| / 512.
  - DAP = max_{dx!=0, dy} #{x : S(x) XOR S(x XOR dx) = dy} / 256.
"""
from __future__ import annotations

import numpy as np

N_BITS = 8
SIZE = 1 << N_BITS  # 256

# Nilai acuan (dari literatur populer, dipakai hanya sebagai pembanding).
NL_TARGET = 112          # acuan dari catatan tugas
SAC_IDEAL = 0.5
DAP_REF = 4 / 256        # = 0.015625 (nilai yang umum dikutip untuk S-Box AES)
LAP_REF = 32 / 512       # = 0.0625   (nilai yang umum dikutip untuk S-Box AES)


def _make_parity_table() -> np.ndarray:
    table = np.zeros(SIZE, dtype=np.int8)
    for v in range(SIZE):
        table[v] = bin(v).count("1") & 1
    return table


PARITY = _make_parity_table()
X_ALL = np.arange(SIZE, dtype=np.int64)
MASKS = np.arange(1, SIZE, dtype=np.int64)       # mask keluaran b = 1..255
BIT_INDEX = np.arange(N_BITS, dtype=np.int64)


def hadamard_matrix(n: int = SIZE) -> np.ndarray:
    """Matriks Hadamard Sylvester: H[i, j] = (-1)^popcount(i & j)."""
    h = np.ones((1, 1), dtype=np.int64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


# --------------------------------------------------------------------------
# Utilitas dasar
# --------------------------------------------------------------------------
def as_sbox_array(sbox) -> np.ndarray:
    """Ubah input (list/array 256 elemen atau 16x16) menjadi array int64 (256,)."""
    arr = np.asarray(sbox)
    if arr.size != SIZE:
        raise ValueError(f"S-Box harus berisi {SIZE} nilai, tetapi diberikan {arr.size}.")
    arr = arr.reshape(-1)
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.mod(arr, 1) == 0):
            raise ValueError("Nilai S-Box harus bilangan bulat.")
    arr = arr.astype(np.int64)
    if arr.min() < 0 or arr.max() > 255:
        raise ValueError("Nilai S-Box harus berada pada rentang 0..255.")
    return arr


def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform pada sumbu terakhir (panjang harus 2^n)."""
    a = np.array(a, dtype=np.int64, copy=True)
    n = a.shape[-1]
    lead = a.shape[:-1]
    h = 1
    while h < n:
        a = a.reshape(lead + (n // (2 * h), 2, h))
        x = a[..., 0, :]
        y = a[..., 1, :]
        a = np.stack((x + y, x - y), axis=-2)
        a = a.reshape(lead + (n,))
        h *= 2
    return a


def walsh_spectrum(sbox) -> np.ndarray:
    """Spektrum Walsh seluruh fungsi komponen.

    Mengembalikan array (255, 256): baris = mask b-1 (b = 1..255), kolom = a.
    """
    s = as_sbox_array(sbox)
    bits = PARITY[MASKS[:, None] & s[None, :]].astype(np.int64)
    return fwht(1 - 2 * bits)


def nonlinearity_per_mask(walsh: np.ndarray) -> np.ndarray:
    """Nonlinearity tiap fungsi komponen (array panjang 255, indeks = mask-1)."""
    return (SIZE - np.abs(walsh).max(axis=1)) // 2


def difference_distribution_table(sbox) -> np.ndarray:
    """DDT (256x256): ddt[dx, dy] = #{x : S(x) ^ S(x^dx) = dy}."""
    s = as_sbox_array(sbox)
    dx = X_ALL[:, None]
    dy = s[None, :] ^ s[X_ALL[None, :] ^ dx]
    flat = (dx * SIZE + dy).ravel()
    return np.bincount(flat, minlength=SIZE * SIZE).reshape(SIZE, SIZE)


# --------------------------------------------------------------------------
# Pengujian individual
# --------------------------------------------------------------------------
def check_bijective(sbox) -> dict:
    s = as_sbox_array(sbox)
    uniq = np.unique(s)
    missing = sorted(set(range(SIZE)) - set(uniq.tolist()))
    return {
        "is_bijective": bool(len(uniq) == SIZE),
        "unique_values": int(len(uniq)),
        "missing_values": [int(v) for v in missing[:20]],
    }


def sac_matrix(sbox) -> np.ndarray:
    """Matriks SAC 8x8: [i, j] = P(bit keluaran j berubah | bit masukan i dibalik)."""
    s = as_sbox_array(sbox)
    out = np.zeros((N_BITS, N_BITS), dtype=np.float64)
    for i in range(N_BITS):
        d = s ^ s[X_ALL ^ (1 << i)]
        out[i] = ((d[:, None] >> BIT_INDEX[None, :]) & 1).mean(axis=0)
    return out


def bic_pairs():
    return [(j, k) for j in range(N_BITS) for k in range(j + 1, N_BITS)]


def bic_sac_matrix(sbox) -> np.ndarray:
    """BIC-SAC: array (28, 8). Baris = pasangan bit keluaran (j<k), kolom = bit masukan i.

    Nilai = P(f_j XOR f_k berubah | bit masukan i dibalik). Ideal 0.5.
    """
    s = as_sbox_array(sbox)
    pairs = bic_pairs()
    out = np.zeros((len(pairs), N_BITS), dtype=np.float64)
    for i in range(N_BITS):
        d = s ^ s[X_ALL ^ (1 << i)]
        bits = (d[:, None] >> BIT_INDEX[None, :]) & 1
        for p, (j, k) in enumerate(pairs):
            out[p, i] = (bits[:, j] ^ bits[:, k]).mean()
    return out


# --------------------------------------------------------------------------
# Evaluasi lengkap
# --------------------------------------------------------------------------
def evaluate_sbox(sbox) -> dict:
    """Hitung seluruh pengujian. Hasil berupa dict yang aman diubah ke JSON."""
    s = as_sbox_array(sbox)

    bij = check_bijective(s)
    walsh = walsh_spectrum(s)
    nl_mask = nonlinearity_per_mask(walsh)
    ddt = difference_distribution_table(s)
    sac = sac_matrix(s)
    bic_sac = bic_sac_matrix(s)

    nl_min = int(nl_mask.min())
    max_walsh_nonzero_a = int(np.abs(walsh[:, 1:]).max())  # a != 0, b != 0
    du = int(ddt[1:, :].max())

    bic_masks = np.array([(1 << j) | (1 << k) for j, k in bic_pairs()])
    bic_nl = nl_mask[bic_masks - 1]

    return {
        "bijective": bij,
        "nonlinearity": {
            "min": nl_min,
            "max": int(nl_mask.max()),
            "mean": float(nl_mask.mean()),
            "target": NL_TARGET,
            "reached_target": bool(nl_min >= NL_TARGET),
            "per_mask": [int(v) for v in nl_mask],
        },
        "sac": {
            "mean": float(sac.mean()),
            "min": float(sac.min()),
            "max": float(sac.max()),
            "max_abs_deviation": float(np.abs(sac - SAC_IDEAL).max()),
            "ideal": SAC_IDEAL,
            "matrix": sac.tolist(),
        },
        "bic_nl": {
            "min": int(bic_nl.min()),
            "mean": float(bic_nl.mean()),
            "max": int(bic_nl.max()),
        },
        "bic_sac": {
            "mean": float(bic_sac.mean()),
            "min": float(bic_sac.min()),
            "max": float(bic_sac.max()),
            "max_abs_deviation": float(np.abs(bic_sac - SAC_IDEAL).max()),
            "ideal": SAC_IDEAL,
        },
        "dap": {
            "max": float(du / SIZE),
            "differential_uniformity": du,
            "reference": DAP_REF,
        },
        "lap": {
            "max": float(max_walsh_nonzero_a / (2 * SIZE)),
            "max_abs_walsh": max_walsh_nonzero_a,
            "reference": LAP_REF,
        },
        "extra": {
            "fixed_points": int(np.sum(s == X_ALL)),
            "opposite_fixed_points": int(np.sum(s == (X_ALL ^ 255))),
        },
    }


def metric_rows(report: dict) -> list[dict]:
    """Ubah hasil evaluate_sbox menjadi baris tabel (untuk tampilan dan CSV)."""
    r = report
    nl = r["nonlinearity"]
    rows = [
        {"Pengujian": "Bijectivity", "Metrik": "S-Box bijektif",
         "Nilai": "Ya" if r["bijective"]["is_bijective"] else "Tidak",
         "Acuan": "Ya (256 nilai unik)",
         "Keterangan": f"{r['bijective']['unique_values']} nilai unik dari 256"},
        {"Pengujian": "NL", "Metrik": "Nonlinearity minimum",
         "Nilai": nl["min"], "Acuan": f"sekitar {NL_TARGET}",
         "Keterangan": "Target tercapai" if nl["reached_target"] else f"Selisih terhadap {NL_TARGET}: {NL_TARGET - nl['min']}"},
        {"Pengujian": "NL", "Metrik": "Nonlinearity rata-rata",
         "Nilai": round(nl["mean"], 4), "Acuan": "-", "Keterangan": "Rata-rata 255 fungsi komponen"},
        {"Pengujian": "NL", "Metrik": "Nonlinearity maksimum",
         "Nilai": nl["max"], "Acuan": "-", "Keterangan": "Komponen terbaik"},
        {"Pengujian": "SAC", "Metrik": "Rata-rata matriks SAC",
         "Nilai": round(r["sac"]["mean"], 5), "Acuan": "0.5", "Keterangan": "Semakin dekat 0.5 semakin baik"},
        {"Pengujian": "SAC", "Metrik": "Nilai SAC minimum",
         "Nilai": round(r["sac"]["min"], 5), "Acuan": "0.5", "Keterangan": ""},
        {"Pengujian": "SAC", "Metrik": "Nilai SAC maksimum",
         "Nilai": round(r["sac"]["max"], 5), "Acuan": "0.5", "Keterangan": ""},
        {"Pengujian": "SAC", "Metrik": "Deviasi maksimum dari 0.5",
         "Nilai": round(r["sac"]["max_abs_deviation"], 5), "Acuan": "0", "Keterangan": ""},
        {"Pengujian": "BIC", "Metrik": "BIC-NL minimum",
         "Nilai": r["bic_nl"]["min"], "Acuan": f"sekitar {NL_TARGET}", "Keterangan": "Dari 28 pasangan bit keluaran"},
        {"Pengujian": "BIC", "Metrik": "BIC-NL rata-rata",
         "Nilai": round(r["bic_nl"]["mean"], 4), "Acuan": "-", "Keterangan": ""},
        {"Pengujian": "BIC", "Metrik": "BIC-SAC rata-rata",
         "Nilai": round(r["bic_sac"]["mean"], 5), "Acuan": "0.5", "Keterangan": ""},
        {"Pengujian": "BIC", "Metrik": "BIC-SAC deviasi maksimum dari 0.5",
         "Nilai": round(r["bic_sac"]["max_abs_deviation"], 5), "Acuan": "0", "Keterangan": ""},
        {"Pengujian": "DAP", "Metrik": "DAP maksimum",
         "Nilai": round(r["dap"]["max"], 6), "Acuan": f"{DAP_REF:.6f} (acuan AES)",
         "Keterangan": f"Differential uniformity = {r['dap']['differential_uniformity']}"},
        {"Pengujian": "LAP", "Metrik": "LAP maksimum",
         "Nilai": round(r["lap"]["max"], 6), "Acuan": f"{LAP_REF:.6f} (acuan AES)",
         "Keterangan": f"Maks |Walsh| = {r['lap']['max_abs_walsh']}"},
        {"Pengujian": "Tambahan", "Metrik": "Titik tetap S(x)=x",
         "Nilai": r["extra"]["fixed_points"], "Acuan": "0", "Keterangan": ""},
        {"Pengujian": "Tambahan", "Metrik": "Titik tetap berlawanan S(x)=~x",
         "Nilai": r["extra"]["opposite_fixed_points"], "Acuan": "0", "Keterangan": ""},
    ]
    return rows


# --------------------------------------------------------------------------
# Evaluasi cepat (dipakai untuk memilih S-Box terbaik di notebook)
# --------------------------------------------------------------------------
def quick_evaluate(sbox, with_sac: bool = False) -> dict:
    """Evaluasi ringan: NL, differential uniformity, LAP (opsional SAC)."""
    s = as_sbox_array(sbox)
    walsh = walsh_spectrum(s)
    nl_min = int((SIZE - np.abs(walsh).max()) // 2)
    max_w_a_nonzero = int(np.abs(walsh[:, 1:]).max())
    du = int(difference_distribution_table(s)[1:, :].max())
    out = {"nl": nl_min, "du": du, "lap": max_w_a_nonzero / (2 * SIZE),
           "bijective": bool(len(np.unique(s)) == SIZE)}
    if with_sac:
        out["sac_dev"] = float(np.abs(sac_matrix(s) - SAC_IDEAL).max())
    return out


def selection_key(q: dict) -> tuple:
    """Kunci pengurutan (kecil = lebih baik): NL tinggi, DU rendah, LAP rendah, SAC baik."""
    return (0 if q.get("bijective", True) else 1, -q["nl"], q["du"], q["lap"], q.get("sac_dev", 0.0))
