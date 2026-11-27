# -*- coding: utf-8 -*-
"""
Verifikasi rumus matematika yang dipakai pada loss GAN di notebook Colab.

Notebook menghitung (dalam PyTorch) ukuran kualitas S-Box langsung dari matriks
one-hot M[x, k] = 1 jika S(x) = k, memakai perkalian matriks dengan matriks
Hadamard H:

    W   = H @ M @ H                 (spektrum Walsh, W[a, b])
    DDT = H @ (W * W) @ H / 65536   (tabel distribusi selisih)
    SAC = rata-rata P(bit keluaran berubah) dari p = M @ BITS

File ini membuktikan (dengan NumPy) bahwa rumus tersebut sama persis dengan
perhitungan brute-force pada modul sbox_metrics.  Jalankan:

    python tests/test_gan_math.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sbox_metrics as sm  # noqa: E402

H = sm.hadamard_matrix(256).astype(np.float64)
BITS = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)
IDX = np.stack([np.arange(256) ^ (1 << i) for i in range(8)])      # (8, 256)


def one_hot(s):
    m = np.zeros((256, 256))
    m[np.arange(256), s] = 1.0
    return m


def sac_from_M(m):
    p = m @ BITS                                   # (256, 8)  P(bit j = 1 | x)
    out = np.zeros((8, 8))
    for i in range(8):
        p2 = p[IDX[i]]
        flip = p * (1 - p2) + (1 - p) * p2
        out[i] = flip.mean(axis=0)
    return out


def test_walsh_formula():
    for seed in range(3):
        s = np.random.default_rng(seed).permutation(256)
        m = one_hot(s)
        w = H @ m @ H                              # W[a, b]
        ref = sm.walsh_spectrum(s)                 # [b-1, a]
        assert np.array_equal(w[:, 1:].T, ref)


def test_ddt_formula():
    for seed in range(3):
        s = np.random.default_rng(10 + seed).permutation(256)
        m = one_hot(s)
        w = H @ m @ H
        ddt = H @ (w * w) @ H / 65536.0
        ref = sm.difference_distribution_table(s)
        assert np.allclose(ddt, ref, atol=1e-6), np.abs(ddt - ref).max()


def test_sac_formula():
    for seed in range(3):
        s = np.random.default_rng(20 + seed).permutation(256)
        assert np.allclose(sac_from_M(one_hot(s)), sm.sac_matrix(s), atol=1e-12)


def test_hinge_terms_on_hard_sbox():
    s = np.random.default_rng(30).permutation(256)
    m = one_hot(s)
    w = H @ m @ H
    max_w = np.abs(w[:, 1:]).max()
    assert (256 - max_w) // 2 == sm.evaluate_sbox(s)["nonlinearity"]["min"]
    ddt = H @ (w * w) @ H / 65536.0
    assert int(round(ddt[1:].max())) == sm.evaluate_sbox(s)["dap"]["differential_uniformity"]


def test_softsort_orientation():
    """Replika NumPy dari soft_sort di notebook: P[k, x] ~ 1 bila elemen x berperingkat k."""
    rng = np.random.default_rng(1)
    s = rng.standard_normal(256)
    s = (s - s.mean()) / s.std()
    tau = 0.003
    srt = np.sort(s)
    diff = np.abs(srt[:, None] - s[None, :])
    logits = -diff / tau
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)              # softmax terhadap elemen x
    rank = np.argsort(np.argsort(s))               # S(x) = peringkat elemen x
    m_soft = p.T                                   # M[x, k] = P[k, x]
    assert np.array_equal(m_soft.argmax(axis=1), rank)
    assert np.allclose(m_soft.sum(axis=0), 1.0)    # tiap k punya total bobot 1


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[LULUS] {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[GAGAL] {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} pengujian lulus.")
    sys.exit(1 if failed else 0)
