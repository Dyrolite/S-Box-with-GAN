# -*- coding: utf-8 -*-
"""
sbox_dataset.py
===============
Persiapan data latih untuk GAN S-Box (dipakai di notebook Google Colab).

GAN membutuhkan contoh "S-Box nyata" agar Discriminator punya pembanding.
Di proyek ini contoh tersebut dibuat dengan cara SEDERHANA dan TIDAK memakai
konstruksi invers GF(2^8):

  1. Buat permutasi acak 0..255.
  2. Perbaiki sedikit dengan hill-climbing (tukar dua nilai bila kualitas
     nonlinearity membaik).
  3. Perbanyak data dengan transformasi afin (S'(x) = L2(S(L1(x) ^ d)) ^ c).
     Transformasi ini mempertahankan NL, DAP, dan LAP, sehingga data baru
     tetap "sebaik" S-Box asalnya.

Catatan: kualitas dataset ini menjadi batas atas kasar bagi kualitas yang bisa
ditiru GAN. Dataset ini TIDAK berisi S-Box AES.
"""
from __future__ import annotations

import numpy as np

from .sbox_metrics import (MASKS, PARITY, SIZE, as_sbox_array, hadamard_matrix,
                           nonlinearity_per_mask, walsh_spectrum)

_H_F = hadamard_matrix(SIZE).astype(np.float32)


def random_sbox(rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(SIZE).astype(np.int64)


def _signs(sbox: np.ndarray) -> np.ndarray:
    """(255, 256) float32 berisi (-1)^(b . S(x))."""
    return (1 - 2 * PARITY[MASKS[:, None] & sbox[None, :]]).astype(np.float32)


def hill_climb(sbox, steps: int, rng: np.random.Generator, power: int = 3) -> np.ndarray:
    """Hill-climbing sederhana: tukar dua entri bila biaya sum|W|^power turun.

    Pembaruan spektrum Walsh dilakukan secara inkremental (cepat).
    """
    s = as_sbox_array(sbox).copy()
    signs = _signs(s)                       # (255, 256)
    walsh = signs @ _H_F                    # (255, 256)  W[b, a]
    cost = float(np.sum(np.abs(walsh) ** power))
    for _ in range(int(steps)):
        i, j = rng.choice(SIZE, size=2, replace=False)
        d = signs[:, j] - signs[:, i]       # perubahan kolom bila S[i] <-> S[j]
        new_w = walsh + np.outer(d, _H_F[i] - _H_F[j])
        new_cost = float(np.sum(np.abs(new_w) ** power))
        if new_cost < cost:
            walsh, cost = new_w, new_cost
            s[i], s[j] = s[j], s[i]
            signs[:, [i, j]] = signs[:, [j, i]]
    return s


def _random_linear_table(rng: np.random.Generator) -> np.ndarray:
    """Tabel transformasi linear 8-bit yang bijektif (acak)."""
    while True:
        rows = rng.integers(1, SIZE, size=8)
        table = np.zeros(SIZE, dtype=np.int64)
        for bit in range(8):
            table |= PARITY[rows[bit] & np.arange(SIZE)].astype(np.int64) << bit
        if len(np.unique(table)) == SIZE:
            return table


def affine_equivalent(sbox, rng: np.random.Generator) -> np.ndarray:
    """S'(x) = L2( S( L1(x) ^ d ) ) ^ c  -> NL, DAP, LAP sama dengan S asal."""
    s = as_sbox_array(sbox)
    l1 = _random_linear_table(rng)
    l2 = _random_linear_table(rng)
    d = int(rng.integers(0, SIZE))
    c = int(rng.integers(0, SIZE))
    x = np.arange(SIZE)
    return (l2[s[l1[x] ^ d]] ^ c).astype(np.int64)


def build_dataset(pool_size: int, hc_steps: int, aug_per_pool: int, seed: int = 0,
                  verbose: bool = True):
    """Bangun dataset S-Box.

    Mengembalikan (data uint8 [N,256], nl_pool list[int]).
    """
    rng = np.random.default_rng(seed)
    pool, nl_pool = [], []
    for n in range(pool_size):
        s = hill_climb(random_sbox(rng), hc_steps, rng)
        nl = int(nonlinearity_per_mask(walsh_spectrum(s)).min())
        pool.append(s)
        nl_pool.append(nl)
        if verbose and (n + 1) % max(1, pool_size // 10) == 0:
            print(f"  pool {n + 1}/{pool_size} | NL terbaik sejauh ini: {max(nl_pool)}")
    data = []
    for s in pool:
        data.append(s)
        for _ in range(aug_per_pool):
            data.append(affine_equivalent(s, rng))
    data = np.stack(data).astype(np.uint8)
    perm = rng.permutation(len(data))
    return data[perm], nl_pool
