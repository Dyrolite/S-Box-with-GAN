# -*- coding: utf-8 -*-
"""
visualization.py
================
Fungsi pembuat grafik (matplotlib) untuk S-Box dan analisis citra.
Semua fungsi mengembalikan objek Figure; gunakan fig_to_png_bytes() untuk unduhan.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # aman untuk server/Streamlit Cloud (tanpa layar)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .image_metrics import histograms  # noqa: E402
from .sbox_metrics import as_sbox_array  # noqa: E402


def fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def fig_sbox_grid(sbox, title: str = "Tabel S-Box 16 x 16 (heksadesimal)"):
    grid = as_sbox_array(sbox).reshape(16, 16)
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    ax.imshow(grid, cmap="viridis", vmin=0, vmax=255, alpha=0.35)
    for r in range(16):
        for c in range(16):
            ax.text(c, r, f"{grid[r, c]:02X}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(16))
    ax.set_yticks(range(16))
    ax.set_xticklabels([f"{i:X}" for i in range(16)])
    ax.set_yticklabels([f"{i:X}" for i in range(16)])
    ax.xaxis.tick_top()
    ax.set_xlabel("Nibble rendah dari x")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("Nibble tinggi dari x")
    ax.set_title(title, pad=28)
    return fig


def fig_sac_heatmap(sac_matrix):
    m = np.asarray(sac_matrix)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(m, cmap="coolwarm", vmin=0.3, vmax=0.7)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Bit keluaran j")
    ax.set_ylabel("Bit masukan i yang dibalik")
    ax.set_title("Matriks SAC (ideal = 0.5)")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


def fig_nl_distribution(per_mask):
    vals = np.asarray(per_mask)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    lo, hi = int(vals.min()), int(vals.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)
    ax.hist(vals, bins=bins, color="#4C78A8", edgecolor="white")
    ax.axvline(112, color="#E45756", linestyle="--", label="Acuan NL = 112")
    ax.set_xlabel("Nonlinearity fungsi komponen")
    ax.set_ylabel("Jumlah fungsi komponen (dari 255)")
    ax.set_title("Sebaran nonlinearity 255 fungsi komponen")
    ax.legend()
    return fig


def fig_ddt_histogram(ddt):
    d = np.asarray(ddt)[1:, :].ravel()
    vals, counts = np.unique(d, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar([str(v) for v in vals], counts, color="#59A14F")
    ax.set_yscale("log")
    ax.set_xlabel("Nilai entri DDT")
    ax.set_ylabel("Jumlah entri (skala log)")
    ax.set_title("Sebaran entri tabel DDT (dx != 0)")
    return fig


_COLORS = {"R": "#E15759", "G": "#59A14F", "B": "#4E79A7", "Gray": "#555555"}


def fig_histograms(images: dict):
    """images: {"Citra asli": arr, "Citra terenkripsi": arr, ...} -> satu baris histogram."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4), squeeze=False)
    for ax, (title, arr) in zip(axes[0], images.items()):
        for name, h in histograms(arr).items():
            ax.plot(np.arange(256), h, color=_COLORS[name], linewidth=0.9, alpha=0.85, label=name)
        ax.set_title(title)
        ax.set_xlabel("Nilai intensitas (0-255)")
        ax.set_ylabel("Jumlah")
        ax.set_xlim(0, 255)
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
