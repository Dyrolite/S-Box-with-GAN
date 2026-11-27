# -*- coding: utf-8 -*-
"""
sbox_io.py
==========
Membaca dan menulis S-Box (JSON/CSV) serta utilitas S-Box invers.

Format berkas JSON (dihasilkan notebook Colab, dibaca aplikasi):
{
  "format_version": 1,
  "name": "...",
  "source": "gan" | "placeholder_hillclimb" | ...,
  "is_gan_generated": true/false,
  "created_at": "...",
  "sbox": [256 bilangan bulat 0..255],
  "training": {... catatan pelatihan ...}
}
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np

from .sbox_metrics import SIZE, as_sbox_array

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SBOX_PATH = PROJECT_ROOT / "sbox" / "best_sbox.json"


def is_bijective(sbox) -> bool:
    return len(np.unique(as_sbox_array(sbox))) == SIZE


def inverse_sbox(sbox) -> np.ndarray:
    """S-Box invers. Hanya valid jika S-Box bijektif."""
    s = as_sbox_array(sbox)
    if len(np.unique(s)) != SIZE:
        raise ValueError("S-Box tidak bijektif, sehingga tidak punya invers (dekripsi tidak mungkin).")
    inv = np.zeros(SIZE, dtype=np.int64)
    inv[s] = np.arange(SIZE)
    return inv


def sbox_grid(sbox) -> np.ndarray:
    return as_sbox_array(sbox).reshape(16, 16)


def parse_sbox_json(data) -> tuple[np.ndarray, dict]:
    """Terima dict/list hasil json.load. Mengembalikan (sbox, meta)."""
    if isinstance(data, dict):
        if "sbox" not in data:
            raise ValueError("Berkas JSON tidak memiliki kunci 'sbox'.")
        values = data["sbox"]
        meta = {k: v for k, v in data.items() if k != "sbox"}
    else:
        values = data
        meta = {}
    arr = np.asarray(values)
    return as_sbox_array(arr), meta


def parse_sbox_csv(text: str) -> np.ndarray:
    """Baca 256 angka (desimal) yang dipisah koma/spasi/baris baru."""
    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    try:
        values = [int(t) for t in tokens]
    except ValueError as exc:
        raise ValueError("CSV hanya boleh berisi bilangan bulat desimal.") from exc
    return as_sbox_array(values)


def load_sbox_file(path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return parse_sbox_csv(path.read_text(encoding="utf-8")), {"source": "csv"}
    with open(path, "r", encoding="utf-8") as f:
        return parse_sbox_json(json.load(f))


def load_sbox_bytes(content: bytes, filename: str) -> tuple[np.ndarray, dict]:
    if filename.lower().endswith(".csv"):
        return parse_sbox_csv(content.decode("utf-8")), {"source": "csv-unggahan"}
    return parse_sbox_json(json.loads(content.decode("utf-8")))


def load_default_sbox() -> tuple[np.ndarray, dict]:
    if not DEFAULT_SBOX_PATH.exists():
        raise FileNotFoundError(
            f"Berkas S-Box bawaan tidak ditemukan: {DEFAULT_SBOX_PATH}. "
            "Salin best_sbox.json hasil Colab ke folder 'sbox/'."
        )
    return load_sbox_file(DEFAULT_SBOX_PATH)


def save_sbox_json(path, sbox, meta: dict | None = None) -> None:
    s = as_sbox_array(sbox)
    payload = {"format_version": 1}
    payload.update(meta or {})
    payload["sbox"] = [int(v) for v in s]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sbox_to_csv_text(sbox, hexadecimal: bool = False) -> str:
    grid = sbox_grid(sbox)
    out = io.StringIO()
    for row in grid:
        out.write(",".join((f"{int(v):02X}" if hexadecimal else str(int(v))) for v in row))
        out.write("\n")
    return out.getvalue()
