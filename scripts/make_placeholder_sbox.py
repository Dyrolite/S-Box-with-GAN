# -*- coding: utf-8 -*-
"""
Membuat S-Box CADANGAN (placeholder) agar aplikasi Streamlit langsung bisa dijalankan.

PENTING: S-Box ini BUKAN hasil GAN. Dibuat dengan hill-climbing sederhana
(tukar dua nilai bila nonlinearity membaik) dan TIDAK memakai konstruksi
invers GF(2^8). Setelah pelatihan di Colab selesai, timpa berkas
sbox/best_sbox.json dengan hasil dari notebook.

Cara pakai (dari folder proyek):
    python scripts/make_placeholder_sbox.py
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sbox_dataset as ds          # noqa: E402
from core import sbox_io, sbox_metrics as sm  # noqa: E402

SEED = 2024
RESTARTS = 3
STEPS = 120_000

rng = np.random.default_rng(SEED)
best, best_key = None, None
for r in range(RESTARTS):
    s = ds.hill_climb(ds.random_sbox(rng), STEPS, rng)
    q = sm.quick_evaluate(s, with_sac=True)
    key = sm.selection_key(q)
    print(f"restart {r + 1}/{RESTARTS}: {q}")
    if best_key is None or key < best_key:
        best, best_key = s, key

meta = {
    "name": "S-Box cadangan (placeholder)",
    "source": "placeholder_hillclimb",
    "is_gan_generated": False,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "notes": ("BUKAN hasil GAN. S-Box ini hanya agar aplikasi bisa dijalankan sebelum "
              "pelatihan di Colab selesai. Timpa dengan best_sbox.json dari notebook."),
    "seed": SEED, "restarts": RESTARTS, "hill_climb_steps": STEPS,
}
out = ROOT / "sbox" / "best_sbox.json"
sbox_io.save_sbox_json(out, best, meta)
print("Disimpan ke", out)
