# -*- coding: utf-8 -*-
"""
Membangun notebooks/GAN_SBox_Training_Colab.ipynb.

Notebook dibuat dari skrip ini supaya modul pengujian di notebook (sbox_metrics.py,
sbox_dataset.py, sbox_io.py) SAMA PERSIS dengan yang dipakai aplikasi Streamlit.
Jalankan (dari folder proyek):  python scripts/build_notebook.py
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "GAN_SBox_Training_Colab.ipynb"

cells = []


def md(text):
    text = textwrap.dedent(text).strip("\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    text = textwrap.dedent(text).strip("\n")
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


def writefile_cell(rel_path):
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    code_text = f"%%writefile /content/gan_work/{rel_path}\n{src}"
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": code_text.splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md("""
# Konstruksi S-Box dengan GAN (Custom Loss) - Notebook Pelatihan

Notebook ini adalah bagian **pelatihan** dari proyek *"Konstruksi S-Box Menggunakan GAN dengan Custom Loss
dan Implementasinya pada Enkripsi Citra"*. Hasilnya (`best_sbox.json`) dipakai oleh aplikasi Streamlit,
sehingga aplikasi **tidak perlu melatih ulang**.

**Alur notebook**

1. Pasang modul proyek (pengujian S-Box, pembuat dataset).
2. Atur konfigurasi (pilih `PRESET`).
3. **Persiapan data**: membuat contoh S-Box "nyata" untuk Discriminator.
4. Mendefinisikan GAN (Generator, Discriminator) dan **custom loss**.
5. Uji kesehatan kode (memastikan rumus loss sama dengan pengujian sebenarnya).
6. Pelatihan.
7. Memilih S-Box terbaik dari banyak sampel Generator, lalu evaluasi lengkap.
8. Menyimpan hasil dan mengunduh `hasil_pelatihan_gan.zip`.

**Sebelum mulai (disarankan):** menu *Runtime > Change runtime type* (Indonesia: *Runtime > Ubah jenis runtime*),
pilih **GPU** (mis. T4). Tanpa GPU tetap jalan, hanya lebih lambat.

> **Penting dan jujur:** GAN tidak menjamin Nonlinearity (NL) = 112 (nilai S-Box AES yang dibuat dengan struktur aljabar).
> Hasil nyata biasanya lebih rendah. Angka yang tampil di sini dihitung program dan **tidak diubah manual**.
> Terima hasilnya apa adanya, lalu bahas keterbatasannya pada laporan.
""")

md("## 1. Pasang modul proyek\nTiga sel berikut hanya menulis berkas modul ke `/content/gan_work/core/` (tidak ada yang perlu diunggah).")
code("""
import os
os.makedirs("/content/gan_work/core", exist_ok=True)
""")
code("%%writefile /content/gan_work/core/__init__.py\n\"\"\"Paket inti proyek.\"\"\"\n")
writefile_cell("core/sbox_metrics.py")
writefile_cell("core/sbox_dataset.py")
writefile_cell("core/sbox_io.py")

md("## 2. Impor dan konfigurasi")
code("""
import sys, os, json, time, copy, shutil, platform
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/content/gan_work")
from core import sbox_metrics as sm, sbox_dataset as ds, sbox_io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False   # presisi penuh agar hitungan NL/DDT tepat
torch.backends.cudnn.allow_tf32 = False
print("Python", platform.python_version(), "| torch", torch.__version__, "| perangkat:", DEVICE)
if DEVICE == "cpu":
    print("PERINGATAN: GPU tidak aktif. Pelatihan tetap jalan tetapi lebih lambat (pakai PRESET 'cepat').")
""")
code("""
# ===================== KONFIGURASI (ubah di sini) =====================
# "cepat"   : untuk mencoba apakah semuanya berjalan (beberapa menit).
# "standar" : untuk hasil yang layak dilaporkan (lebih lama).
# "panjang" : paling lama, peluang hasil lebih baik.
# Perkiraan waktu sangat kasar dan belum diukur; tergantung CPU/GPU Colab yang Anda dapat.
PRESET = "cepat"

PRESETS = {
    "cepat":   dict(POOL_SIZE=40,  HC_STEPS=1500, AUG_PER_POOL=25, N_STEPS=600,   BATCH=16, EVAL_EVERY=100, FINAL_SAMPLES=2000,  TOP_K=30),
    "standar": dict(POOL_SIZE=200, HC_STEPS=4000, AUG_PER_POOL=50, N_STEPS=5000,  BATCH=32, EVAL_EVERY=250, FINAL_SAMPLES=20000, TOP_K=100),
    "panjang": dict(POOL_SIZE=400, HC_STEPS=6000, AUG_PER_POOL=60, N_STEPS=20000, BATCH=32, EVAL_EVERY=500, FINAL_SAMPLES=50000, TOP_K=200),
}
CFG = dict(PRESETS[PRESET])
CFG.update(
    PRESET=PRESET, SEED=42, Z_DIM=128,
    LR_G=5e-4, LR_D=2e-4,
    TAU_START=0.08, TAU_END=0.01,     # suhu soft-sort: besar = kabur (mudah dilatih), kecil = tajam
    NL_TARGET=112, DU_TARGET=4,       # target loss: NL >= 112 dan entri DDT <= 4
    # bobot RELATIF tiap komponen loss (tiap komponen sudah dinormalkan otomatis ke skala ~1 di awal)
    W_ADV=0.2, W_NL=1.0, W_DAP=1.0, W_SAC=1.0, W_BIJ=0.5, W_DIV=0.2,
    LOG_EVERY=25, EVAL_N=200,
)
np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
print("Konfigurasi:", json.dumps(CFG, indent=1))
""")

md("""
## 3. Persiapan data

GAN butuh contoh "S-Box nyata" supaya Discriminator punya pembanding. Contoh dibuat **tanpa konstruksi invers GF(2^8)**:

1. permutasi acak 0..255,
2. diperbaiki sedikit dengan *hill-climbing* (tukar dua nilai bila nonlinearity membaik),
3. diperbanyak dengan transformasi afin (mempertahankan NL, DAP, dan LAP).

Citra **Mandrill tidak dipakai di sini** (hanya untuk menguji enkripsi di aplikasi).
""")
code("""
t0 = time.time()
data, nl_pool = ds.build_dataset(CFG["POOL_SIZE"], CFG["HC_STEPS"], CFG["AUG_PER_POOL"], seed=CFG["SEED"])
print(f"Dataset: {data.shape[0]} S-Box ({time.time()-t0:.0f} detik)")
print(f"NL pool  -> maks {max(nl_pool)}, rata-rata {np.mean(nl_pool):.1f}, min {min(nl_pool)}")
q_data = [sm.quick_evaluate(s) for s in data[:300]]
print(f"DU data  -> min {min(q['du'] for q in q_data)}, rata-rata {np.mean([q['du'] for q in q_data]):.2f}")
DATA_BEST_NL = int(max(nl_pool))
os.makedirs("/content/hasil_gan", exist_ok=True)
np.save("/content/hasil_gan/dataset_sbox.npy", data)
plt.figure(figsize=(5, 3.2)); plt.hist(nl_pool, bins=range(min(nl_pool), max(nl_pool) + 2), color="#4C78A8")
plt.xlabel("NL S-Box pada pool data"); plt.ylabel("Jumlah"); plt.title("Kualitas data latih"); plt.show()
""")

md("""
## 4. Model GAN dan custom loss

**Cara kerja singkat (bahasa sederhana)**

- S-Box harus *permutasi* (setiap angka 0..255 muncul tepat sekali), padahal jaringan saraf menghasilkan angka kontinu.
- Solusi: **Generator** menghasilkan 256 skor, lalu S-Box = *peringkat* tiap skor (`argsort`). Dengan begitu hasilnya
  **selalu bijektif**.
- Agar bisa dilatih dengan gradien, dipakai **soft-sort** (versi "lembut" dari pengurutan) dengan
  *straight-through estimator*: nilai yang dihitung loss adalah S-Box asli (tegas), sedangkan gradiennya lewat versi lembut.
- **Discriminator** membedakan S-Box dari dataset dan S-Box buatan Generator.
- **Custom loss** menambah tujuan kriptografi di luar loss GAN biasa:
  `loss_NL` (hukum spektrum Walsh yang melebihi batas NL=112), `loss_DAP` (hukum entri DDT di atas 4),
  `loss_SAC` (jarak SAC ke 0.5), `loss_bij` (penjumlahan soft-permutasi harus 1), dan `loss_div` (agar sampel beragam).
- Tiap komponen loss dinormalkan otomatis di awal pelatihan sehingga bobotnya mudah diatur.

Rumus Walsh/DDT/SAC pada loss telah diverifikasi sama dengan perhitungan brute-force (lihat `tests/test_gan_math.py`).
""")
code("""
Z = CFG["Z_DIM"]
H     = torch.tensor(sm.hadamard_matrix(256), dtype=torch.float32, device=DEVICE)           # Hadamard 256x256
BITS  = torch.tensor((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1, dtype=torch.float32, device=DEVICE)
KVAL  = torch.arange(256, dtype=torch.float32, device=DEVICE)
IDX   = torch.tensor(np.stack([np.arange(256) ^ (1 << i) for i in range(8)]), dtype=torch.long, device=DEVICE)  # (8,256)
W_T   = float(256 - 2 * CFG["NL_TARGET"])   # NL>=112  <=>  max|Walsh| <= 32
DU_T  = float(CFG["DU_TARGET"])


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
        )
    def forward(self, z):
        s = self.net(z)                                               # 256 skor
        return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
    def forward(self, v):
        return self.net(v)


def soft_sort(s, tau):
    # P[b, k, x] ~ 1 bila elemen x berperingkat k (SoftSort). Baris k dinormalkan terhadap x.
    s_sorted, _ = torch.sort(s, dim=-1)
    diff = (s_sorted.unsqueeze(-1) - s.unsqueeze(-2)).abs()
    return torch.softmax(-diff / tau, dim=-1)


def decode_hard(s):
    # S(x) = peringkat skor ke-x  -> selalu permutasi 0..255
    return torch.argsort(torch.argsort(s, dim=-1), dim=-1)


def sbox_from_scores(s, tau):
    rank = decode_hard(s)
    M_hard = F.one_hot(rank, 256).float()          # M[b, x, k] = 1 bila S(x) = k
    P = soft_sort(s, tau)
    M_soft = P.transpose(1, 2)
    M = M_hard + M_soft - M_soft.detach()          # straight-through: maju = tegas, gradien = lembut
    return M, P, rank


def walsh(M):                                      # W[a, b] = (H M H)[a, b]
    return torch.matmul(torch.matmul(H, M), H)


def loss_terms(M, P, s):
    W = walsh(M)
    # NL: hukum |Walsh| yang melebihi batas (mask keluaran b >= 1)
    nl_pen = ((F.relu(W[:, :, 1:].abs() - W_T) / W_T) ** 2).sum(dim=(1, 2)).mean()
    # DAP: DDT = H (W*W) H / 65536 ; hukum entri (dx >= 1) di atas DU_T
    ddt = torch.matmul(torch.matmul(H, W * W), H) / 65536.0
    dap_pen = (F.relu(ddt[:, 1:, :] - DU_T) ** 2).sum(dim=(1, 2)).mean()
    # SAC: peluang bit keluaran j berubah saat bit masukan i dibalik, ideal 0.5
    p = torch.matmul(M, BITS)                      # (B,256,8)
    p2 = p[:, IDX, :]                              # (B,8,256,8)
    p1 = p.unsqueeze(1)
    flip = p1 * (1 - p2) + (1 - p1) * p2
    sac = flip.mean(dim=2)                         # (B,8,8)
    sac_pen = ((sac - 0.5) ** 2).mean()
    # Bijektivitas lembut: tiap x harus 'dipakai' tepat sekali oleh soft-permutasi
    bij_pen = ((P.sum(dim=1) - 1.0) ** 2).mean()
    # Keragaman: sampel dalam satu batch jangan saling berkorelasi
    sn = s - s.mean(1, keepdim=True)
    sn = sn / (sn.norm(dim=1, keepdim=True) + 1e-8)
    corr = sn @ sn.t()
    off = corr - torch.diag(torch.diag(corr))
    n = s.shape[0]
    div_pen = (off ** 2).sum() / max(n * (n - 1), 1)
    return {"nl": nl_pen, "dap": dap_pen, "sac": sac_pen, "bij": bij_pen, "div": div_pen}, sac, W, ddt


G = Generator(Z).to(DEVICE)
D = Discriminator().to(DEVICE)
print("Parameter Generator:", sum(p.numel() for p in G.parameters()),
      "| Discriminator:", sum(p.numel() for p in D.parameters()))
""")

md("""
## 5. Uji kesehatan kode

Sel ini memastikan (a) hasil perhitungan `PyTorch` untuk NL, DU, dan SAC **sama** dengan modul pengujian `sbox_metrics`,
dan (b) gradien benar-benar mengalir ke Generator. Bila ada `AssertionError`, **jangan lanjut melatih**.
""")
code("""
with torch.no_grad():
    s = G(torch.randn(4, Z, device=DEVICE))
    M, P, rank = sbox_from_scores(s, CFG["TAU_START"])
    terms, sac_t, W_t, ddt_t = loss_terms(M, P, s)
for b in range(4):
    sb = rank[b].cpu().numpy()
    assert len(np.unique(sb)) == 256, "S-Box dari Generator tidak bijektif"
    q = sm.quick_evaluate(sb)
    nl_torch = int((256 - W_t[b][:, 1:].abs().max().item()) // 2)
    du_torch = int(round(ddt_t[b][1:].max().item()))
    assert nl_torch == q["nl"], (nl_torch, q["nl"])
    assert du_torch == q["du"], (du_torch, q["du"])
    assert np.allclose(sac_t[b].cpu().numpy(), sm.sac_matrix(sb), atol=1e-4)
print("OK: NL, DU, dan SAC versi PyTorch sama dengan modul pengujian.")

s = G(torch.randn(CFG["BATCH"], Z, device=DEVICE))
M, P, rank = sbox_from_scores(s, CFG["TAU_START"])
t, *_ = loss_terms(M, P, s)
total = sum(t.values())
G.zero_grad(); total.backward()
gn = sum(p.grad.norm().item() ** 2 for p in G.parameters() if p.grad is not None) ** 0.5
assert np.isfinite(gn) and gn > 0, f"Gradien tidak valid: {gn}"
print(f"OK: gradien mengalir ke Generator (norma {gn:.4g}).")
print("Nilai awal komponen loss:", {k: round(v.item(), 4) for k, v in t.items()})
G.zero_grad()
""")

md("""
## 6. Pelatihan

Selama pelatihan, tiap `EVAL_EVERY` langkah dibuat sampel S-Box dari Generator lalu dihitung NL dan DU sebenarnya.
Perhatikan kolom `nl_best` (NL terbaik pada sampel) dan `nl_mean`. Kalau tidak membaik sama sekali, lihat bagian
*Tips* di akhir notebook.
""")
code("""
def estimate_scales(n_batches=8):
    acc = {k: 0.0 for k in ("nl", "dap", "sac", "bij", "div")}
    with torch.no_grad():
        for _ in range(n_batches):
            s = G(torch.randn(CFG["BATCH"], Z, device=DEVICE))
            M, P, _ = sbox_from_scores(s, CFG["TAU_START"])
            t, *_ = loss_terms(M, P, s)
            for k in acc:
                acc[k] += float(t[k]) / n_batches
    return {k: 1.0 / max(v, 1e-8) for k, v in acc.items()}


@torch.no_grad()
def sample_sboxes(n, batch=512):
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        out.append(decode_hard(G(torch.randn(b, Z, device=DEVICE))).cpu().numpy().astype(np.int64))
    return np.concatenate(out)


def tau_at(step):
    frac = (step - 1) / max(CFG["N_STEPS"] - 1, 1)
    return CFG["TAU_START"] * (CFG["TAU_END"] / CFG["TAU_START"]) ** frac


SCALES = estimate_scales()
print("Skala normalisasi loss:", {k: round(v, 4) for k, v in SCALES.items()})

optG = torch.optim.Adam(G.parameters(), lr=CFG["LR_G"], betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=CFG["LR_D"], betas=(0.5, 0.999))
real_pool = torch.tensor(data, dtype=torch.float32, device=DEVICE) / 255.0
bce = F.binary_cross_entropy_with_logits
WEIGHTS = {"nl": CFG["W_NL"], "dap": CFG["W_DAP"], "sac": CFG["W_SAC"], "bij": CFG["W_BIJ"], "div": CFG["W_DIV"]}

log_rows, eval_rows = [], []
best = {"key": None, "sbox": None, "step": None}
t_start = time.time()
B = CFG["BATCH"]

for step in range(1, CFG["N_STEPS"] + 1):
    tau = tau_at(step)

    # ---------- (1) latih Discriminator ----------
    real = real_pool[torch.randint(0, real_pool.shape[0], (B,), device=DEVICE)]
    with torch.no_grad():
        fake = decode_hard(G(torch.randn(B, Z, device=DEVICE))).float() / 255.0
    d_real, d_fake = D(real), D(fake)
    loss_d = bce(d_real, torch.full_like(d_real, 0.9)) + bce(d_fake, torch.zeros_like(d_fake))
    optD.zero_grad(set_to_none=True); loss_d.backward(); optD.step()

    # ---------- (2) latih Generator (loss GAN + custom loss) ----------
    s = G(torch.randn(B, Z, device=DEVICE))
    M, P, rank = sbox_from_scores(s, tau)
    fake_v = (M * KVAL).sum(dim=-1) / 255.0
    d_out = D(fake_v)
    adv = bce(d_out, torch.ones_like(d_out))
    t, *_ = loss_terms(M, P, s)
    custom = sum(WEIGHTS[k] * SCALES[k] * t[k] for k in WEIGHTS)
    loss_g = CFG["W_ADV"] * adv + custom
    optG.zero_grad(set_to_none=True); loss_g.backward(); optG.step()

    if step % CFG["LOG_EVERY"] == 0 or step == 1:
        log_rows.append({"step": step, "tau": tau, "loss_d": loss_d.item(), "loss_g": loss_g.item(),
                         "adv": adv.item(), **{f"loss_{k}": float(v) for k, v in t.items()}})

    if step % CFG["EVAL_EVERY"] == 0 or step == CFG["N_STEPS"]:
        samples = sample_sboxes(CFG["EVAL_N"])
        qs = [sm.quick_evaluate(x) for x in samples]
        nls = np.array([q["nl"] for q in qs]); dus = np.array([q["du"] for q in qs])
        uniq = len({x.tobytes() for x in samples})
        for x, q in zip(samples, qs):
            key = sm.selection_key(q)
            if best["key"] is None or key < best["key"]:
                best.update(key=key, sbox=x.copy(), step=step)
        eval_rows.append({"step": step, "nl_best": int(nls.max()), "nl_mean": float(nls.mean()),
                          "du_best": int(dus.min()), "du_mean": float(dus.mean()), "unik": uniq})
        e = eval_rows[-1]
        print(f"langkah {step:>6}/{CFG['N_STEPS']} | tau {tau:.3f} | loss_g {loss_g.item():8.3f} | "
              f"NL terbaik {e['nl_best']:>3} rata-rata {e['nl_mean']:6.2f} | DU terbaik {e['du_best']:>2} | "
              f"S-Box unik {uniq}/{CFG['EVAL_N']} | {time.time()-t_start:5.0f} dtk")

log_df, eval_df = pd.DataFrame(log_rows), pd.DataFrame(eval_rows)
print(f"Selesai dalam {time.time()-t_start:.0f} detik.")
""")

md("""
## 7. Memilih S-Box terbaik dan evaluasi lengkap

Generator yang sudah dilatih dipakai membuat banyak S-Box. Semuanya dinilai dengan pengujian sebenarnya (NL, DU, LAP, SAC),
lalu S-Box terbaik dipilih. Semua S-Box dari Generator **bijektif** karena cara pembuatannya.
""")
code("""
t0 = time.time()
samples = sample_sboxes(CFG["FINAL_SAMPLES"])
uniq = {x.tobytes(): x for x in samples}
samples = list(uniq.values())
print(f"{len(samples)} S-Box unik dari {CFG['FINAL_SAMPLES']} sampel.")
quick = [sm.quick_evaluate(x) for x in samples]
order = sorted(range(len(samples)), key=lambda i: sm.selection_key(quick[i]))[:CFG["TOP_K"]]
cands = [(samples[i], sm.quick_evaluate(samples[i], with_sac=True)) for i in order]
cands.append((best["sbox"], sm.quick_evaluate(best["sbox"], with_sac=True)))     # S-Box terbaik selama pelatihan
cands.sort(key=lambda c: sm.selection_key(c[1]))
best_sbox, best_q = cands[0]
assert len(np.unique(best_sbox)) == 256
print(f"Pemilihan selesai ({time.time()-t0:.0f} detik). Terbaik: {best_q}")
nl_all = np.array([q["nl"] for q in quick])
print("Sebaran NL pada seluruh sampel:", dict(zip(*np.unique(nl_all, return_counts=True))))
print(f"Pembanding: NL terbaik pada dataset latih = {DATA_BEST_NL}")

report = sm.evaluate_sbox(best_sbox)
display(pd.DataFrame(sm.metric_rows(report)))
""")
code("""
grid = best_sbox.reshape(16, 16)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].imshow(grid, cmap="viridis", alpha=0.35, vmin=0, vmax=255)
for r in range(16):
    for c in range(16):
        ax[0].text(c, r, f"{grid[r, c]:02X}", ha="center", va="center", fontsize=7)
ax[0].set_title("S-Box terbaik (heksadesimal)"); ax[0].set_xticks(range(16)); ax[0].set_yticks(range(16))
ax[1].plot(log_df["step"], log_df["loss_g"], label="loss_g"); ax[1].plot(log_df["step"], log_df["loss_d"], label="loss_d")
ax[1].set_xlabel("langkah"); ax[1].set_title("Loss pelatihan"); ax[1].legend()
ax[2].plot(eval_df["step"], eval_df["nl_best"], label="NL terbaik"); ax[2].plot(eval_df["step"], eval_df["nl_mean"], label="NL rata-rata")
ax[2].axhline(112, color="r", linestyle="--", label="acuan 112"); ax[2].set_xlabel("langkah"); ax[2].set_title("NL pada sampel Generator"); ax[2].legend()
plt.tight_layout(); plt.savefig("/content/hasil_gan/training_curves.png", dpi=130); plt.show()
""")

md("""
## 8. Simpan hasil dan unduh

Berkas yang dihasilkan di `/content/hasil_gan/` lalu di-zip menjadi `hasil_pelatihan_gan.zip`:

| Berkas | Fungsi |
|---|---|
| `best_sbox.json` | **S-Box terbaik** - salin ke folder `sbox/` proyek (dibaca aplikasi) |
| `generator.pt` | bobot Generator terlatih (opsional, simpan di `models/`) |
| `training_log.csv`, `eval_log.csv` | catatan pelatihan |
| `sbox_report.json` | hasil pengujian lengkap S-Box terbaik |
| `training_curves.png` | grafik pelatihan |
| `dataset_sbox.npy` | dataset latih yang dipakai |
""")
code("""
OUT = "/content/hasil_gan"
os.makedirs(OUT, exist_ok=True)
nl_final = int(report["nonlinearity"]["min"])
meta = {
    "name": "S-Box hasil GAN (custom loss)",
    "source": "gan",
    "is_gan_generated": True,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "training": {
        "config": CFG,
        "device": DEVICE,
        "dataset_size": int(data.shape[0]),
        "dataset_best_nl": DATA_BEST_NL,
        "final_samples": CFG["FINAL_SAMPLES"],
        "selected_nl_min": nl_final,
        "selected_dap_uniformity": int(report["dap"]["differential_uniformity"]),
        "catatan": "NL, DAP, LAP, SAC, BIC dihitung ulang oleh aplikasi dari tabel S-Box; angka di sini hanya ringkasan.",
    },
}
sbox_io.save_sbox_json(f"{OUT}/best_sbox.json", best_sbox, meta)
torch.save({"state_dict": G.state_dict(), "z_dim": Z, "config": CFG}, f"{OUT}/generator.pt")
log_df.to_csv(f"{OUT}/training_log.csv", index=False)
eval_df.to_csv(f"{OUT}/eval_log.csv", index=False)
with open(f"{OUT}/sbox_report.json", "w", encoding="utf-8") as f:
    json.dump({"sbox": [int(v) for v in best_sbox], "hasil_pengujian": report}, f, indent=2)

# pemeriksaan: baca ulang berkas yang baru disimpan
chk, chk_meta = sbox_io.load_sbox_file(f"{OUT}/best_sbox.json")
assert np.array_equal(chk, best_sbox) and sbox_io.is_bijective(chk)
print("best_sbox.json valid dan bijektif. NL minimum =", nl_final)

shutil.make_archive("/content/hasil_pelatihan_gan", "zip", OUT)
print("ZIP dibuat: /content/hasil_pelatihan_gan.zip")
try:
    from google.colab import files
    files.download("/content/hasil_pelatihan_gan.zip")
except ImportError:
    print("Bukan di Google Colab: ambil berkas dari folder", OUT)
""")
code("""
# (Opsional) simpan juga ke Google Drive agar tidak hilang saat sesi Colab berakhir.
# Hapus tanda # pada tiga baris berikut bila ingin memakainya.
# from google.colab import drive; drive.mount("/content/drive")
# shutil.copytree("/content/hasil_gan", "/content/drive/MyDrive/hasil_gan_sbox", dirs_exist_ok=True)
# print("Tersalin ke Google Drive: MyDrive/hasil_gan_sbox")
""")

md("""
## Langkah berikutnya

1. Ekstrak `hasil_pelatihan_gan.zip`.
2. Salin **`best_sbox.json`** ke folder `sbox/` proyek (timpa berkas lama). Salin `generator.pt` ke `models/` bila ingin menyimpannya.
3. Jalankan aplikasi: `streamlit run app.py`. Aplikasi akan menampilkan S-Box ini dan menghitung ulang seluruh pengujiannya.

## Tips bila hasil kurang baik

- Jalankan ulang dengan `PRESET = "standar"` atau `"panjang"`; ubah `SEED` untuk hasil berbeda.
- Jika `nl_best` tidak membaik, naikkan `TAU_START` (mis. 0.15) agar gradien lebih halus, atau naikkan `W_NL`.
- Jika semua sampel hampir sama (kolom `unik` kecil), naikkan `W_DIV` atau turunkan `LR_G`.
- NL = 112 membutuhkan struktur aljabar khusus; mencapainya murni dari GAN tanpa struktur hampir tidak realistis.
  Sampaikan hasil sebenarnya pada laporan beserta keterbatasan (kualitas data latih, relaksasi permutasi, sensitivitas hiperparameter).
""")

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "GAN_SBox_Training_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook dibuat:", OUT, "| jumlah sel:", len(cells))
