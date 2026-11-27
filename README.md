# Konstruksi S-Box Menggunakan GAN dengan Custom Loss dan Implementasinya pada Enkripsi Citra

Proyek mata kuliah Kriptografi. Alur kerja:

```
Google Colab (latih GAN)  -->  best_sbox.json  -->  Aplikasi Streamlit (uji S-Box + enkripsi citra)
```

Aplikasi **tidak melatih GAN**; ia hanya membaca S-Box hasil pelatihan.

## Struktur folder

```
GAN_SBOX_STREAMLIT/
├── app.py                          # halaman utama Streamlit
├── requirements.txt                # dependensi aplikasi
├── README.md
├── PANDUAN_PENGGUNAAN_DAN_DEPLOYMENT.docx
├── .gitignore
├── .streamlit/config.toml
├── core/
│   ├── sbox_metrics.py             # NL, SAC, BIC, DAP, LAP, bijectivity
│   ├── sbox_dataset.py             # persiapan data latih GAN (dipakai di Colab)
│   ├── sbox_io.py                  # baca/tulis S-Box (JSON/CSV), S-Box invers
│   ├── image_crypto.py             # enkripsi & dekripsi citra (substitusi, permutasi, difusi)
│   ├── image_metrics.py            # entropy, histogram, NPCR, UACI, korelasi
│   └── visualization.py            # grafik matplotlib
├── notebooks/GAN_SBox_Training_Colab.ipynb
├── sbox/best_sbox.json             # S-Box yang dipakai aplikasi (SAAT INI: placeholder, BUKAN GAN)
├── models/                         # generator.pt (opsional)
├── sample_images/                  # taruh mandrill.png di sini
├── scripts/                        # pembuat placeholder, citra contoh, notebook
└── tests/                          # pengujian otomatis
```

## Menjalankan di Windows (ringkas)

```
cd C:\Proyek\GAN_SBOX_STREAMLIT
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Langkah lengkap (Colab, GitHub, Streamlit Cloud, pemecahan masalah) ada di
`PANDUAN_PENGGUNAAN_DAN_DEPLOYMENT.docx`.

## Komponen teknis

- **GAN**: Generator menghasilkan 256 skor, S-Box = peringkat skor (selalu bijektif). Soft-sort + straight-through
  membuat loss kriptografi bisa dilatih. **Custom loss**: loss GAN + hukuman NL (Walsh > 32), DAP (DDT > 4),
  SAC, bijektivitas lembut, dan keragaman. Tidak memakai invers GF(2^8).
- **Enkripsi citra**: substitusi S-Box -> permutasi berbasis kunci -> difusi maju & mundur (rantai S-Box). Dekripsi
  membalik semuanya dan identik 100% dengan citra asli.
- **Evaluasi citra**: entropy, histogram, NPCR, UACI (uji satu-bit), korelasi piksel bertetangga, MSE/PSNR.

## Catatan penting dan jujur

- S-Box di `sbox/best_sbox.json` saat ini **placeholder** (hill-climbing, NL=100, bukan GAN). Ganti dengan hasil Colab.
- **NL = 112 tidak dijamin** oleh GAN. Angka yang tampil dihitung program, tidak diatur manual.
- **MBCR**: definisi belum jelas, **tidak dihitung dan tidak dikarang**; konfirmasikan ke dosen.
  **"Distogram"** diasumsikan **histogram**; mohon dikonfirmasi.
- Skema enkripsi untuk keperluan belajar, belum dianalisis kriptografi formal.
- Citra Mandrill tidak tersedia saat proyek dibuat; letakkan sendiri di `sample_images/mandrill.png`.

## Pengujian

```
python tests/test_core.py
python tests/test_gan_math.py
```

Diuji dengan Python 3.12, numpy 2.5, pandas 3.0, Pillow 13, matplotlib 3.11. Pelatihan GAN (PyTorch) dan
Streamlit asli **belum dijalankan** di lingkungan pembuatan; lihat bagian status di panduan.
