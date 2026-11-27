# -*- coding: utf-8 -*-
"""
app.py - Aplikasi Streamlit
"Konstruksi S-Box Menggunakan GAN dengan Custom Loss dan Implementasinya pada Enkripsi Citra"

Menjalankan secara lokal (dari folder proyek):
    streamlit run app.py

Aplikasi TIDAK melatih GAN. Aplikasi hanya membaca S-Box hasil pelatihan
(sbox/best_sbox.json) atau berkas S-Box yang diunggah di sidebar.
"""
import hashlib
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core import image_crypto as ic
from core import image_metrics as im
from core import sbox_io
from core import sbox_metrics as sm
from core import visualization as viz

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_images"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

st.set_page_config(page_title="S-Box GAN & Enkripsi Citra", page_icon="🔐", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
      h1 {font-size: 1.9rem !important;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); padding: 0.7rem 0.9rem;
                                   border-radius: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Fungsi bantu
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_report(sbox_tuple: tuple) -> dict:
    return sm.evaluate_sbox(np.array(sbox_tuple))


@st.cache_data(show_spinner=False)
def cached_ddt(sbox_tuple: tuple) -> np.ndarray:
    return sm.difference_distribution_table(np.array(sbox_tuple))


def find_mandrill():
    if not SAMPLE_DIR.exists():
        return None
    for p in sorted(SAMPLE_DIR.iterdir()):
        if p.stem.lower().startswith("mandrill") and p.suffix.lower() in IMG_EXT:
            return p
    return None


def to_json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")


def to_csv_bytes(rows) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")  # utf-8-sig: aman dibuka Excel


def load_active_sbox():
    """Pilih S-Box aktif dari sidebar. Mengembalikan (sbox, meta, label)."""
    choice = st.sidebar.radio(
        "Sumber S-Box",
        ["Bawaan proyek (sbox/best_sbox.json)", "Unggah berkas S-Box (.json / .csv)"],
        key="sbox_source",
    )
    if choice.startswith("Unggah"):
        up = st.sidebar.file_uploader("Berkas S-Box", type=["json", "csv"], key="sbox_upload")
        if up is not None:
            try:
                sbox, meta = sbox_io.load_sbox_bytes(up.getvalue(), up.name)
                meta = dict(meta)
                meta.setdefault("is_gan_generated", None)
                return sbox, meta, f"Unggahan: {up.name}"
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Berkas tidak valid: {exc}")
        else:
            st.sidebar.info("Belum ada berkas. Memakai S-Box bawaan proyek.")
    try:
        sbox, meta = sbox_io.load_default_sbox()
        return sbox, meta, "Bawaan proyek"
    except Exception as exc:  # noqa: BLE001
        st.error(f"Tidak dapat memuat S-Box bawaan: {exc}")
        st.stop()


def sig_of(sbox_t, key, arr) -> str:
    h = hashlib.sha256()
    h.update(repr(sbox_t).encode())
    h.update(key.encode("utf-8"))
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("Pengaturan")
sbox, meta, sbox_label = load_active_sbox()
sbox_t = tuple(int(v) for v in sbox)
is_bij = sbox_io.is_bijective(sbox)

st.sidebar.markdown("---")
key_text = st.sidebar.text_input(
    "Kunci enkripsi (teks)", value="kunci-contoh-123", key="enc_key",
    help="Kunci yang sama dipakai untuk enkripsi dan dekripsi. Kunci berbeda menghasilkan hasil berbeda.",
)
max_side = st.sidebar.selectbox(
    "Ukuran maksimum sisi citra (piksel)", [128, 256, 512, 1024], index=2, key="max_side",
    help="Citra lebih besar akan diperkecil agar proses tetap cepat. Mandrill 512x512 tidak diubah.",
)
n_trials = st.sidebar.slider("Jumlah uji sensitivitas NPCR/UACI", 1, 20, 5, key="n_trials")
st.sidebar.caption("Aplikasi ini tidak melatih GAN. S-Box dibaca dari berkas hasil pelatihan Colab.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Konstruksi S-Box dengan GAN (Custom Loss) dan Enkripsi Citra")
st.caption("Proyek mata kuliah Kriptografi - S-Box 8-bit (16x16) hasil GAN, pengujian kriptografi, "
           "dan enkripsi/dekripsi citra yang dapat dibalik.")

is_gan = meta.get("is_gan_generated")
if is_gan is False:
    st.warning(
        "**S-Box yang aktif BUKAN hasil GAN** (S-Box cadangan agar aplikasi bisa dicoba). "
        "Latih GAN di Google Colab, lalu ganti berkas `sbox/best_sbox.json` dengan hasilnya "
        "(atau unggah lewat sidebar). Lihat panduan di PANDUAN_PENGGUNAAN_DAN_DEPLOYMENT.docx."
    )
elif is_gan is None:
    st.info("Asal S-Box tidak tercatat pada berkas ini (tidak ada penanda `is_gan_generated`).")
if not is_bij:
    st.error("S-Box ini TIDAK bijektif, sehingga enkripsi/dekripsi dinonaktifkan. "
             "Pengujian S-Box tetap bisa dilihat.")

tab_sbox, tab_test, tab_enc, tab_ana, tab_note = st.tabs(
    ["1. S-Box", "2. Pengujian S-Box", "3. Enkripsi & Dekripsi", "4. Analisis Citra", "5. Catatan"]
)

# ----------------------------------------------------------------------------
# Tab 1 - S-Box
# ----------------------------------------------------------------------------
with tab_sbox:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = viz.fig_sbox_grid(sbox)
        st.pyplot(fig)
        sbox_png = viz.fig_to_png_bytes(fig)
    with c2:
        st.subheader("Informasi")
        info_rows = {
            "Sumber tampilan": sbox_label,
            "Nama": meta.get("name", "-"),
            "Jenis": meta.get("source", "-"),
            "Hasil GAN": {True: "Ya", False: "Tidak", None: "Tidak diketahui"}[is_gan],
            "Dibuat": meta.get("created_at", "-"),
        }
        for k, v in info_rows.items():
            st.markdown(f"**{k}:** {v}")
        training = meta.get("training")
        if isinstance(training, dict) and training:
            with st.expander("Catatan pelatihan dari notebook"):
                st.json(training)
        fmt = st.radio("Format tabel", ["Heksadesimal", "Desimal"], horizontal=True, key="tbl_fmt")
        grid = sbox_io.sbox_grid(sbox)
        labels = [f"{i:X}" for i in range(16)]
        if fmt == "Heksadesimal":
            df = pd.DataFrame([[f"{int(v):02X}" for v in row] for row in grid], index=labels, columns=labels)
        else:
            df = pd.DataFrame(grid, index=labels, columns=labels)
        st.dataframe(df)

    st.markdown("**Unduh S-Box**")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("JSON (S-Box + info)", to_json_bytes({"format_version": 1, **meta, "sbox": list(sbox_t)}),
                       "sbox.json", "application/json", key="dl_sbox_json")
    d2.download_button("CSV desimal 16x16", sbox_io.sbox_to_csv_text(sbox).encode(), "sbox_desimal.csv",
                       "text/csv", key="dl_sbox_csv")
    d3.download_button("CSV heksadesimal 16x16", sbox_io.sbox_to_csv_text(sbox, True).encode(),
                       "sbox_hex.csv", "text/csv", key="dl_sbox_hex")
    d4.download_button("Gambar tabel (PNG)", sbox_png, "sbox_tabel.png", "image/png", key="dl_sbox_png")

# ----------------------------------------------------------------------------
# Tab 2 - Pengujian S-Box
# ----------------------------------------------------------------------------
with tab_test:
    report = cached_report(sbox_t)
    st.caption("Seluruh angka di bawah dihitung langsung oleh program dari S-Box aktif, bukan nilai tetap.")

    m = st.columns(6)
    m[0].metric("Bijektif", "Ya" if report["bijective"]["is_bijective"] else "Tidak")
    m[1].metric("NL minimum", report["nonlinearity"]["min"], delta=f"{report['nonlinearity']['min'] - sm.NL_TARGET} dari 112",
                delta_color="off")
    m[2].metric("SAC rata-rata", f"{report['sac']['mean']:.4f}")
    m[3].metric("BIC-NL min", report["bic_nl"]["min"])
    m[4].metric("DAP maks", f"{report['dap']['max']:.5f}")
    m[5].metric("LAP maks", f"{report['lap']['max']:.5f}")

    if report["nonlinearity"]["reached_target"]:
        st.success("Nonlinearity minimum mencapai atau melampaui acuan 112.")
    else:
        st.info(
            f"Nonlinearity minimum = {report['nonlinearity']['min']}, di bawah acuan 112. "
            "Ini wajar untuk S-Box hasil pencarian/pembelajaran mesin tanpa struktur aljabar; "
            "lihat tab Catatan untuk penjelasannya. Angka ini tidak disesuaikan secara manual."
        )

    rows = sm.metric_rows(report)
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(viz.fig_sac_heatmap(report["sac"]["matrix"]))
    with g2:
        st.pyplot(viz.fig_nl_distribution(report["nonlinearity"]["per_mask"]))
    g3, g4 = st.columns(2)
    with g3:
        st.pyplot(viz.fig_ddt_histogram(cached_ddt(sbox_t)))
    with g4:
        st.markdown("**Cara membaca hasil**")
        st.markdown(
            "- **NL**: makin tinggi makin tahan serangan linear (acuan tugas ~112).\n"
            "- **SAC**: tiap sel idealnya 0.5.\n"
            "- **BIC**: kebebasan antar bit keluaran; BIC-NL tinggi dan BIC-SAC dekat 0.5 lebih baik.\n"
            "- **DAP**: makin kecil makin tahan serangan diferensial.\n"
            "- **LAP**: makin kecil makin tahan serangan linear.\n"
            "- Nilai acuan DAP/LAP pada tabel adalah nilai yang umum dikutip untuk S-Box AES "
            "(sebagai pembanding saja)."
        )

    full = {"sbox": list(sbox_t), "info": meta, "hasil_pengujian": report}
    e1, e2 = st.columns(2)
    e1.download_button("Unduh hasil pengujian (JSON lengkap)", to_json_bytes(full), "hasil_pengujian_sbox.json",
                       "application/json", key="dl_test_json")
    e2.download_button("Unduh ringkasan pengujian (CSV)", to_csv_bytes(rows), "hasil_pengujian_sbox.csv",
                       "text/csv", key="dl_test_csv")

# ----------------------------------------------------------------------------
# Tab 3 - Enkripsi & Dekripsi
# ----------------------------------------------------------------------------
with tab_enc:
    if not is_bij:
        st.error("Enkripsi dinonaktifkan karena S-Box tidak bijektif.")
    else:
        mandrill = find_mandrill()
        options = []
        if mandrill is not None:
            options.append("Citra Mandrill (sample_images)")
        options += ["Citra uji sintetis bawaan", "Unggah citra sendiri"]
        src = st.radio("Sumber citra", options, horizontal=True, key="img_src")
        if mandrill is None:
            st.caption("Citra Mandrill belum ditemukan. Simpan sebagai `sample_images/mandrill.png` "
                       "agar pilihan Mandrill muncul (lihat panduan).")

        pil_img = None
        label = ""
        try:
            if src.startswith("Citra Mandrill"):
                pil_img, label = Image.open(mandrill), mandrill.name
            elif src.startswith("Citra uji"):
                pil_img, label = Image.open(SAMPLE_DIR / "uji_sintetis_512.png"), "uji_sintetis_512.png"
            else:
                up = st.file_uploader("Unggah citra (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                                      key="img_upload")
                if up is not None:
                    pil_img, label = Image.open(BytesIO(up.getvalue())), up.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Citra tidak dapat dibuka: {exc}")

        if pil_img is not None:
            orig_size = pil_img.size
            arr = ic.pil_to_array(pil_img, max_side=int(max_side))
            if arr.shape[1] != orig_size[0] or arr.shape[0] != orig_size[1]:
                st.info(f"Citra diperkecil dari {orig_size[0]}x{orig_size[1]} menjadi "
                        f"{arr.shape[1]}x{arr.shape[0]} piksel (batas sisi {max_side}).")
            mode_txt = "abu-abu" if arr.ndim == 2 else "berwarna (RGB)"
            st.caption(f"Citra: {label} | {arr.shape[1]}x{arr.shape[0]} piksel | {mode_txt}")

            if not key_text:
                st.warning("Isi kunci enkripsi di sidebar terlebih dahulu.")
            go = st.button("Enkripsi lalu dekripsi", type="primary", disabled=not key_text, key="btn_run")
            cur_sig = sig_of(sbox_t, key_text, arr)

            if go:
                with st.spinner("Memproses enkripsi dan dekripsi..."):
                    t0 = time.time()
                    stages = ic.encrypt_image(arr, sbox, key_text, return_stages=True)
                    t_enc = time.time() - t0
                    t0 = time.time()
                    dec = ic.decrypt_image(stages["cipher"], sbox, key_text)
                    t_dec = time.time() - t0
                st.session_state["result"] = {
                    "sig": cur_sig, "orig": arr, "stages": stages, "cipher": stages["cipher"],
                    "decrypted": dec, "t_enc": t_enc, "t_dec": t_dec, "label": label,
                    "key": key_text, "sens": None,
                }

            res = st.session_state.get("result")
            if res is not None:
                if res["sig"] != cur_sig:
                    st.warning("Pengaturan (S-Box, kunci, atau citra) berubah sejak hasil ini dibuat. "
                               "Tekan tombol di atas untuk memperbarui.")
                ok = np.array_equal(res["orig"], res["decrypted"])
                if ok:
                    st.success(f"Dekripsi BERHASIL: citra hasil dekripsi identik 100% dengan citra asli "
                               f"(enkripsi {res['t_enc']:.2f} detik, dekripsi {res['t_dec']:.2f} detik).")
                else:
                    st.error("Citra hasil dekripsi berbeda dari citra asli.")
                ca, cb, cc = st.columns(3)
                ca.image(res["orig"], caption="Citra asli", clamp=True)
                cb.image(res["cipher"], caption="Citra terenkripsi", clamp=True)
                cc.image(res["decrypted"], caption="Hasil dekripsi", clamp=True)

                with st.expander("Lihat tahapan enkripsi (substitusi, permutasi, difusi)"):
                    s1, s2, s3, s4 = st.columns(4)
                    s1.image(res["stages"]["substitution"], caption="1. Substitusi S-Box", clamp=True)
                    s2.image(res["stages"]["permutation"], caption="2. Permutasi berbasis kunci", clamp=True)
                    s3.image(res["stages"]["diffusion_forward"], caption="3. Difusi maju", clamp=True)
                    s4.image(res["stages"]["cipher"], caption="4. Difusi mundur (hasil akhir)", clamp=True)
                    st.caption("Substitusi saja belum menyembunyikan pola citra; permutasi dan difusi "
                               "yang membuat hasil akhir tampak acak.")

                if st.checkbox("Demo: dekripsi dengan kunci yang SALAH", key="chk_wrong"):
                    wrong = ic.decrypt_image(res["cipher"], sbox, res["key"] + "x")
                    st.image(wrong, caption="Dekripsi dengan kunci salah (tetap berupa derau)", clamp=True)

                st.markdown("**Unduh citra** (format PNG agar tanpa kehilangan data)")
                u1, u2, u3 = st.columns(3)
                u1.download_button("Citra asli (PNG)", ic.array_to_png_bytes(res["orig"]), "citra_asli.png",
                                   "image/png", key="dl_orig")
                u2.download_button("Citra terenkripsi (PNG)", ic.array_to_png_bytes(res["cipher"]),
                                   "citra_terenkripsi.png", "image/png", key="dl_cipher")
                u3.download_button("Hasil dekripsi (PNG)", ic.array_to_png_bytes(res["decrypted"]),
                                   "citra_hasil_dekripsi.png", "image/png", key="dl_dec")

    st.markdown("---")
    st.subheader("Dekripsi dari berkas citra terenkripsi")
    st.caption("Unggah PNG terenkripsi yang diunduh dari aplikasi ini. S-Box aktif dan kunci di sidebar "
               "harus sama dengan saat enkripsi.")
    if is_bij:
        up_c = st.file_uploader("Citra terenkripsi (PNG)", type=["png"], key="cipher_upload")
        if up_c is not None:
            try:
                c_arr = ic.pil_to_array(Image.open(BytesIO(up_c.getvalue())))
                if st.button("Dekripsi berkas ini", key="btn_dec_file", disabled=not key_text):
                    d_arr = ic.decrypt_image(c_arr, sbox, key_text)
                    st.image(d_arr, caption="Hasil dekripsi berkas", clamp=True)
                    st.download_button("Unduh hasil dekripsi (PNG)", ic.array_to_png_bytes(d_arr),
                                       "hasil_dekripsi_berkas.png", "image/png", key="dl_dec_file")
                    st.caption("Jika S-Box atau kunci berbeda dari saat enkripsi, hasilnya berupa derau.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Gagal memproses berkas: {exc}")

# ----------------------------------------------------------------------------
# Tab 4 - Analisis citra
# ----------------------------------------------------------------------------
with tab_ana:
    res = st.session_state.get("result")
    if res is None:
        st.info("Jalankan enkripsi pada tab 3 terlebih dahulu.")
    else:
        if st.button("Jalankan uji sensitivitas NPCR/UACI", key="btn_sens",
                     help="Mengubah 1 bit pada satu byte citra asli, mengenkripsi ulang, lalu membandingkan "
                          "dua citra sandi. Diulang sesuai jumlah uji di sidebar."):
            prog = st.progress(0.0, text="Menjalankan uji sensitivitas...")
            res["sens"] = im.plaintext_sensitivity(
                res["orig"], sbox, res["key"], cipher_ref=res["cipher"], n_trials=int(n_trials),
                progress_cb=lambda i, n: prog.progress(i / n, text=f"Uji {i} dari {n}"))
            prog.empty()
            st.session_state["result"] = res

        rep = im.build_image_report(res["orig"], res["cipher"], res["decrypted"], res.get("sens"))
        ent = rep["entropy"]
        k = st.columns(4)
        k[0].metric("Entropy citra asli", f"{ent['asli']:.4f}")
        k[1].metric("Entropy citra sandi", f"{ent['terenkripsi']:.4f}", delta=f"{ent['terenkripsi'] - 8:.4f} dari 8",
                    delta_color="off")
        sens = res.get("sens")
        k[2].metric("NPCR (uji sensitivitas)", f"{sens['npcr_mean']:.4f} %" if sens else "belum diuji")
        k[3].metric("UACI (uji sensitivitas)", f"{sens['uaci_mean']:.4f} %" if sens else "belum diuji")
        st.caption(f"Nilai harapan citra acak ideal 8-bit: entropy mendekati 8, NPCR sekitar {im.NPCR_EXPECTED:.4f}%, "
                   f"UACI sekitar {im.UACI_EXPECTED:.4f}%.")

        st.subheader("Histogram")
        hist_fig = viz.fig_histograms({"Citra asli": res["orig"], "Citra terenkripsi": res["cipher"],
                                       "Hasil dekripsi": res["decrypted"]})
        st.pyplot(hist_fig)
        st.caption("Histogram citra terenkripsi yang baik tampak datar/merata, berbeda dari citra asli.")

        st.subheader("Tabel hasil")
        rows_img = im.report_rows(rep)
        st.dataframe(pd.DataFrame(rows_img), hide_index=True)

        d1, d2, d3 = st.columns(3)
        d1.download_button("Hasil analisis (CSV)", to_csv_bytes(rows_img), "hasil_analisis_citra.csv", "text/csv",
                           key="dl_img_csv")
        d2.download_button("Hasil analisis (JSON)", to_json_bytes(rep), "hasil_analisis_citra.json",
                           "application/json", key="dl_img_json")
        d3.download_button("Gambar histogram (PNG)", viz.fig_to_png_bytes(hist_fig), "histogram.png", "image/png",
                           key="dl_hist")
        st.warning("**MBCR** tidak dihitung karena definisinya belum jelas pada catatan tugas "
                   "(tidak dikarang). Mohon konfirmasi ke dosen.")

# ----------------------------------------------------------------------------
# Tab 5 - Catatan
# ----------------------------------------------------------------------------
with tab_note:
    st.subheader("Hal yang perlu dikonfirmasi ke dosen")
    st.markdown(
        "- **MBCR**: istilah ini tidak memiliki definisi baku yang bisa dipastikan dari catatan tugas, "
        "jadi rumusnya tidak dibuat-buat dan tidak dihitung di aplikasi. Tanyakan rumus atau rujukan yang dimaksud.\n"
        "- **\"Distogram\"**: kemungkinan besar salah ketik dari **histogram** (distribusi nilai intensitas piksel). "
        "Aplikasi menampilkan histogram. Mohon dipastikan ke dosen."
    )
    st.subheader("Keterbatasan GAN untuk membuat S-Box")
    st.markdown(
        "- GAN membutuhkan **data latih** (contoh S-Box). Di proyek ini data dibuat dari permutasi acak yang "
        "diperbaiki sedikit, lalu diperbanyak dengan transformasi afin; kualitasnya membatasi apa yang bisa ditiru.\n"
        "- S-Box adalah **permutasi diskret**. Agar bisa dilatih dengan gradien, dipakai relaksasi (soft-sort) dan "
        "custom loss untuk NL, DAP, SAC, dan bijektivitas.\n"
        "- Nonlinearity 112 (setara S-Box AES) sangat sulit dicapai tanpa struktur aljabar. Hasil GAN umumnya "
        "lebih rendah dari 112. **Angka NL pada aplikasi adalah hasil hitung sebenarnya.**\n"
        "- Hasil pelatihan bergantung pada seed, jumlah langkah, dan bobot loss; hasil tiap pelatihan bisa berbeda."
    )
    st.subheader("Catatan keamanan")
    st.markdown(
        "Skema enkripsi citra ini dibuat untuk **tujuan belajar/tugas kuliah** dan belum dianalisis secara "
        "kriptografis formal. Jangan dipakai untuk melindungi data sensitif sesungguhnya. Enkripsi bersifat "
        "deterministik: citra dan kunci yang sama menghasilkan citra sandi yang sama."
    )
    st.subheader("Alur enkripsi")
    st.markdown(
        "1. **Substitusi**: tiap byte diganti lewat S-Box.\n"
        "2. **Permutasi**: posisi byte diacak berdasarkan kunci.\n"
        "3. **Difusi maju dan mundur**: tiap byte bergantung pada byte sebelumnya/sesudahnya dan aliran kunci, "
        "sehingga perubahan kecil menyebar ke seluruh citra.\n"
        "Dekripsi menjalankan langkah yang sama secara terbalik menggunakan S-Box invers."
    )
