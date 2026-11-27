# -*- coding: utf-8 -*-
"""
Pengujian dasar proyek (tanpa pytest pun bisa dijalankan):

    python tests/test_core.py

Jika pytest terpasang:  python -m pytest tests -q
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import image_crypto as ic          # noqa: E402
from core import image_metrics as im         # noqa: E402
from core import sbox_dataset as ds          # noqa: E402
from core import sbox_io, sbox_metrics as sm  # noqa: E402


def _rand_sbox(seed=0):
    return np.random.default_rng(seed).permutation(256)


# ---------------------------------------------------------------- S-Box
def test_identity_properties():
    r = sm.evaluate_sbox(np.arange(256))
    assert r["bijective"]["is_bijective"]
    assert r["nonlinearity"]["min"] == 0          # fungsi linear -> NL 0
    assert r["dap"]["differential_uniformity"] == 256
    assert np.allclose(np.diag(np.array(r["sac"]["matrix"])), 1.0)


def test_non_bijective_detected():
    s = _rand_sbox(1).copy()
    s[5] = s[6]
    r = sm.check_bijective(s)
    assert not r["is_bijective"] and r["unique_values"] == 255
    try:
        sbox_io.inverse_sbox(s)
        raise AssertionError("seharusnya ValueError")
    except ValueError:
        pass


def test_nl_matches_bruteforce():
    s = _rand_sbox(2)
    x = np.arange(256)
    par = sm.PARITY
    nl_fast = sm.nonlinearity_per_mask(sm.walsh_spectrum(s))
    for b in (1, 7, 100, 255):                    # beberapa mask diuji manual
        f = par[b & s]
        best = 256
        for a in range(256):
            lin = par[a & x]
            best = min(best, int(np.sum(f != lin)), int(np.sum(f == lin)))
        assert best == nl_fast[b - 1], (b, best, nl_fast[b - 1])


def test_ddt_row_sums_and_dap():
    s = _rand_sbox(3)
    ddt = sm.difference_distribution_table(s)
    assert np.all(ddt.sum(axis=1) == 256)
    assert np.all(ddt[1:] % 2 == 0)               # permutasi -> entri genap
    r = sm.evaluate_sbox(s)
    assert abs(r["dap"]["max"] - ddt[1:].max() / 256) < 1e-12


def test_lap_formula_bruteforce():
    s = _rand_sbox(4)
    x = np.arange(256)
    par = sm.PARITY
    best = 0.0
    for b in range(1, 256):
        for a in range(1, 256):
            cnt = int(np.sum(par[a & x] == par[b & s]))
            best = max(best, abs(cnt - 128) / 256)
    assert abs(best - sm.evaluate_sbox(s)["lap"]["max"]) < 1e-12


def test_sac_bruteforce():
    s = _rand_sbox(5)
    sac = sm.sac_matrix(s)
    for i in (0, 3, 7):
        for j in (0, 4, 7):
            cnt = sum(((s[x] ^ s[x ^ (1 << i)]) >> j) & 1 for x in range(256))
            assert abs(cnt / 256 - sac[i, j]) < 1e-12


def test_affine_equivalence_preserves_metrics():
    rng = np.random.default_rng(6)
    s = ds.hill_climb(ds.random_sbox(rng), 500, rng)
    a = ds.affine_equivalent(s, rng)
    q1, q2 = sm.quick_evaluate(s), sm.quick_evaluate(a)
    assert (q1["nl"], q1["du"], q1["lap"]) == (q2["nl"], q2["du"], q2["lap"])


def test_default_sbox_loads_and_is_bijective():
    s, meta = sbox_io.load_default_sbox()
    assert sbox_io.is_bijective(s)
    assert "is_gan_generated" in meta


def test_sbox_json_csv_roundtrip(tmp_path=None):
    import tempfile
    s = _rand_sbox(7)
    d = Path(tempfile.mkdtemp())
    sbox_io.save_sbox_json(d / "t.json", s, {"source": "uji"})
    s2, meta = sbox_io.load_sbox_file(d / "t.json")
    assert np.array_equal(s, s2) and meta["source"] == "uji"
    s3 = sbox_io.parse_sbox_csv(sbox_io.sbox_to_csv_text(s))
    assert np.array_equal(s, s3)


# ---------------------------------------------------------------- Citra
def _test_image(h=96, w=96, rgb=True):
    y, x = np.mgrid[0:h, 0:w]
    if rgb:
        return np.stack([(x * 3) % 256, (y * 3) % 256, (x + y) % 256], -1).astype(np.uint8)
    return ((x * 3 + y) % 256).astype(np.uint8)


def test_roundtrip_rgb_gray_and_default_sbox():
    s, _ = sbox_io.load_default_sbox()
    for rgb in (True, False):
        img = _test_image(rgb=rgb)
        c = ic.encrypt_image(img, s, "kunci-uji")
        assert c.shape == img.shape and c.dtype == np.uint8
        assert not np.array_equal(c, img)
        assert np.array_equal(ic.decrypt_image(c, s, "kunci-uji"), img)


def test_roundtrip_through_png_bytes():
    from io import BytesIO
    from PIL import Image
    s, _ = sbox_io.load_default_sbox()
    img = _test_image()
    c = ic.encrypt_image(img, s, "k1")
    png = ic.array_to_png_bytes(c)
    c2 = ic.pil_to_array(Image.open(BytesIO(png)))
    assert np.array_equal(c, c2)
    assert np.array_equal(ic.decrypt_image(c2, s, "k1"), img)


def test_wrong_key_and_determinism():
    s, _ = sbox_io.load_default_sbox()
    img = _test_image()
    c1 = ic.encrypt_image(img, s, "kunci-A")
    assert np.array_equal(c1, ic.encrypt_image(img, s, "kunci-A"))
    assert not np.array_equal(c1, ic.encrypt_image(img, s, "kunci-B"))
    assert not np.array_equal(ic.decrypt_image(c1, s, "kunci-B"), img)


def test_roundtrip_with_random_sboxes_and_odd_sizes():
    img = _test_image(37, 53)
    for seed in range(3):
        s = _rand_sbox(100 + seed)
        c = ic.encrypt_image(img, s, f"k{seed}")
        assert np.array_equal(ic.decrypt_image(c, s, f"k{seed}"), img)


def test_stages_consistent():
    s, _ = sbox_io.load_default_sbox()
    img = _test_image()
    st = ic.encrypt_image(img, s, "k", return_stages=True)
    assert np.array_equal(st["cipher"], ic.encrypt_image(img, s, "k"))
    assert np.array_equal(st["substitution"], s[img.reshape(-1)].astype(np.uint8).reshape(img.shape))


def test_empty_key_rejected():
    s, _ = sbox_io.load_default_sbox()
    try:
        ic.encrypt_image(_test_image(), s, "")
        raise AssertionError("seharusnya ValueError")
    except ValueError:
        pass


def test_image_metrics_sanity():
    s, _ = sbox_io.load_default_sbox()
    img = _test_image(128, 128)
    c = ic.encrypt_image(img, s, "kunci")
    assert im.shannon_entropy(c) > 7.98 > im.shannon_entropy(img)
    sens = im.plaintext_sensitivity(img, s, "kunci", cipher_ref=c, n_trials=4)
    assert abs(sens["npcr_mean"] - im.NPCR_EXPECTED) < 0.5, sens["npcr_mean"]
    assert abs(sens["uaci_mean"] - im.UACI_EXPECTED) < 0.8, sens["uaci_mean"]
    assert im.npcr(img, img) == 0 and im.uaci(img, img) == 0
    dec = ic.decrypt_image(c, s, "kunci")
    rep = im.build_image_report(img, c, dec, sens)
    assert rep["verifikasi_dekripsi"]["identik_dengan_asli"]
    assert "MBCR" in rep["mbcr"]
    assert len(im.report_rows(rep)) > 8


def test_speed_512_rgb():
    s, _ = sbox_io.load_default_sbox()
    img = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    t = time.time()
    c = ic.encrypt_image(img, s, "kunci")
    t_enc = time.time() - t
    t = time.time()
    d = ic.decrypt_image(c, s, "kunci")
    t_dec = time.time() - t
    assert np.array_equal(d, img)
    print(f"   (512x512 RGB: enkripsi {t_enc:.2f} s, dekripsi {t_dec:.2f} s)")


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
