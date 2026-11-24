#!/usr/bin/env python3
"""
program/inti/inti_04_volume_elementer.py
===================================
Langkah 04 (luaran wajib 3): volume elementer representatif (REV) melalui uji ukuran domain bertingkat,
dikerjakan pada resolusi penuh (Bab 8 menyatakan REV per tingkat bukan pokok proyek ini).

Aturan Pertemuan 4 ("Dua cara mengambil subvolume", "Koefisien variasi"):
  * subvolume ACAK pada tiap sisi L; dari n nilai dihitung rerata mu, simpangan baku s, dan CV = s / mu;
  * paling sedikit 20 subvolume per ukuran untuk porositas dan luas spesifik (di sini 40),
    paling sedikit 6 subvolume per ukuran untuk permeabilitas (di sini 6);
  * L_REV = sisi terkecil yang sebarannya di bawah 5 persen dari rerata (kesepakatan mata kuliah).
Permeabilitas subvolume dihitung dengan pemecah LBM yang sama (arah x, penyangga fisis 9 um),
porositas tembus dengan ketetanggaan 6, luas spesifik dengan cacah muka voxel (P11).
"""

import os
import sys
import csv
import json
import time

import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.katalog_batuan import muat_semua_batuan, dapatkan_subdirektori_batuan, dapatkan_batas_langkah_batuan, hasil_sah, tunak_sah

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DX_UM = 2.25
SISI_GEOMETRI = [32, 48, 64, 96, 128, 160, 192]
N_GEOMETRI = 40
SISI_ALIRAN = [64, 96, 128]
N_ALIRAN = 6
BATAS_CV = 0.05
BENIH = 2026
SUMBU_X = 2                                   # C-order [z, y, x]


def muat_solver(backend):
    import importlib.util
    jalur = os.path.join(ROOT, "program", backend, f"{backend}_solver_lbm.py")
    spec = importlib.util.spec_from_file_location(f"solver_lbm_{backend}", jalur)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baca_citra(data_dir, berkas, n=256):
    for kand in [os.path.join(data_dir, "sumber", "citra-biner", berkas),
                 os.path.join(data_dir, "citra-biner", berkas)]:
        if os.path.exists(kand):
            from program.citra import baca_citra_biner
            return baca_citra_biner(kand, n=n)
    raise FileNotFoundError(f"Citra biner {berkas} tidak ditemukan")


def porositas_tembus(pori, sumbu=SUMBU_X):
    lbl, n = ndi.label(pori, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return 0.0
    a = set(np.unique(np.take(lbl, 0, axis=sumbu))) - {0}
    b = set(np.unique(np.take(lbl, -1, axis=sumbu))) - {0}
    s = a & b
    return float(np.isin(lbl, list(s)).mean()) if s else 0.0


def luas_spesifik_um(pori):
    p = pori.astype(np.int8)
    return sum(int(np.abs(np.diff(p, axis=a)).sum()) for a in range(3)) / (pori.size * DX_UM)


def asal_acak(rng, L, n, N=256):
    return [tuple(int(v) for v in rng.integers(0, N - L + 1, size=3)) for _ in range(n)]


def statistik(nilai):
    v = np.asarray(nilai, float)
    mu = float(v.mean())
    s = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return mu, s, (s / mu if mu > 0 else float("inf"))


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=None, backend="gpu", periksa=500,
             verbose=False, **_):
    katalog = muat_semua_batuan(data_dir)
    rocks = rocks or ["Bentheimer", "Parker"]
    out_dir = out_dir or os.path.join(ROOT, "hasil", backend)
    solver = muat_solver(backend)

    print("=" * 82)
    print(f"LANGKAH 04 [{backend.upper()}]: VOLUME ELEMENTER REPRESENTATIF (UJI UKURAN BERTINGKAT, P4)")
    print(f"  * Geometri : sisi {SISI_GEOMETRI} voxel, {N_GEOMETRI} subvolume acak per sisi")
    print(f"  * Aliran   : sisi {SISI_ALIRAN} voxel, {N_ALIRAN} subvolume acak per sisi, LBM arah x")
    print(f"  * Kriteria : L_REV = sisi terkecil dengan CV < {BATAS_CV*100:.0f}% (kesepakatan P4)")
    print("=" * 82, flush=True)

    hasil_semua = {}
    for kunci in rocks:
        info = katalog.get(kunci, {})
        dirb = dapatkan_subdirektori_batuan(out_dir, kunci)
        sim_dir = os.path.join(dirb, "volume_elementer")
        os.makedirs(sim_dir, exist_ok=True)
        pori = baca_citra(data_dir, info.get("berkas", f"{kunci}_2d25um_binary_256.raw"))
        rng = np.random.default_rng(BENIH)
        print(f"\n>>> {kunci} (phi seluruh citra = {pori.mean():.4f})", flush=True)

        baris = []
        for L in [s for s in SISI_GEOMETRI if s < pori.shape[0]]:
            phi, phi_th, sv = [], [], []
            for (z, y, x) in asal_acak(rng, L, N_GEOMETRI, N=pori.shape[0]):
                sub = pori[z:z + L, y:y + L, x:x + L]
                phi.append(float(sub.mean()))
                phi_th.append(porositas_tembus(sub))
                sv.append(luas_spesifik_um(sub))
            m_p, s_p, cv_p = statistik(phi)
            m_t, s_t, cv_t = statistik(phi_th)
            m_s, s_s, cv_s = statistik(sv)
            baris.append({"batuan": kunci, "sisi_voxel": L, "sisi_um": L * DX_UM, "n_geometri": N_GEOMETRI,
                          "phi_rerata": m_p, "phi_cv": cv_p, "phi_th_rerata": m_t, "phi_th_cv": cv_t,
                          "Sv_rerata_um1": m_s, "Sv_cv": cv_s,
                          "n_aliran": 0, "k_rerata_mD": float("nan"), "k_cv": float("nan"),
                          "subvolume_menembus": ""})
            print(f"    L={L:>3} ({L*DX_UM:6.1f} um): CV phi {cv_p*100:5.2f}% | CV phi_th {cv_t*100:5.2f}% | "
                  f"CV Sv {cv_s*100:5.2f}%", flush=True)

        rng_k = np.random.default_rng(BENIH + 1)
        for L in [s for s in SISI_ALIRAN if s < pori.shape[0]]:
            ks, tembus = [], 0
            for i, (z, y, x) in enumerate(asal_acak(rng_k, L, N_ALIRAN, N=pori.shape[0])):
                jalur = os.path.join(sim_dir, f"{kunci}_L{L}_s{i}_x.json")
                d = None
                maks = dapatkan_batas_langkah_batuan(katalog, kunci, tingkat=1 if L >= 96 else 2, arah="x")
                if os.path.exists(jalur):
                    with open(jalur, encoding="utf-8") as f:
                        d = json.load(f)
                    # dipakai ulang hanya bila sah DAN (tunak, atau sudah mencapai batas langkah yang berlaku)
                    if d.get("asal") != [z, y, x] or not hasil_sah(d) or not (
                            tunak_sah(d) or int(d.get("langkah", 0)) >= maks or float(d.get("phi_th", 1)) == 0.0):
                        d = None
                if d is None:
                    sub = np.ascontiguousarray(pori[z:z + L, y:y + L, x:x + L])
                    d = solver.jalankan_kasus_lbm(sub, kunci, 0, f"rev_L{L}_s{i}", "x", out_file=None,
                                                  maks_langkah=maks, periksa=periksa, verbose=verbose)
                    d.update({"asal": [z, y, x], "sisi_voxel": L})
                    with open(jalur, "w", encoding="utf-8") as f:
                        json.dump(d, f, indent=2)
                k = float(d.get("K_mD", 0.0))
                tembus += int(k > 0)
                ks.append(k)
            m_k, s_k, cv_k = statistik(ks)
            r = next(b for b in baris if b["sisi_voxel"] == L)
            r.update({"n_aliran": N_ALIRAN, "k_rerata_mD": m_k, "k_cv": cv_k,
                      "subvolume_menembus": f"{tembus} dari {N_ALIRAN}"})
            print(f"    L={L:>3} aliran x: k rerata {m_k:9.2f} mD | CV k {cv_k*100:6.2f}% | menembus {tembus}/{N_ALIRAN}",
                  flush=True)

        def L_rev(kolom):
            for b in baris:
                if np.isfinite(b[kolom]) and b[kolom] < BATAS_CV:
                    return b["sisi_voxel"]
            return None

        rev = {"phi": L_rev("phi_cv"), "phi_th": L_rev("phi_th_cv"), "Sv": L_rev("Sv_cv"), "k": L_rev("k_cv")}
        ringkas = {kk: (f"{v} voxel ({v*DX_UM:.0f} um)" if v else f"belum tercapai sampai {max(SISI_GEOMETRI if kk != 'k' else SISI_ALIRAN)} voxel")
                   for kk, v in rev.items()}
        hasil = {"batuan": kunci, "backend": backend, "benih": BENIH, "batas_cv": BATAS_CV,
                 "L_REV_voxel": rev, "L_REV_teks": ringkas, "tabel": baris}
        hasil_semua[kunci] = hasil
        print("    L_REV: " + " | ".join(f"{k} {v}" for k, v in ringkas.items()), flush=True)

        with open(os.path.join(dirb, "volume_elementer.json"), "w", encoding="utf-8") as f:
            json.dump(hasil, f, indent=2)
        with open(os.path.join(dirb, "tabel_volume_elementer.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
            w.writeheader()
            w.writerows(baris)
        plot_rev({kunci: hasil}, os.path.join(dirb, "grafik", "04_volume_elementer.png"))

    if len(hasil_semua) > 1:
        plot_rev(hasil_semua, os.path.join(out_dir, "grafik", "04_volume_elementer.png"))
    print("\n" + "=" * 82)
    print("SELESAI LANGKAH 04.")
    print("=" * 82)
    return hasil_semua


def plot_rev(hasil_semua, out_img):
    """Luaran 3: koefisien variasi terhadap sisi subvolume (uji ukuran bertingkat P4) dan batas 5 %."""
    from program import gaya_grafik as G
    from matplotlib.ticker import FixedLocator, NullLocator
    rocks = list(hasil_semua)
    fig, axes = plt.subplots(1, len(rocks), figsize=(5.8 * len(rocks), 4.3), squeeze=False)
    for i, kunci in enumerate(rocks):
        ax = axes[0, i]
        t = hasil_semua[kunci]["tabel"]
        for kol, nama, warna, mk in [("phi_cv", "porositas φ", G.BIRU, "o"), ("phi_th_cv", "porositas tembus", G.ORANYE, "s"),
                                     ("Sv_cv", "luas spesifik Sv", G.AQUA, "^"), ("k_cv", "permeabilitas $k_x$", G.TINTA, "D")]:
            x = [b["sisi_voxel"] for b in t if np.isfinite(b[kol])]
            y = [b[kol] * 100 for b in t if np.isfinite(b[kol])]
            if y:
                ax.plot(x, y, "-", marker=mk, color=warna, lw=1.6 if kol != "k_cv" else 1.2,
                        ls="-" if kol != "k_cv" else "--", label=nama)
        G.garis_acuan(ax, y=BATAS_CV * 100, label="batas CV 5 %")
        sisi = [b["sisi_voxel"] for b in t]
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sisi))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{L}\n{G.angka(L * DX_UM, 0)}" for L in sisi])
        ax.set_yscale("log")
        G.log_angka(ax, "y")
        ax.set_xlabel("Sisi subvolume (voxel / µm)")
        ax.set_ylabel("Koefisien variasi (%)")
        ax.set_title(f"{kunci}: volume elementer representatif")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.simpan(fig, out_img)
