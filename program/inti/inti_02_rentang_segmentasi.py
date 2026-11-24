#!/usr/bin/env python3
"""
program/inti/inti_02_rentang_segmentasi.py
=====================================
Langkah 02 (Bab 8 langkah 1; luaran wajib 2 dan 7): rentang segmentasi pada resolusi penuh.

Citra yang tersedia sudah biner, sehingga penyapuan ambang aras keabuan tidak dapat dikerjakan
langsung. Pedoman Seksi 5.2 mengizinkan penyapuan setara yang MENGGESER LETAK ANTARMUKA pori-padatan
sebesar sebagian voxel pada kedua arah. Cara yang dipakai di sini:

  1. Medan pori biner dihaluskan dengan tapis Gauss sigma = 0,8 voxel, s(x) = G_sigma * 1[pori].
     Pada antarmuka datar s = Phi(d/sigma), d = jarak bertanda ke antarmuka (positif di dalam pori).
  2. Pori pada geseran delta (voxel, positif = pori melebar) adalah s > T(delta), T = Phi(-delta/sigma).
  3. delta disapu pada 7 nilai -0,5 ... +0,5 voxel. Batas 0,5 voxel adalah ketidakpastian letak tepi
     akibat efek volume parsial yang dipakai P2 (nilai_kelayakan: galat_r = 0,5 / r_voxel).
  4. Pada tiap delta: porositas total, porositas tembus (ketetanggaan 6, P4), luas spesifik
     (cacah muka voxel, P11), dan permeabilitas LBM D3Q19 tiga arah memakai pemecah yang sama
     dengan simulasi batuan (BUKAN proksi).

Keluaran (P13 dan Bab 8 langkah 1):
  setengah lebar rentang nisbi  Delta_seg = (k_maks - k_min) / 2 / median sapuan x 100 %  (P16 Soal 5;
  rerata geometrik dan per arah); k_ref = permeabilitas citra biner asli dari Langkah 01 tetap menjadi acuan
  simpangan resolusi (langkah 06); kepekaan d ln k / d delta,
  dan ketidakpastian baku tipe B u = Delta_seg / sqrt(3) (P13 Contoh Soal 1).
"""

import os
import sys
import csv
import json
import math
import time
import importlib.util

import numpy as np
from scipy import ndimage as ndi
from scipy.special import ndtr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.katalog_batuan import muat_semua_batuan, dapatkan_subdirektori_batuan, dapatkan_batas_langkah_batuan, hasil_sah, tunak_sah

DEFAULT_DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil")
DX_UM = 2.25
DARCY_M2 = 9.869233e-13
SIGMA = 0.8
DELTA_VOXEL = [-0.5, -1.0 / 3.0, -1.0 / 6.0, 0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5]
ARAH = ["x", "y", "z"]
SUMBU = {"x": 2, "y": 1, "z": 0}       # C-order [z, y, x]


def muat_solver(backend):
    jalur = os.path.join(ROOT, "program", backend, f"{backend}_solver_lbm.py")
    spec = importlib.util.spec_from_file_location(f"solver_lbm_{backend}", jalur)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baca_citra(data_dir, berkas, n=256):
    for kand in [os.path.join(data_dir, "sumber", "citra-biner", berkas),
                 os.path.join(data_dir, "citra-biner", berkas),
                 os.path.join(DEFAULT_DATA_DIR, "sumber", "citra-biner", berkas)]:
        if os.path.exists(kand):
            from program.citra import baca_citra_biner
            return baca_citra_biner(kand, n=n)
    raise FileNotFoundError(f"Citra biner {berkas} tidak ditemukan")


def porositas_tembus(pori, sumbu):
    lbl, n = ndi.label(pori, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return 0.0
    a = set(np.unique(np.take(lbl, 0, axis=sumbu))) - {0}
    b = set(np.unique(np.take(lbl, -1, axis=sumbu))) - {0}
    sambung = a & b
    return float(np.isin(lbl, list(sambung)).mean()) if sambung else 0.0


def luas_spesifik_um(pori, dx_um=DX_UM):
    """Cacah muka voxel pori-padatan per volume (P11 luas_spesifik), dalam 1/um."""
    p = pori.astype(np.int8)
    muka = sum(int(np.abs(np.diff(p, axis=a)).sum()) for a in range(3))
    return muka / (pori.size * dx_um)


def k_ref_langkah0(dir_batuan, info_bat):
    """Permeabilitas citra biner asli dari T0 sah pemecah sendiri (Langkah 01, kriteria berlaku).
    Data acuan dosen tidak dipakai sebagai k_ref (pemecah dan kriteria berbeda)."""
    k = {}
    kunci = os.path.basename(os.path.normpath(dir_batuan))
    for a in ARAH:
        j = os.path.join(dir_batuan, "simulasi_multiresolusi", f"{kunci}_T0_resolusi_penuh_{a}.json")
        if os.path.exists(j):
            with open(j, encoding="utf-8") as f:
                d = json.load(f)
            if tunak_sah(d):
                k[a] = float(d["K_mD"])
    if len(k) < 3:
        raise FileNotFoundError(f"T0 sah untuk {kunci} belum lengkap ({sorted(k)}); "
                                f"jalankan Langkah 01 dengan --mode_t0 simulasi lebih dahulu")
    return k, "Langkah 01 (T0 pemecah sendiri, kriteria P7/P11)"


def rerata_geometrik(nilai):
    return float(np.exp(np.mean(np.log(nilai)))) if all(v > 0 for v in nilai) else 0.0


def kasus_tersimpan(jalur, maks):
    if not os.path.exists(jalur):
        return None
    try:
        with open(jalur, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if d.get("sigma_gauss") != SIGMA:
        return None
    if not hasil_sah(d):
        return None
    if d.get("konvergen") is True or d.get("langkah", 0) >= maks or d.get("phi_th", 1.0) == 0.0:
        return d
    return None


def jalankan(rocks=None, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR, backend="gpu",
             periksa=500, delta_voxel=None, verbose=True):
    katalog = muat_semua_batuan(data_dir)
    rocks = rocks or ["Bentheimer", "Parker"]
    delta_voxel = list(delta_voxel or DELTA_VOXEL)
    solver = muat_solver(backend)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 82)
    print(f"LANGKAH 02 [{backend.upper()}]: RENTANG SEGMENTASI PADA RESOLUSI PENUH (LBM TIAP AMBANG)")
    print(f"  * Cara        : geseran antarmuka setara (Pedoman 5.2), Gauss sigma = {SIGMA} voxel")
    print(f"  * Geseran     : {', '.join(f'{d:+.3f}' for d in delta_voxel)} voxel "
          f"({len(delta_voxel)} ambang, batas +/-0,5 voxel dari P2)")
    print(f"  * Sampel      : {', '.join(rocks)}")
    print("=" * 82, flush=True)

    hasil_semua = {}
    for kunci in rocks:
        info = katalog.get(kunci, {})
        dir_batuan = dapatkan_subdirektori_batuan(out_dir, kunci)
        sim_dir = os.path.join(dir_batuan, "rentang_segmentasi")
        os.makedirs(sim_dir, exist_ok=True)

        pori0 = baca_citra(data_dir, info.get("berkas", f"{kunci}_2d25um_binary_256.raw"))
        k_ref, sumber_ref = k_ref_langkah0(dir_batuan, info)
        k_ref_geo = rerata_geometrik([k_ref[a] for a in ARAH])
        halus = ndi.gaussian_filter(pori0.astype(np.float32), sigma=SIGMA)

        print(f"\n>>> {kunci}: k_ref ({sumber_ref}) = kx {k_ref['x']:.2f}, ky {k_ref['y']:.2f}, "
              f"kz {k_ref['z']:.2f}, k_geo {k_ref_geo:.2f} mD", flush=True)

        baris = []
        for j, delta in enumerate(delta_voxel):
            T = float(ndtr(-delta / SIGMA))
            pori = halus > T
            phi = float(pori.mean())
            phi_th = porositas_tembus(pori, SUMBU["x"])
            Sv = luas_spesifik_um(pori)
            k_arah, konv, lang, diag = {}, {}, {}, {}
            for arah in ARAH:
                jalur = os.path.join(sim_dir, f"{kunci}_T0_ambang{j}_{arah}.json")
                maks = dapatkan_batas_langkah_batuan(katalog, kunci, tingkat=0, arah=arah)
                d = kasus_tersimpan(jalur, maks)
                if d is None:
                    d = solver.jalankan_kasus_lbm(pori, kunci, 0, f"ambang{j}", arah, out_file=None,
                                                  maks_langkah=maks, periksa=periksa, verbose=verbose)
                    d.update({"delta_voxel": delta, "delta_um": delta * DX_UM, "ambang_T": T,
                              "sigma_gauss": SIGMA})
                    with open(jalur, "w", encoding="utf-8") as f:
                        json.dump(d, f, indent=2)
                k_arah[arah] = float(d["K_mD"])
                konv[arah] = bool(d.get("konvergen", False))
                lang[arah] = int(d.get("langkah", 0))
                diag[arah] = d.get("diagnostik", {})
            k_geo = rerata_geometrik([k_arah[a] for a in ARAH])
            baris.append({
                "indeks": j, "delta_voxel": delta, "delta_um": delta * DX_UM, "T": T,
                "phi_tot": phi, "phi_th": phi_th, "Sv_um1": Sv,
                "kx_mD": k_arah["x"], "ky_mD": k_arah["y"], "kz_mD": k_arah["z"], "k_mD": k_geo,
                "delta_k_persen": (k_geo / k_ref_geo - 1.0) * 100.0 if k_ref_geo > 0 else float("nan"),
                "tau_u_x": diag["x"].get("tau_u", float("nan")),
                "konvergen_xyz": all(konv.values()), "langkah_xyz": lang,
            })
            print(f"    delta={delta:+.3f} vox (T={T:.3f}) phi={phi:.4f} phi_th={phi_th:.4f} Sv={Sv:.4f} | "
                  f"kx={k_arah['x']:.1f} ky={k_arah['y']:.1f} kz={k_arah['z']:.1f} k_geo={k_geo:.1f} mD"
                  f"{'' if all(konv.values()) else '  [BELUM KONVERGEN]'}", flush=True)

        def setengah_lebar(kunci_k):
            """P16 Soal 5 (notebook): setengah lebar = (maks - min) / 2 / median sapuan x 100 %."""
            ks = [b[kunci_k] for b in baris]
            med = float(np.median(ks))
            return (max(ks) - min(ks)) / (2.0 * med) * 100.0 if med > 0 else float("nan")

        dseg_geo = setengah_lebar("k_mD")
        dseg_arah = {a: setengah_lebar(f"k{a}_mD") for a in ARAH}
        dv = np.array([b["delta_voxel"] for b in baris])
        kg = np.array([b["k_mD"] for b in baris])
        if (kg > 0).all():
            j0 = int(np.argmin(np.abs(dv)))
            kepekaan = float(np.gradient(np.log(kg), dv)[j0])
        else:
            kepekaan = float("nan")

        hasil = {
            "batuan": kunci, "nama": info.get("nama", kunci), "backend": backend,
            "dx_um": DX_UM, "N": int(pori0.shape[0]), "cara": "geseran antarmuka Gauss (Pedoman 5.2)",
            "sigma_gauss_voxel": SIGMA, "delta_voxel": delta_voxel,
            "sumber_permeabilitas_acuan": sumber_ref,
            "kx_ref_mD": k_ref["x"], "ky_ref_mD": k_ref["y"], "kz_ref_mD": k_ref["z"],
            "k_geo_ref_mD": k_ref_geo, "k_ref_nominal_mD": k_ref_geo,
            "k_min_mD": float(kg.min()), "k_max_mD": float(kg.max()), "k_median_sapuan_mD": float(np.median(kg)),
            "rumus_setengah_lebar": "(k_maks - k_min) / 2 / median sapuan x 100 % (P16 Soal 5)",
            "setengah_lebar_persen": dseg_geo,
            "setengah_lebar_arah_persen": dseg_arah,
            "kepekaan_dlnk_ddelta_per_voxel": kepekaan,
            "u_ambang_tipeB_persen": dseg_geo / math.sqrt(3.0),
            "phi_nominal": float(pori0.mean()),
            "phi_th_nominal": porositas_tembus(pori0, SUMBU["x"]),
            "Sv_nominal_um1": luas_spesifik_um(pori0),
            "semua_konvergen": all(b["konvergen_xyz"] for b in baris),
            "sapuan_ambang": baris,
        }
        hasil_semua[kunci] = hasil

        simpan = dict(hasil)
        simpan[kunci] = hasil
        for nama in ("rentang_segmentasi_acuan.json", "basis_acuan.json"):
            with open(os.path.join(dir_batuan, nama), "w", encoding="utf-8") as f:
                json.dump(simpan, f, indent=2)
        tulis_csv(os.path.join(dir_batuan, "tabel_rentang_segmentasi.csv"), baris, kunci)
        plot_rentang({kunci: hasil}, os.path.join(dir_batuan, "grafik", "02_rentang_segmentasi.png"))

        print(f"    [HASIL {kunci}] Delta_seg (k_geo) = +/-{dseg_geo:.2f}% | per arah: "
              + ", ".join(f"{a} +/-{dseg_arah[a]:.2f}%" for a in ARAH)
              + f" | d ln k/d delta = {kepekaan:.2f} /voxel | u_B = {dseg_geo/math.sqrt(3):.2f}%"
              + ("" if hasil["semua_konvergen"] else "  [ADA KASUS BELUM KONVERGEN]"), flush=True)

    if len(rocks) > 1:
        semua = [dict(b, batuan=k) for k, h in hasil_semua.items() for b in h["sapuan_ambang"]]
        tulis_csv(os.path.join(out_dir, "tabel_rentang_segmentasi.csv"), semua, None)
        with open(os.path.join(out_dir, "rentang_segmentasi_acuan.json"), "w", encoding="utf-8") as f:
            json.dump(hasil_semua, f, indent=2)
        plot_rentang(hasil_semua, os.path.join(out_dir, "grafik", "02_rentang_segmentasi.png"))

    print("\n" + "=" * 82)
    print("SELESAI LANGKAH 02: rentang segmentasi resolusi penuh tersimpan.")
    print("=" * 82)
    return hasil_semua


def tulis_csv(jalur, baris, kunci):
    kolom = ["batuan", "indeks", "delta_voxel", "delta_um", "T", "phi_tot", "phi_th", "Sv_um1",
             "kx_mD", "ky_mD", "kz_mD", "k_mD", "delta_k_persen", "tau_u_x", "konvergen_xyz"]
    with open(jalur, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        for b in baris:
            w.writerow(dict(b, batuan=b.get("batuan", kunci)))


def plot_rentang(hasil_semua, out_img):
    """Luaran 2 dan 7: k dan porositas terhadap geseran antarmuka delta (Pedoman 5.2). Dua panel per batuan
    (permeabilitas | porositas), bukan sumbu ganda."""
    from program import gaya_grafik as G
    rocks = list(hasil_semua)
    fig, axes = plt.subplots(len(rocks), 2, figsize=(11, 3.7 * len(rocks)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for i, kunci in enumerate(rocks):
        d = hasil_semua[kunci]
        s_ = d["sapuan_ambang"]
        x = [b["delta_um"] / DX_UM for b in s_]
        kref, dseg = d["k_geo_ref_mD"], d["setengah_lebar_persen"]
        ax = axes[i, 0]
        kg = [b["k_mD"] for b in s_]
        # pita = rentang yang sebenarnya [k_min, k_maks]; Delta_seg = setengah lebarnya / k_ref (Pedoman 5.2).
        # Sebaran k tidak simetris terhadap k_ref, jadi pita simetris k_ref(1 +/- Delta_seg) tidak digambar di sini.
        G.pita(ax, min(kg), max(kg),
               label=f"rentang $k_{{geo}}$ {G.angka(min(kg), 0)}–{G.angka(max(kg), 0)} mD "
                     f"(setengah lebar Δ$_{{seg}}$ = ±{G.angka(dseg, 1)} % dari median)")
        G.garis_acuan(ax, y=kref, label=f"$k_{{ref}}$ citra asli = {G.angka(kref, 1)} mD")
        for a_ in ("x", "y", "z"):
            ax.plot(x, [b[f"k{a_}_mD"] for b in s_], "-", marker=G.PENANDA_ARAH[a_], ms=4, lw=1.0,
                    color=G.ARAH[a_], label=f"$k_{a_}$")
        ax.plot(x, [b["k_mD"] for b in s_], "-D", color=G.TINTA, lw=1.8, ms=5, label="$k_{geo}$")
        ax.set_ylabel("Permeabilitas (mD)")
        ax.set_title(f"{d.get('nama', kunci)}: permeabilitas")
        ax.legend(loc="upper left", fontsize=7.5)
        ax2 = axes[i, 1]
        ax2.plot(x, [b["phi_tot"] for b in s_], "-o", color=G.BIRU, label="φ total")
        ax2.plot(x, [b["phi_th"] for b in s_], "-s", color=G.ORANYE, label="φ tembus")
        ax2.set_ylabel("Porositas")
        ax2.set_title(f"{d.get('nama', kunci)}: porositas")
        ax2.legend(loc="upper left", fontsize=7.5)
        for a2 in (ax, ax2):
            G.koma(a2)
            a2.set_xlabel("Geseran antarmuka δ (voxel, + = pori melebar)")
    G.simpan(fig, out_img)
