#!/usr/bin/env python3
"""
program/inti/inti_06_evaluasi_kelayakan_resolusi.py
==============================================
Langkah 06 (Bab 8 langkah 4, 6, 7, 8; luaran khusus 9-12): evaluasi kelayakan resolusi.

Masukan per batuan (folder hasil/<backend>/<batuan>/):
  * karakteristik_leher_pori.json (Langkah 03) -> d_t leher penentu P16/P11 (per arah; utama = terkecil),
                                                 pembanding: granulometri r50 (P2) dan median skeleton
  * rentang_segmentasi_acuan.json (Langkah 02) -> setengah lebar rentang Delta_seg (geo dan per arah)
  * tabel_multiresolusi.csv (Langkah 05)        -> phi_tot, phi_th, Sv tiap tingkat dan cara
  * simulasi_multiresolusi/*.json (Langkah 01/05) -> k tiga arah, konvergensi, diagnostik (tau_u)

Rumus:
  * Kelayakan (P2 nilai_kelayakan): nisbah = d_t/dx, r_voxel = nisbah/2, galat_r = 0,5/r_voxel,
    galat_k = 2 galat_r. Nyquist: nisbah < 2. Patokan Pertemuan 2: 4 sampai 10 voxel per leher pori.
  * Simpangan nisbi (Bab 8 langkah 6): Delta_k = (k_t - k_T0)/k_T0 x 100 %, per arah dan rerata geometrik
    (P11: rerata geometrik sebagai satu angka pengganti).
  * Tingkat kritis (Bab 8 langkah 7): tingkat TERKASAR yang simpangannya masih di bawah garis Delta_seg,
    ditelusuri berurutan T1, T2, T3 dan berhenti pada kegagalan pertama.
  * Medan pandang (Bab 8 langkah 8): pada cacah voxel detektor tetap, FOV sebanding dx, sehingga
    lipatan per arah = dx_kritis / 2,25 um dan lipatan volume = (dx_kritis / 2,25 um)^3.
"""

import os
import sys
import csv
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from program.katalog_batuan import dapatkan_subdirektori_batuan

DEFAULT_OUT_DIR = os.path.join(ROOT, "hasil")
DX0 = 2.25
TINGKAT = [(0, "T0", 256, 2.25, ["resolusi_penuh"]),
           (1, "T1", 128, 4.50, ["majority", "block_avg"]),
           (2, "T2", 64, 9.00, ["majority", "block_avg"]),
           (3, "T3", 32, 18.00, ["majority", "block_avg"])]
CARA = ["majority", "block_avg"]
NAMA_CARA = {"resolusi_penuh": "Resolusi penuh", "majority": "Suara terbanyak", "block_avg": "Rerata blok"}
ARAH = ["x", "y", "z"]
BATAS_NYQUIST = 2.0          # P2: d_t/dx < 2 -> di bawah batas Nyquist


def nilai_kelayakan(d_t_um, dx_um):
    """Salinan langsung P2 sel 30 (nilai_kelayakan)."""
    nisbah = d_t_um / dx_um
    r_voxel = nisbah / 2.0
    galat_r = 0.5 / r_voxel if r_voxel > 0 else float("inf")
    galat_k = 2.0 * galat_r
    if nisbah < 2:
        putusan = "TIDAK LAYAK (di bawah batas Nyquist)"
    elif nisbah < 4:
        putusan = "LAYAK DENGAN CATATAN (di bawah kaidah 4 voxel)"
    elif nisbah < 8:
        putusan = "LAYAK"
    else:
        putusan = "LAYAK DAN BAIK (8-10 voxel)"
    return nisbah, galat_r, galat_k, putusan


def geo(nilai):
    return float(np.exp(np.mean(np.log(nilai)))) if all(v > 0 for v in nilai) else 0.0


def baca_json(jalur):
    with open(jalur, encoding="utf-8") as f:
        return json.load(f)


def wajib(jalur, langkah):
    if not os.path.exists(jalur):
        raise FileNotFoundError(f"{os.path.basename(jalur)} tidak ada; jalankan Langkah {langkah} lebih dahulu "
                                f"({jalur})")
    return jalur


def muat_simulasi(sim_dir, kunci, kode, cara):
    k, konv, tau = {}, {}, {}
    for a in ARAH:
        j = os.path.join(sim_dir, f"{kunci}_{kode}_{cara}_{a}.json")
        if not os.path.exists(j):
            k[a], konv[a], tau[a] = float("nan"), False, float("nan")
            continue
        d = baca_json(j)
        k[a] = float(d.get("K_mD", 0.0))
        konv[a] = bool(d.get("konvergen", False))
        tau[a] = float(d.get("diagnostik", {}).get("tau_u", float("nan")))
    return k, konv, tau


def jalankan(rocks=None, out_dir=DEFAULT_OUT_DIR, **_):
    rocks = rocks or ["Bentheimer", "Parker"]
    print("=" * 82)
    print("LANGKAH 06: EVALUASI KELAYAKAN RESOLUSI (NYQUIST, SIMPANGAN, TINGKAT KRITIS, MEDAN PANDANG)")
    print(f"  * Sampel : {', '.join(rocks)}")
    print("=" * 82, flush=True)

    rekap_semua, kritis_semua = {}, []
    for kunci in rocks:
        dirb = dapatkan_subdirektori_batuan(out_dir, kunci)
        sim_dir = os.path.join(dirb, "simulasi_multiresolusi")
        leher = baca_json(wajib(os.path.join(dirb, "karakteristik_leher_pori.json"), "03"))
        rentang = baca_json(wajib(os.path.join(dirb, "rentang_segmentasi_acuan.json"), "02"))
        if "dt_um" not in leher:
            raise ValueError(f"karakteristik_leher_pori.json {kunci} belum memuat leher penentu P16; jalankan ulang Langkah 03")
        d_t = float(leher["dt_um"])
        d_c_arah = leher["leher_penentu"]["d_c_um"]
        d_granulo = float(leher.get("dt_granulometri_um", float("nan")))
        d_c = float(leher.get("dc_skeleton_median_um", float("nan")))
        dseg = float(rentang["setengah_lebar_persen"])
        dseg_arah = rentang.get("setengah_lebar_arah_persen", {a: dseg for a in ARAH})

        multi = {}
        with open(wajib(os.path.join(dirb, "tabel_multiresolusi.csv"), "05"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["batuan"] == kunci:
                    multi[(int(r["tingkat"]), r["cara"])] = r

        # --- tabel 9: nisbah voxel per leher pori -------------------------------------------
        tabel_kelayakan = []
        for t, kode, N, dx, _ in TINGKAT:
            nis, gr, gk, put = nilai_kelayakan(d_t, dx)
            tabel_kelayakan.append({
                "batuan": kunci, "tingkat": kode,
                "N": int(next((r["N"] for (tt, _), r in multi.items() if tt == t), N)),   # dari langkah 05
                "dx_um": dx,
                "dt_per_dx": nis,
                "dt_per_dx_x": d_c_arah["x"] / dx, "dt_per_dx_y": d_c_arah["y"] / dx, "dt_per_dx_z": d_c_arah["z"] / dx,
                "dskeleton_per_dx": d_c / dx, "dgranulometri_per_dx": d_granulo / dx,
                "galat_r_persen": 100 * gr, "galat_k_persen": 100 * gk,
                "memenuhi_4_10_voxel": nis >= 4.0,
                "di_bawah_nyquist": nis < 2.0, "putusan": put,
            })

        # --- tabel 10: dua cara penurunan resolusi ----------------------------------------------
        k0, konv0, tau0 = muat_simulasi(sim_dir, kunci, "T0", "resolusi_penuh")
        k0_geo = geo([k0[a] for a in ARAH])
        tabel_cara, tabel_simp = [], []
        for t, kode, N, dx, daftar in TINGKAT:
            k_cara = {}
            for cara in daftar:
                k, konv, tau = muat_simulasi(sim_dir, kunci, kode, cara)
                kg = geo([k[a] for a in ARAH])
                k_cara[cara] = kg
                m = multi.get((t, cara), {})
                tabel_cara.append({
                    "batuan": kunci, "tingkat": kode, "dx_um": dx, "cara": cara,
                    "phi_tot": float(m.get("phi_tot", "nan")), "phi_th": float(m.get("phi_th", "nan")),
                    "Sv_um1": float(m.get("Sv_um1", "nan")),
                    # bagian ruang pori yang tidak ikut menembus (kantong terputus), arah x
                    "fraksi_pori_terputus_persen": (1 - float(m.get("phi_th", "nan")) / float(m.get("phi_tot", "nan"))) * 100
                    if float(m.get("phi_tot", 0) or 0) > 0 else float("nan"),
                    "kx_mD": k["x"], "ky_mD": k["y"], "kz_mD": k["z"], "k_geo_mD": kg,
                    "tau_u_rerata": float(np.nanmean([tau[a] for a in ARAH])) if any(
                        np.isfinite(tau[a]) for a in ARAH) else float("nan"),
                    "konvergen_xyz": all(konv.values()),
                    "selisih_k_antar_cara_persen": float("nan"),
                })
                if t > 0:
                    simp = {a: (k[a] / k0[a] - 1) * 100 if k0[a] > 0 and np.isfinite(k[a]) else float("nan")
                            for a in ARAH}
                    simp_geo = (kg / k0_geo - 1) * 100 if k0_geo > 0 else float("nan")
                    tabel_simp.append({
                        "batuan": kunci, "tingkat": kode, "cara": cara, "dx_um": dx,
                        "dt_per_dx": d_t / dx,
                        "delta_kx_persen": simp["x"], "delta_ky_persen": simp["y"],
                        "delta_kz_persen": simp["z"], "delta_kgeo_persen": simp_geo,
                        "dalam_pita_geo": bool(abs(simp_geo) <= dseg) if np.isfinite(simp_geo) else False,
                        "dalam_pita_x": bool(abs(simp["x"]) <= dseg_arah["x"]) if np.isfinite(simp["x"]) else False,
                        "dalam_pita_y": bool(abs(simp["y"]) <= dseg_arah["y"]) if np.isfinite(simp["y"]) else False,
                        "dalam_pita_z": bool(abs(simp["z"]) <= dseg_arah["z"]) if np.isfinite(simp["z"]) else False,
                        "konvergen_xyz": all(konv.values()),
                    })
            if t > 0 and k_cara.get("majority", 0) > 0:
                selisih = (k_cara["block_avg"] / k_cara["majority"] - 1) * 100
                for r in tabel_cara[-2:]:
                    r["selisih_k_antar_cara_persen"] = selisih

        # --- tingkat kritis dan medan pandang ----------------------------------------------------
        kritis_batu = []
        for cara in CARA:
            baris_cara = [r for r in tabel_simp if r["cara"] == cara]
            kode_k, dx_k, dev_k = "T0", DX0, 0.0
            for r in baris_cara:                      # berurutan T1, T2, T3; berhenti pada kegagalan pertama
                if not r["dalam_pita_geo"]:
                    break
                kode_k, dx_k, dev_k = r["tingkat"], r["dx_um"], r["delta_kgeo_persen"]
            berikut = next((r for r in baris_cara if r["dx_um"] > dx_k), None)
            lipat = dx_k / DX0
            pernyataan = (f"{kunci} ({NAMA_CARA[cara].lower()}): citra dapat dikasarkan sampai {kode_k}, "
                          f"dx = {dx_k:.2f} um (simpangan k_geo {dev_k:+.1f}% terhadap pita +/-{dseg:.1f}%), "
                          f"sehingga medan pandang dapat diperluas {lipat:.0f} kali per arah "
                          f"({lipat**3:.0f} kali volume) pada cacah voxel detektor tetap.")
            if berikut:
                pernyataan += (f" Pada tingkat berikutnya {berikut['tingkat']} (dx = {berikut['dx_um']:.2f} um) "
                               f"simpangannya {berikut['delta_kgeo_persen']:+.1f}%.")
            belum = [r["tingkat"] for r in baris_cara if not r["konvergen_xyz"] and r["dx_um"] <= (berikut or {}).get("dx_um", 99)]
            if belum:
                pernyataan += f" Catatan: simulasi {', '.join(belum)} belum konvergen penuh."
            # Kriteria gabungan Bab 8 + Nyquist (P2): tingkat juga harus memenuhi d_t/dx >= 2;
            # berhenti pada kegagalan pertama salah satu syarat. Jawaban Bab 8 di atas tetap dilaporkan apa adanya.
            kode_n, dx_n, dev_n = "T0", DX0, 0.0
            for r in baris_cara:
                if not (r["dalam_pita_geo"] and r["dt_per_dx"] >= BATAS_NYQUIST):
                    break
                kode_n, dx_n, dev_n = r["tingkat"], r["dx_um"], r["delta_kgeo_persen"]
            lipat_n = dx_n / DX0
            if kode_n != kode_k:
                pernyataan += (f" Dengan syarat Nyquist (d_t/dx >= {BATAS_NYQUIST:g}) batasnya {kode_n}, dx = {dx_n:.2f} um "
                               f"(d_t/dx = {d_t / dx_n:.2f}), medan pandang {lipat_n:.0f} kali per arah ({lipat_n**3:.0f} kali volume).")
            rk = {
                "batuan": kunci, "cara": cara, "setengah_lebar_segmentasi_persen": dseg,
                "tingkat_kritis": kode_k, "dx_kritis_um": dx_k, "dt_per_dx_kritis": d_t / dx_k,
                "simpangan_kgeo_kritis_persen": dev_k,
                "tingkat_berikutnya": berikut["tingkat"] if berikut else "",
                "simpangan_kgeo_berikutnya_persen": berikut["delta_kgeo_persen"] if berikut else float("nan"),
                "lipatan_fov_per_arah": lipat, "lipatan_fov_volume": lipat ** 3,
                "tingkat_kritis_nyquist": kode_n, "dx_kritis_nyquist_um": dx_n, "dt_per_dx_kritis_nyquist": d_t / dx_n,
                "simpangan_kgeo_kritis_nyquist_persen": dev_n,
                "lipatan_fov_nyquist_per_arah": lipat_n, "lipatan_fov_nyquist_volume": lipat_n ** 3,
                "pernyataan": pernyataan,
            }
            kritis_batu.append(rk)
            print(f"    * {pernyataan}", flush=True)
        kritis_semua.extend(kritis_batu)

        rekap = {"batuan": kunci, "dt_um": d_t, "dt_takrif": leher.get("takrif_dt", ""),
                 "dc_penentu_arah_um": d_c_arah, "dt_granulometri_um": d_granulo, "dc_skeleton_median_um": d_c,
                 "setengah_lebar_segmentasi_persen": dseg, "setengah_lebar_arah_persen": dseg_arah,
                 "k_T0_mD": k0, "k_T0_geo_mD": k0_geo,
                 "tabel_kelayakan": tabel_kelayakan, "tabel_dua_cara": tabel_cara,
                 "tabel_simpangan": tabel_simp, "tingkat_kritis": kritis_batu}
        rekap_semua[kunci] = rekap

        cetak_tabel(kunci, d_t, tabel_kelayakan, tabel_cara, tabel_simp, dseg)
        tulis_csv(os.path.join(dirb, "tabel_evaluasi_kelayakan_resolusi.csv"), tabel_kelayakan)
        tulis_csv(os.path.join(dirb, "tabel_dua_cara_penurunan.csv"), tabel_cara)
        tulis_csv(os.path.join(dirb, "tabel_simpangan_permeabilitas.csv"), tabel_simp)
        tulis_csv(os.path.join(dirb, "tabel_tingkat_kritis_fov.csv"), kritis_batu)
        with open(os.path.join(dirb, "evaluasi_kelayakan_resolusi.json"), "w", encoding="utf-8") as f:
            json.dump(rekap, f, indent=2)
        with open(os.path.join(dirb, "rekap_tingkat_kritis_fov.json"), "w", encoding="utf-8") as f:
            json.dump(kritis_batu, f, indent=2)
        plot_simpangan({kunci: rekap}, os.path.join(dirb, "grafik", "06b_simpangan_permeabilitas_dan_kritis.png"))
        plot_kelayakan({kunci: rekap}, os.path.join(dirb, "grafik", "06a_kelayakan_resolusi.png"))

    if len(rocks) > 1:
        for nama, kunci_tabel in [("tabel_evaluasi_kelayakan_resolusi.csv", "tabel_kelayakan"),
                                  ("tabel_dua_cara_penurunan.csv", "tabel_dua_cara"),
                                  ("tabel_simpangan_permeabilitas.csv", "tabel_simpangan")]:
            tulis_csv(os.path.join(out_dir, nama), [r for k in rocks for r in rekap_semua[k][kunci_tabel]])
        tulis_csv(os.path.join(out_dir, "tabel_tingkat_kritis_fov.csv"), kritis_semua)
        with open(os.path.join(out_dir, "evaluasi_kelayakan_resolusi.json"), "w", encoding="utf-8") as f:
            json.dump(rekap_semua, f, indent=2)
        plot_simpangan(rekap_semua, os.path.join(out_dir, "grafik", "06b_simpangan_permeabilitas_dan_kritis.png"))
        plot_kelayakan(rekap_semua, os.path.join(out_dir, "grafik", "06a_kelayakan_resolusi.png"))

    print("\n" + "=" * 82)
    print("SELESAI LANGKAH 06.")
    print("=" * 82)
    return rekap_semua


def cetak_tabel(kunci, d_t, kel, cara, simp, dseg):
    print(f"\n>>> {kunci}: d_t = {d_t:.2f} um, Delta_seg = +/-{dseg:.2f}%")
    print(f"    {'Tingkat':<7} {'dx(um)':>7} {'d_t/dx':>7} {'galat k':>8}  Putusan")
    for r in kel:
        print(f"    {r['tingkat']:<7} {r['dx_um']:7.2f} {r['dt_per_dx']:7.2f} {r['galat_k_persen']:7.1f}%  {r['putusan']}")
    print(f"    {'Tingkat':<7} {'Cara':<15} {'phi':>7} {'phi_th':>7} {'putus':>6} {'Sv':>7} {'k_geo':>9} {'antar cara':>10}")
    for r in cara:
        s = r["selisih_k_antar_cara_persen"]
        print(f"    {r['tingkat']:<7} {NAMA_CARA[r['cara']]:<15} {r['phi_tot']:7.4f} {r['phi_th']:7.4f} {r['fraksi_pori_terputus_persen']:5.1f}% "
              f"{r['Sv_um1']:7.4f} {r['k_geo_mD']:9.2f} {('' if not np.isfinite(s) else f'{s:+.1f}%'):>10}"
              f"{'' if r['konvergen_xyz'] else '  *belum konvergen'}")


def tulis_csv(jalur, baris):
    if not baris:
        return
    with open(jalur, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)


def plot_simpangan(rekap_semua, out_img):
    """Luaran 11: simpangan nisbi k_geo terhadap dx bersama pita setengah lebar rentang segmentasi.
    Satu panel per batuan; garis tipis = arah x, y, z dari cara yang sama."""
    from program import gaya_grafik as G
    rocks = list(rekap_semua)
    fig, axes = plt.subplots(1, len(rocks), figsize=(5.6 * len(rocks), 4.4), squeeze=False)
    for i, kunci in enumerate(rocks):
        ax = axes[0, i]
        r = rekap_semua[kunci]
        dseg = r["setengah_lebar_segmentasi_persen"]
        G.pita(ax, -dseg, dseg, label=f"rentang segmentasi ±{G.angka(dseg, 1)} %")
        batas_n = r["dt_um"] / BATAS_NYQUIST                       # dx di atas ini: d_t/dx < 2 (P2)
        ax.axvspan(batas_n, 18.0 * 1.25, facecolor="none", edgecolor=G.TINTA_3, hatch="///", lw=0, alpha=0.5,
                   zorder=0, label=f"di bawah Nyquist ($d_t$/Δx < 2, Δx > {G.angka(batas_n, 2)} µm)")
        G.garis_acuan(ax, y=0, gaya="-", warna=G.TINTA_3, lw=0.8)
        for cara in ("majority", "block_avg"):
            sub = [t for t in r["tabel_simpangan"] if t["cara"] == cara]
            x = [DX0] + [t["dx_um"] for t in sub]
            y = [0] + [t["delta_kgeo_persen"] for t in sub]
            for a in ARAH:
                ax.plot(x, [0] + [t[f"delta_k{a}_persen"] for t in sub], "-", color=G.CARA[cara], lw=0.7, alpha=0.35)
            ax.plot(x, y, "-", marker=G.PENANDA_CARA[cara], color=G.CARA[cara], lw=1.8,
                    label=f"{G.NAMA_CARA[cara]} (k geometrik)")
            kr = next((k for k in r["tingkat_kritis"] if k["cara"] == cara), None)
            if kr:
                # lingkaran tebal = jawaban Bab 8 + Nyquist (kesimpulan); putus-putus = Bab 8 saja bila berbeda
                jn = int(kr.get("tingkat_kritis_nyquist", kr["tingkat_kritis"])[1])
                jb = int(kr["tingkat_kritis"][1])
                ax.plot(x[jn], y[jn], "o", ms=14, mfc="none", mec=G.CARA[cara], mew=2.0)
                teks = f"kritis T{jn}" if jn == jb else f"kritis T{jn} (+ Nyquist)"
                G.label_ujung(ax, x[jn], y[jn], teks, dx=10, dy=(11 if cara == "block_avg" else -11),
                              va=("bottom" if cara == "block_avg" else "top"), warna=G.TINTA_2)
                if jb != jn:
                    ax.scatter([x[jb]], [y[jb]], s=230, facecolors="none", edgecolors=G.CARA[cara],
                               linewidths=1.0, linestyle=(0, (2, 2)), zorder=5)
                    G.label_ujung(ax, x[jb], y[jb], f"T{jb}: pita saja", dx=-10, ha="right",
                                  dy=(11 if cara == "block_avg" else -11),
                                  va=("bottom" if cara == "block_avg" else "top"), warna=G.TINTA_3, ukuran=7.5)
        ax.plot([], [], "-", color=G.TINTA_3, lw=0.7, label="arah x, y, z (garis tipis)")
        ax.plot([], [], "o", ms=10, mfc="none", mec=G.TINTA_2, mew=2.0, label="tingkat kritis (pita + Nyquist)")
        if any(k.get("tingkat_kritis_nyquist", k["tingkat_kritis"]) != k["tingkat_kritis"] for k in r["tingkat_kritis"]):
            ax.scatter([], [], s=100, facecolors="none", edgecolors=G.TINTA_3, linewidths=1.0, linestyle=(0, (2, 2)),
                       label="kritis menurut pita saja")
        G.sumbu_dx(ax)
        G.koma(ax, "y")
        ax.set_ylabel("Simpangan nisbi Δk / $k_{T0}$ (%)")
        ax.set_title(f"{kunci}: simpangan permeabilitas terhadap resolusi")
        ax.legend(loc="upper left")
    G.simpan(fig, out_img)


def plot_kelayakan(rekap_semua, out_img):
    """Luaran 9 (grafik pendamping): d_t/dx terhadap patokan 4-10 voxel (P2) dan batas Nyquist."""
    from program import gaya_grafik as G
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    dx = [2.25, 4.5, 9.0, 18.0]
    G.pita(ax, 4, 10, label="patokan 4–10 voxel")
    G.garis_acuan(ax, y=2, label="batas Nyquist (2 voxel)")
    warna = [G.BIRU, G.ORANYE, G.AQUA]
    banyak = len(rekap_semua) > 3
    for i, (kunci, r) in enumerate(rekap_semua.items()):
        y = [t["dt_per_dx"] for t in r["tabel_kelayakan"]]
        if banyak:      # lebih dari 3 batuan: tinta netral + label langsung, bukan warna ke-4 dst.
            ax.plot(dx, y, "-o", color=G.TINTA_3, lw=1.0, ms=3.5)
            G.label_ujung(ax, dx[0], y[0], kunci, dx=-6, ha="right", ukuran=7)
        else:
            ax.plot(dx, y, "-", marker="osD"[i], color=warna[i], label=f"{kunci} ($d_t$ = {G.angka(r['dt_um'], 1)} µm)")
    G.sumbu_dx(ax)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: G.angka(v)))
    ax.set_ylabel("Voxel per leher pori $d_t$ / Δx")
    ax.set_title("Kecukupan resolusi terhadap leher pori penentu")
    ax.legend(loc="lower left")
    G.simpan(fig, out_img)
